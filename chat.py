import asyncio
import json
import os
from collections.abc import Iterable
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionToolUnionParam,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

MCP_SERVER_CONFIG = {
    "command": "uv",
    "args": [
        "run",
        "python",
        "mcp_server.py",
    ],
    "env": None,
}


def convert_mcp_tool_to_function(tool) -> ChatCompletionToolUnionParam:
    """Convert MCP tool schema to OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema.get("properties", {}),
                "required": tool.inputSchema.get("required", []),
            },
        },
    }


class MCPAgent:
    """
    An LLM agent that bridges MCP (Model Context Protocol) tools to OpenRouter's
    function calling API, enabling the LLM to use external tools seamlessly.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        debug: bool = False,
    ):
        """
        Initialize the MCP Agent.

        Args:
            api_key: OpenRouter API key
            model: Model identifier to use
            debug: Enable debug logging
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model
        self.debug = debug
        self.console = Console()
        self.messages = []
        self.available_tools: Iterable[ChatCompletionToolUnionParam] = []

    async def connect_to_mcp_server(self, mcp_server_config: dict):
        """
        Connect to an MCP server and cache available tools.

        Args:
            mcp_server_config: Server configuration with command, args, and env
        """
        server_params = StdioServerParameters(**mcp_server_config)
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # Cache available tools
        response = await self.session.list_tools()
        self.available_tools = [
            convert_mcp_tool_to_function(tool) for tool in response.tools
        ]

        # Display connection success
        tool_names = [tool.name for tool in response.tools]
        self.console.print(
            Panel(
                f"[green]Connected successfully![/green]\n\n"
                f"[cyan]Available tools:[/cyan] {', '.join(tool_names)}",
                title="MCP Server",
                border_style="green",
            )
        )

    async def __call__(self, query: str) -> str:
        """
        Process a query through the LLM with tool support.

        Args:
            query: User query string

        Returns:
            The assistant's response as a string
        """
        self.messages.append({"role": "user", "content": query})

        if self.debug:
            self._debug_log("Request", self.messages)

        # First API call
        response = await self._call_llm()
        self.messages.append(response.choices[0].message.model_dump())

        content = response.choices[0].message

        # Check if the model wants to call a tool
        if content.tool_calls and len(content.tool_calls) > 0:
            tool_call: ChatCompletionMessageToolCallUnion = content.tool_calls[0]

            if type(tool_call) is ChatCompletionMessageFunctionToolCall:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")

                # Display tool call
                self._display_tool_call(tool_name, tool_args)

                # Execute tool
                tool_result = await self._execute_tool(
                    tool_name, tool_args, tool_call.id
                )

                # Display tool result
                self._display_tool_result(tool_name, tool_result)

                # Get final response from LLM after tool execution
                response = await self._call_llm()
                self.messages.append(response.choices[0].message.model_dump())

                return response.choices[0].message.content or ""

        return content.content or ""

    async def _call_llm(self):
        """Make an API call to the LLM."""
        try:
            if self.available_tools:
                return self.openai.chat.completions.create(
                    model=self.model,
                    tools=self.available_tools,
                    messages=self.messages,
                )
            else:
                return self.openai.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                )
        except Exception as e:
            self.console.print(f"[red]API Error:[/red] {e}")
            raise

    async def _execute_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str
    ) -> str:
        """
        Execute a tool call via MCP and add result to message history.

        Returns:
            The tool result as a string
        """
        if not self.session:
            raise RuntimeError("MCP session not initialized")

        try:
            result = await self.session.call_tool(tool_name, tool_args)

            # Convert MCP result content to string
            tool_result_text = ""
            for item in result.content:
                if hasattr(item, "text"):
                    tool_result_text += item.text
                else:
                    tool_result_text += str(item)

            # Add tool result to message history
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result_text,
                }
            )

            return tool_result_text

        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            self.console.print(f"[red]{error_msg}[/red]")
            # Add error to message history so LLM knows the tool failed
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": error_msg,
                }
            )
            return error_msg

    def _display_tool_call(self, tool_name: str, tool_args: dict):
        """Display a tool call in a nice panel."""
        args_json = json.dumps(tool_args, indent=2)
        syntax = Syntax(args_json, "json", theme="monokai", line_numbers=False)

        self.console.print(
            Panel(
                syntax,
                title=f"[yellow]Tool Call:[/yellow] {tool_name}",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    def _display_tool_result(self, tool_name: str, result: str):
        """Display tool result in a panel."""
        # Truncate very long results
        display_result = result if len(result) < 500 else result[:500] + "..."

        self.console.print(
            Panel(
                display_result,
                title=f"[blue]Tool Result:[/blue] {tool_name}",
                border_style="blue",
                padding=(1, 2),
            )
        )

    def _debug_log(self, title: str, data):
        """Display debug information."""
        json_str = json.dumps(data, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        self.console.print(Panel(syntax, title=f"[DEBUG] {title}", border_style="dim"))

    async def chat_loop(self):
        """Run an interactive chat loop with rich formatting."""
        self.console.print(
            "\n[bold cyan]MCP Agent Ready![/bold cyan]\n"
            "Type your queries or [bold]'quit'[/bold] to exit.\n"
        )

        while True:
            try:
                # Get user input with a minimal prompt
                query = Prompt.ask("\n>").strip()

                if query.lower() in ["quit", "exit"]:
                    self.console.print("\n[yellow]Goodbye![/yellow]")
                    break

                if not query:
                    continue

                # Display user query in a panel
                self.console.print(
                    Panel(
                        query,
                        title="[bold green]You[/bold green]",
                        border_style="green",
                        padding=(1, 2),
                    )
                )

                # Process query
                result = await self(query)

                # Display assistant response
                if result:
                    self.console.print(
                        Panel(
                            Markdown(result),
                            title="[bold magenta]Assistant[/bold magenta]",
                            border_style="magenta",
                            padding=(1, 2),
                        )
                    )

            except KeyboardInterrupt:
                self.console.print("\n\n[yellow]Interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"\n[red]Error:[/red] {e}")
                if self.debug:
                    raise

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


async def main():
    """Main entry point."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", None)

    if openrouter_api_key is None:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    model = os.getenv("OPENROUTER_MODEL", "x-ai/grok-code-fast-1")

    agent = MCPAgent(debug=False, model=model, api_key=openrouter_api_key)

    try:
        await agent.connect_to_mcp_server(MCP_SERVER_CONFIG)
        await agent.chat_loop()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
