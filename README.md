<div align="center">

# 3-2-1 Vector Store

Vector search through issues of [3-2-1 by James Clear](https://jamesclear.com/3-2-1)

</div>

[3-2-1 by James Clear](https://jamesclear.com/3-2-1) is by far my favorite newsletter.

It is sent out every Thursday and it features 3 ideas from James, 2 quotes from others and 1 question for the reader.

As it has been going for years now with this project we index its winsdom into a Vector Store for semantic search.

## Data Pipeline

### Installing dependencies

`uv sync`

Then, activate the virtual environment:

```source .venv/bin/activate```

Here we are going to use `dagster` to run jobs from the command line.

Alternativelly launch Dagster UI with `dagster dev` and open it [in your browser](http://localhost:3000) to manually launch jobs and monitor their execution.

### Download Newsletter Issues

Launch the download job from command line:

`dagster job execute -m pipeline.definitions -j download_pipeline`

This will download all issues in HTML format into `data/raw/html` in around 10 minutes

### Parse, Encode and Store

After downloading all issues in HTML format this job will:
- Covert them to markdown
- Prepare and text data and split it into chunks
- Encode the documents into 384-long embedding vectors with `all-MiniLM-L6-v2`
- Store each document with vector and payload into Qdrant

Before running the pipeline run a local instance of Qdrant with `docker run -p 6333:6333 qdrant/qdrant`

Then launch the processing job with `dagster job execute -m pipeline.definitions -j full_processing_pipeline`

## Query

Once the ingestion pipeline successfully completed it is possible to query the vector store.

The query will:
- Encode the question with `all-MiniLM-L6-v2`
- Query the vector store for near vectors using dot distance as we previously normalized during encoding
- Rerank each of the 10 results together with the user question with `ms-marco-MiniLM-L-6-v2`
- Present the results sorted by this new score

### CLI

From the command line use the `query.py` script to target the docker instance of Qdrant which we previously loaded data into.

`uv run query.py "What has been said about the importance of priorites?"`


![query in CLI](art/query-cli.png "Query from CLI")


### Web

From the web browser launch the Streamlit `app.py`.

This will use an in memory version of Qdrant loaded directly from the Parquet file.

`uv run streamlit run app.py`

![streamlit](art/streamlit.png "Query from Streamlit")



