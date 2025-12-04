import pandas as pd

df = pd.read_parquet("data/parquet/newsletter_embeddings.parquet")

print(df.sample(20))
