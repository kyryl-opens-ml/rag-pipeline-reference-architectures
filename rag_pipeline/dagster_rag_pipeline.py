import os
import re
import requests
import lancedb
from sentence_transformers import SentenceTransformer
from dagster import (
    asset,
    Definitions,
    ScheduleDefinition,
    AssetSelection,
    define_asset_job,
)

# Asset to load text data
@asset
def text_data() -> str:
    """
    Downloads and loads text from a file URL using requests.
    """
    file_url = "https://raw.githubusercontent.com/lancedb/vectordb-recipes/main/tutorials/RAG-from-Scratch/lease.txt"
    response = requests.get(file_url)
    response.raise_for_status()
    return response.text

# Asset to split text into chunks
@asset
def chunks(text_data: str) -> list:
    """
    Splits text into smaller segments recursively.
    """
    max_chunk_length = 100
    overlap = 10
    result = []
    current_chunk_count = 0
    separator = ["\n", " "]
    _splits = re.split(f"({'|'.join(separator)})", text_data)
    splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

    while current_chunk_count < len(splits):
        chunk_start = max(0, current_chunk_count - overlap)
        chunk_end = current_chunk_count + max_chunk_length
        chunk = "".join(splits[chunk_start:chunk_end])
        if chunk:
            result.append(chunk)
        current_chunk_count += max_chunk_length

    return result

# Asset to generate embeddings
@asset
def embeddings(chunks: list):
    """
    Generates embeddings for a list of text chunks.
    """
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

# Asset to prepare data for insertion
@asset
def prepared_data(chunks: list, embeddings) -> list:
    """
    Prepares data for insertion into LanceDB.
    """
    return [{"text": chunk, "vector": embed} for chunk, embed in zip(chunks, embeddings)]

# Asset to insert data into LanceDB
@asset
def lancedb_inserted(prepared_data: list):
    """
    Inserts data into LanceDB.
    """
    db_path = "/tmp/lancedb"
    db = lancedb.connect(db_path)
    table_name = "scratch"

    try:
        # Try to open the table to check if it exists
        table = db.open_table(table_name)
        # If the table exists, add to the existing data
        table.add(prepared_data)
        print("Data added to existing table.")
    except Exception as ex:
        print("Table does not exist. Creating a new table.")
        # If the table does not exist, create a new table
        table = db.create_table(table_name, data=prepared_data, mode="create")
        print("New table created and data inserted.")

# Define the list of all assets
all_assets = [
    text_data,
    chunks,
    embeddings,
    prepared_data,
    lancedb_inserted,
]

# Define the asset job to materialize all assets
asset_job = define_asset_job(
    name="materialize_all_assets",
    selection=AssetSelection.assets(*all_assets),
)

daily_schedule = ScheduleDefinition(
    job=asset_job,
    cron_schedule="0 0 * * *",  # Every day at midnight UTC
)

# Combine everything into a Definitions object
defs = Definitions(
    assets=all_assets,
    schedules=[daily_schedule],
)