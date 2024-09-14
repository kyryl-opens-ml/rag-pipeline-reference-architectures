from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import requests
import re
import lancedb
from sentence_transformers import SentenceTransformer
import os
import pickle

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Directory to store intermediate data
INTERMEDIATE_DATA_DIR = '/tmp/airflow_data'
os.makedirs(INTERMEDIATE_DATA_DIR, exist_ok=True)

# Define the DAG and its schedule
with DAG(
    'ingestion_dag',
    default_args=default_args,
    description='A simple ingestion DAG',
    schedule_interval=timedelta(days=1),  # Runs daily
    start_date=datetime(2023, 10, 1),
    catchup=False,
) as dag:

    # Task 1: Load text data
    def load_text():
        file_url = "https://raw.githubusercontent.com/lancedb/vectordb-recipes/main/tutorials/RAG-from-Scratch/lease.txt"
        response = requests.get(file_url)
        response.raise_for_status()
        text = response.text
        # Save the text data to a file
        with open(os.path.join(INTERMEDIATE_DATA_DIR, 'text_data.pkl'), 'wb') as f:
            pickle.dump(text, f)

    load_text_task = PythonOperator(
        task_id='load_text_task',
        python_callable=load_text,
    )

    # Task 2: Split text into chunks
    def recursive_text_splitter():
        # Load the text data from the file
        with open(os.path.join(INTERMEDIATE_DATA_DIR, 'text_data.pkl'), 'rb') as f:
            text = pickle.load(f)

        max_chunk_length = 100
        overlap = 10
        result = []
        current_chunk_count = 0
        separator = ["\n", " "]
        _splits = re.split(f"({'|'.join(separator)})", text)
        splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

        while current_chunk_count < len(splits):
            chunk_start = max(0, current_chunk_count - overlap)
            chunk_end = current_chunk_count + max_chunk_length
            chunk = "".join(splits[chunk_start:chunk_end])
            if chunk:
                result.append(chunk)
            current_chunk_count += max_chunk_length

        # Save the chunks to a file
        with open(os.path.join(INTERMEDIATE_DATA_DIR, 'chunks.pkl'), 'wb') as f:
            pickle.dump(result, f)

    split_text_task = PythonOperator(
        task_id='split_text_task',
        python_callable=recursive_text_splitter,
    )

    # Task 3: Generate embeddings
    def embedder():
        # Load the chunks from the file
        with open(os.path.join(INTERMEDIATE_DATA_DIR, 'chunks.pkl'), 'rb') as f:
            chunks = pickle.load(f)

        model_name = 'all-MiniLM-L6-v2'
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks)

        # Save the embeddings to a file
        with open(os.path.join(INTERMEDIATE_DATA_DIR, 'embeddings.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)

    embedder_task = PythonOperator(
        task_id='embedder_task',
        python_callable=embedder,
    )

    # Task 4: Prepare data for insertion
    def prepare_data():
        # Load the chunks and embeddings from files
        with open(os.path.join(INTERMEDIATE_DATA_DIR, 'chunks.pkl'), 'rb') as f:
            chunks = pickle.load(f)
        with open(os.path.join(INTERMEDIATE_DATA_DIR, 'embeddings.pkl'), 'rb') as f:
            embeddings = pickle.load(f)

        data = [{"text": chunk, "vector": embed.tolist()} for chunk, embed in zip(chunks, embeddings)]

        # Save the prepared data to a file
        with open(os.path.join(INTERMEDIATE_DATA_DIR, 'prepared_data.pkl'), 'wb') as f:
            pickle.dump(data, f)

    prepare_data_task = PythonOperator(
        task_id='prepare_data_task',
        python_callable=prepare_data,
    )

    # Task 5: Insert data into LanceDB
    def insert_to_lancedb():
        # Load the prepared data from the file
        with open(os.path.join(INTERMEDIATE_DATA_DIR, 'prepared_data.pkl'), 'rb') as f:
            data = pickle.load(f)

        db_path = "/tmp/lancedb"
        db = lancedb.connect(db_path)
        table_name = "scratch"

        try:
            # Try to open the table to check if it exists
            table = db.open_table(table_name)
            # If the table exists, add to the existing data
            table.add(data)
            print("Data added to existing table.")
        except Exception as ex:
            print("Table does not exist. Creating a new table.")
            # If the table does not exist, create a new table
            table = db.create_table(table_name, data=data, mode="create")
            print("New table created and data inserted.")

    insert_to_lancedb_task = PythonOperator(
        task_id='insert_to_lancedb_task',
        python_callable=insert_to_lancedb,
    )

    # Define task dependencies
    load_text_task >> split_text_task >> embedder_task >> prepare_data_task >> insert_to_lancedb_task