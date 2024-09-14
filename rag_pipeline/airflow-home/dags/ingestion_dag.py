from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import requests
import re
import lancedb
from sentence_transformers import SentenceTransformer
import numpy as np

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['you@example.com'],  # Update with your email
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

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
    def load_text(**kwargs):
        file_url = "https://raw.githubusercontent.com/lancedb/vectordb-recipes/main/tutorials/RAG-from-Scratch/lease.txt"
        response = requests.get(file_url)
        response.raise_for_status()
        text = response.text
        # Push the result to XCom
        kwargs['ti'].xcom_push(key='text_data', value=text)

    load_text_task = PythonOperator(
        task_id='load_text_task',
        python_callable=load_text,
        provide_context=True,
    )

    # Task 2: Split text into chunks
    def recursive_text_splitter(**kwargs):
        text = kwargs['ti'].xcom_pull(key='text_data', task_ids='load_text_task')
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

        # Push the chunks to XCom
        kwargs['ti'].xcom_push(key='chunks', value=result)

    split_text_task = PythonOperator(
        task_id='split_text_task',
        python_callable=recursive_text_splitter,
        provide_context=True,
    )

    # Task 3: Generate embeddings
    def embedder(**kwargs):
        chunks = kwargs['ti'].xcom_pull(key='chunks', task_ids='split_text_task')
        model_name = 'all-MiniLM-L6-v2'
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks)
        # Convert embeddings to list for XCom
        embeddings_list = embeddings.tolist()
        # Push embeddings to XCom
        kwargs['ti'].xcom_push(key='embeddings', value=embeddings_list)

    embedder_task = PythonOperator(
        task_id='embedder_task',
        python_callable=embedder,
        provide_context=True,
    )

    # Task 4: Prepare data for insertion
    def prepare_data(**kwargs):
        chunks = kwargs['ti'].xcom_pull(key='chunks', task_ids='split_text_task')
        embeddings = kwargs['ti'].xcom_pull(key='embeddings', task_ids='embedder_task')
        data = [{"text": chunk, "vector": embed} for chunk, embed in zip(chunks, embeddings)]
        # Push data to XCom
        kwargs['ti'].xcom_push(key='prepared_data', value=data)

    prepare_data_task = PythonOperator(
        task_id='prepare_data_task',
        python_callable=prepare_data,
        provide_context=True,
    )

    # Task 5: Insert data into LanceDB
    def insert_to_lancedb(**kwargs):
        data = kwargs['ti'].xcom_pull(key='prepared_data', task_ids='prepare_data_task')
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
        provide_context=True,
    )

    # Define task dependencies
    load_text_task >> split_text_task >> embedder_task >> prepare_data_task >> insert_to_lancedb_task