import re
from sentence_transformers import SentenceTransformer
import lancedb
import openai
import requests
import os
from openai import OpenAI

def load_text(file_url):
    """
    Downloads and loads text from a file URL using requests.
    """
    response = requests.get(file_url)
    response.raise_for_status()  # Check for request errors
    return response.text


def recursive_text_splitter(text, max_chunk_length=1000, overlap=100):
    """
    Splits text into smaller segments recursively.
    """
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

    return result


def embedder(text_chunks, model_name='all-MiniLM-L6-v2'):
    """
    Generates embeddings for a list of text chunks.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks)
    return embeddings


def prepare_data(chunks, embeddings):
    """
    Prepares data for insertion into LanceDB.
    """
    return [{"text": chunk, "vector": embed} for chunk, embed in zip(chunks, embeddings)]


def insert_to_lancedb(data, db_path="/tmp/lancedb"):
    """
    Inserts data into LanceDB. If the table exists, add to the existing data;
    if it does not exist, create a new table.
    """
    db = lancedb.connect(db_path)
    table_name = "scratch"

    try:
        # Try to open the table to check if it exists
        table = db.open_table(table_name)
        # If the table exists, add to the existing data
        table.add(data)
    except Exception as ex:
        print(ex)
        # If the table does not exist, create a new table
        table = db.create_table(table_name, data=data, mode="create")

    return table


def retrieve_context(table, question, model_name='all-MiniLM-L6-v2', top_k=5):
    """
    Retrieves the most relevant context from LanceDB based on a question.
    """
    query_embedding = embedder([question], model_name=model_name)[0]
    results = table.search(query_embedding).limit(top_k).to_list()
    return [r["text"] for r in results]


def generate_prompt(question, context):
    """
    Creates a prompt for the OpenAI model.
    """
    base_prompt = """You are an AI assistant. Your task is to understand the user question and provide an answer using the provided contexts. Every answer you generate should have citations in this pattern "Answer [position].", for example: "Earth is round [1][2].," if it's relevant.

    Your answers are correct, high-quality, and written by a domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."

    User question: {}

    Contexts:
    {}
    """
    return base_prompt.format(question, "\n".join(context))


# def generate_answer(prompt, model_name="gpt-4o-mini-2024-07-18"):
def generate_answer(prompt, model_name="gpt-4o"):
    """
    Generates an answer from OpenAI using the new API format and GPT-4 model.
    """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model=model_name,
    )
    return response.choices[0].message.content


def main():
    # Step 1: Load the text data
    text_data = load_text("https://raw.githubusercontent.com/lancedb/vectordb-recipes/main/tutorials/RAG-from-Scratch/lease.txt")

    # Step 2: Split the text into chunks
    chunks = recursive_text_splitter(text_data, max_chunk_length=100, overlap=10)
    print("Number of Chunks:", len(chunks))

    # Step 3: Generate embeddings for each chunk
    embeddings = embedder(chunks)

    # Step 4: Insert data into LanceDB
    data = prepare_data(chunks, embeddings)
    table = insert_to_lancedb(data)

    # Step 5: Retrieve relevant context based on a question
    question = "What is the issue date of lease?"
    context = retrieve_context(table, question)

    # Step 6: Generate the prompt for the AI model
    prompt = generate_prompt(question, context)
    print(prompt)
    # Step 7: Generate an answer from OpenAI
    answer = generate_answer(prompt)
    print(answer)


if __name__ == "__main__":
    main()