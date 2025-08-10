import os
from chunking import load_and_chunk
# Assuming embedding.py has the following functions:
# def get_embedding_model(): ...
# def embed_text(model, texts: list[str]): ...
# def embed_query(model, query: str): ...
from embedding import get_embedding_model, embed_text, embed_query

# Assuming vector_store.py has a VectorStore class:
# class VectorStore:
#   def __init__(self, dimension): ...
#   def add(self, embeddings, texts): ...
#   def search(self, query_vector, top_k): ...
from vector_store import VectorStore
from langchain_ollama import OllamaLLM
from pathlib import Path

# Use an absolute path to find the 'data' folder
# This makes the code independent of the current working directory
project_root = Path(__file__).parent.parent
data_folder_path = project_root / "data"

# Step 1: Read the text files and create chunks
# We use the absolute path we just created
chunks = load_and_chunk(str(data_folder_path), chunk_size=300, chunk_overlap=50)

# Step 2: Chunk the text and get the text content from the chunks
# The load_and_chunk function returns a list of dictionaries.
# We need to extract the 'chunk' key from each dictionary.
chunk_texts = [c["chunk"] for c in chunks]

# Step 3: Embed the chunked text
# Get the embedding model from your embedding.py file.
model = get_embedding_model()

# Now, pass the list of strings (chunk_texts) to the embedding function.
chunk_embeddings = embed_text(model, chunk_texts)

# Step 4: Store it in the vector database (Faiss)
# Get the dimension of the embedding vectors.
dimension = len(chunk_embeddings[0])

# Initialize the vector store.
vector_db = VectorStore(dimension)

# Add the embeddings and their corresponding text to the vector store.
vector_db.add(chunk_embeddings, chunk_texts)

# Step 5: Take a user query
query = "How can abith become a good engineer and what are the skills he needs more to crack MAANG?"

# Step 6: Embed the user query
query_vector = embed_query(model, query)

# Step 7: Search the vector database for the top 3 most relevant chunks
top_chunks = vector_db.search(query_vector, top_k=3)

# Step 8: Create the context from the retrieved chunks
context = "\n".join(top_chunks)
print("--- Retrieved Context ---")
print(context)
print("\n" + "="*30 + "\n")

# Step 9: Initialize the LLM
llm = OllamaLLM(model="llama3")

# Step 10: Create a prompt using the context and query
prompt = f"Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"

# Step 11: Invoke the LLM to get the final response
response = llm.invoke(prompt)

# Step 12: Print the final response
print("--- LLM Response ---")
print(response)