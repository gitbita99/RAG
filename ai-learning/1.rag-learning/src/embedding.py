from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_db(chunks, persist_directory=None):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        chunks,
        embedding=embedding,
        persist_directory=persist_directory  # Optional: save DB for later use
    )
    return vector_db