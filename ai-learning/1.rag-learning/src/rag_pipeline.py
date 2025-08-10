from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from pathlib import Path
from chunking import load_and_split_pdf
from embedding import create_vector_db
from retrieval import query_vector_db

project_root = Path(__file__).parent.parent
pdf_file_path = project_root / "data" / "DE_latest.pdf"

chunks = load_and_split_pdf(pdf_file_path)

vector_db = create_vector_db(chunks)

query = "Who is Mohamed Abith?"
results = query_vector_db(vector_db, query)

llm = OllamaLLM(model="llama3")

context = "\n".join([doc.page_content for doc in results])
prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}"

response = llm.invoke(prompt)
print(response)
