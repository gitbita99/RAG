from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_and_split_pdf(pdf_path: Path, chunk_size=300, chunk_overlap=50 ):

    document = PyPDFLoader(str(pdf_path))
    document_load = document.load() 
    splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    chunks = splitter.split_documents(document_load)

    return chunks