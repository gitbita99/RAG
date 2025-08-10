import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk(
    folder_path: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> list:
    """
    Loads all .txt files from the given folder, splits them into chunks,
    and returns a list of dicts with 'chunk' and 'metadata'.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            chunks = text_splitter.split_text(file_content)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "chunk": chunk,
                    "metadata": {
                        "source": filename,
                        "chunk_index": i
                    }
                })

    return all_chunks
