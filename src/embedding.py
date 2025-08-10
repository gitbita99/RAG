from sentence_transformers import SentenceTransformer
from typing import List

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load and return the SentenceTransformer embedding model.
    """
    return SentenceTransformer(model_name)

def embed_text(model: SentenceTransformer, texts: List[str]):
    """
    Embed a list of chunks.
    Each chunk should be a dict with a 'chunk' key containing the text.
    """
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_tensor=False
    )
    return embeddings

def embed_query(model: SentenceTransformer, query: str):
    """
    Embed a single query string.
    """
    return model.encode([query], convert_to_tensor=False)[0]
