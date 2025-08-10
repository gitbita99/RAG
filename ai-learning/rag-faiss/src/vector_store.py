import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.vectors = []  # Keep local copy if you want (optional)
        self.texts = []    # Fixed: was 'self.text' before

    def add(self, vectors: list, texts: list):
        """
        Adds vectors and corresponding texts to the index.
        - vectors: List of embedding vectors (list of lists or numpy array)
        - texts: List of strings (same length as vectors)
        """
        vectors_np = np.array(vectors).astype('float32')
        self.index.add(vectors_np)
        self.vectors.extend(vectors)  # Optional: for local reference
        self.texts.extend(texts)

    def search(self, query_vector: list, top_k: int = 3):
        """
        Searches the index for the nearest vectors to the query.
        Returns the top_k matching texts.
        """
        query_np = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_np, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1:  # Valid FAISS index
                results.append(self.texts[idx])
        return results