
def query_vector_db(vectorDB, query, k=3):
    results = vectorDB.similarity_search(query, k=k)
    return results
