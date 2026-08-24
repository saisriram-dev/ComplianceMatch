import numpy as np

def build_corpus(X):
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X_norm

def cosine_similarity_matrix(corpus_norm, query):
    query = query / np.linalg.norm(query)
    return corpus_norm @ query

def top_k(query, corpus_norm, k):
    matrix = cosine_similarity_matrix(corpus_norm, query)

    top_k_recommendations = np.argpartition(matrix, -k)[-k:]
    ordering = np.argsort(matrix[top_k_recommendations])[::-1]
    candidates = top_k_recommendations[ordering]


    return candidates, matrix[candidates]
