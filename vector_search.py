from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
import numpy as np
from pydantic import BaseModel, Field
from datetime import datetime, timezone

app = FastAPI()

SEARCH_LOGS: list[dict] = []

def log_query(k: int, top_score: float, storage: str = "memory"):
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "top_score": float(top_score)
    }

    SEARCH_LOGS.append(log_entry)
    with open("search_audit.log", "a") as f:
        f.write(f"{log_entry['timestamp']} | k={log_entry['k']} | top_score={log_entry['top_score']:.4f}\n")

# Mock corpus initialization (10 vectors of dimension 4)
# Replace with your actual vector database or array initialization
X = np.random.rand(10, 4)

def build_corpus(X):
    if X.size == 0:
        return X
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X_norm

corpus_norm = build_corpus(X)

class QueryRequest(BaseModel):
    query: list[float]
    k: int = Field(1, ge=1, le=100)

class Topk(BaseModel):
    indices: list[int]
    topk_similarities: list[float]

def cosine_similarity_matrix(request: QueryRequest):
    query_vec = np.array(request.query)
    norm = np.linalg.norm(query_vec)

    if norm == 0:
        raise HTTPException(status_code=400, detail="Query vector cannot be all zeros.")

    query_norm = query_vec / norm
    return corpus_norm @ query_norm

@app.post("/search_db", response_model=Topk)
def top_k(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    matrix = Depends(cosine_similarity_matrix),
):
    # Validate dimensions dynamically against X
    if len(request.query) != X.shape[1]:
        raise HTTPException(
            status_code=422, 
            detail=f"Dimension mismatch: expected {X.shape[1]}, got {len(request.query)}"
        )

    # Use request.k and cap it to corpus size
    k = min(request.k, X.shape[0])

    top_k_recommendations = np.argpartition(matrix, -k)[-k:]
    ordering = np.argsort(matrix[top_k_recommendations])[::-1]
    candidates = top_k_recommendations[ordering]
    scores = matrix[candidates]

    top_score = scores[0] if len(scores) > 0 else 0.0

    background_tasks.add_task(log_query, k=k, top_score=top_score)

    return {
        "indices": candidates.tolist(),
        "topk_similarities": matrix[candidates].tolist()
    }
