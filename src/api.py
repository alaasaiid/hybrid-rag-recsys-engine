import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.catalog import get_default_catalog
from src.hybrid_retriever import HybridRRFEngine
from src.reranker import TwoStageLightGBMReranker

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = HybridRRFEngine()
    retriever.index_catalog(get_default_catalog())
    state["retriever"] = retriever
    state["reranker"] = TwoStageLightGBMReranker()
    yield
    state.clear()

app = FastAPI(
    title="Hybrid RAG & Two-Stage Recommendation Engine",
    version="1.0.0",
    lifespan=lifespan,
)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, examples=["combine bm25 and vector search with rrf"])
    candidate_pool_size: int = Field(default=5, ge=2, le=8)
    top_k: int = Field(default=3, ge=1, le=5)

@app.get("/health")
def health_check():
    return {"status": "healthy", "indexed_items": len(get_default_catalog())}

@app.post("/api/v1/search")
def hybrid_search_and_rerank(payload: SearchRequest):
    if "retriever" not in state:
        raise HTTPException(status_code=503, detail="Search engine not initialized.")

    t0 = time.perf_counter()
    candidates = state["retriever"].retrieve_hybrid(payload.query, top_k=payload.candidate_pool_size)
    reranked = state["reranker"].rerank(payload.query, candidates, top_k=payload.top_k)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "query": payload.query,
        "latency_ms": elapsed_ms,
        "results": reranked,
    }
