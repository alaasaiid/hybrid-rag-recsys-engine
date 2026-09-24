from fastapi.testclient import TestClient
from src.api import app
from src.catalog import get_default_catalog
from src.hybrid_retriever import HybridRRFEngine
from src.reranker import TwoStageLightGBMReranker

def test_hybrid_rrf_retrieval():
    engine = HybridRRFEngine(rrf_k=60)
    engine.index_catalog(get_default_catalog())
    results = engine.retrieve_hybrid("BM25 Reciprocal Rank Fusion RRF", top_k=3)
    assert len(results) == 3
    assert results[0]["id"] == 4
    assert "rrf_score" in results[0]

def test_lightgbm_reranker():
    engine = HybridRRFEngine(rrf_k=60)
    engine.index_catalog(get_default_catalog())
    candidates = engine.retrieve_hybrid("ONNX INT8 quantization latency", top_k=4)
    reranker = TwoStageLightGBMReranker()
    reranked = reranker.rerank("ONNX INT8 quantization latency", candidates, top_k=2)
    assert len(reranked) == 2
    assert "lgbm_rank_score" in reranked[0]

def test_fastapi_search_endpoint():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        res = client.post("/api/v1/search", json={"query": "Qdrant HNSW vector search", "candidate_pool_size": 4, "top_k": 2})
        assert res.status_code == 200
        body = res.json()
        assert body["latency_ms"] > 0
        assert len(body["results"]) == 2
