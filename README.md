# Hybrid RAG & Two-Stage Recommendation Engine

Production retrieval and ranking microservice combining Dense Vector Search (Qdrant), Lexical Search (BM25) via Reciprocal Rank Fusion (RRF, k=60), Second-Stage LightGBM Re-Ranking, and ONNX Runtime INT8 Quantization.

## Quickstart
- pip install -r requirements.txt
- python -m pytest tests/ -v
- python -m src.evaluate_rag
- python -m src.quantize_onnx
- uvicorn src.api:app --reload
