from typing import List, Dict, Any

def get_default_catalog() -> List[Dict[str, Any]]:
    """Returns a structured catalog of items for hybrid retrieval and re-ranking."""
    return [
        {"id": 1, "title": "Distributed PySpark ETL Pipeline Guide", "text": "Optimize Apache Spark shuffle partitions, broadcast joins, and Parquet columnar storage for high-throughput batch ETL data engineering pipelines.", "category": "data_engineering", "popularity": 92},
        {"id": 2, "title": "Qdrant Vector Database HNSW Indexing", "text": "Configure Hierarchical Navigable Small World (HNSW) graphs in Qdrant for sub-15ms approximate nearest neighbor dense vector similarity search.", "category": "vector_search", "popularity": 88},
        {"id": 3, "title": "ONNX Runtime INT8 Dynamic Quantization", "text": "Accelerate Transformer encoder CPU inference by 2.8x using ONNX Runtime dynamic INT8 weight quantization with minimal accuracy degradation.", "category": "mlops", "popularity": 95},
        {"id": 4, "title": "Hybrid RAG with BM25 and Reciprocal Rank Fusion", "text": "Combine lexical BM25 exact keyword matching with dense semantic embeddings using Reciprocal Rank Fusion (RRF k=60) to improve context precision.", "category": "rag", "popularity": 97},
        {"id": 5, "title": "LightGBM Learning-to-Rank Two-Stage RecSys", "text": "Second-stage gradient boosted decision tree ranker optimizing NDCG@10 over candidate sets using query-item interaction and similarity features.", "category": "recsys", "popularity": 85},
        {"id": 6, "title": "FastAPI Asynchronous Microservices in Docker", "text": "Productionize machine learning inference endpoints with FastAPI, Pydantic v2 schema validation, lifespan state management, and multi-stage Docker builds.", "category": "mlops", "popularity": 90},
        {"id": 7, "title": "Pydantic v2 Data Contracts for Biomedical NLP", "text": "Enforce strict JSONL schema validation and entity linking normalization across PubMed, ATC, and UMLS CUI clinical datasets.", "category": "data_engineering", "popularity": 84},
        {"id": 8, "title": "MLflow Experiment Tracking and Model Registry", "text": "Log hyperparameter sweeps, retrieval evaluation metrics (Context Precision, Recall@K, NDCG), and serialized model artifacts in MLflow.", "category": "mlops", "popularity": 89},
    ]
