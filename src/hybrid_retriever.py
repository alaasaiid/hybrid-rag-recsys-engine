import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

class HybridRRFEngine:
    """Combines Qdrant dense vector retrieval with BM25 lexical search via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.embedder = SentenceTransformer(model_name)
        self.vector_size = self.embedder.get_embedding_dimension()
        self.client = QdrantClient(":memory:")
        self.collection_name = "hybrid_knowledge_base"
        self.catalog: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

    def index_catalog(self, items: List[Dict[str, Any]]) -> None:
        self.catalog = items
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

        corpus_texts = [f"{item['title']} {item['text']}" for item in items]
        embeddings = self.embedder.encode(corpus_texts, normalize_embeddings=True)

        points = [
            PointStruct(id=item["id"], vector= embeddings[idx].tolist(), payload=item)
            for idx, item in enumerate(items)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.catalog or self.bm25 is None:
            raise ValueError("Catalog has not been indexed yet.")

        # 1. Dense Retrieval via Qdrant
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        dense_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=len(self.catalog),
        ).points

        dense_ranks = {hit.id: rank for rank, hit in enumerate(dense_hits, start=1)}
        dense_scores = {hit.id: float(hit.score) for hit in dense_hits}

        # 2. Lexical Retrieval via BM25
        tokenized_query = query.lower().split()
        bm25_raw = self.bm25.get_scores(tokenized_query)
        bm25_sorted_indices = np.argsort(bm25_raw)[::-1]

        bm25_ranks = {}
        bm25_scores = {}
        for rank, idx in enumerate(bm25_sorted_indices, start=1):
            item_id = self.catalog[idx]["id"]
            bm25_ranks[item_id] = rank
            bm25_scores[item_id] = float(bm25_raw[idx])

        # 3. Reciprocal Rank Fusion (RRF, k=60)
        fused_results = []
        for item in self.catalog:
            item_id = item["id"]
            r_dense = dense_ranks.get(item_id, len(self.catalog) + 1)
            r_bm25 = bm25_ranks.get(item_id, len(self.catalog) + 1)
            rrf_score = (1.0 / (self.rrf_k + r_dense)) + (1.0 / (self.rrf_k + r_bm25))

            fused_results.append({
                **item,
                "dense_score": round(dense_scores.get(item_id, 0.0), 4),
                "bm25_score": round(bm25_scores.get(item_id, 0.0), 4),
                "rrf_score": round(rrf_score, 6),
            })

        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_results[:top_k]
