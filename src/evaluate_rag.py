import warnings
import numpy as np

# Suppress external Pydantic protected namespace warning on entry
warnings.filterwarnings("ignore", category=UserWarning, message=".*protected_namespaces.*")

import mlflow
from sklearn.metrics import ndcg_score
from src.catalog import get_default_catalog
from src.hybrid_retriever import HybridRRFEngine
from src.reranker import TwoStageLightGBMReranker

EVAL_QUERIES = [
    {"query": "how to speed up transformer cpu inference with int8 quantization", "relevant_ids": {3, 6}},
    {"query": "combine bm25 keyword search with vector embeddings rrf", "relevant_ids": {4, 2}},
    {"query": "spark shuffle partitions and broadcast join etl", "relevant_ids": {1, 7}},
    {"query": "two stage learning to rank lightgbm ndcg", "relevant_ids": {5, 8}},
]


def evaluate_pipeline() -> dict:
    engine = HybridRRFEngine()
    engine.index_catalog(get_default_catalog())
    reranker = TwoStageLightGBMReranker()

    precisions_hybrid = []
    ndcgs_reranked = []

    for sample in EVAL_QUERIES:
        q = sample["query"]
        rel = sample["relevant_ids"]

        hybrid_candidates = engine.retrieve_hybrid(q, top_k=4)
        hits = sum(1 for c in hybrid_candidates[:2] if c["id"] in rel)
        precisions_hybrid.append(hits / 2.0)

        reranked = reranker.rerank(q, hybrid_candidates, top_k=4)
        y_true = np.asarray([[1.0 if c["id"] in rel else 0.0 for c in reranked]])
        y_score = np.asarray([[c["lgbm_rank_score"] for c in reranked]])
        if y_true.sum() > 0:
            ndcgs_reranked.append(float(ndcg_score(y_true, y_score)))

    metrics = {
        "context_precision_at_2": round(float(np.mean(precisions_hybrid)), 4),
        "ndcg_at_4": round(float(np.mean(ndcgs_reranked)), 4),
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hybrid_rag_recsys_eval")
    with mlflow.start_run(run_name="rrf_lgbm_eval"):
        mlflow.log_params({"rrf_k": 60, "embedder": "all-MiniLM-L6-v2", "reranker": "LGBMRanker"})
        mlflow.log_metrics(metrics)

    print("Evaluation Metrics:", metrics)
    return metrics


if __name__ == "__main__":
    evaluate_pipeline()