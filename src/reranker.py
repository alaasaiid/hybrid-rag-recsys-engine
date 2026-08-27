import numpy as np
import lightgbm as lgb
from typing import List, Dict, Any

class TwoStageLightGBMReranker:
    """Second-stage GBDT re-ranker scoring candidates on retrieval + metadata features."""

    def __init__(self):
        self.model: lgb.LGBMRanker | None = None
        self._train_default_ranker()

    def _extract_features(self, query: str, candidate: Dict[str, Any]) -> List[float]:
        q_tokens = set(query.lower().split())
        doc_tokens = set(f"{candidate['title']} {candidate['text']}".lower().split())
        overlap = len(q_tokens.intersection(doc_tokens)) / max(len(q_tokens), 1)
        return [
            float(candidate.get("dense_score", 0.0)),
            float(candidate.get("bm25_score", 0.0)),
            float(candidate.get("rrf_score", 0.0)),
            float(candidate.get("popularity", 50.0)) / 100.0,
            float(overlap),
        ]

    def _train_default_ranker(self) -> None:
        """Trains a lightweight LGBMRanker on synthetic query-candidate relevance groups."""
        rng = np.random.default_rng(42)
        X_list, y_list, groups = [], [], []
        for _ in range(30):
            n_cand = 6
            dense = rng.uniform(0.1, 0.95, size=n_cand)
            bm25 = rng.uniform(0.0, 5.0, size=n_cand)
            rrf = rng.uniform(0.01, 0.033, size=n_cand)
            pop = rng.uniform(0.3, 1.0, size=n_cand)
            ovlp = rng.uniform(0.0, 1.0, size=n_cand)
            X_group = np.column_stack([dense, bm25, rrf, pop, ovlp])
            # Relevance grade (0, 1, 2) driven primarily by hybrid signals + token overlap
            signal = 0.45 * dense + 0.25 * (bm25 / 5.0) + 0.20 * ovlp + 0.10 * pop
            y_group = np.digitize(signal, bins=[0.4, 0.65])
            X_list.append(X_group)
            y_list.append(y_group)
            groups.append(n_cand)

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=40,
            learning_rate=0.08,
            num_leaves=15,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X_train, y_train, group=groups)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        features = np.array([self._extract_features(query, c) for c in candidates])
        scores = self.model.predict(features)
        enriched = []
        for cand, score in zip(candidates, scores):
            enriched.append({**cand, "lgbm_rank_score": round(float(score), 4)})
        enriched.sort(key=lambda x: x["lgbm_rank_score"], reverse=True)
        return enriched[:top_k]
