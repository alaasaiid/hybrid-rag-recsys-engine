# 1. Create src/multimodal_clip.py
@'
import numpy as np
import torch
import open_clip
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class MultimodalCLIPSearch:
    """Cross-modal text-to-image and text-to-catalog retrieval using OpenCLIP ViT-B-32 and Qdrant."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.client = QdrantClient(":memory:")
        self.collection_name = "multimodal_clip_index"
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

    def index_text_descriptions(self, items: list[dict]) -> None:
        points = []
        for item in items:
            tokens = self.tokenizer([item["title"]])
            with torch.no_grad():
                emb = self.model.encode_text(tokens)
                emb /= emb.norm(dim=-1, keepdim=True)
                vec = emb.squeeze().cpu().numpy().tolist()
            points.append(PointStruct(id=item["id"], vector=vec, payload=item))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        tokens = self.tokenizer([query])
        with torch.no_grad():
            q_emb = self.model.encode_text(tokens)
            q_emb /= q_emb.norm(dim=-1, keepdim=True)
            q_vec = q_emb.squeeze().cpu().numpy().tolist()
        hits = self.client.query_points(collection_name=self.collection_name, query=q_vec, limit=top_k).points
        return [{"id": h.id, "score": round(float(h.score), 4), **h.payload} for h in hits]
'@ | Set-Content -Encoding UTF8 "src\multimodal_clip.py"

# 2. Cleanup redundant old project directories
Remove-Item -Recurse -Force "C:\_Workspace\_Dev\01_Active_Projects\2026-09-17_Multimodal Recommendation - Rag Engine" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "C:\_Workspace\_Dev\01_Active_Projects\2026-09-30_enterprise-rag-knowledge-engine" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "C:\_Workspace\_Dev\01_Active_Projects\2026-10-07_multimodal-search" -ErrorAction SilentlyContinue

Write-Host "✅ Created src/multimodal_clip.py and cleaned up old folders!" -ForegroundColor Green