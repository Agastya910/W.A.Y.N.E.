from flashrank import Ranker, RerankRequest
from config import TOP_K_RERANK


class Reranker:
    """
    CPU-optimized reranker using FlashRank with a ~30MB ONNX model.
    Implements retrieve-then-rerank for higher retrieval precision.
    """

    def __init__(self):
        # ms-marco-MiniLM-L-12-v2 is ~30MB, runs fast on CPU, high precision
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="opt")

    def rerank(self, query: str, candidates: list, top_k: int = TOP_K_RERANK) -> list:
        """
        Rerank a list of candidate dicts by relevance to the query.

        Each candidate must have a "content" key with the text to rank.
        Returns top_k most relevant candidates, preserving original dict structure.
        """
        if not candidates:
            return []

        passages = [
            {"id": str(i), "text": c.get("content", ""), "meta": c}
            for i, c in enumerate(candidates)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        # Reconstruct original dicts in reranked order
        reranked = []
        for r in results[:top_k]:
            original = r["meta"]
            original["rerank_score"] = r.get("score", 0.0)
            reranked.append(original)

        return reranked
