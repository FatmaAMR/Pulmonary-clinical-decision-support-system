import math
from typing import Dict, Any, List, Optional

from core.schemas import (
    DocumentChunk,
    RetrievalResult,
    EvalQueryCase,
    RetrievalEvalReport,
)
from core.interfaces import (
    BaseDocumentStore,
    BaseEmbedder,
    BaseVectorStore,
    BaseReranker,
)
from core.registry import registry
import components.embedders
import components.vectorstores
import components.rerankers


# Design Pattern: Facade Pattern. Orchestrates Vector Indexing, Hybrid Search, Cross-Reranking, Abstention Gating, and Evaluation.
class RetrievalPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        document_store: BaseDocumentStore,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        reranker: Optional[BaseReranker] = None,
    ):
        self.config = config
        self.document_store = document_store
        ret_cfg = config.get("retrieval", {})

        if embedder is not None:
            self.embedder = embedder
        else:
            emb_cfg = ret_cfg.get("embedder", {"name": "hash_embedder", "params": {"dim": 384}})
            emb_cls = registry.get_embedder(emb_cfg["name"])
            self.embedder = emb_cls(**emb_cfg.get("params", {}))

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            vs_cfg = ret_cfg.get("vector_store", {"provider": "in_memory_vector_store", "params": {}})
            vs_cls = registry.get_vector_store(vs_cfg["provider"])
            self.vector_store = vs_cls(**vs_cfg.get("params", {}))

        if reranker is not None:
            self.reranker = reranker
        else:
            rerank_cfg = ret_cfg.get("reranker", {"name": "hybrid_overlap_reranker", "params": {}})
            rerank_cls = registry.get_reranker(rerank_cfg["name"])
            self.reranker = rerank_cls(**rerank_cfg.get("params", {}))

        self.abstention_threshold: float = float(ret_cfg.get("abstention_threshold", 0.25))

    def index_file_chunks(self, file_id: str) -> Dict[str, Any]:
        chunks = self.document_store.get_chunks_by_file(file_id)
        if not chunks:
            raise ValueError(f"No chunks found for file_id '{file_id}'. Run chunking first.")

        texts = [
            f"{c.metadata.page_or_section or ''} | {c.text}".strip()
            for c in chunks
        ]
        vectors = self.embedder.embed_documents(texts)
        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec

        self.vector_store.delete_by_file_id(file_id)
        indexed_count = self.vector_store.upsert_chunks(chunks)

        return {
            "file_id": file_id,
            "indexed_chunks_count": indexed_count,
            "embedding_dimension": self.embedder.dimension,
            "total_vectors_in_store": self.vector_store.count_vectors(),
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        abstention_threshold_override: Optional[float] = None,
    ) -> RetrievalResult:
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        q_vec = self.embedder.embed_query(query)
        candidates = self.vector_store.search(
            query_embedding=q_vec,
            query_text=query,
            top_k=max(top_k * 2, 10),
            mode=mode,
            file_id_filter=file_id_filter,
            content_type_filter=content_type_filter,
        )

        ranked = self.reranker.rerank(query=query, chunks=candidates, top_k=top_k)
        max_score = max((c.rerank_score or 0.0 for c in ranked), default=0.0)
        threshold = (
            abstention_threshold_override
            if abstention_threshold_override is not None
            else self.abstention_threshold
        )

        if not ranked or max_score < threshold:
            return RetrievalResult(
                query=query,
                ranked_chunks=[],
                max_confidence_score=max_score,
                abstained=True,
                abstention_reason=f"Max confidence score ({max_score:.3f}) is below threshold ({threshold:.3f}).",
                search_mode=mode,
                telemetry={"candidate_count": len(candidates), "threshold": threshold},
            )

        return RetrievalResult(
            query=query,
            ranked_chunks=ranked,
            max_confidence_score=max_score,
            abstained=False,
            abstention_reason=None,
            search_mode=mode,
            telemetry={"candidate_count": len(candidates), "threshold": threshold},
        )

    @staticmethod
    def _is_chunk_relevant(chunk: DocumentChunk, case: EvalQueryCase) -> bool:
        text_lower = chunk.text.lower()
        sec_lower = (chunk.metadata.page_or_section or "").lower()
        if case.expected_section_substr and case.expected_section_substr.lower() in sec_lower:
            return True
        if case.expected_keywords:
            return any(kw.lower() in text_lower for kw in case.expected_keywords)
        return False

    def evaluate_retrieval(
        self,
        test_cases: List[EvalQueryCase],
        top_k: int = 5,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
    ) -> RetrievalEvalReport:
        if not test_cases:
            raise ValueError("test_cases cannot be empty.")

        hits = 0
        rr_sum = 0.0
        ndcg_sum = 0.0
        precision_sum = 0.0
        answerable_count = 0
        abstention_correct = 0
        details: List[Dict[str, Any]] = []

        for case in test_cases:
            res = self.retrieve(
                query=case.query,
                top_k=top_k,
                mode=mode,
                file_id_filter=file_id_filter,
            )

            if case.should_abstain:
                is_correct_abstain = res.abstained
                if is_correct_abstain:
                    abstention_correct += 1
                details.append({
                    "query": case.query,
                    "type": "unanswerable",
                    "abstained": res.abstained,
                    "passed": is_correct_abstain,
                    "max_confidence_score": res.max_confidence_score,
                })
                continue

            answerable_count += 1
            if not res.abstained:
                abstention_correct += 1

            rel_flags = [1 if self._is_chunk_relevant(c, case) else 0 for c in res.ranked_chunks]
            is_hit = any(rel_flags)
            if is_hit:
                hits += 1

            first_rank = next((i + 1 for i, flag in enumerate(rel_flags) if flag == 1), 0)
            rr = (1.0 / first_rank) if first_rank > 0 else 0.0
            rr_sum += rr

            dcg = sum(flag / math.log2(idx + 2) for idx, flag in enumerate(rel_flags))
            ideal_flags = sorted(rel_flags, reverse=True)
            idcg = sum(flag / math.log2(idx + 2) for idx, flag in enumerate(ideal_flags))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            ndcg_sum += ndcg

            prec = (sum(rel_flags) / len(rel_flags)) if rel_flags else 0.0
            precision_sum += prec

            details.append({
                "query": case.query,
                "type": "answerable",
                "hit": is_hit,
                "first_relevant_rank": first_rank,
                "reciprocal_rank": round(rr, 4),
                "ndcg": round(ndcg, 4),
                "context_precision": round(prec, 4),
                "max_confidence_score": res.max_confidence_score,
                "top_chunk_section": res.ranked_chunks[0].metadata.page_or_section if res.ranked_chunks else None,
            })

        denom = max(1, answerable_count)
        return RetrievalEvalReport(
            total_queries=len(test_cases),
            hit_rate_at_k=round(hits / denom, 4),
            mrr_score=round(rr_sum / denom, 4),
            ndcg_at_k=round(ndcg_sum / denom, 4),
            context_precision_at_k=round(precision_sum / denom, 4),
            abstention_accuracy=round(abstention_correct / len(test_cases), 4),
            per_query_details=details,
        )
