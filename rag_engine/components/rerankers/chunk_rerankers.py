import re
from typing import List, Set, Optional
from core.interfaces import BaseReranker
from core.schemas import ScoredChunk
from core.registry import registry

DEFAULT_STOPWORDS: Set[str] = {
    "what", "is", "are", "how", "does", "do", "the", "a", "an", "in", "on", "of",
    "to", "for", "and", "or", "with", "by", "from", "about", "explain", "tell",
    "me", "work", "works", "working", "can", "you", "who", "where", "when", "why"
}


@registry.register_reranker("hybrid_overlap_reranker")
class HybridOverlapReranker(BaseReranker):
    def __init__(
        self,
        semantic_weight: float = 0.60,
        lexical_weight: float = 0.40,
        section_boost: float = 0.15,
        stopwords: Optional[List[str]] = None,
    ):
        self.semantic_weight = float(semantic_weight)
        self.lexical_weight = float(lexical_weight)
        self.section_boost = float(section_boost)
        self.stopwords: Set[str] = set(stopwords) if stopwords is not None else DEFAULT_STOPWORDS

    def _extract_meaningful_stems(self, text: str) -> Set[str]:
        raw_tokens = re.findall(r"[a-zA-Z0-9_\u0600-\u06FF]{2,}", text.lower())
        stems = set()
        for t in raw_tokens:
            if t in self.stopwords:
                continue
            stem = re.sub(r"(ing|ed|es|s)$", "", t) if len(t) > 4 else t
            stems.add(stem)
        return stems

    def rerank(
        self,
        query: str,
        candidates: Optional[List[ScoredChunk]] = None,
        chunks: Optional[List[ScoredChunk]] = None,
        top_k: int = 5,
    ) -> List[ScoredChunk]:
        items = candidates if candidates is not None else (chunks or [])
        if not items:
            return []

        q_stems = self._extract_meaningful_stems(query)
        if not q_stems:
            q_stems = set(re.findall(r"[a-zA-Z0-9_\u0600-\u06FF]{2,}", query.lower()))

        for cand in items:
            c_stems = self._extract_meaningful_stems(cand.text)
            sec_stems = self._extract_meaningful_stems(cand.metadata.page_or_section or "")

            overlap_ratio = len(q_stems & c_stems) / len(q_stems) if q_stems else 0.0
            sec_boost = self.section_boost if (q_stems & sec_stems) else 0.0

            sem_clamped = max(0.0, min(1.0, cand.similarity_score))
            lex_combined = max(cand.lexical_score, overlap_ratio)

            final_score = (
                (self.semantic_weight * sem_clamped)
                + (self.lexical_weight * lex_combined)
                + sec_boost
            )
            cand.rerank_score = round(min(1.0, final_score), 4)

        ranked = sorted(items, key=lambda x: x.rerank_score, reverse=True)
        return ranked[:top_k]