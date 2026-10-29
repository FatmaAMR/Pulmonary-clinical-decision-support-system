import re
from typing import Dict, Any, Optional, List, Set
from core.schemas import GenerationResult
from core.interfaces import BaseGenerator
from pipeline.retrieval_pipeline import RetrievalPipeline
from core.registry import registry
import components.generators


def _tokenize_content(text: str) -> Set[str]:
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "was", "were",
        "are", "have", "has", "had", "not", "but", "what", "who", "when",
        "where", "how", "which", "into", "their", "they", "them", "can"
    }
    return {
        w for w in re.findall(r"[a-zA-Z0-9_\u0600-\u06FF]{3,}", text.lower())
        if w not in stop
    }


def _clean_markdown_artifacts(text: str) -> str:
    cleaned = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    return cleaned.strip()


class EndToEndRAGPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        retrieval_pipeline: RetrievalPipeline,
        generator: Optional[BaseGenerator] = None,
    ):
        self.config = config
        self.retrieval_pipeline = retrieval_pipeline

        guard_cfg = config.get("guardrails", {})
        self.min_citation_overlap = int(guard_cfg.get("min_citation_token_overlap", 3))
        self.max_snippet_chars = int(guard_cfg.get("max_snippet_preview_chars", 220))
        self.refusal_phrases = tuple(
            guard_cfg.get(
                "refusal_phrases",
                [
                    "couldn't find specific pediatric clinical guidelines",
                    "do not specifically discuss",
                    "does not contain",
                    "do not cover",
                    "cannot answer",
                    "insufficient",
                    "how can i help",
                    "feel free to ask",
                ],
            )
        )

        if generator is not None:
            self.generator = generator
        else:
            gen_cfg = config.get("generation", {"name": "groq_generator", "params": {}})
            gen_cls = registry.get_generator(gen_cfg["name"])
            self.generator = gen_cls(**gen_cfg.get("params", {}))

    def run(
        self,
        query: str,
        top_k: int = 3,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
        abstention_threshold_override: Optional[float] = None,
    ) -> GenerationResult:
        retrieval_res = self.retrieval_pipeline.retrieve(
            query=query,
            top_k=top_k,
            mode=mode,
            file_id_filter=file_id_filter,
            abstention_threshold_override=abstention_threshold_override,
        )

        if retrieval_res.abstained:
            return GenerationResult(
                query=query,
                answer="I couldn't find relevant clinical guidelines for this question in the knowledge base. Please ask a specific pediatric pulmonology question or upload the relevant guideline.",
                sources_cited=[],
                abstained=True,
                model_name="none",
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        gen_res = self.generator.generate(
            query=query,
            context_chunks=retrieval_res.ranked_chunks,
        )
        raw_answer = gen_res["answer"]

        is_refusal_or_greeting = any(p.lower() in raw_answer.lower() for p in self.refusal_phrases)

        if is_refusal_or_greeting:
            clean_ans = re.sub(r"\[Source\s+\d+\]", "", raw_answer, flags=re.IGNORECASE).strip()
            clean_ans = _clean_markdown_artifacts(clean_ans)
            return GenerationResult(
                query=query,
                answer=clean_ans,
                sources_cited=[],
                abstained=False,
                model_name=gen_res["model_name"],
                token_usage=gen_res.get("token_usage", {}),
            )

        cited_indices = {
            int(m) for m in re.findall(r"\[Source\s+(\d+)\]", raw_answer, flags=re.IGNORECASE)
        }

        if not cited_indices and retrieval_res.ranked_chunks:
            ans_tokens = _tokenize_content(raw_answer)
            inferred_refs = []
            for idx, c in enumerate(retrieval_res.ranked_chunks, start=1):
                c_tokens = _tokenize_content(c.text)
                if len(ans_tokens & c_tokens) >= self.min_citation_overlap:
                    cited_indices.add(idx)
                    inferred_refs.append(f"[Source {idx}]")
            if inferred_refs:
                raw_answer = f"{raw_answer.rstrip()} {' '.join(inferred_refs)}"

        clean_answer = _clean_markdown_artifacts(raw_answer)

        sources: List[Dict[str, Any]] = []
        for idx, c in enumerate(retrieval_res.ranked_chunks, start=1):
            if idx not in cited_indices:
                continue
            raw_link = getattr(c.metadata, "source_link", "") or ""
            if raw_link.startswith("http://") or raw_link.startswith("https://"):
                resolved_link = raw_link
            else:
                resolved_link = f"/documents/{c.file_id}/view"

            clean_snippet = _clean_markdown_artifacts(c.text)
            sources.append(
                {
                    "source_ref": f"[Source {idx}]",
                    "cited_in_answer": True,
                    "chunk_id": c.chunk_id,
                    "file_id": c.file_id,
                    "section": _clean_markdown_artifacts(c.metadata.page_or_section or "Guideline Section"),
                    "source_name": c.metadata.source_name if c.metadata.source_name != "string" else "Clinical Guideline Document",
                    "source_link": resolved_link,
                    "rerank_score": c.rerank_score,
                    "snippet": clean_snippet[: self.max_snippet_chars]
                    + ("..." if len(clean_snippet) > self.max_snippet_chars else ""),
                }
            )

        return GenerationResult(
            query=query,
            answer=clean_answer,
            sources_cited=sources,
            abstained=False,
            model_name=gen_res["model_name"],
            token_usage=gen_res.get("token_usage", {}),
        )
