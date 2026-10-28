import re
import hashlib
from dataclasses import replace
from typing import List, Optional, Any
from core.interfaces import BaseChunker
from core.schemas import RawDocument, DocumentChunk
from core.registry import registry


def _generate_deterministic_chunk_id(doc_id: str, index: int, text: str) -> str:
    short_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}_chk_{index}_{short_hash}"


def _detect_content_type(text: str) -> str:
    stripped = text.strip()
    if "|" in stripped and re.search(r"\|[\s:-]+\|", stripped):
        return "table"
    if stripped.startswith("```") or "def " in stripped or "class " in stripped:
        return "code"
    if re.search(r"^(\s*[-*•]|\s*\d+\.)\s+", stripped, re.MULTILINE):
        return "list"
    return "prose"


@registry.register_chunker("recursive_chunker")
class RecursiveHierarchyChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: Any = 512,
        chunk_overlap: Any = 64,
        boundary_lookback_ratio: Any = 0.6,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        # Defensive casting in case values come as unparsed template strings or raw types
        cs = 512 if isinstance(chunk_size, str) and "${" in chunk_size else int(chunk_size)
        co = 64 if isinstance(chunk_overlap, str) and "${" in chunk_overlap else int(chunk_overlap)
        blr = 0.6 if isinstance(boundary_lookback_ratio, str) and "${" in boundary_lookback_ratio else float(boundary_lookback_ratio)

        if cs <= 0:
            raise ValueError("chunk_size must be strictly positive.")
        if co < 0 or co >= cs:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

        self.chunk_size = cs
        self.chunk_overlap = co
        self.boundary_lookback_ratio = blr
        self.separators = separators or ["\n\n", "\n", ". ", "؟ ", "! ", " "]

    def _split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        pattern = r"(\n\n|\n|(?<=[.؟!])\s+)"
        raw_parts = re.split(pattern, text)
        segments = [p for p in raw_parts if p and not p.isspace()]

        chunks: List[str] = []
        current_chunk = ""

        for seg in segments:
            candidate = f"{current_chunk} {seg}".strip() if current_chunk else seg.strip()
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    tail_start = len(current_chunk) - self.chunk_overlap
                    if tail_start > 0 and current_chunk[tail_start - 1] not in " \n\t":
                        space_idx = current_chunk.find(" ", tail_start)
                        if space_idx != -1:
                            tail_start = space_idx + 1
                    overlap_tail = current_chunk[max(0, tail_start):].strip() if self.chunk_overlap > 0 else ""
                    current_chunk = f"{overlap_tail} {seg}".strip() if overlap_tail else seg.strip()
                else:
                    current_chunk = seg.strip()

                min_cut = int(self.chunk_size * self.boundary_lookback_ratio)
                while len(current_chunk) > self.chunk_size:
                    cut_point = self.chunk_size
                    while cut_point > min_cut and current_chunk[cut_point - 1] not in " \n\t":
                        cut_point -= 1
                    if cut_point <= min_cut:
                        cut_point = self.chunk_size

                    chunks.append(current_chunk[:cut_point].strip())

                    next_start = max(0, cut_point - self.chunk_overlap)
                    if next_start > 0 and next_start < cut_point and current_chunk[next_start - 1] not in " \n\t":
                        first_space = current_chunk.find(" ", next_start, cut_point)
                        if first_space != -1:
                            next_start = first_space + 1
                    current_chunk = current_chunk[next_start:].strip()

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk(self, document: RawDocument) -> List[DocumentChunk]:
        text_pieces = self._split_text(document.content)
        result: List[DocumentChunk] = []

        base_section = document.metadata.page_or_section or "Document"
        for idx, piece in enumerate(text_pieces):
            chunk_meta = replace(
                document.metadata,
                page_or_section=f"{base_section} | Part {idx + 1}",
            )
            chunk_id = _generate_deterministic_chunk_id(document.doc_id, idx, piece)
            result.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    file_id=document.file_id,
                    chunk_index=idx,
                    text=piece,
                    metadata=chunk_meta,
                    content_type=_detect_content_type(piece),
                )
            )
        return result


@registry.register_chunker("markdown_section_chunker")
class MarkdownSectionChunker(BaseChunker):
    HEADER_REGEX = re.compile(r"^(#{1,4}\s+.+|\d+(?:\.\d+)*\s+[A-Z\u0600-\u06FF].+)$", re.MULTILINE)

    def __init__(
        self,
        chunk_size: Any = 512,
        chunk_overlap: Any = 64,
        boundary_lookback_ratio: Any = 0.6,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        self.inner_chunker = RecursiveHierarchyChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            boundary_lookback_ratio=boundary_lookback_ratio,
            separators=separators,
            **kwargs,
        )

    def chunk(self, document: RawDocument) -> List[DocumentChunk]:
        content = document.content.strip()
        if not content:
            return []

        matches = list(self.HEADER_REGEX.finditer(content))
        if not matches:
            return self.inner_chunker.chunk(document)

        all_chunks: List[DocumentChunk] = []
        chunk_counter = 0

        if matches[0].start() > 0:
            intro_text = content[: matches[0].start()].strip()
            if intro_text:
                intro_doc = RawDocument(
                    doc_id=document.doc_id,
                    file_id=document.file_id,
                    content=intro_text,
                    metadata=replace(document.metadata, page_or_section="Introduction"),
                )
                for c in self.inner_chunker.chunk(intro_doc):
                    c.chunk_index = chunk_counter
                    c.chunk_id = _generate_deterministic_chunk_id(document.doc_id, chunk_counter, c.text)
                    all_chunks.append(c)
                    chunk_counter += 1

        for i, match in enumerate(matches):
            section_title = match.group(1).lstrip("#").strip()
            start_idx = match.start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_body = content[start_idx:end_idx].strip()

            section_meta = replace(
                document.metadata,
                page_or_section=section_title,
            )
            section_doc = RawDocument(
                doc_id=document.doc_id,
                file_id=document.file_id,
                content=section_body,
                metadata=section_meta,
            )

            sub_chunks = self.inner_chunker.chunk(section_doc)
            for sub in sub_chunks:
                sub.chunk_index = chunk_counter
                sub.chunk_id = _generate_deterministic_chunk_id(document.doc_id, chunk_counter, sub.text)
                all_chunks.append(sub)
                chunk_counter += 1

        return all_chunks
