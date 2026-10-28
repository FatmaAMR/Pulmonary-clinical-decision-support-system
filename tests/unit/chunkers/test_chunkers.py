import pytest
from core.schemas import SourceMetadata, RawDocument
from core.registry import registry
import components.chunkers


def test_chunkers_registered_in_registry():
    assert registry.get_chunker("recursive_chunker") is not None
    assert registry.get_chunker("markdown_section_chunker") is not None


def test_recursive_chunker_validates_parameters():
    chunker_cls = registry.get_chunker("recursive_chunker")
    with pytest.raises(ValueError, match="chunk_size must be strictly positive"):
        chunker_cls(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be >= 0 and < chunk_size"):
        chunker_cls(chunk_size=100, chunk_overlap=100)


def test_recursive_chunker_splits_long_text_and_preserves_metadata():
    chunker = registry.get_chunker("recursive_chunker")(chunk_size=80, chunk_overlap=15)
    meta = SourceMetadata(
        source_name="WHO Chest X-Ray Guideline",
        version_date="2025-01",
        source_link="https://who.int/cxr",
        file_format="pdf",
        page_or_section="Page 4",
    )
    text = (
        "Sentence one describes right lower lobe consolidation. "
        "Sentence two evaluates pleural effusion meniscus sign. "
        "Sentence three correlates fever and hypoxia with radiography."
    )
    doc = RawDocument(doc_id="doc_99", file_id="file_1", content=text, metadata=meta)

    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert len(c.text) <= 80
        assert c.doc_id == "doc_99"
        assert c.metadata.source_name == "WHO Chest X-Ray Guideline"
        assert f"Page 4 | Part {idx + 1}" == c.metadata.page_or_section

    # Verify deterministic chunk_id generation
    chunks_second_run = chunker.chunk(doc)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunks_second_run]


def test_section_aware_chunker_binds_clinical_headers_to_citation_metadata():
    chunker = registry.get_chunker("markdown_section_chunker")(chunk_size=200, chunk_overlap=20)
    meta = SourceMetadata(
        source_name="BTS Thoracic Protocol",
        version_date="2024",
        source_link="https://bts.org.uk",
        file_format="md",
    )
    markdown_guideline = (
        "# 1. Pneumonia Findings\n"
        "Airspace opacity with air bronchograms requires clinical correlation with fever.\n\n"
        "## 2. الإرتشاح البلوري\n"
        "يتم تقييم زاوية الضلع والحجاب الحاجز في أشعة الصدر الخلفية الأمامية."
    )
    doc = RawDocument(doc_id="doc_bts", content=markdown_guideline, metadata=meta)

    chunks = chunker.chunk(doc)

    assert len(chunks) == 2
    assert "1. Pneumonia Findings" in chunks[0].metadata.page_or_section
    assert "Airspace opacity" in chunks[0].text
    assert "2. الإرتشاح البلوري" in chunks[1].metadata.page_or_section
    assert "زاوية الضلع" in chunks[1].text