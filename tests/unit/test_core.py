import pytest
from dataclasses import FrozenInstanceError
from core.schemas import SourceMetadata, RawDocument, DocumentChunk
from core.interfaces import BaseFileParser, BaseTextNormalizer
from core.registry import ComponentRegistry

def test_source_metadata_validation_and_immutability():
    meta = SourceMetadata(
        source_name="WHO Chest Radiography Guideline",
        version_date="2025-01",
        source_link="https://who.int/guidelines/cxr",
        file_format="pdf",
        page_or_section="Section 3.2"
    )
    assert meta.source_name == "WHO Chest Radiography Guideline"

    # Verify immutability (frozen=True)
    with pytest.raises(FrozenInstanceError):
        meta.source_name = "Modified Name"

    # Verify empty source_name raises ValueError
    with pytest.raises(ValueError, match="source_name cannot be empty"):
        SourceMetadata(source_name="   ", version_date="2025", source_link="link", file_format="pdf")

def test_registry_registers_and_retrieves_valid_parser():
    temp_registry = ComponentRegistry()

    @temp_registry.register_parser(".dicom")
    class DummyDicomParser(BaseFileParser):
        def parse(self, file_path, base_metadata):
            return []

    # Should retrieve whether we pass '.dicom' or 'DICOM'
    retrieved_cls = temp_registry.get_parser("DICOM")
    assert retrieved_cls is DummyDicomParser

def test_registry_rejects_class_not_inheriting_interface():
    temp_registry = ComponentRegistry()

    with pytest.raises(TypeError, match="must inherit from BaseTextNormalizer"):
        @temp_registry.register_normalizer("bad_normalizer")
        class InvalidNormalizer:
            def normalize(self, text: str) -> str:
                return text

def test_registry_raises_key_error_for_unregistered_format():
    temp_registry = ComponentRegistry()
    with pytest.raises(KeyError, match="No parser registered for extension '.xyz'"):
        temp_registry.get_parser(".xyz")