import re
import unicodedata
from typing import List, Dict, Any, Optional
from core.interfaces import BaseTextNormalizer
from core.schemas import RawDocument
from core.registry import registry

@registry.register_normalizer("unicode_whitespace_cleaner")
class UnicodeWhitespaceNormalizer(BaseTextNormalizer):
    def __init__(self, preserve_paragraphs: bool = True):
        self.preserve_paragraphs = preserve_paragraphs

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        
        # SWE Fix: Strip NUL bytes (0x00) to prevent PostgreSQL DataError
        text = text.replace("\x00", "")
        
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        if self.preserve_paragraphs:
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
            cleaned = "\n".join(lines)
            return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        return re.sub(r"\s+", " ", text).strip()


@registry.register_normalizer("arabic_text_normalizer")
class ArabicTextNormalizer(BaseTextNormalizer):
    def __init__(
        self,
        remove_diacritics: bool = True,
        remove_tatweel: bool = True,
        unify_alef: bool = True,
        unify_yeh: bool = True,
    ):
        self.remove_diacritics = remove_diacritics
        self.remove_tatweel = remove_tatweel
        self.unify_alef = unify_alef
        self.unify_yeh = unify_yeh

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        if self.remove_diacritics:
            text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
        if self.remove_tatweel:
            text = re.sub(r"\u0640+", "", text)
        if self.unify_alef:
            text = re.sub(r"[إأآٱ]", "ا", text)
        if self.unify_yeh:
            text = re.sub(r"ى", "ي", text)
        return text


@registry.register_normalizer("dictionary_abbreviation_expander")
class DictionaryAbbreviationNormalizer(BaseTextNormalizer):
    def __init__(self, custom_abbreviations: Optional[Dict[str, str]] = None):
        self.abbreviations = custom_abbreviations or {}

    def normalize(self, text: str) -> str:
        if not text or not self.abbreviations:
            return text
        for abbr, full_term in self.abbreviations.items():
            pattern = rf"\b{re.escape(abbr)}\b(?!\s*\({re.escape(full_term)}\))"
            text = re.sub(pattern, f"{abbr} ({full_term})", text)
        return text


@registry.register_normalizer("regex_pii_redactor")
class RegexPIIRedactor(BaseTextNormalizer):
    def __init__(self, replacement_token: str = "[REDACTED]"):
        self.replacement_token = replacement_token

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            self.replacement_token,
            text,
        )
        text = re.sub(
            r"\b(?:MRN|PID|ID)\s*[:#-]\s*\d+\b",
            f"ID: {self.replacement_token}",
            text,
            flags=re.IGNORECASE,
        )
        return text


class NormalizerChain:
    def __init__(self, chain_config: List[Dict[str, Any]]):
        self.steps: List[BaseTextNormalizer] = []
        for step_cfg in chain_config:
            name = step_cfg["name"]
            params = step_cfg.get("params", {})
            norm_cls = registry.get_normalizer(name)
            self.steps.append(norm_cls(**params))

    def apply(self, text: str) -> str:
        result = text
        for normalizer in self.steps:
            result = normalizer.normalize(result)
        return result

    def apply_to_documents(self, documents: List[RawDocument]) -> List[RawDocument]:
        normalized_docs: List[RawDocument] = []
        for doc in documents:
            cleaned_content = self.apply(doc.content)
            normalized_docs.append(
                RawDocument(
                    doc_id=doc.doc_id,
                    file_id=doc.file_id,
                    content=cleaned_content,
                    metadata=doc.metadata,
                )
            )
        return normalized_docs




@registry.register_normalizer("regex_noise_cleaner")
class RegexNoiseCleaner(BaseTextNormalizer):
    def __init__(self, patterns_to_remove: List[str]):
        self.patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns_to_remove]

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        for pattern in self.patterns:
            text = pattern.sub("", text)
        return text.strip()