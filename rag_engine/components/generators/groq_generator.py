import os
from typing import List, Dict, Any, Optional
from core.interfaces import BaseGenerator
from core.registry import registry
from core.config_loader import load_dotenv_file

DEFAULT_SYSTEM_PROMPT = (
    "You are a strict, evidence-based technical AI assistant.\n"
    "RULES:\n"
    "1. Answer the question using ONLY the provided Context Chunks.\n"
    "2. You MUST cite the exact source number at the end of every factual sentence using [Source 1], [Source 2], etc.\n"
    "3. Never use outside knowledge. If the context does not contain the answer, state clearly that the provided documents do not cover that specific detail."
)

DEFAULT_USER_TEMPLATE = (
    "Context Chunks:\n{context_text}\n\n"
    "Question: {query}\n\n"
    "Provide a clear answer with mandatory inline [Source X] citations:"
)


@registry.register_generator("groq_generator")
class GroqGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str = "allam-2-7b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        preferred_defaults: Optional[List[str]] = None,
        excluded_model_keywords: Optional[List[str]] = None,
    ):
        load_dotenv_file()
        self.model_name = str(model_name)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        self.user_prompt_template = (user_prompt_template or DEFAULT_USER_TEMPLATE).strip()
        self.preferred_defaults = tuple(
            preferred_defaults or ("allam-2-7b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b")
        )
        self.excluded_keywords = tuple(
            excluded_model_keywords or ("whisper", "orpheus", "tts", "guard", "playai", "embed")
        )
        self._explicit_api_key = api_key
        self._client = None
        self._available_models: Optional[List[str]] = None

    def _get_client(self):
        if self._client is None:
            load_dotenv_file()
            resolved_key = self._explicit_api_key or os.getenv("GROQ_API_KEY")
            if not resolved_key:
                raise ValueError("Groq API key is missing. Set GROQ_API_KEY in .env or environment.")
            from groq import Groq
            self._client = Groq(api_key=resolved_key)
        return self._client

    def list_available_models(self) -> List[str]:
        if self._available_models is None:
            client = self._get_client()
            raw_models = [m.id for m in client.models.list().data]
            self._available_models = sorted([
                m for m in raw_models
                if not any(ex in m.lower() for ex in self.excluded_keywords)
            ])
        return self._available_models

    def _resolve_active_model(self) -> str:
        available = self.list_available_models()
        if not available or self.model_name in available:
            return self.model_name
        for pref in self.preferred_defaults:
            if pref in available:
                self.model_name = pref
                return self.model_name
        self.model_name = available[0]
        return self.model_name

    def generate(self, query: str, context_chunks: List[Any]) -> Dict[str, Any]:
        client = self._get_client()
        active_model = self._resolve_active_model()

        context_text = "\n\n".join(
            f"[Source {i+1}] (Section: {c.metadata.page_or_section} | Source: {c.metadata.source_name})\n{c.text}"
            for i, c in enumerate(context_chunks)
        )

        user_prompt = self.user_prompt_template.format(
            context_text=context_text,
            query=query,
        )

        response = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = (response.choices[0].message.content or "").strip()
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }

        return {
            "answer": answer,
            "model_name": active_model,
            "token_usage": usage,
        }


@registry.register_generator("extractive_mock_generator")
class ExtractiveMockGenerator(BaseGenerator):
    def __init__(self, model_name: str = "extractive-deterministic-v1", **kwargs):
        self.model_name = model_name

    def list_available_models(self) -> List[str]:
        return [self.model_name]

    def generate(self, query: str, context_chunks: List[Any]) -> Dict[str, Any]:
        if not context_chunks:
            return {
                "answer": "Insufficient context to answer.",
                "model_name": self.model_name,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        top = context_chunks[0]
        synthesized = f"Based on {top.metadata.page_or_section}: {top.text} [Source 1]"
        return {
            "answer": synthesized,
            "model_name": self.model_name,
            "token_usage": {
                "prompt_tokens": len(query.split()),
                "completion_tokens": len(synthesized.split()),
                "total_tokens": len(query.split()) + len(synthesized.split()),
            },
        }