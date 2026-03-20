"""
ollama_client.py - Kuwasiliana na Ollama API
Inasimamia LLM (DeepSeek Coder 4-bit) na Embeddings (Granite Multilingual).
"""

import time
import logging
from typing import Generator
import ollama
from src.config import config

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Mteja wa Ollama - anashughulikia mawasiliano yote na Ollama API.
    
    Mifano:
    Kutoa jibu:
        client = OllamaClient()
        jibu = client.generate("Swali langu hapa")

    Kutoa embeddings:
        vekta = client.embed("Maandishi yangu")
    """

    def __init__(self):
        self.client = ollama.Client(host=config.ollama.base_url)
        self.llm_model = config.ollama.llm_model
        self.embed_model = config.ollama.embed_model
        self._verify_connection()

    def _verify_connection(self):
        """Hakikisha Ollama inaendesha na modeli zipo."""
        try:
            models = self.client.list()
            available = [m.model for m in models.models]
            logger.info(f"Ollama imeunganishwa. Modeli zinazopatikana: {len(available)}")

            if not any(self.llm_model.split(":")[0] in m for m in available):
                logger.warning(
                    f"⚠️  LLM modeli '{self.llm_model}' haijapatikana.\n"
                    f"   Tekeleza: ollama pull {self.llm_model}"
                )
            if not any(self.embed_model.split(":")[0] in m for m in available):
                logger.warning(
                    f"⚠️  Embedding modeli '{self.embed_model}' haijapatikana.\n"
                    f"   Tekeleza: ollama pull {self.embed_model}"
                )
        except Exception as e:
            logger.error(f"❌ Haiwezekani kuungana na Ollama: {e}")
            raise ConnectionError(
                f"Ollama haiendesha kwenye {config.ollama.base_url}.\n"
                "Tekeleza: ollama serve"
            ) from e

    # LLM Generation
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str | Generator:
        """
        Toa jibu kutoka LLM (DeepSeek Coder).

        Parameters,
        prompt      : Swali au amri kwa LLM
        system      : Ujumbe wa mfumo (system prompt)
        temperature : Kiwango cha ubunifu (0=makini, 1=ubunifu)
        max_tokens  : Idadi ya juu ya tokeni katika jibu
        stream      : Kama True, rejesha generator ya majibu kwa wakati halisi

        Returns,
        str au Generator
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        try:
            if stream:
                return self._stream_generate(messages, options)
            else:
                response = self.client.chat(
                    model=self.llm_model,
                    messages=messages,
                    options=options,
                    stream=False,
                )
                return response.message.content

        except Exception as e:
            logger.error(f"Hitilafu wakati wa kutengeneza jibu: {e}")
            raise

    def _stream_generate(self, messages: list, options: dict) -> Generator:
        """Generator ya majibu ya wakati halisi (streaming)."""
        for chunk in self.client.chat(
            model=self.llm_model,
            messages=messages,
            options=options,
            stream=True,
        ):
            if chunk.message.content:
                yield chunk.message.content

    def chat_with_history(
        self,
        messages: list[dict],
        system: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Ongea na LLM ukitumia historia ya mazungumzo.

        Parameters,
        messages : Orodha ya ujumbe {'role': 'user'/'assistant', 'content': '...'}
        system   : Ujumbe wa mfumo
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        try:
            response = self.client.chat(
                model=self.llm_model,
                messages=full_messages,
                options={"temperature": temperature},
                stream=False,
            )
            return response.message.content
        except Exception as e:
            logger.error(f"Hitilafu katika mazungumzo: {e}")
            raise
            
    # Embeddings
    def embed(self, text: str) -> list[float]:
        """
        Tengeneza embedding vector kwa maandishi moja.

        Parameters,
        text : Maandishi ya kugeuzwa kuwa vector

        Returns,
        list[float] - Vector ya embedding
        """
        try:
            response = self.client.embeddings(
                model=self.embed_model,
                prompt=text,
            )
            return response.embedding
        except Exception as e:
            logger.error(f"Hitilafu katika embedding: {e}")
            raise

    def embed_batch(self, texts: list[str], delay: float = 0.05) -> list[list[float]]:
        """
        Tengeneza embeddings kwa orodha ya maandishi.

        Parameters,
        texts  : Orodha ya maandishi
        delay  : Muda wa kusubiri kati ya maombi (sekunde)

        Returns,
        list[list[float]] - Orodha ya vectors
        """
        embeddings = []
        for i, text in enumerate(texts):
            emb = self.embed(text)
            embeddings.append(emb)
            if delay > 0 and i < len(texts) - 1:
                time.sleep(delay)
        logger.info(f"Embeddings {len(embeddings)} zimekamilika.")
        return embeddings

    # Model Info
    def get_model_info(self) -> dict:
        """Pata taarifa za modeli zinazotumika."""
        try:
            llm_info = self.client.show(self.llm_model)
            return {
                "llm_model": self.llm_model,
                "embed_model": self.embed_model,
                "llm_parameters": llm_info.get("details", {}),
            }
        except Exception as e:
            logger.warning(f"Haiwezekani kupata taarifa za modeli: {e}")
            return {"llm_model": self.llm_model, "embed_model": self.embed_model}

    def list_models(self) -> list[str]:
        """Orodha ya modeli zote zilizowekwa kwenye Ollama."""
        models = self.client.list()
        return [m.model for m in models.models]
