"""
rag_engine.py - Injini Kuu ya RAG (Retrieval-Augmented Generation).

Mtiririko wa kazi (Pipeline):
1.INGIZA HATI  → process_file() → embed → store in SemanticMemory
2.ULIZA SWALI  → embed swali → lookup cache
                             ↓ (cache miss)
                           search SemanticMemory
                           search EpisodicMemory (historia)
                           build prompt
                           query LLM (DeepSeek Coder)
                           store kwenye Cache
                           store kwenye EpisodicMemory
                             ↓
                        JIBU LINAREJESHWA
"""

import uuid
import logging
from typing import Generator, Optional
from dataclasses import dataclass, field

from src.config import config
from src.ollama_client import OllamaClient
from src.memory_mapping import MemoryManager
from src.document_processor import DocumentProcessor, DocumentChunk
from src.utils import (
    detect_language,
    format_context,
    build_rag_prompt,
    build_system_prompt,
    truncate_text,
)

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Jibu kamili la mfumo wa RAG."""
    answer: str
    session_id: str
    question: str
    sources: list[str] = field(default_factory=list)
    from_cache: bool = False
    language: str = "auto"
    context_chunks: int = 0
    episodic_turns: int = 0
    metadata: dict = field(default_factory=dict)


class RAGEngine:
    """
    Injini Kuu ya RAG - inaunganisha vipande vyote vya mfumo.

    Mfano wa Matumizi:
        engine = RAGEngine()

        # 1.Ingiza hati
        engine.ingest_file("data/documents/swahili_tech.pdf")

        # 2.Uliza maswali
        session = "mazungumzo_01"
        jibu = engine.ask("Eleza akili bandia", session_id=session)
        print(jibu.answer)

        # 3.Endelea mazungumzo
        jibu2 = engine.ask("Niambie zaidi", session_id=session)
    """

    def __init__(self):
        logger.info("Inaanzisha RAGEngine...")
        self.ollama = OllamaClient()
        self.memory = MemoryManager()
        self.processor = DocumentProcessor(
            chunk_size=config.documents.chunk_size,
            chunk_overlap=config.documents.chunk_overlap,
        )
        logger.info("✅ RAGEngine imeanzishwa.")

    # UINGIZAJI WA HATI (Document Ingestion)
    def ingest_file(self, filepath: str) -> dict:
        """
        Ingiza faili kwenye mfumo wa kumbukumbu.

        Mtiririko:
        1. Soma na ugawanye faili katika vipande
        2. Tengeneza embeddings kwa kila kipande
        3. Hifadhi kwenye SemanticMemory (ChromaDB)

        Parameters,
        filepath : Njia ya faili (PDF, TXT, DOCX, MD)

        Returns,
        dict - Muhtasari wa uingizaji
        """
        logger.info(f"Inaingiza faili: {filepath}")

        # 1.Sindika faili
        chunks = self.processor.process_file(filepath)
        if not chunks:
            return {"success": False, "message": "Hakuna maandishi yaliyopatikana.", "chunks": 0}

        # 2.Tengeneza embeddings
        texts = [c.text for c in chunks]
        logger.info(f"Inatengeneza embeddings kwa vipande {len(texts)}...")
        embeddings = self.ollama.embed_batch(texts)

        # 3.Hifadhi kwenye SemanticMemory
        source_name = chunks[0].source
        ids = self.memory.semantic.add_chunks(
            chunks=texts,
            embeddings=embeddings,
            source=source_name,
            metadata_extra={"total_chunks": len(chunks)},
        )

        result = {
            "success": True,
            "source": source_name,
            "chunks": len(chunks),
            "ids": len(ids),
            "message": f"✅ '{source_name}' imesindikwa kikamilifu.",
        }
        logger.info(result["message"])
        return result

    def ingest_directory(self, directory: str) -> list[dict]:
        """
        Ingiza faili zote kutoka saraka.

        Parameters,
        directory : Saraka yenye faili za kuingiza

        Returns,
        list[dict] - Matokeo ya kila faili
        """
        results = []
        seen_sources = set()
        chunks_buffer: list[DocumentChunk] = []

        for chunk in self.processor.process_directory(directory):
            chunks_buffer.append(chunk)
            seen_sources.add(chunk.source)

        # Panga kwa chanzo na uingize kwa kundi
        from itertools import groupby
        chunks_buffer.sort(key=lambda c: c.source)

        for source, group in groupby(chunks_buffer, key=lambda c: c.source):
            group_chunks = list(group)
            texts = [c.text for c in group_chunks]
            embeddings = self.ollama.embed_batch(texts)
            ids = self.memory.semantic.add_chunks(
                chunks=texts,
                embeddings=embeddings,
                source=source,
            )
            results.append({
                "success": True,
                "source": source,
                "chunks": len(group_chunks),
                "ids": len(ids),
            })
            logger.info(f"✅ '{source}': vipande {len(group_chunks)}")

        return results

    def ingest_text(self, text: str, source_name: str = "manual_input") -> dict:
        """
        Ingiza maandishi moja kwa moja (bila faili).

        Parameters
        text        : Maandishi ya kuingiza
        source_name : Jina la chanzo
        """
        chunks = self.processor.process_text(text, source=source_name)
        if not chunks:
            return {"success": False, "message": "Maandishi tupu.", "chunks": 0}

        texts = [c.text for c in chunks]
        embeddings = self.ollama.embed_batch(texts)
        ids = self.memory.semantic.add_chunks(texts, embeddings, source=source_name)

        return {
            "success": True,
            "source": source_name,
            "chunks": len(chunks),
            "ids": len(ids),
        }

    # MASWALI NA MAJIBU (Question Answering)
    def ask(
        self,
        question: str,
        session_id: str | None = None,
        top_k: int | None = None,
        use_cache: bool = True,
        use_history: bool = True,
        stream: bool = False,
    ) -> RAGResponse | Generator:
        """
        Uliza swali na upate jibu likiwa na muktadha wa nyaraka.

        Parameters,
        question    : Swali la mtumiaji
        session_id  : Kitambulisho cha kipindi (hutolewa kama hakipo)
        top_k       : Idadi ya vipande vya muktadha
        use_cache   : Tumia cache au la
        use_history : Tumia historia ya mazungumzo
        stream      : Streamu jibu kwa wakati halisi

        Returns
        RAGResponse au Generator (kama stream=True)
        """
        session_id = session_id or str(uuid.uuid4())
        top_k = top_k or config.memory.semantic_top_k
        if config.language != "auto":
            language = config.language
        else:
            language = detect_language(question)

        logger.info(f"[{session_id[:8]}...] Swali: {truncate_text(question, 80)}")

        # 1.Tengeneza embedding ya swali
        q_embedding = self.ollama.embed(question)

        # 2.Angalia cache
        if use_cache:
            cached = self.memory.cache.lookup(q_embedding)
            if cached:
                logger.info(f"✅ Jibu limetoka cache (score={cached['score']}).")
                # Hifadhi mazungumzo hata kama jibu limetoka cache
                self._save_to_episodic(session_id, question, cached["answer"], q_embedding)
                return RAGResponse(
                    answer=cached["answer"],
                    session_id=session_id,
                    question=question,
                    from_cache=True,
                    language=language,
                    metadata={"cache_score": cached["score"]},
                )

        # 3.Tafuta kwenye SemanticMemory
        semantic_results = self.memory.semantic.search(q_embedding, top_k=top_k)
        logger.info(f"Matokeo ya semantic: {len(semantic_results)} vipande.")

        # 4.Pata historia ya mazungumzo
        history = []
        episodic_context = []
        if use_history:
            history = self.memory.episodic.get_formatted_history(session_id)
            # Pia tafuta mazungumzo yanayofanana kutoka sessions zote
            episodic_results = self.memory.episodic.search_similar_turns(
                q_embedding,
                top_k=config.memory.episodic_top_k,
            )
            episodic_context = [r["content"] for r in episodic_results]

        # 5.Jenga prompt
        context = format_context(semantic_results, episodic_context)
        sources = list({r["source"] for r in semantic_results})
        system = build_system_prompt(language)
        prompt = build_rag_prompt(question, context, history)

        # 6.Uliza LLM
        if stream:
            return self._stream_answer(
                prompt=prompt,
                system=system,
                session_id=session_id,
                question=question,
                q_embedding=q_embedding,
                sources=sources,
                language=language,
                context_chunks=len(semantic_results),
                episodic_turns=len(history),
            )

        answer = self.ollama.generate(prompt=prompt, system=system, temperature=0.7)

        # 7.Hifadhi matokeo
        self._save_to_episodic(session_id, question, answer, q_embedding)
        if use_cache and semantic_results:
            self.memory.cache.store(question, q_embedding, answer)

        return RAGResponse(
            answer=answer,
            session_id=session_id,
            question=question,
            sources=sources,
            from_cache=False,
            language=language,
            context_chunks=len(semantic_results),
            episodic_turns=len(history),
        )

    def _stream_answer(
        self,
        prompt: str,
        system: str,
        session_id: str,
        question: str,
        q_embedding: list[float],
        sources: list[str],
        language: str,
        context_chunks: int,
        episodic_turns: int,
    ) -> Generator:
        """Generator ya majibu ya wakati halisi."""
        full_answer = ""
        for token in self.ollama.generate(prompt=prompt, system=system, stream=True):
            full_answer += token
            yield token

        # Hifadhi baada ya kukamilika
        self._save_to_episodic(session_id, question, full_answer, q_embedding)
        if sources:
            self.memory.cache.store(question, q_embedding, full_answer)

    def _save_to_episodic(
        self, session_id: str, question: str, answer: str, q_embedding: list[float]
    ):
        """Hifadhi zamu za mazungumzo kwenye EpisodicMemory."""
        # Hifadhi swali la mtumiaji
        self.memory.episodic.add_turn(
            session_id=session_id,
            role="user",
            content=question,
            embedding=q_embedding,
        )
        # Hifadhi jibu la msaidizi
        a_embedding = self.ollama.embed(answer[:500])  
        # Epuka embedding kubwa sana
        self.memory.episodic.add_turn(
            session_id=session_id,
            role="assistant",
            content=answer,
            embedding=a_embedding,
        )

  
    # USIMAMIZI (Management)
    def get_status(self) -> dict:
        """Pata hali ya mfumo wote."""
        stats = self.memory.get_stats()
        model_info = self.ollama.get_model_info()
        return {
            "status": "online",
            "app": config.app_name,
            "version": config.version,
            "models": model_info,
            "memory": stats,
        }

    def list_documents(self) -> list[str]:
        """Orodha ya nyaraka zilizohifadhiwa."""
        return self.memory.semantic.get_sources()

    def delete_document(self, source: str):
        """Futa nyaraka kutoka kumbukumbu."""
        self.memory.semantic.delete_source(source)

    def new_session(self) -> str:
        """Anza kipindi kipya cha mazungumzo."""
        return str(uuid.uuid4())

    def get_session_history(self, session_id: str) -> list[dict]:
        """Pata historia ya mazungumzo ya kipindi."""
        return self.memory.episodic.get_session_history(session_id)

    def clear_session(self, session_id: str):
        """Futa historia ya kipindi fulani."""
        self.memory.episodic.delete_session(session_id)

    def reset_all(self):
        """⚠️ Futa KILA KITU - nyaraka na mazungumzo yote."""
        self.memory.reset_all()
