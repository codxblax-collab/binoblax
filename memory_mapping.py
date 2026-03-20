"""
memory_mapping.py - Mfumo wa Kumbukumbu (Memory Mapping)

Aina tatu za kumbukumbu:

1. SEMANTIC MEMORY (Kumbukumbu ya Maana)
   -Hifadhi ya nyaraka zilizosindikwa
   -Vector search kwa maudhui ya hati
   -Inabaki daima (persistent)

2. EPISODIC MEMORY (Kumbukumbu ya Matukio)
   -Historia ya mazungumzo
   -Mfululizo wa maswali na majibu
   -Inabaki daima (persistent)

3. SEMANTIC CACHE (Cache ya Maana)
   -Hifadhi ya maswali yaliyoulizwa awali
   -Kuepuka kuuliza LLM tena kwa maswali yanayofanana
   -Inabaki daima (persistent)
"""

import uuid
import logging
from datetime import datetime
from typing import Optional
import chromadb
from chromadb.config import Settings
from src.config import config

logger = logging.getLogger(__name__)

# Msingi wa Memory (Base Class)

class BaseMemory:
    """Darasa la msingi kwa aina zote za kumbukumbu."""

    def __init__(self, collection_name: str, persist_dir: str):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Mkusanyiko '{collection_name}' umepakiwa. Vipande: {self.collection.count()}")

    def count(self) -> int:
        """Idadi ya vipande vilivyohifadhiwa."""
        return self.collection.count()

    def delete_all(self):
        """Futa kumbukumbu yote katika mkusanyiko huu."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Mkusanyiko '{self.collection_name}' umefutwa na kuanzishwa upya.")

# 1. SEMANTIC MEMORY - Kumbukumbu ya Maana (Nyaraka)

class SemanticMemory(BaseMemory):
    """
    Kumbukumbu ya Maana - hifadhi ya nyaraka na vipande vyake.
    
    Mfano:
        mem = SemanticMemory()
        mem.add_chunks(chunks, embeddings, source="hati.pdf")
        matokeo = mem.search("swali langu", embedding=[...])
    """

    def __init__(self):
        super().__init__(
            collection_name=config.chroma.nyaraka_collection,
            persist_dir=config.chroma.persist_dir,
        )

    def add_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        source: str,
        metadata_extra: dict | None = None,
    ) -> list[str]:
        """
        Ongeza vipande vya hati kwenye kumbukumbu ya maana.

        Parameters,
        chunks         : Orodha ya vipande vya maandishi
        embeddings     : Vectors za kila kipande
        source         : Jina la faili chanzo
        metadata_extra : Metadata ya ziada (optional)

        Returns,
        list[str] - IDs za vipande vilivyohifadhiwa
        """
        if not chunks or not embeddings:
            return []

        ids = [str(uuid.uuid4()) for _ in chunks]
        timestamp = datetime.now().isoformat()

        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = {
                "source": source,
                "chunk_index": i,
                "timestamp": timestamp,
                "char_count": len(chunk),
            }
            if metadata_extra:
                meta.update(metadata_extra)
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(f"Vipande {len(chunks)} vimehifadhiwa kutoka '{source}'.")
        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Tafuta vipande vinavyofanana na swali.

        Parameters,
        query_embedding : Vector ya swali
        top_k           : Idadi ya matokeo
        source_filter   : Chuja kwa faili maalum

        Returns,
        list[dict] - Matokeo yenye 'text', 'source', 'score', 'metadata'
        """
        top_k = top_k or config.memory.semantic_top_k
        where = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({
                    "text": doc,
                    "source": meta.get("source", "Haijulikani"),
                    "score": round(1 - dist, 4),  # cosine similarity
                    "metadata": meta,
                })
        return output

    def get_sources(self) -> list[str]:
        """Pata orodha ya vyanzo vyote vilivyohifadhiwa."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        sources = list({m.get("source", "") for m in results["metadatas"]})
        return sorted(sources)

    def delete_source(self, source: str):
        """Futa vipande vyote kutoka chanzo fulani."""
        self.collection.delete(where={"source": source})
        logger.info(f"Vipande vya '{source}' vimefutwa.")

# 2. EPISODIC MEMORY - Kumbukumbu ya Matukio (Mazungumzo)

class EpisodicMemory(BaseMemory):
    """
    Kumbukumbu ya Matukio - hifadhi ya historia ya mazungumzo.

    Mfano:
        mem = EpisodicMemory()
        mem.add_turn(session_id="abc", role="user", content="Habari?", embedding=[...])
        historia = mem.get_session_history("abc")
    """

    def __init__(self):
        super().__init__(
            collection_name=config.chroma.memory_collection,
            persist_dir=config.chroma.persist_dir,
        )

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: list[float],
        extra_meta: dict | None = None,
    ) -> str:
        """
        Ongeza zamu moja ya mazungumzo (swali au jibu).

        Parameters,
        session_id : Kitambulisho cha kipindi (session)
        role       : 'user' au 'assistant'
        content    : Maandishi ya ujumbe
        embedding  : Vector ya ujumbe
        extra_meta : Metadata ya ziada
        """
        turn_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        meta = {
            "session_id": session_id,
            "role": role,
            "timestamp": timestamp,
            "char_count": len(content),
        }
        if extra_meta:
            meta.update(extra_meta)

        self.collection.add(
            ids=[turn_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta],
        )

        # Simamia ukubwa wa kumbukumbu
        self._enforce_memory_limit(session_id)
        return turn_id

    def get_session_history(self, session_id: str) -> list[dict]:
        """
        Pata historia yote ya mazungumzo ya kipindi fulani.
        Imepangwa kwa wakati (timestamp).
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )

        if not results["documents"]:
            return []

        turns = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            turns.append({
                "role": meta.get("role", "user"),
                "content": doc,
                "timestamp": meta.get("timestamp", ""),
                "session_id": session_id,
            })

        # Panga kwa wakati
        turns.sort(key=lambda x: x["timestamp"])
        return turns

    def search_similar_turns(
        self,
        query_embedding: list[float],
        session_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Tafuta mazungumzo yanayofanana na swali.
        Unaweza kuchuja kwa session maalum au kutafuta kote.
        """
        top_k = top_k or config.memory.episodic_top_k
        if self.collection.count() == 0:
            return []

        where = {"session_id": session_id} if session_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({
                    "content": doc,
                    "role": meta.get("role", "user"),
                    "session_id": meta.get("session_id", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "score": round(1 - dist, 4),
                })
        return output

    def get_formatted_history(self, session_id: str) -> list[dict]:
        """
        Pata historia katika muundo wa Ollama messages.
        [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        """
        turns = self.get_session_history(session_id)
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    def list_sessions(self) -> list[str]:
        """Orodha ya vitambulisho vyote vya mazungumzo."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        sessions = list({m.get("session_id", "") for m in results["metadatas"]})
        return sorted(sessions)

    def delete_session(self, session_id: str):
        """Futa historia ya kipindi fulani."""
        self.collection.delete(where={"session_id": session_id})
        logger.info(f"Historia ya session '{session_id}' imefutwa.")

    def _enforce_memory_limit(self, session_id: str):
        """Ondoa mazungumzo ya zamani zaidi ukizidi kikomo."""
        if self.collection.count() == 0:
            return

        results = self.collection.get(
            where={"session_id": session_id},
            include=["metadatas"],
        )

        if len(results["ids"]) > config.memory.max_episodic_memory:
            # Pata mazungumzo ya zamani zaidi
            paired = list(zip(results["ids"], results["metadatas"]))
            paired.sort(key=lambda x: x[1].get("timestamp", ""))
            excess = len(paired) - config.memory.max_episodic_memory
            ids_to_delete = [p[0] for p in paired[:excess]]
            self.collection.delete(ids=ids_to_delete)
            logger.debug(f"Kumbukumbu {excess} za zamani zimefutwa.")

# 3. SEMANTIC CACHE - Cache ya Maana.

class SemanticCache(BaseMemory):
    """
    Cache ya Maana - hifadhi ya majibu ya maswali yaliyoulizwa awali.
    Epuka kuuliza LLM tena kwa maswali yanayofanana sana.

    Mfano:
        cache = SemanticCache()
        
        # Hifadhi jibu
        cache.store("Swali langu", embedding=[...], answer="Jibu langu")
        
        # Tafuta jibu la awali
        jibu = cache.lookup("Swali sawa", embedding=[...])
        if jibu:
            print("Cache inajibu:", jibu)
    """

    def __init__(self):
        super().__init__(
            collection_name=config.chroma.cache_collection,
            persist_dir=config.chroma.persist_dir,
        )
        self.threshold = config.memory.cache_similarity_threshold

    def store(
        self,
        question: str,
        embedding: list[float],
        answer: str,
        metadata: dict | None = None,
    ) -> str:
        """
        Hifadhi swali na jibu lake kwenye cache.

        Parameters,
        question  : Swali lililoulizwa
        embedding : Vector ya swali
        answer    : Jibu lililotolewa na LLM
        metadata  : Taarifa za ziada
        """
        cache_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        meta = {
            "answer": answer[:1000],  # Epuka metadata kubwa sana
            "timestamp": timestamp,
            "question_length": len(question),
            "answer_length": len(answer),
        }
        if metadata:
            meta.update(metadata)

        self.collection.add(
            ids=[cache_id],
            embeddings=[embedding],
            documents=[question],
            metadatas=[meta],
        )
        logger.debug(f"Jibu limehifadhiwa kwenye cache (ID: {cache_id[:8]}...).")
        return cache_id

    def lookup(
        self,
        query_embedding: list[float],
        threshold: float | None = None,
    ) -> Optional[dict]:
        """
        Tafuta jibu sawa kwenye cache.

        Parameters,
        query_embedding : Vector ya swali kipya
        threshold       : Kikomo cha ufanani (0-1)

        Returns,
        dict au None - {'question': ..., 'answer': ..., 'score': ...}
        """
        if self.collection.count() == 0:
            return None

        threshold = threshold or self.threshold

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1 - distance

        if similarity >= threshold:
            return {
                "question": results["documents"][0][0],
                "answer": results["metadatas"][0][0].get("answer", ""),
                "score": round(similarity, 4),
                "timestamp": results["metadatas"][0][0].get("timestamp", ""),
                "from_cache": True,
            }
        return None

    def get_cache_stats(self) -> dict:
        """Pata takwimu za cache."""
        return {
            "total_cached": self.collection.count(),
            "threshold": self.threshold,
            "collection": self.collection_name,
        }

    def clear_old_cache(self, older_than_days: int = 7):
        """Futa cache za zamani zaidi ya siku fulani."""
        # ChromaDB haiwezi kuchuja kwa tarehe moja kwa moja,
        # kwa hivyo tunafuta kwa kuhesabu mwenyewe
        if self.collection.count() == 0:
            return

        cutoff = datetime.now().timestamp() - (older_than_days * 86400)
        results = self.collection.get(include=["metadatas"])

        ids_to_delete = []
        for id_, meta in zip(results["ids"], results["metadatas"]):
            ts = meta.get("timestamp", "")
            if ts:
                try:
                    entry_time = datetime.fromisoformat(ts).timestamp()
                    if entry_time < cutoff:
                        ids_to_delete.append(id_)
                except ValueError:
                    pass

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Cache {len(ids_to_delete)} za zamani zimefutwa.")

# MEMORY MANAGER - Msimamizi wa Kumbukumbu Zote

class MemoryManager:
    """
    Msimamia mkuu wa kumbukumbu zote tatu.
    Hutoa kiolesura kimoja cha kupata semantic, episodic, na cache.

    Mfano:
        mm = MemoryManager()
        
        # Hifadhi nyaraka
        mm.semantic.add_chunks(chunks, embeddings, source="doc.pdf")
        
        # Ongeza mazungumzo
        mm.episodic.add_turn(session_id="s1", role="user", ...)
        
        # Hifadhi na tafuta cache
        mm.cache.store("swali", embedding, "jibu")
        jibu = mm.cache.lookup(embedding)
    """

    def __init__(self):
        logger.info("Inaanzisha MemoryManager...")
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.cache = SemanticCache()
        logger.info("✅ MemoryManager imeanzishwa kikamilifu.")

    def get_stats(self) -> dict:
        """Pata takwimu za kumbukumbu zote."""
        return {
            "semantic_memory": {
                "vipande": self.semantic.count(),
                "vyanzo": self.semantic.get_sources(),
            },
            "episodic_memory": {
                "zamu": self.episodic.count(),
                "sessions": self.episodic.list_sessions(),
            },
            "cache": self.cache.get_cache_stats(),
        }

    def reset_all(self):
        """⚠️ Futa kumbukumbu ZOTE. Tumia kwa tahadhari!"""
        self.semantic.delete_all()
        self.episodic.delete_all()
        self.cache.delete_all()
        logger.warning("⚠️ Kumbukumbu zote zimefutwa!")

    def reset_conversations(self):
        """Futa mazungumzo yote (episodic memory)."""
        self.episodic.delete_all()
        self.cache.delete_all()
        logger.info("Mazungumzo yote yamefutwa.")
