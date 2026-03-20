"""
test_memory.py - Majaribio ya Mfumo wa Kumbukumbu,

Tekeleza:  pytest tests/test_memory.py -v
"""

import pytest
import sys
import os
import tempfile
import shutil

# Ongeza src kwenye njia ya Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Maandalizi (Fixtures)
@pytest.fixture(scope="module")
def temp_chroma_dir():
    """Saraka ya muda kwa ChromaDB wakati wa majaribio."""
    tmp = tempfile.mkdtemp(prefix="rag_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def mock_embedding():
    """Embedding ya bandia kwa majaribio (dimension 384)."""
    return [0.1] * 384


@pytest.fixture(scope="module")
def mock_embedding_alt():
    """Embedding tofauti kidogo."""
    return [0.2] * 384

# Majaribio ya TextSplitter
class TestTextSplitter:
    """Majaribio ya kigawanyaji cha maandishi."""

    def setup_method(self):
        from src.document_processor import TextSplitter
        self.splitter = TextSplitter(chunk_size=200, chunk_overlap=20)

    def test_split_short_text(self):
        """Maandishi mafupi hayagawanywi."""
        text = "Hii ni sentensi fupi sana."
        chunks = self.splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        """Maandishi marefu yanagawanywa vizuri."""
        text = " ".join(["Hii ni sentensi ndefu sana."] * 30)
        chunks = self.splitter.split(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0

    def test_split_empty_text(self):
        """Maandishi tupu hurejesha orodha tupu."""
        assert self.splitter.split("") == []
        assert self.splitter.split("   ") == []

    def test_split_with_paragraphs(self):
        """Maandishi yenye aya yanagawanywa kwa aya."""
        text = "Aya ya kwanza yenye maneno mengi.\n\nAya ya pili tofauti kabisa.\n\nAya ya tatu."
        chunks = self.splitter.split(text)
        assert len(chunks) >= 1

    def test_chunk_size_respected(self):
        """Vipande havizidi ukubwa uliowekwa."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
        text = "x " * 500  # 1000 herufi
        chunks = splitter.split(text)
        for chunk in chunks:
            assert len(chunk) <= 150  # Ruhusa kidogo zaidi kwa overlap

# Majaribio ya DocumentProcessor
class TestDocumentProcessor:
    """Majaribio ya kisindika hati."""

    def setup_method(self):
        from src.document_processor import DocumentProcessor
        self.processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)

    def test_process_text(self):
        """Kusindika maandishi ya moja kwa moja."""
        text = "Akili Bandia ni tawi la sayansi ya kompyuta. Inajumuisha ujifunzaji wa mashine."
        chunks = self.processor.process_text(text, source="jaribio")
        assert len(chunks) >= 1
        assert all(c.source == "jaribio" for c in chunks)
        assert all(len(c.text) > 0 for c in chunks)

    def test_process_txt_file(self, tmp_path):
        """Kusindika faili ya maandishi."""
        test_file = tmp_path / "jaribio.txt"
        test_file.write_text(
            "Hii ni jaribio la kwanza.\n\nHii ni aya ya pili ya maandishi.",
            encoding="utf-8",
        )
        chunks = self.processor.process_file(test_file)
        assert len(chunks) >= 1
        assert chunks[0].source == "jaribio.txt"

    def test_unsupported_file(self, tmp_path):
        """Faili isiyoungwa mkono itoe kosa."""
        bad_file = tmp_path / "jaribio.xyz"
        bad_file.write_text("data")
        with pytest.raises(ValueError, match="Aina ya faili"):
            self.processor.process_file(bad_file)

    def test_file_not_found(self):
        """Faili isiyopo itoe kosa."""
        with pytest.raises(FileNotFoundError):
            self.processor.process_file("/hapana/faili/hii.pdf")

    def test_get_file_info(self, tmp_path):
        """Pata taarifa za faili."""
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf")
        info = self.processor.get_file_info(f)
        assert info["name"] == "test.pdf"
        assert info["extension"] == ".pdf"
        assert info["supported"] is True

# Majaribio ya SemanticMemory
class TestSemanticMemory:
    """Majaribio ya kumbukumbu ya maana."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_chroma_dir, monkeypatch):
        """Sanidi kumbukumbu ya muda."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", temp_chroma_dir)
        monkeypatch.setenv("CHROMA_NYARAKA_COLLECTION", "test_nyaraka")

        # Reimport na mipangilio mpya
        import importlib
        import src.config as cfg_module
        importlib.reload(cfg_module)
        cfg_module.config.chroma.persist_dir = temp_chroma_dir
        cfg_module.config.chroma.nyaraka_collection = "test_nyaraka"

        from src.memory_mapping import SemanticMemory
        self.memory = SemanticMemory()

    def test_add_and_search(self, mock_embedding, mock_embedding_alt):
        """Ongeza vipande na vitafute."""
        chunks = ["Habari za Kiswahili", "Teknolojia ya kisasa"]
        embeddings = [mock_embedding, mock_embedding_alt]

        ids = self.memory.add_chunks(chunks, embeddings, source="jaribio.txt")
        assert len(ids) == 2

        results = self.memory.search(mock_embedding, top_k=2)
        assert len(results) > 0
        assert "text" in results[0]
        assert "source" in results[0]
        assert "score" in results[0]

    def test_count(self, mock_embedding):
        """Hesabu vipande vilivyohifadhiwa."""
        initial = self.memory.count()
        self.memory.add_chunks(["Kipande kipya"], [mock_embedding], source="kipimo.txt")
        assert self.memory.count() == initial + 1

    def test_get_sources(self, mock_embedding):
        """Pata orodha ya vyanzo."""
        self.memory.add_chunks(["A"], [mock_embedding], source="chanzo_a.pdf")
        sources = self.memory.get_sources()
        assert "chanzo_a.pdf" in sources

    def test_delete_source(self, mock_embedding):
        """Futa vipande vya chanzo fulani."""
        self.memory.add_chunks(["Futa hii"], [mock_embedding], source="kufutwa.txt")
        self.memory.delete_source("kufutwa.txt")
        sources = self.memory.get_sources()
        assert "kufutwa.txt" not in sources

# Majaribio ya EpisodicMemory
class TestEpisodicMemory:
    """Majaribio ya kumbukumbu ya matukio."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_chroma_dir):
        import src.config as cfg_module
        cfg_module.config.chroma.persist_dir = temp_chroma_dir
        cfg_module.config.chroma.memory_collection = "test_memory"

        from src.memory_mapping import EpisodicMemory
        self.memory = EpisodicMemory()

    def test_add_and_retrieve(self, mock_embedding):
        """Ongeza na pata historia."""
        session_id = "jaribio_session_001"
        self.memory.add_turn(session_id, "user", "Swali langu", mock_embedding)
        self.memory.add_turn(session_id, "assistant", "Jibu langu", mock_embedding)

        history = self.memory.get_session_history(session_id)
        assert len(history) >= 2

    def test_session_isolation(self, mock_embedding):
        """Kila session ina historia yake."""
        self.memory.add_turn("session_A", "user", "Swali A", mock_embedding)
        self.memory.add_turn("session_B", "user", "Swali B", mock_embedding)

        history_a = self.memory.get_session_history("session_A")
        history_b = self.memory.get_session_history("session_B")

        contents_a = [h["content"] for h in history_a]
        contents_b = [h["content"] for h in history_b]

        assert not any(c in contents_b for c in contents_a if c)

    def test_formatted_history(self, mock_embedding):
        """Historia ina muundo unaofaa kwa Ollama."""
        session_id = "format_test"
        self.memory.add_turn(session_id, "user", "Hello", mock_embedding)
        self.memory.add_turn(session_id, "assistant", "Habari", mock_embedding)

        fmt = self.memory.get_formatted_history(session_id)
        for turn in fmt:
            assert "role" in turn
            assert "content" in turn
            assert turn["role"] in ("user", "assistant")

    def test_list_sessions(self, mock_embedding):
        """Pata orodha ya sessions."""
        self.memory.add_turn("sess_list_1", "user", "A", mock_embedding)
        self.memory.add_turn("sess_list_2", "user", "B", mock_embedding)
        sessions = self.memory.list_sessions()
        assert "sess_list_1" in sessions
        assert "sess_list_2" in sessions

# Majaribio ya SemanticCache
class TestSemanticCache:
    """Majaribio ya cache ya maana."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_chroma_dir):
        import src.config as cfg_module
        cfg_module.config.chroma.persist_dir = temp_chroma_dir
        cfg_module.config.chroma.cache_collection = "test_cache"
        cfg_module.config.memory.cache_similarity_threshold = 0.90

        from src.memory_mapping import SemanticCache
        self.cache = SemanticCache()

    def test_store_and_lookup_hit(self, mock_embedding):
        """Cache inapata jibu la awali kwa swali linalolingana."""
        self.cache.store("Swali la jaribio", mock_embedding, "Jibu la jaribio")
        result = self.cache.lookup(mock_embedding, threshold=0.90)
        assert result is not None
        assert result["answer"] == "Jibu la jaribio"
        assert result["from_cache"] is True
        assert result["score"] >= 0.90

    def test_lookup_miss_on_empty(self, mock_embedding):
        """Cache tupu haileti jibu."""
        from src.memory_mapping import SemanticCache
        import src.config as cfg_module
        cfg_module.config.chroma.cache_collection = "test_cache_empty"
        empty_cache = SemanticCache()
        result = empty_cache.lookup(mock_embedding)
        assert result is None

    def test_cache_stats(self, mock_embedding):
        """Takwimu za cache zinaonyesha idadi sahihi."""
        initial = self.cache.get_cache_stats()["total_cached"]
        self.cache.store("Swali jipya", mock_embedding, "Jibu jipya")
        stats = self.cache.get_cache_stats()
        assert stats["total_cached"] == initial + 1

# Majaribio ya Utils
class TestUtils:
    """Majaribio ya misaada."""

    def test_detect_swahili(self):
        from src.utils import detect_language
        assert detect_language("Je, unaweza kunieleza kuhusu teknolojia?") == "sw"

    def test_detect_english(self):
        from src.utils import detect_language
        assert detect_language("What is artificial intelligence and how does it work?") == "en"

    def test_truncate_text(self):
        from src.utils import truncate_text
        text = "A" * 300
        result = truncate_text(text, max_chars=100)
        assert len(result) <= 100
        assert result.endswith("...")

    def test_truncate_short_text(self):
        from src.utils import truncate_text
        text = "Habari"
        assert truncate_text(text, max_chars=100) == text

    def test_format_context_empty(self):
        from src.utils import format_context
        result = format_context([], [])
        assert result == "" or isinstance(result, str)

    def test_build_system_prompt(self):
        from src.utils import build_system_prompt
        sw_prompt = build_system_prompt("sw")
        en_prompt = build_system_prompt("en")
        assert "Kiswahili" in sw_prompt or len(sw_prompt) > 10
        assert len(en_prompt) > 10

    def test_build_rag_prompt_with_context(self):
        from src.utils import build_rag_prompt
        prompt = build_rag_prompt(
            question="Swali langu?",
            context="Hii ni muktadha.",
            history=[{"role": "user", "content": "Habari"}, {"role": "assistant", "content": "Nzuri"}],
        )
        assert "Swali langu?" in prompt
        assert "Hii ni muktadha." in prompt

# Tekeleza kama script
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
