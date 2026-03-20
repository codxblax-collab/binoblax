"""
document_processor.py - Kusoma na Kusindika Nyaraka

Inasaidia:
-PDF (kupitia pypdf)
-Maandishi ya kawaida (.txt, .md)
 Word documents (.docx)

Kazi kuu:
-Kusoma faili
-Kugawanya maandishi katika vipande (chunking)
-Kuondoa maandishi yasiyo na maana
"""

import re
import logging
from pathlib import Path
from typing import Generator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Kipande kimoja cha hati iliyosindikwa."""
    text: str
    source: str
    chunk_index: int
    page_number: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        return f"Chunk(source={self.source!r}, idx={self.chunk_index}, chars={len(self.text)})"


class TextSplitter:
    """
    Kigawanyaji cha maandishi katika vipande vya ukubwa unaofaa.
    Inazingatia mpaka wa sentensi au aya.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        """
        Gawanya maandishi katika vipande.

        Parameters,
        text : Maandishi kamili ya kugawanya

        Returns,
        list[str] - Orodha ya vipande
        """
        if not text or not text.strip():
            return []

        # Safisha maandishi
        text = self._clean_text(text)

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        # Gawanya kwa aya kwanza
        paragraphs = self._split_by_paragraphs(text)

        current_chunk = ""
        for para in paragraphs:
            if len(para) > self.chunk_size:
                # Aya ndefu - igawanye zaidi kwa sentensi
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                sub_chunks = self._split_long_paragraph(para)
                chunks.extend(sub_chunks)
            elif len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk += ("\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Anza kipande kipya na mwingiliano (overlap)
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + ("\n" if overlap_text else "") + para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Chuja vipande vifupi sana
        chunks = [c for c in chunks if len(c) > 30]
        return chunks

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Gawanya kwa aya (mistari miwili au zaidi ya nafasi)."""
        paragraphs = re.split(r"\n\s*\n+", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_long_paragraph(self, text: str) -> list[str]:
        """Gawanya aya ndefu kwa sentensi."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= self.chunk_size:
                current += (" " if current else "") + sent
            else:
                if current:
                    chunks.append(current.strip())
                overlap_text = self._get_overlap(current)
                current = overlap_text + (" " if overlap_text else "") + sent

                # Sentensi ndefu zaidi ya chunk_size - igawanye kwa nguvu
                if len(current) > self.chunk_size:
                    while len(current) > self.chunk_size:
                        chunks.append(current[:self.chunk_size].strip())
                        current = current[self.chunk_size - self.chunk_overlap:]
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _get_overlap(self, text: str) -> str:
        """Pata sehemu ya mwisho ya kipande kwa mwingiliano."""
        if not text or self.chunk_overlap <= 0:
            return ""
        words = text.split()
        overlap_words = words[-max(1, self.chunk_overlap // 5):]
        return " ".join(overlap_words)

    def _clean_text(self, text: str) -> str:
        """Safisha maandishi yasiyo na maana."""
        # Ondoa herufi nyingi za nafasi
        text = re.sub(r" {3,}", " ", text)
        # Ondoa mistari tupu mingi
        text = re.sub(r"\n{4,}", "\n\n", text)
        # Ondoa herufi zisizoweza kusomwa
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()


class DocumentProcessor:
    """
    Kisindika Hati - husoma faili za aina mbalimbali na kuzigawanya.

    Mfano:
        proc = DocumentProcessor()
        
        # Sindika faili moja
        chunks = proc.process_file("data/documents/hati.pdf")
        
        # Sindika saraka nzima
        for chunk in proc.process_directory("data/documents/"):
            print(chunk.text[:100])
    """

    SUPPORTED = {".pdf", ".txt", ".md", ".docx"}

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(f"DocumentProcessor imeanzishwa (chunk_size={chunk_size}, overlap={chunk_overlap}).")

    # Kiolesura cha Umma

    def process_file(self, filepath: str | Path) -> list[DocumentChunk]:
        """
        Sindika faili moja na urudishe vipande.

        Parameters,
        filepath : Njia ya faili

        Returns,
        list[DocumentChunk] - Orodha ya vipande vilivyosindikwa
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Faili haipatikani: {filepath}")
        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(f"Aina ya faili '{path.suffix}' haiwezi kusomwa.")

        logger.info(f"Inasindika: {path.name}")

        # Soma maandishi kulingana na aina ya faili
        if path.suffix.lower() == ".pdf":
            pages = self._read_pdf(path)
        elif path.suffix.lower() == ".docx":
            pages = self._read_docx(path)
        else:
            pages = self._read_text(path)

        # Gawanya katika vipande
        all_chunks = []
        chunk_idx = 0
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            raw_chunks = self.splitter.split(page_text)
            for raw in raw_chunks:
                chunk = DocumentChunk(
                    text=raw,
                    source=path.name,
                    chunk_index=chunk_idx,
                    page_number=page_num,
                    char_start=0,
                    char_end=len(raw),
                    metadata={"filepath": str(path), "extension": path.suffix},
                )
                all_chunks.append(chunk)
                chunk_idx += 1

        logger.info(f"✅ {path.name}: vipande {len(all_chunks)} vimetolewa.")
        return all_chunks

    def process_directory(self, directory: str | Path) -> Generator[DocumentChunk, None, None]:
        """
        Sindika faili zote katika saraka.

        Parameters,
        directory : Saraka yenye faili

        Yields,
        DocumentChunk - Vipande moja kwa moja
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Saraka haipatikani: {directory}")

        files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED
        ]

        if not files:
            logger.warning(f"Hakuna faili zinazoweza kusindikwa katika '{directory}'.")
            return

        logger.info(f"Faili {len(files)} zimepatikana katika '{directory}'.")
        for f in files:
            try:
                chunks = self.process_file(f)
                for chunk in chunks:
                    yield chunk
            except Exception as e:
                logger.error(f"Hitilafu kusindika '{f.name}': {e}")

    def process_text(self, text: str, source: str = "manual_input") -> list[DocumentChunk]:
        """
        Sindika maandishi yaliyoingizwa moja kwa moja (bila faili).

        Parameters,
        text   : Maandishi ya kusindika
        source : Jina la chanzo (kwa kumbukumbu)
        """
        raw_chunks = self.splitter.split(text)
        return [
            DocumentChunk(
                text=raw,
                source=source,
                chunk_index=i,
                page_number=0,
            )
            for i, raw in enumerate(raw_chunks)
        ]

    # Wasomaji wa Faili

    def _read_pdf(self, path: Path) -> list[tuple[int, str]]:
        """Soma faili ya PDF, rejesha orodha ya (ukurasa, maandishi)."""
        try:
            import pypdf
        except ImportError:
            raise ImportError("Sakinisha pypdf: pip install pypdf")

        pages = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    pages.append((i + 1, text))
                except Exception as e:
                    logger.warning(f"Ukurasa {i+1} hauwezi kusomwa: {e}")
        return pages

    def _read_docx(self, path: Path) -> list[tuple[int, str]]:
        """Soma faili ya Word (.docx)."""
        try:
            import docx
        except ImportError:
            raise ImportError("Sakinisha python-docx: pip install python-docx")

        doc = docx.Document(str(path))
        full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        return [(1, full_text)]

    def _read_text(self, path: Path) -> list[tuple[int, str]]:
        """Soma faili ya maandishi (.txt, .md)."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                text = path.read_text(encoding=enc)
                return [(1, text)]
            except (UnicodeDecodeError, LookupError):
                continue
        raise ValueError(f"Haiwezekani kusoma '{path.name}' kwa mwandiko wowote.")

    # Msaada

    def get_file_info(self, filepath: str | Path) -> dict:
        """Pata taarifa za faili bila kuisindika."""
        path = Path(filepath)
        return {
            "name": path.name,
            "extension": path.suffix,
            "size_kb": round(path.stat().st_size / 1024, 2) if path.exists() else 0,
            "supported": path.suffix.lower() in self.SUPPORTED,
        }
