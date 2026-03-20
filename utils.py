"""
utils.py - Misaada ya Mfumo (Helper Functions),

Inajumuisha:
-Kutambua lugha
-Kujenga prompts
-Kuandaa muktadha (context)
-Misaada mingine ya matumizi
"""

import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# KUTAMBUA LUGHA (Language Detection)
# Maneno ya kawaida ya Kiswahili
SWAHILI_MARKERS = {
    "na", "ya", "wa", "ni", "kwa", "katika", "je", "hii",
    "hiyo", "kwamba", "lakini", "pia", "au", "nini", "vipi",
    "jinsi", "tafadhali", "asante", "karibu", "habari", "swali",
    "jibu", "eleza", "niambie", "sema", "fanya", "pata",
}

# Maneno ya kawaida ya Kiingereza
ENGLISH_MARKERS = {
    "the", "is", "are", "was", "were", "what", "how", "why",
    "when", "where", "who", "which", "that", "this", "these",
    "those", "and", "or", "but", "with", "from", "into",
    "please", "explain", "tell", "give", "show", "find",
}


def detect_language(text: str) -> str:
    """
    Tambua lugha ya maandishi (Kiswahili au Kiingereza).

    Parameters,
    text : Maandishi ya kuchunguza

    Returns,
    str - 'sw' (Swahili), 'en' (English), au 'other'
    """
    if not text:
        return "en"

    words = set(re.findall(r"\b\w+\b", text.lower()))

    sw_count = len(words & SWAHILI_MARKERS)
    en_count = len(words & ENGLISH_MARKERS)

    if sw_count > en_count:
        return "sw"
    elif en_count > sw_count:
        return "en"
    elif sw_count > 0 or en_count > 0:
        return "sw"  # Chaguomsingi ni Kiswahili
    return "en"

# KUJENGA PROMPTS
SYSTEM_PROMPTS = {
    "sw": """Wewe ni msaidizi wa akili bandia anayeongea kwa Kiswahili na lugha nyingine.
Jibu kwa Kiswahili kama mtumiaji anaongea Kiswahili.
Tumia muktadha uliotolewa kujibu maswali kwa usahihi.
Kama jibu halipo kwenye muktadha, sema hilo wazi badala ya kubuni.
Jibu kwa ufupi na uwazi. Hakikisha maelezo yanaeleweka.""",

    "en": """You are a multilingual AI assistant with expertise in answering questions.
Reply in the same language the user is using.
Use the provided context to answer accurately.
If the answer is not in the context, say so clearly rather than guessing.
Be concise and clear in your explanations.""",

    "default": """You are a helpful multilingual AI assistant.
Answer in the same language the user is writing in.
Use the provided context documents to answer accurately.
If you cannot find the answer in the context, say so clearly.
Be helpful, accurate, and concise.""",
}

RAG_PROMPT_TEMPLATE = """
{history_section}

## Muktadha wa Nyaraka (Context from Documents):
{context}

## Swali / Question:
{question}

## Jibu / Answer:
""".strip()

RAG_PROMPT_NO_CONTEXT = """
{history_section}

Swali: {question}

(Hakuna nyaraka zinazohusiana zilizopatikana kwenye kumbukumbu.)

Jibu:
""".strip()


def build_system_prompt(language: str = "sw") -> str:
    """
    Jenga system prompt kulingana na lugha.

    Parameters,
    language : Msimbo wa lugha ('sw', 'en', n.k.)
    """
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["default"])


def build_rag_prompt(
    question: str,
    context: str,
    history: list[dict] | None = None,
    max_history_turns: int = 6,
) -> str:
    """
    Jenga prompt kamili ya RAG.

    Parameters,
    question         : Swali la mtumiaji
    context          : Muktadha uliotoka kwenye nyaraka
    history          : Historia ya mazungumzo
    max_history_turns: Idadi ya juu ya zamu za historia

    Returns,
    str - Prompt iliyoandaliwa
    """
    # Andaa historia
    history_section = ""
    if history:
        recent = history[-max_history_turns:]
        lines = ["## Historia ya Mazungumzo / Conversation History:"]
        for turn in recent:
            role_label = "Mtumiaji" if turn["role"] == "user" else "Msaidizi"
            lines.append(f"{role_label}: {truncate_text(turn['content'], 200)}")
        history_section = "\n".join(lines)

    if context.strip():
        return RAG_PROMPT_TEMPLATE.format(
            history_section=history_section,
            context=context,
            question=question,
        )
    else:
        return RAG_PROMPT_NO_CONTEXT.format(
            history_section=history_section,
            question=question,
        )


def format_context(
    semantic_results: list[dict],
    episodic_context: list[str] | None = None,
    max_context_chars: int = 3000,
) -> str:
    """
    Andaa muktadha kutoka kwenye matokeo ya utafutaji.

    Parameters,
    semantic_results  : Matokeo ya semantic search
    episodic_context  : Historia inayohusiana
    max_context_chars : Ukubwa wa juu wa muktadha (herufi)

    Returns,
    str - Muktadha ulioandaliwa
    """
    sections = []
    total_chars = 0

    # Nyaraka za semantic memory
    if semantic_results:
        sections.append("### Kutoka Nyaraka (From Documents):")
        for i, result in enumerate(semantic_results, 1):
            source = result.get("source", "Haijulikani")
            text = result.get("text", "")
            score = result.get("score", 0)

            entry = f"[{i}] ({source}, ufanani={score:.2f}):\n{text}"
            if total_chars + len(entry) > max_context_chars:
                break
            sections.append(entry)
            total_chars += len(entry)

    # Historia inayohusiana (episodic)
    if episodic_context:
        sections.append("\n### Kutoka Historia (From Conversation History):")
        for ctx in episodic_context[:2]:
            entry = truncate_text(ctx, 300)
            if total_chars + len(entry) > max_context_chars:
                break
            sections.append(entry)
            total_chars += len(entry)

    return "\n\n".join(sections)

# MISAADA MINGINE (Other Helpers)
def truncate_text(text: str, max_chars: int = 200, suffix: str = "...") -> str:
    """Funga maandishi kwa urefu uliopewa."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def clean_filename(name: str) -> str:
    """Safisha jina la faili ili liwe salama."""
    cleaned = re.sub(r"[^\w\s\-_.]", "", name)
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.lower()


def format_timestamp(dt: datetime | None = None) -> str:
    """Rudisha timestamp iliyoandikwa vizuri."""
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_size(bytes_count: int) -> str:
    """Badilisha baiti kuwa muundo unaosomeka na binadamu."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_count < 1024:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f} TB"


def setup_logging(level: str = "INFO") -> None:
    """Sanidi mfumo wa kurekodi matukio (logging)."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Punguza kelele za maktaba za nje
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def print_banner():
    """Chapisha ujumbe wa karibuni."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          🌍  RAG Multilingual System  🌍                    
║                                                              
║  LLM   : DeepSeek Coder (4-bit quantized via Ollama)        
║  Embed : Granite Multilingual Embeddings                     
║  DB    : ChromaDB (Persistent)                               
║  Mem   : Semantic + Episodic + Cache                         
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_sources(sources: list[str]):
    """Chapisha vyanzo kwa muundo mzuri."""
    if not sources:
        return
    print("\n📚 Vyanzo / Sources:")
    for i, src in enumerate(sources, 1):
        print(f"   {i}. {src}")


def print_stats(stats: dict):
    """Chapisha takwimu za mfumo."""
    print("\n📊 Takwimu za Mfumo / System Statistics:")
    mem = stats.get("memory", {})
    sem = mem.get("semantic_memory", {})
    epi = mem.get("episodic_memory", {})
    cache = mem.get("cache", {})

    print(f"   Nyaraka (vipande)  : {sem.get('vipande', 0)}")
    print(f"   Vyanzo             : {len(sem.get('vyanzo', []))}")
    print(f"   Historia (zamu)    : {epi.get('zamu', 0)}")
    print(f"   Mazungumzo         : {len(epi.get('sessions', []))}")
    print(f"   Cache              : {cache.get('total_cached', 0)}")
