"""
run.py - Script Kuu ya Kuanzisha Mfumo wa RAG Multilingual.

Matumizi:
    python run.py   # Anza mazungumzo ya kawaida
    python run.py --ingest <faili>   # Ingiza faili moja
    python run.py --ingest-dir <dir> # Ingiza saraka nzima
    python run.py --status  # Angalia hali ya mfumo
    python run.py --reset  # Futa kumbukumbu zote
    python run.py --session <id>  # Endelea mazungumzo ya awali
"""

import sys
import argparse
import logging
from pathlib import Path

from src.utils import setup_logging, print_banner, print_sources, print_stats
from src.config import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG Multilingual - Mfumo wa Maswali kwa Kiswahili na Lugha Nyingine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mifano:
  python run.py
  python run.py --ingest data/documents/hati.pdf
  python run.py --ingest-dir data/documents/
  python run.py --status
  python run.py --reset
  python run.py --session mazungumzo_01
        """,
    )
    parser.add_argument("--ingest", metavar="FAILI", help="Ingiza faili moja kwenye mfumo")
    parser.add_argument("--ingest-dir", metavar="SARAKA", help="Ingiza faili zote kutoka saraka")
    parser.add_argument("--status", action="store_true", help="Onyesha hali ya mfumo")
    parser.add_argument("--reset", action="store_true", help="Futa kumbukumbu zote")
    parser.add_argument("--reset-chats", action="store_true", help="Futa mazungumzo tu")
    parser.add_argument("--list-docs", action="store_true", help="Orodha ya nyaraka zilizohifadhiwa")
    parser.add_argument("--session", metavar="ID", help="Kitambulisho cha kipindi cha kuendelea")
    parser.add_argument("--stream", action="store_true", help="Onyesha jibu kwa wakati halisi")
    parser.add_argument("--no-cache", action="store_true", help="Zima cache")
    parser.add_argument("--top-k", type=int, default=None, help="Idadi ya vipande vya muktadha")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def initialize_engine():
    """Anzisha injini ya RAG kwa usalama."""
    try:
        from src.rag_engine import RAGEngine
        return RAGEngine()
    except ConnectionError as e:
        print(f"\n❌ HITILAFU: {e}")
        print("\n💡 Suluhisho:")
        print("   1. Sakinisha Ollama: https://ollama.ai")
        print("   2. Anzisha Ollama:   ollama serve")
        print(f"   3. Pakua LLM:       ollama pull {config.ollama.llm_model}")
        print(f"   4. Pakua Embed:     ollama pull {config.ollama.embed_model}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Hitilafu isiyotarajiwa: {e}")
        logging.exception("Hitilafu katika kuanzisha injini:")
        sys.exit(1)


def cmd_ingest(engine, filepath: str):
    """Ingiza faili moja."""
    print(f"\n📂 Inaingiza: {filepath}")
    try:
        result = engine.ingest_file(filepath)
        if result["success"]:
            print(f"✅ {result['message']}")
            print(f"   Vipande: {result['chunks']}")
        else:
            print(f"❌ {result['message']}")
    except FileNotFoundError:
        print(f"❌ Faili haipatikani: {filepath}")
    except ValueError as e:
        print(f"❌ Hitilafu ya faili: {e}")


def cmd_ingest_dir(engine, directory: str):
    """Ingiza faili zote kutoka saraka."""
    print(f"\n📁 Inaingiza saraka: {directory}")
    try:
        results = engine.ingest_directory(directory)
        if not results:
            print("⚠️  Hakuna faili zilizosindikwa.")
            return
        print(f"\n✅ Faili {len(results)} zimesindikwa:")
        for r in results:
            print(f"   • {r['source']}: vipande {r['chunks']}")
    except FileNotFoundError:
        print(f"❌ Saraka haipatikani: {directory}")


def cmd_status(engine):
    """Onyesha hali ya mfumo."""
    print("\n🔍 Inaangalia hali ya mfumo...")
    status = engine.get_status()
    print(f"\n✅ Mfumo: {status['status'].upper()}")
    print(f"   Jina   : {status['app']} v{status['version']}")
    print(f"   LLM    : {status['models']['llm_model']}")
    print(f"   Embed  : {status['models']['embed_model']}")
    print_stats(status)


def cmd_list_docs(engine):
    """Orodha ya nyaraka."""
    docs = engine.list_documents()
    if not docs:
        print("\n📚 Hakuna nyaraka zilizohifadhiwa.")
        return
    print(f"\n📚 Nyaraka {len(docs)} zilizohifadhiwa:")
    for i, doc in enumerate(docs, 1):
        print(f"   {i}. {doc}")


def cmd_reset(engine, chats_only: bool = False):
    """Futa kumbukumbu."""
    if chats_only:
        confirm = input("\n⚠️  Hakikisha: Futa mazungumzo yote? [n/Y] ").strip().lower()
        if confirm in ("y", "yes", "ndiyo", "ndio"):
            engine.memory.reset_conversations()
            print("✅ Mazungumzo yote yamefutwa.")
        else:
            print("❌ Imesimamishwa.")
    else:
        confirm = input("\n⚠️  TAHADHARI: Hii itafuta KILA KITU (nyaraka + mazungumzo). Endelea? [n/Y] ").strip().lower()
        if confirm in ("y", "yes", "ndiyo", "ndio"):
            engine.reset_all()
            print("✅ Kumbukumbu zote zimefutwa.")
        else:
            print("❌ Imesimamishwa.")


def interactive_mode(engine, args):
    """Hali ya mazungumzo ya kawaida ya mtumiaji."""
    print_banner()

    # Angalia nyaraka zilizopo
    docs = engine.list_documents()
    if docs:
        print(f"📚 Nyaraka zilizopakiwa: {len(docs)}")
        for d in docs[:5]:
            print(f"   • {d}")
        if len(docs) > 5:
            print(f"   ... na nyingine {len(docs)-5}")
    else:
        print("💡 Hakuna nyaraka. Ingiza kwa: python run.py --ingest <faili>")

    # Anza kipindi
    session_id = args.session or engine.new_session()
    print(f"\n🆔 Kipindi: {session_id[:16]}...")

    # Angalia historia ya awali
    history = engine.get_session_history(session_id)
    if history:
        print(f"📝 Historia ya awali: zamu {len(history)}")

    print("\n" + "─" * 60)
    print("Andika swali lako. Amri maalum:")
    print("  /exit   - Toka")
    print("  /status - Hali ya mfumo")
    print("  /docs   - Orodha ya nyaraka")
    print("  /clear  - Futa mazungumzo ya kipindi hiki")
    print("  /new    - Anza kipindi kipya")
    print("─" * 60 + "\n")

    while True:
        try:
            user_input = input("🗣  Wewe: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Kwaheri! Mazungumzo yamehifadhiwa.")
            break

        if not user_input:
            continue

        # Amri maalum
        if user_input.lower() in ("/exit", "/toka", "exit", "quit"):
            print("\n👋 Kwaheri!")
            break
        elif user_input.lower() == "/status":
            cmd_status(engine)
            continue
        elif user_input.lower() in ("/docs", "/nyaraka"):
            cmd_list_docs(engine)
            continue
        elif user_input.lower() == "/clear":
            engine.clear_session(session_id)
            print("✅ Mazungumzo ya kipindi hiki yamefutwa.")
            continue
        elif user_input.lower() == "/new":
            session_id = engine.new_session()
            print(f"✅ Kipindi kipya: {session_id[:16]}...")
            continue
        elif user_input.lower().startswith("/ingest "):
            filepath = user_input[8:].strip()
            cmd_ingest(engine, filepath)
            continue

        # Uliza swali
        print("\n🤖 Msaidizi: ", end="", flush=True)

        try:
            if args.stream:
                # Streaming mode
                full_answer = ""
                for token in engine.ask(
                    question=user_input,
                    session_id=session_id,
                    top_k=args.top_k,
                    use_cache=not args.no_cache,
                    stream=True,
                ):
                    print(token, end="", flush=True)
                    full_answer += token
                print()  # Mstari mpya baada ya jibu

            else:
                # Normal mode
                response = engine.ask(
                    question=user_input,
                    session_id=session_id,
                    top_k=args.top_k,
                    use_cache=not args.no_cache,
                    stream=False,
                )
                print(response.answer)

                # Onyesha vyanzo
                if response.sources:
                    print_sources(response.sources)

                # Onyesha kama jibu limetoka cache
                if response.from_cache:
                    print("   💾 (Jibu limetoka cache)")

        except Exception as e:
            print(f"\n❌ Hitilafu: {e}")
            logger.exception("Hitilafu wakati wa kujibu:")

        print()  # Nafasi kati ya mazungumzo


def main():
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Mfumo wa RAG Multilingual unaanzishwa...")

    # Anzisha injini
    engine = initialize_engine()

    # Tekeleza amri za mstari wa amri
    if args.ingest:
        cmd_ingest(engine, args.ingest)
    elif args.ingest_dir:
        cmd_ingest_dir(engine, args.ingest_dir)
    elif args.status:
        cmd_status(engine)
    elif args.reset:
        cmd_reset(engine, chats_only=False)
    elif args.reset_chats:
        cmd_reset(engine, chats_only=True)
    elif args.list_docs:
        cmd_list_docs(engine)
    else:
        # Hali ya mazungumzo
        interactive_mode(engine, args)


if __name__ == "__main__":
    main()
