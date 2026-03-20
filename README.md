RAG Multilingual.
Mfumo wa Retrieval-Augmented Generation (RAG) wenye lugha nyingi, ukitumia:

-LLM, DeepSeek Coder 6.7B (4-bit quantized via Ollama) 
-Embeddings, Granite Multilingual (via Ollama) 
-Vector DB, ChromaDB (Persistent) 
-Memory, Semantic + Episodic + Cache 
-Lugha, Kiswahili, Kiingereza, na nyingine 

sanidi wa Haraka.

#Linux and macOS
chmod +x setup.sh
./setup.sh
source venv/bin/activate

#Windows (PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\venv\Scripts\Activate.ps1

#Kwa Mkono.
python -m venv venv
source venv/bin/activate #Windows: venv\Scripts\activate
pip install -r requirements.txt

#Sakinisha na anzisha Ollama
ollama serve &
ollama pull deepseek-coder:6.7b-instruct-q4_K_M
ollama pull granite-embedding:multilingual

Matumizi.

#Ingiza Nyaraka.
#Faili moja
python run.py --ingest data/documents/hati.pdf

#Saraka nzima.
python run.py --ingest-dir data/documents/

#Anzisha Mazungumzo
#Mazungumzo ya kawaida
python run.py

#Stream majibu kwa wakati halisi
python run.py --stream

#Endelea mazungumzo ya awali
python run.py --session <session_id>

#Zima cache (daima uliza LLM)
python run.py --no-cache

#Amri Nyingine
python run.py --status        # Hali ya mfumo
python run.py --list-docs     # Orodha ya nyaraka
python run.py --reset         # Futa KILA KITU
python run.py --reset-chats   # Futa mazungumzo tu

#Amri Wakati wa Mazungumzo
/exit   - Toka
/status - Hali ya mfumo
/docs   - Orodha ya nyaraka
/clear  - Futa mazungumzo ya kipindi hiki
/new    - Anza kipindi kipya
/ingest <faili>  - Ingiza faili moja kwa moja

#Mfumo wa Kumbukumbu.

1.Semantic Memory (Kumbukumbu ya Maana)
-Hifadhi ya vipande vya nyaraka
-Vector search kwa maudhui yanayohusiana
-Inabaki daima (persistent)
-ChromaDB collection: `nyaraka/`

2.Episodic Memory (Kumbukumbu ya Matukio)
-Historia ya mazungumzo
-Mfululizo wa maswali na majibu
-Kila kipindi kina historia yake
-ChromaDB collection: `memory/`

3.Semantic Cache (Cache ya Maana)
-Majibu ya maswali yaliyoulizwa awali
-Epuka kuuliza LLM kwa maswali yanayofanana sana
-Threshold ya ufanani: 0.92 (inaweza kubadilishwa)
-ChromaDB collection: `cache/`

#Usanidi (.env)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=deepseek-coder:6.7b-instruct-q4_K_M
OLLAMA_EMBED_MODEL=granite-embedding:multilingual

CHROMA_PERSIST_DIR=./chroma_data
CHUNK_SIZE=512
CHUNK_OVERLAP=64

CACHE_SIMILARITY_THRESHOLD=0.92
SEMANTIC_TOP_K=5
EPISODIC_TOP_K=3

#Majaribio.
pytest tests/ -v
pytest tests/test_memory.py -v --tb=short

#Muundo wa Mradi.

rag-multilingual/
├── src/
│   ├── config.py            # Mipangilio ya mfumo
│   ├── ollama_client.py     # LLM + Embeddings (Ollama)
│   ├── memory_mapping.py    # Semantic + Episodic + Cache
│   ├── rag_engine.py        # Core RAG pipeline
│   ├── document_processor.py# Kusoma PDF, TXT, DOCX
│   └── utils.py             # Misaada ya kila aina
├── data/documents/          # Weka nyaraka hapa
├── chroma_data/             # ChromaDB (persistent)
├── tests/                   # Majaribio ya mfumo
├── run.py                   # Script kuu
├── setup.sh                 # Usanidi (Linux/macOS)
└── setup.ps1                # Usanidi (Windows)


#Aina za Faili Zinazounganishwa.
PDF, Word (docx), Maandishi (txt) and Markdown (md).

#Vidokezo.
-Anza na --status kuhakikisha mfumo unafanya kazi
-Ingiza nyaraka ndogo kwanza kujaribu
-Tumia --stream kwa majibu ya haraka zaidi (yaonekana)
-Cache inasaidia sana kwa maswali yanayorudiwa mara kwa mara
-Episodic memory inaruhusu mazungumzo ya kina na yanayoendelea

#Mahitaji ya Mfumo.

-Python 3.10+
-RAM: 8GB+ (16GB inapendekezwa kwa DeepSeek 6.7B)
-Disk: 10GB+ kwa modeli
-OS: Linux, macOS, Windows 10+
