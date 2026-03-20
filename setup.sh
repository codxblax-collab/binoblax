#!/bin/bash
# setup.sh - Script ya Kusanidi Mazingira (Linux/macOS)

set -e  # Simama ukikutana na hitilafu

BLUE="\033[34m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

info()    { echo -e "${BLUE}[INFO]${RESET} $1"; }
success() { echo -e "${GREEN}[✅ OK]${RESET} $1"; }
warning() { echo -e "${YELLOW}[⚠️ ]${RESET} $1"; }
error()   { echo -e "${RED}[❌]${RESET} $1"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "     🌍 RAG Multilingual - Setup Script 🌍     "
echo "╚══════════════════════════════════════════════════╝"
echo ""

# 1. Angalia Python
info "Kuangalia Python..."
if ! command -v python3 &>/dev/null; then
    error "Python 3 haijapatikana. Sakinisha kwanza: https://python.org"
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
info "Python $PYTHON_VERSION imepatikana."
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    success "Toleo la Python linakubalika."
else
    error "Inahitajika Python 3.10+. Una $PYTHON_VERSION."
fi

# 2. Tengeneza mazingira ya virtual
if [ ! -d "venv" ]; then
    info "Inatengeneza mazingira ya virtual (venv)..."
    python3 -m venv venv
    success "Mazingira ya virtual yameundwa."
else
    info "Mazingira ya virtual tayari yanawepo."
fi

# Amilisha
source venv/bin/activate
success "Mazingira ya virtual imeamilishwa."

# 3. Sakinisha packages
info "Inasanikisha Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
success "Packages zote zimesanikishwa."

# 4. Angalia na sanidi Ollama
info "Kuangalia Ollama..."
if ! command -v ollama &>/dev/null; then
    warning "Ollama haijapatikana. Inasanikisha..."
    curl -fsSL https://ollama.ai/install.sh | sh
    success "Ollama imesanikishwa."
else
    success "Ollama imepatikana: $(ollama --version 2>&1 | head -1)"
fi

# Angalia kama Ollama inaendesha
if ! ollama list &>/dev/null; then
    warning "Ollama haiendesha. Inaanzisha..."
    ollama serve &
    sleep 3
    success "Ollama imeanzishwa."
fi

# 5. Pakua modeli
LLM_MODEL="deepseek-coder:6.7b-instruct-q4_K_M"
EMBED_MODEL="granite-embedding:multilingual"

info "Inaangalia modeli za Ollama..."

if ollama list 2>/dev/null | grep -q "deepseek-coder"; then
    success "DeepSeek Coder tayari imewekwa."
else
    info "Inazipakua LLM: $LLM_MODEL (inaweza kuchukua muda)..."
    ollama pull "$LLM_MODEL"
    success "LLM imesanikishwa."
fi

if ollama list 2>/dev/null | grep -q "granite-embedding"; then
    success "Granite Embedding tayari imewekwa."
else
    info "Inapakua Embedding: $EMBED_MODEL..."
    ollama pull "$EMBED_MODEL"
    success "Embedding imesanikishwa."
fi

# 6. Tengeneza muundo wa saraka
info "Inatengeneza muundo wa saraka..."
mkdir -p data/documents data/uploads data/processed
mkdir -p chroma_data/nyaraka chroma_data/memory chroma_data/cache
mkdir -p models notebooks tests
success "Muundo wa saraka umeundwa."

# 7. Nakili .env kama haipo
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        success ".env imeundwa kutoka .env.example."
    fi
fi

# 8. Jaribu mfumo haraka
info "Inajaribu mfumo..."
python3 -c "
from src.config import config
from src.document_processor import DocumentProcessor
dp = DocumentProcessor()
chunks = dp.process_text('Habari! Hii ni jaribio la mfumo wa RAG Multilingual.')
print(f'  ✅ DocumentProcessor: vipande {len(chunks)}')
print('  ✅ Mipangilio imepakiwa vizuri.')
" && success "Mtihani wa msingi umefaulu." || warning "Hitilafu katika mtihani wa msingi."

# Mwisho
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "            ✅ Usanidi Umekamilika!            "
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Hatua zifuatazo:"
echo "  1.Amilisha mazingira:  source venv/bin/activate"
echo "  2.Ingiza nyaraka:      python run.py --ingest data/documents/hati.pdf"
echo "  3.Anzisha mazungumzo:  python run.py"
echo ""
