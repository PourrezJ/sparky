#!/usr/bin/env bash
# Sparky for Apple Silicon: Python environment + llama.cpp with Metal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$ROOT/runtime"
LLAMA_SRC="$RUNTIME/llama/source"
PYTHON="$RUNTIME/venv/bin/python"
WITH_COMFY=0
WITH_VOICE=0
for arg in "$@"; do
  case "$arg" in
    --with-comfy) WITH_COMFY=1 ;;
    --with-voice) WITH_VOICE=1 ;;
    -h|--help)
      echo "Usage: ./install-macos.sh [--with-comfy] [--with-voice]"
      echo "Defaults to local GGUF chat (Metal). ComfyUI and Whisper are optional."
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer requires macOS; current system: $(uname -s)." >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Native Apple Silicon (arm64) is required. Run Terminal without Rosetta." >&2
  exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Install Apple's Command Line Tools with: xcode-select --install" >&2
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required: https://brew.sh/" >&2
  exit 1
fi

echo "Installing build dependencies..."
brew install python@3.12 cmake git node
PY3="$(brew --prefix python@3.12)/bin/python3.12"
[[ -x "$PY3" ]] || { echo "Homebrew Python 3.12 not found" >&2; exit 1; }
mkdir -p "$RUNTIME" "$ROOT/data" "$ROOT/models/llm" "$ROOT/models/mmproj"
if [[ ! -x "$PYTHON" ]]; then "$PY3" -m venv "$RUNTIME/venv"; fi
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$ROOT/requirements.txt" numpy

echo "Building llama.cpp with Apple's Metal backend..."
mkdir -p "$(dirname "$LLAMA_SRC")"
if [[ ! -d "$LLAMA_SRC/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_SRC"
else
  echo "Using existing llama.cpp source. Update it manually to change versions."
fi
cmake -S "$LLAMA_SRC" -B "$LLAMA_SRC/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF
cmake --build "$LLAMA_SRC/build" --config Release --target llama-server -j "$(sysctl -n hw.ncpu)"
LLAMA_BIN="$LLAMA_SRC/build/bin/llama-server"
[[ -x "$LLAMA_BIN" ]] || { echo "Metal llama-server executable not found" >&2; exit 1; }

if (( WITH_VOICE )); then
  "$PYTHON" -m pip install torch transformers accelerate sentencepiece
fi
if (( WITH_COMFY )); then
  echo "Installing ComfyUI with Apple's MPS PyTorch backend (experimental)..."
  if [[ ! -d "$RUNTIME/ComfyUI/.git" ]]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$RUNTIME/ComfyUI"
  fi
  "$PYTHON" -m pip install torch torchvision torchaudio
  "$PYTHON" -m pip install -r "$RUNTIME/ComfyUI/requirements.txt"
fi

# Leave user-supplied paths and settings intact on reinstall.
SPARKY_LLAMA_BIN="$LLAMA_BIN" SPARKY_PYTHON_BIN="$PYTHON" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
# The venv python is a symlink to Homebrew Python: resolve() would follow it
# outside this repo, so walk the install path without resolving symlinks.
root = Path(os.environ["SPARKY_PYTHON_BIN"]).absolute().parents[3]
file = root / "config.json"
data = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
data.setdefault("python", os.environ["SPARKY_PYTHON_BIN"])
data.setdefault("llama_server", os.environ["SPARKY_LLAMA_BIN"])
file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY

echo
echo "Sparky is installed for Apple Silicon. Run: ./start-macos.sh"
echo "At first launch, pick a GGUF model in the Library (none is downloaded automatically)."
if (( ! WITH_COMFY )); then
  echo "Image/video generation is not installed. It is optional and MPS support varies by model."
fi
