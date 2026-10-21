#!/usr/bin/env bash
# Sparky for macOS: Apple Silicon (Metal) and Intel (CPU).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$ROOT/runtime"
LLAMA_SRC="$RUNTIME/llama/source"
PYTHON="$RUNTIME/venv/bin/python"
WITH_COMFY=0
WITH_VOICE=0
WITH_METAL=0
for arg in "$@"; do
  case "$arg" in
    --with-comfy) WITH_COMFY=1 ;;
    --with-voice) WITH_VOICE=1 ;;
    --with-metal) WITH_METAL=1 ;;
    -h|--help)
      echo "Usage: ./install-macos.sh [--with-comfy] [--with-voice] [--with-metal]"
      echo "Apple Silicon uses Metal; Intel defaults to CPU. Intel Metal is experimental."
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer requires macOS; current system: $(uname -s)." >&2
  exit 1
fi
ARCH="$(uname -m)"
case "$ARCH" in
  arm64) WITH_METAL=1 ;;
  x86_64) ;;
  *) echo "Unsupported macOS architecture: $ARCH" >&2; exit 1 ;;
esac
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

if [[ "$ARCH" == "x86_64" && "$WITH_METAL" == "0" ]]; then
  echo "Intel Mac: building the CPU backend (no GPU required)."
else
  echo "Building llama.cpp with Apple's Metal backend..."
fi
mkdir -p "$(dirname "$LLAMA_SRC")"
if [[ ! -d "$LLAMA_SRC/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_SRC"
else
  echo "Using existing llama.cpp source. Update it manually to change versions."
fi
METAL_CMAKE=OFF
if (( WITH_METAL )); then METAL_CMAKE=ON; fi
cmake -S "$LLAMA_SRC" -B "$LLAMA_SRC/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
  -DGGML_METAL="$METAL_CMAKE" -DLLAMA_CURL=OFF
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
SPARKY_LLAMA_BIN="$LLAMA_BIN" SPARKY_PYTHON_BIN="$PYTHON" \
SPARKY_GPU_LAYERS="$([[ "$WITH_METAL" == "1" ]] && echo 99 || echo 0)" \
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
# The venv python is a symlink to Homebrew Python: resolve() would follow it
# outside this repo, so walk the install path without resolving symlinks.
root = Path(os.environ["SPARKY_PYTHON_BIN"]).absolute().parents[3]
file = root / "config.json"
data = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
data.setdefault("python", os.environ["SPARKY_PYTHON_BIN"])
data["llama_server"] = os.environ["SPARKY_LLAMA_BIN"]
data["gpu_layers"] = int(os.environ["SPARKY_GPU_LAYERS"])
file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY

echo
echo "Sparky is installed for macOS $ARCH. Run: ./start-macos.sh"
echo "At first launch, pick a GGUF model in the Library (none is downloaded automatically)."
if (( ! WITH_COMFY )); then
  echo "Image/video generation is not installed. It is optional and MPS support varies by model."
fi
