# Sparky for macOS (Apple Silicon)

This fork adds an **experimental, native macOS port** to the upstream Windows application.
It runs Sparky as a local web service and uses **llama.cpp with Metal** for GGUF models.
It does not yet ship a signed or notarized `.app` bundle.

## Requirements

- An **Apple Silicon** Mac (arm64) with macOS and at least 16 GB of unified memory recommended
- Xcode Command Line Tools: `xcode-select --install`
- [Homebrew](https://brew.sh/)
- Free disk space for the llama.cpp build plus GGUF model files (model download is opt-in)
- Network access during installation for Python packages, llama.cpp, and optional models

Intel Macs, Rosetta, and macOS versions without the necessary Metal tools are **not supported** by this installer.

## Installation

```bash
git clone https://github.com/PourrezJ/sparky.git
cd sparky
chmod +x install-macos.sh start-macos.sh stop-macos.sh
./install-macos.sh
./start-macos.sh
```

Open the local web interface at `http://127.0.0.1:7860` and choose a GGUF assistant
from the model library. The installer does **not** download an LLM by default.
The app uses the virtual environment at `runtime/venv` and stores models locally.
Homebrew's Node.js is installed for the generated-project checker and npm-based MCP servers.
Use `./stop-macos.sh` to stop the processes started by the launcher.

### Optional features

```bash
./install-macos.sh --with-voice
./install-macos.sh --with-comfy
```

The base install provides chat, locally served documents, and local GGUF inference via
Metal. Whisper transcription is optional and uses PyTorch MPS where supported.
ComfyUI is **experimental** on Metal/MPS: image models may require extra attention,
and GPU-specific video pipelines designed around CUDA may not work on Macs.
Neither image nor video generation has been verified on macOS.

The optional **Computer** mode obeys the existing per-chat approval levels.
macOS may ask for Screen Recording permission when taking screenshots; grant it
to the app/terminal launching Sparky in System Settings > Privacy & Security.

## Custom installation paths

`config.json` can override `llama_server`, `comfy_dir`, model paths, and ports.
The macOS installer creates missing configuration keys without overwriting existing
choices. The llama.cpp binary and related dynamic libraries remain together in
`runtime/llama/source/build/bin/`. Re-run the installer to rebuild.

## Build / tests without a Mac

The Python tests and script syntax can be checked on Linux:

```bash
python3 -m pip install -r requirements.txt numpy pytest ruff
python3 -m ruff check app tests
python3 -m pytest
bash -n install-macos.sh start-macos.sh stop-macos.sh
```

GitHub Actions runs Python tests on Windows, Linux, and macOS; the separate
`Build llama.cpp Metal for macOS arm64` workflow compiles `llama-server` with
Apple's SDK on a hosted macOS runner and exposes an arm64 binary artifact.
Linux checks cannot verify Metal, macOS permissions, hardware performance, or
model compatibility. The build workflow is an engine build, **not** a complete
standalone Sparky application.

For Windows installation instructions, see [README.md](README.md).
