# Sparky for macOS (Apple Silicon and Intel)

This fork adds an **experimental, native macOS port** to the upstream Windows application.
It runs Sparky as a local web service using **llama.cpp**: Metal on Apple Silicon,
CPU inference by default on Intel.
It does not yet ship a signed or notarized `.app` bundle.

## Requirements

- An **Apple Silicon** (arm64) or **Intel** (x86_64) Mac
- 16 GB RAM recommended; small 0.8B/2B models can run with less
- Xcode Command Line Tools: `xcode-select --install`
- [Homebrew](https://brew.sh/)
- Free disk space for the llama.cpp build plus GGUF model files (model download is opt-in)
- Network access during installation for Python packages, llama.cpp, and optional models

Run the installer in your Mac's native architecture, without Rosetta.
Intel Hackintosh support is experimental and defaults to CPU-only inference.

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

The base install provides chat, local documents and GGUF inference via Metal on
Apple Silicon or CPU on Intel. Whisper transcription is optional: MPS where
supported, CPU otherwise.
ComfyUI is **experimental** on Metal/MPS: image models may require extra attention,
and GPU-specific video pipelines designed around CUDA may not work on Macs.
Neither image nor video generation has been verified on macOS.

### Intel Hackintosh (Lenovo Yoga L390)

The original ARM64 build cannot run on this Intel laptop. Run
`./install-macos.sh` on the machine to compile an x86_64 CPU backend. This
sets `gpu_layers: 0` in `config.json` and does not depend on Metal or patched
graphics drivers. Start with Qwen3.5 **2B Q4_K_M**, or **0.8B Q4_K_M**
for reduced RAM use and quicker answers.

Only experiment with `./install-macos.sh --with-metal` if graphics acceleration
has been stable on the Hackintosh; previous GPU freezes make CPU mode a safer
choice. Image/video generation is not recommended on the Intel laptop CPU.

The optional **Computer** mode obeys the existing per-chat approval levels.
macOS may ask for Screen Recording permission when taking screenshots; grant it
to the app/terminal launching Sparky in System Settings > Privacy & Security.

## Custom installation paths

`config.json` can override `llama_server`, `gpu_layers`, `comfy_dir`,
model paths and ports. Re-running the installer updates the GPU-layer setting
to the chosen CPU/Metal mode without changing model paths or ports. The
llama.cpp binary and related dynamic libraries remain together in
`runtime/llama/source/build/bin/`. Re-run the installer to rebuild.

## Build / tests without a Mac

The Python tests and script syntax can be checked on Linux:

```bash
python3 -m pip install -r requirements.txt numpy pytest ruff
python3 -m ruff check app tests
python3 -m pytest
bash -n install-macos.sh start-macos.sh stop-macos.sh
```

GitHub Actions runs Python tests on Windows, Linux, macOS Apple Silicon and
macOS Intel; separate workflows compile arm64 Metal and Intel x86_64 CPU
versions of `llama-server`. Both produce architecture-specific artifacts.
Linux checks cannot verify Metal, macOS permissions, hardware performance, or
model compatibility. The build workflow is an engine build, **not** a complete
standalone Sparky application.

For Windows installation instructions, see [README.md](README.md).
