<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app/logo.png">
    <img src="docs/logo-on-dark.png" width="96" alt="Sparky">
  </picture>
</p>

<h1 align="center">Sparky</h1>

<p align="center"><b>Your private AI assistant. 100% local.</b><br>
Chat, see images, generate and edit pictures, build apps and dictate by voice —<br>
everything runs on your own PC. Nothing is sent to the internet.</p>

<p align="center">
  <img alt="Windows 10/11" src="https://img.shields.io/badge/Windows-10%20%7C%2011-0078D6">
  <img alt="x64 and ARM64" src="https://img.shields.io/badge/x64%20%7C%20ARM64-NVIDIA%20CUDA-76B900">
  <img alt="100% local" src="https://img.shields.io/badge/100%25-local-8c7cf0">
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="English and French" src="https://img.shields.io/badge/UI-English%20%7C%20Fran%C3%A7ais-lightgrey">
</p>

<p align="center"><img src="docs/demo-artifact.gif" alt="Sparky writes a Breakout game live in the side panel, then runs it" width="900"></p>

---

## What it does

- **Chat like ChatGPT, offline** — conversation history, plus **temporary chats** that are never saved.
- **Code & apps panel** — ask for a game, a web page, a tool or a script: it is written live in a panel next to the chat, with a **running preview**, the code, versions, copy and download.
- **Multi-file projects** — "build me a mini project": a website with `index.html` + `css/` + `js/`, or a Python program split in modules. Files are written one by one, the preview serves the whole project, the assistant can **generate the pictures** of a website itself (hero banner, products…) and put them in the project, and when it is done Sparky **tests the project automatically** — website opened in a headless browser (JavaScript errors, missing files, broken layout, every button clicked), Python and Node.js programs executed (`npm install` included) — and the model **fixes the errors until it works**. **▶ Run** re-runs it anytime, and the project downloads as a ZIP. Node.js is installed automatically (portable).
- **Images** — generate and edit pictures by simply asking ("make it snow", "remove the bottle"). The agent checks its own result and retries when needed.
- **Vision** — drop photos, screenshots or scanned documents: the assistant sees and analyses them.
- **Sparky Bot (beta)** — a companion that remembers you, learns your style and grows with you, with a memory you can see and edit ([details](#-sparky-bot--personal-companion-with-memory-beta-in-progress)).
- **Voice dictation, offline** — click the mic, talk, stop: Whisper turns your voice into text right in the input box.
- **Adjustable thinking** — Off, Light, Normal or Deep, right from the model menu.
- **System scan** — at install time (and anytime from *Settings > System*) Sparky checks your GPU, memory and disk, then **recommends the models that fit your machine**.
- **Model library** — one-click installs with resumable downloads, and speed / quality / memory indicators *for your hardware*.
- **Only official models** — Qwen, Gemma, gpt-oss, Nemotron, FLUX.2, Whisper… "uncensored", "abliterated" or NSFW variants are excluded from the catalog and from the Hugging Face search.
- **English or French** — chosen during install, switchable in the settings.

## Screenshots

| | |
|---|---|
| <img src="docs/image-edit.png" alt="Image generation and editing"><br>**Images** — generate, then edit by instruction | <img src="docs/thinking.png" alt="Visible reasoning"><br>**Thinking** — see how the model reasoned |
| <img src="docs/welcome.png" alt="Welcome screen with system scan"><br>**First launch** — system scan and recommended models | <img src="docs/picker.png" alt="Model menu with thinking levels"><br>**Model menu** — switch models, pick a thinking level |
| <img src="docs/library.png" alt="Model library"><br>**Model library** — speed, quality and memory for your PC | <img src="docs/settings.png" alt="Settings"><br>**Settings** — language, system, models |

<p align="center"><img src="docs/demo-image.gif" alt="Generating and editing an image" width="800"></p>

<p align="center"><img src="docs/demo-voice.gif" alt="Offline voice dictation" width="800"><br><i>Offline voice dictation: live level bars, then the text lands in the input box.</i></p>

## 🚧 Sparky Bot — personal companion with memory *(beta, in progress)*

> **Status: in active development.** Sparky Bot ships as a beta inside Sparky (sidebar → *Sparky Bot*). It works today, but the memory engine and the personality system are still being tuned and may change.

Sparky Bot is a friend-like companion that **remembers you, learns your style and grows with you** — in the spirit of OpenAI's *Dots* and Grok's companions, but **100% local** and with a memory you can actually see and edit.

<p align="center"><img src="docs/demo-bot.gif" alt="Sparky Bot learns about the user while chatting" width="900"></p>

| | |
|---|---|
| <img src="docs/bot.png" alt="Sparky Bot with its memory"><br>**Memory** — every fact it learned, grouped, editable, pinnable | <img src="docs/bot-journal.png" alt="Sparky Bot journal"><br>**Journal** — it reflects between conversations and remembers to follow up |

**How the memory works**

- **Learns after every message** — a background pass extracts durable facts about you (who you are, tastes, projects, people, goals, events) and decides to *add*, *update* or *forget*, Mem0-style.
- **Guard-rails in code, not just in the prompt** — one fact per memory; an update must be about the *same* fact (checked by meaning) or it becomes a new memory instead of overwriting; near-duplicates are ignored; a new memory must come from *your* words, never from the bot's own reply.
- **Smart retrieval** — before each reply it picks the memories that matter now: meaning (local multilingual embeddings, `multilingual-e5-small`), importance, recency, plus pinned facts.
- **Time-aware** — "next Tuesday" is stored as a real date, and past events are updated with what happened.
- **Learns your style** — tone, length, formality, humor, language.
- **Reflects when idle** — merges duplicates, writes a short journal, updates its summary of your relationship and prepares what to ask next time ("How did the meetup go?").
- **Relationship level** — from *Just met* to *Close friends* as you talk over days. Always a friend: never romantic, never clingy.
- **You stay in control** — see, edit, pin or delete any memory, turn learning off, or *Forget everything* in one click. Nothing leaves your PC.

**Roadmap**

- [x] Persistent memory with add / update / forget, guard-rails and semantic search
- [x] Style profile, journal, follow-ups, relationship level, 4 personalities
- [x] Animated avatar, bilingual UI (English / French)
- [ ] Voice conversations (Whisper in, local text-to-speech out)
- [ ] Proactive check-ins (optional notifications)
- [ ] Memory import / export, per-memory "why do you know this?" with the source message
- [ ] Larger evaluation set for the memory engine across models

## Install (Windows 10/11)

**One line**, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/anisayari/sparky/main/install.ps1 | iex
```

**Or** download the project (*Code > Download ZIP*), unzip it and double-click `install.bat`.

The installer walks you through 7 simple steps:

1. choose your language (English / Français) and check your machine,
2. pick the install folder (default `%LOCALAPPDATA%\Sparky`),
3. install Python 3.12 if needed (via winget),
4. install PyTorch for your GPU and the image engine [ComfyUI](https://github.com/comfyanonymous/ComfyUI),
5. install the AI engine [llama.cpp](https://github.com/ggml-org/llama.cpp) (CUDA build for NVIDIA GPUs, x64 or ARM64),
6. scan your hardware, show the recommended models and offer to download them,
7. create a **Sparky** shortcut on the desktop and in the Start menu.

<p align="center"><img src="docs/installer.png" alt="The step-by-step installer" width="560"></p>

Then double-click the shortcut (or `start.bat`): Sparky opens at <http://127.0.0.1:7860>. To stop everything: `stop.bat`.

### Which machine?

| Machine | Runs well |
|---|---|
| NVIDIA GPU 24 GB+ or 32 GB+ unified memory | 27–35B assistants, Qwen-Image 2.1 |
| NVIDIA GPU 12–16 GB | 9–20B assistants, FLUX.2 Klein |
| NVIDIA GPU 6–8 GB | 4–9B assistants, FLUX.2 Klein (slower) |
| No NVIDIA GPU | small assistants (4B) on the CPU, no image generation |

The system scan picks automatically among these.

## Models

Speeds measured on an RTX Spark N1X (Q4, full power); Sparky adapts them to your GPU.

| Assistant | Size | Speed | GPQA (published) | Vision | Best for |
|---|---|---|---|---|---|
| gpt-oss 20B | 11 GB | 83 tok/s | 71.5 | | fastest |
| Qwen3.5 35B A3B | 21 GB | 74 tok/s | 84.2 | ✓ | best fast all-rounder |
| Nemotron 3.5 Lightning 30B | 24 GB | 74 tok/s | 75.4 | | fast without thinking |
| Qwen3.5 4B | 3 GB | 66 tok/s | 76.2 | ✓ | tiny, even without a GPU |
| Gemma 4 26B A4B | 17 GB | 62 tok/s | 82.3 | ✓ | fast and versatile |
| Qwen3.5 9B | 6 GB | 34 tok/s | 81.7 | ✓ | best small model |
| Gemma 4 12B | 7 GB | 34 tok/s | 78.8 | ✓ | light |
| Qwen3.8 27B | 16 GB | 16 (33 with MTP) | 89.2 | ✓ | smartest, best at code |
| Gemma 4 E4B | 6 GB | – | – | ✓ | small machines |

All of them can think before answering (adjustable level). Vision can be added in one click when it is missing.

| Images | Size | Time per image | Style |
|---|---|---|---|
| FLUX.2 Klein 4B | 7 GB | ~3 s | photo, readable text, edits by instruction |
| Qwen-Image 2.1 | 16 GB | ~35 s | most realistic, edits by instruction |
| Juggernaut XL v9 | 10 GB* | ~15 s | photo (SDXL) |
| Animagine XL 4.0 | 10 GB* | ~15 s | illustration (SDXL) |

| Voice | Size | Note |
|---|---|---|
| Whisper large-v3 turbo | 1.6 GB | very accurate, 99 languages (GPU) |
| Whisper base | 0.3 GB | light, for machines without a GPU |

\* includes the IP-Adapter files (≈ 4 GB, shared by all SDXL models) used to reuse a character or a style.

Every model keeps its own license (Apache 2.0, Gemma Terms of Use, NVIDIA Open Model License, etc.): check its Hugging Face page before any commercial use.

## Existing installation (ComfyUI / llama.cpp already installed)

Create a `config.json` file at the project root to reuse your folders:

```json
{
  "python": "D:\\ai\\venv\\Scripts\\python.exe",
  "comfy_dir": "D:\\ai\\ComfyUI",
  "llama_server": "D:\\ai\\llama\\llama-server.exe",
  "llm_dir": "D:\\ai\\models",
  "mmproj_dir": "D:\\ai\\mmproj"
}
```

Other keys: `voice_dir`, `data_dir`, `app_port` (7860), `llm_port` (8080), `comfy_port` (8188). The `SPARKY_CONFIG` environment variable can point to another config file.

## Command line

```powershell
runtime\venv\Scripts\python.exe app\syscheck.py                  # system scan (add --lang=fr for French)
runtime\venv\Scripts\python.exe app\models_cli.py --list         # catalog
runtime\venv\Scripts\python.exe app\models_cli.py --recommended  # install the recommended models
runtime\venv\Scripts\python.exe app\models_cli.py qwen359        # install one model
```

## Update / uninstall

- **Update**: run the install command again (or `git pull` in the install folder).
- **Uninstall**: `stop.bat`, then delete the install folder (`%LOCALAPPDATA%\Sparky` by default) and the shortcuts.

## Privacy

Everything runs on your machine: conversations, images and models stay in the install folder (`data\`, `models\`). The only outgoing connections download the tools and models (GitHub, PyPI, Hugging Face) and search the model library.

## En français

L'interface et l'installeur existent aussi en français : choisis *Français* au début de l'installation, ou change la langue dans *Paramètres*.

<p align="center"><img src="docs/french.png" alt="Sparky en français" width="760"></p>

## Project layout

```
app/
  server.py     web server + model management
  harness.py    agent: assistant <-> image tools loop, artifacts
  catalog.py    catalog, downloads, indicators
  syscheck.py   system scan and recommendations
  stt.py        offline voice dictation (Whisper)
  companion.py  Sparky Bot: persona, learning, reflection, journal (beta)
  memory.py     Sparky Bot memory: SQLite store + hybrid semantic retrieval
  index.html    interface (English / French)
install.ps1     step-by-step installer
start.ps1 / stop.ps1
docs/           screenshots and demos
```

## License

Code under the [MIT](LICENSE) license. ComfyUI (GPL-3.0) and llama.cpp (MIT) are downloaded separately during install.
