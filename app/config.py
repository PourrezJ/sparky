r"""Chemins de Defend Local IA.

Par defaut tout vit dans le dossier du projet (runtime\, models\, data\). Un fichier config.json
a la racine du projet peut pointer vers une installation existante, par exemple :
  {"comfy_dir": "D:\\ComfyUI", "llama_server": "D:\\llama\\llama-server.exe", "llm_dir": "D:\\models"}
"""
import json, os

APP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP)
_cfg = {}
if os.path.exists(os.path.join(ROOT, "config.json")):
    _cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8-sig"))


def _p(key, default):
    return os.path.abspath(os.path.expandvars(_cfg.get(key) or default))


RUNTIME = _p("runtime_dir", os.path.join(ROOT, "runtime"))
COMFY_DIR = _p("comfy_dir", os.path.join(RUNTIME, "ComfyUI"))
LLAMA_SERVER = _p("llama_server", os.path.join(RUNTIME, "llama", "llama-server.exe"))
LLM_DIR = _p("llm_dir", os.path.join(ROOT, "models", "llm"))
MMPROJ_DIR = _p("mmproj_dir", os.path.join(ROOT, "models", "mmproj"))
DATA = _p("data_dir", os.path.join(ROOT, "data"))
COMFY_MODELS = os.path.join(COMFY_DIR, "models")
COMFY_IN = os.path.join(COMFY_DIR, "input")
COMFY_OUT = os.path.join(COMFY_DIR, "output")
COMFY_PORT = int(_cfg.get("comfy_port", 8188))
LLM_PORT = int(_cfg.get("llm_port", 8080))
APP_PORT = int(_cfg.get("app_port", 7860))
