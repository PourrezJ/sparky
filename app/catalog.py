"""Model catalog: installation (download resumed after interruptions), speed estimate
on this machine (from the system scan) and indicators (quality, memory, vision, thinking).

The catalog only offers official or widely used models; "uncensored",
"abliterated", NSFW, etc. variants are excluded, including from Hugging Face search.
"""
import json, os, re, subprocess, threading, time, urllib.request

import config

HF = "https://huggingface.co/{repo}/resolve/main/{file}"
M, P, CM = config.LLM_DIR, config.MMPROJ_DIR, config.COMFY_MODELS
REF_BW = 230.0  # bandwidth (GB/s) of the benchmark machine (RTX Spark): ref_tps was measured there


def _first(*paths):
    """First existing path (old file names accepted), otherwise the first one (canonical name)."""
    return next((p for p in paths if os.path.exists(p)), paths[0])


# optional MTP draft: doubles Qwen3.8 27B speed (measured: 16 -> 33 words/s)
MTP = _first(os.path.join(M, "mtp-Qwen3.8-27B-Q4_0.gguf"), os.path.join(os.path.dirname(M), "mtp", "mtp-Qwen3.8-27B-Q4_0.gguf"))
BLOCKED = ("uncensor", "abliterat", "heretic", "nsfw", "lustify", "pony", "unfilter", "unaligned", "jailbreak",
           "erotic", "porn", "hentai", "lewd", "nude", "xxx", "18+", "hauhau", "dolphin", "not-for-all-audiences")
NOWIN = config.NOWIN


def blocked(*texts):
    t = " ".join(str(x) for x in texts).lower()
    return any(w in t for w in BLOCKED)


def llm(id, label, quality, repo, file, mmproj, size_gb, active_gb=None, think=True, desc="", ref_tps=None, alt=()):
    """mmproj: vision file in the same repo (None = model that cannot see images).
    ref_tps: speed measured on the benchmark machine; alt: other accepted local names."""
    stem = repo.split("/")[-1].replace("-GGUF", "")
    main = _first(os.path.join(M, file), *[os.path.join(M, a) for a in alt])
    vis = _first(os.path.join(P, f"mmproj-{stem}-F16.gguf"), os.path.join(P, f"mmproj-{stem.replace('-it', '')}-F16.gguf"))
    return {"id": id, "kind": "llm", "label": label, "quality": quality, "repo": repo, "desc": desc,
            "files": [(repo, file, main), (repo, mmproj, vis) if mmproj else None], "ref_tps": ref_tps,
            "size_gb": size_gb, "active_gb": active_gb or size_gb, "think": think, "vision": mmproj is not None}


IPA = "h94/IP-Adapter"
IPADAPTER = [  # character / style references for SDXL, shared by all SDXL models
    (IPA, "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors", os.path.join(CM, "ipadapter", "ip-adapter-plus_sdxl_vit-h.safetensors")),
    (IPA, "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors", os.path.join(CM, "ipadapter", "ip-adapter-plus-face_sdxl_vit-h.safetensors")),
    (IPA, "models/image_encoder/model.safetensors", os.path.join(CM, "clip_vision", "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"))]


def ckpt(id, label, quality, repo, remote, size_gb, style="", desc=""):
    local = os.path.basename(remote)
    return {"id": id, "kind": "image", "engine": "sdxl", "label": label, "quality": quality, "style": style,
            "files": [(repo, remote, os.path.join(CM, "checkpoints", local))] + IPADAPTER, "size_gb": size_gb + 3.9,
            "desc": desc, "ckpt": local, "repo": repo}


QI = "Comfy-Org/Qwen-Image-2.1"


def qwen_image(id, label, quality, desc=""):
    files = [(QI, "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", os.path.join(CM, "diffusion_models", "qwen_image_2.1_int8_convrot.safetensors")),
             (QI, "text_encoders/qwen3vl_8b_int8_convrot.safetensors", os.path.join(CM, "text_encoders", "qwen3vl_8b_int8_convrot.safetensors")),
             (QI, "vae/qwen_image_2.1_vae_bf16.safetensors", os.path.join(CM, "vae", "qwen_image_2.1_vae_bf16.safetensors"))]
    return {"id": id, "kind": "image", "engine": "qwen21", "label": label, "quality": quality, "style": "photo",
            "files": files, "size_gb": 16.1, "desc": desc, "ckpt": id, "lora": None, "repo": QI, "est_secs": 40}


def klein():
    """FLUX.2 Klein 4B: 3 s per image in the benchmark, understands sentences and edits by instruction."""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10, creationflags=NOWIN).stdout.lower()
    except Exception:
        gpu = ""
    # FP4 encoder on Blackwell GPUs (RTX 50, Spark), FP8 elsewhere
    te = "qwen_3_4b_fp4_mixed.safetensors" if any(k in gpu for k in ("spark", "gb10", "rtx 50", " 50")) else "qwen_3_4b_fp8_mixed.safetensors"
    files = [("black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors", os.path.join(CM, "diffusion_models", "flux-2-klein-4b-fp8.safetensors")),
             ("Comfy-Org/z_image_turbo", "split_files/text_encoders/" + te, os.path.join(CM, "text_encoders", te)),
             ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors", os.path.join(CM, "vae", "flux2-vae.safetensors"))]
    return {"id": "klein4b", "kind": "image", "engine": "klein", "label": "FLUX.2 Klein 4B", "quality": 4, "style": "photo",
            "files": files, "size_gb": 3.8 + (3.2 if "fp4" in te else 5.3) + 0.3, "ckpt": "klein4b", "lora": None,
            "repo": files[0][0], "est_secs": 3, "clip": te,
            "desc": "Images en ~3 s : rapide, texte lisible, retouche par instruction."}


LTX = "Lightricks/LTX-2.5"
WAN = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"


def ltx():
    """LTX-2.5 22B distilled (int8): video with synchronized sound, 8 steps; ~40 s for 4 s of 768p on an RTX Spark."""
    names = {"unet": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
             "clip": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
             "vae": "ltx-2.5-video-vae-bf16.safetensors", "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors"}
    files = [(LTX, "diffusion_models/" + names["unet"], os.path.join(CM, "diffusion_models", names["unet"])),
             (LTX, "text_encoders/" + names["clip"], os.path.join(CM, "text_encoders", names["clip"])),
             (LTX, "vae/" + names["vae"], os.path.join(CM, "vae", names["vae"])),
             (LTX, "vae/" + names["audio_vae"], os.path.join(CM, "vae", names["audio_vae"]))]
    return {"id": "ltx25", "kind": "video", "engine": "ltx", "label": "LTX-2.5 22B", "quality": 5, "style": "video + son",
            "files": files, "size_gb": 38.7, "ckpt": "ltx25", "repo": LTX, "est_secs": 60, "est_secs_per_s": 10,
            "long_side": 1024, "audio": True, **names,
            "desc": "Vidéo avec le son (ambiance, voix, musique), texte ou image vers vidéo, jusqu'à 10 s."}


def wan5b():
    """Wan 2.2 TI2V 5B: lighter video model (text or image to video, 720p, no sound)."""
    names = {"unet": "wan2.2_ti2v_5B_fp16.safetensors", "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "vae": "wan2.2_vae.safetensors"}
    files = [(WAN, "split_files/diffusion_models/" + names["unet"], os.path.join(CM, "diffusion_models", names["unet"])),
             (WAN, "split_files/text_encoders/" + names["clip"], os.path.join(CM, "text_encoders", names["clip"])),
             (WAN, "split_files/vae/" + names["vae"], os.path.join(CM, "vae", names["vae"]))]
    return {"id": "wan22-5b", "kind": "video", "engine": "wan", "label": "Wan 2.2 5B", "quality": 3, "style": "vidéo",
            "files": files, "size_gb": 18.2, "ckpt": "wan22-5b", "repo": WAN, "est_secs": 150, "est_secs_per_s": 30,
            "long_side": 1280, "audio": False, **names,
            "desc": "Vidéo plus légère (sans le son), texte ou image vers vidéo."}


WHISPER_FILES = ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json", "special_tokens_map.json", "model.safetensors")


def voice(id, label, quality, repo, size_gb, gpu_secs, cpu_secs, desc=""):
    """Offline voice dictation (Whisper, via transformers). *_secs: time for 10 s of speech."""
    d = os.path.join(config.VOICE_DIR, repo.split("/")[-1])
    return {"id": id, "kind": "voice", "label": label, "quality": quality, "repo": repo, "desc": desc,
            "files": [(repo, f, os.path.join(d, f)) for f in WHISPER_FILES], "dir": d, "size_gb": size_gb,
            "gpu_secs": gpu_secs, "cpu_secs": cpu_secs, "ckpt": id}


# speeds (ref_tps) and qualities: local RTX Spark benchmark (Oct. 2026), published GPQA Diamond
U = "unsloth/"
CATALOG = [
    llm("qwen38", "Qwen3.8 27B", 5, U + "Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf", "mmproj-F16.gguf", 16.2,
        ref_tps=16, desc="Le plus intelligent (GPQA 89) et le meilleur en code ; 2x plus rapide avec MTP."),
    llm("qwen35moe", "Qwen3.5 35B A3B", 5, U + "Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-Q4_K_M.gguf", "mmproj-F16.gguf", 21.3,
        active_gb=1.9, ref_tps=74, desc="Meilleur généraliste rapide (GPQA 84) ; demande beaucoup de mémoire."),
    llm("gemma26", "Gemma 4 26B A4B", 4, U + "gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", "mmproj-F16.gguf", 16.9,
        active_gb=2.6, ref_tps=62, desc="Google. Très rapide et polyvalent (GPQA 82)."),
    llm("gptoss20", "gpt-oss 20B", 3, U + "gpt-oss-20b-GGUF", "gpt-oss-20b-Q4_K_M.gguf", None, 10.8, active_gb=2.4,
        ref_tps=83, desc="OpenAI. Le plus rapide ; bon raisonnement, moins de culture. Ne voit pas les images."),
    llm("nemotron30", "Nemotron 3.5 Lightning 30B", 3, U + "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
        "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_M.gguf", None, 23.5, active_gb=2.2, ref_tps=74,
        alt=("Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_M.gguf",),
        desc="NVIDIA. Rapide, répond bien sans réfléchir ; ne voit pas les images."),
    llm("qwen359", "Qwen3.5 9B", 4, U + "Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf", "mmproj-F16.gguf", 6.2,
        ref_tps=34, desc="Le meilleur petit modèle (GPQA 82) : économe et bon en français."),
    llm("gemma12", "Gemma 4 12B", 3, U + "gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf", "mmproj-F16.gguf", 6.8,
        ref_tps=34, desc="Google. Léger et équilibré."),
    llm("qwen354", "Qwen3.5 4B", 3, U + "Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", "mmproj-F16.gguf", 3.2,
        ref_tps=66, desc="Minuscule mais solide (GPQA 76) ; très rapide, tourne même sans GPU."),
    llm("gemmae4b", "Gemma 4 E4B", 2, U + "gemma-4-e4b-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf", "mmproj-F16.gguf", 5.5,
        ref_tps=50, desc="Google. Pour les petites machines."),
    voice("whisper-turbo", "Whisper large-v3 turbo", 5, "openai/whisper-large-v3-turbo", 1.6, 1, 25,
          desc="Dictée vocale hors ligne, très précise (99 langues)."),
    voice("whisper-base", "Whisper base", 3, "openai/whisper-base", 0.3, 0.5, 3,
          desc="Dictée vocale légère, pour les machines sans GPU."),
    {"id": "e5-small", "kind": "embed", "label": "Multilingual E5 small", "quality": 4,
     "repo": "intfloat/multilingual-e5-small", "dir": os.path.join(config.EMBED_DIR, "multilingual-e5-small"),
     "files": [("intfloat/multilingual-e5-small", f, os.path.join(config.EMBED_DIR, "multilingual-e5-small", f))
               for f in ("config.json", "model.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json",
                         "tokenizer.json", "tokenizer_config.json")],
     "size_gb": 0.47, "gpu_secs": 0.1, "cpu_secs": 0.1, "ckpt": "e5-small",
     "desc": "Mémoire de Sparky Bot : retrouve les souvenirs par le sens (français, anglais…)."},
    klein(),
    qwen_image("qwen21", "Qwen-Image 2.1", 5, desc="Le plus réaliste (~35 s par image), retouche par instruction."),
    ckpt("juggernaut", "Juggernaut XL v9", 4, "RunDiffusion/Juggernaut-XL-v9", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
         6.6, style="photo", desc="Photo réaliste (SDXL), léger."),
    ckpt("animagine", "Animagine XL 4.0", 4, "cagliostrolab/animagine-xl-4.0", "animagine-xl-4.0-opt.safetensors",
         6.5, style="illustration", desc="Illustration, anime, BD (SDXL)."),
    ltx(),
    wan5b(),
]
BY_ID = {m["id"]: m for m in CATALOG}
jobs = {}  # id -> {"done": bytes, "total": bytes, "state": "running|ok|error", "error": str}


def _ok(f):
    return bool(f) and os.path.exists(f[2]) and not os.path.exists(f[2] + ".part")


def installed(m):
    """An assistant is usable as soon as its model is present; vision (mmproj) is an add-on."""
    if m.get("cloud"):  # cloud model: usable once its provider has a key
        import cloud
        return cloud.has_key(m["provider"])
    files = m["files"][:1] if m["kind"] == "llm" else m["files"]
    return all(_ok(f) for f in files if f)


def has_vision(m):
    if m.get("cloud"):
        return m.get("vision", False)
    return m["kind"] == "llm" and len(m["files"]) > 1 and _ok(m["files"][1])


def llm_args(m):
    a = [m["files"][0][2]] + (["--mmproj", m["files"][1][2]] if has_vision(m) else [])
    if m["id"] == "qwen38" and os.path.exists(MTP):
        a += ["--spec-type", "draft-mtp", "--spec-draft-model", MTP, "--spec-draft-n-max", "3"]
    return a


def indicators(m, measured=None, sysinfo=None):
    """speed: 1 slow / 2 fair / 3 fast; fit: 2 fits alongside image generation, 1 tight, 0 too big."""
    if sysinfo is None:
        import syscheck
        sysinfo = syscheck.current()
    if m.get("cloud"):  # runs on the provider's servers: no memory used here
        return {"speed": 3, "fit": 2, "tps": 0, "measured": False}
    a = sysinfo["assess"]
    mem, bw, img = a["model_mem_gb"], a["bandwidth"], a["image_gb"] if a["images"] else 0
    if m["kind"] == "llm":
        mtp = m["id"] == "qwen38" and os.path.exists(MTP)
        est = m["ref_tps"] * bw / REF_BW if m.get("ref_tps") else bw / m["active_gb"] * 0.85
        tps = measured or est * (2.05 if mtp else 1.0)
        speed = 3 if tps >= 40 else 2 if tps >= 20 else 1
        need = m["size_gb"] + 2.5  # model + context
        fit = 2 if need + img <= mem else 1 if need <= mem else 0
        return {"speed": speed, "fit": fit, "tps": round(tps), "measured": bool(measured)}
    if m["kind"] in ("voice", "embed"):
        secs = measured or (m["gpu_secs"] if a["mode"] != "cpu" else m["cpu_secs"])
        return {"speed": 3 if secs <= 3 else 2 if secs <= 10 else 1, "fit": 2, "secs": round(secs, 1), "measured": bool(measured)}
    if m["kind"] == "video":  # the assistant is paused during a video: the model gets the whole memory (and offloads)
        secs = (measured * 5 + 20) if measured else m["est_secs"] * (230 / max(bw, 60)) ** 0.7  # for 5 s of video
        fit = 2 if m["size_gb"] + 4 <= mem else 1 if m["size_gb"] <= mem * 1.6 else 0
        return {"speed": 3 if secs <= 60 else 2 if secs <= 180 else 1, "fit": fit, "secs": round(secs), "measured": bool(measured)}
    secs = measured or m.get("est_secs", 15) * (230 / max(bw, 60)) ** 0.7
    fit = 2 if m["size_gb"] + 6 <= mem else 1 if m["size_gb"] <= mem + 2 else 0
    return {"speed": 3 if secs <= 12 else 2 if secs <= 30 else 1, "fit": fit, "secs": round(secs), "measured": bool(measured)}


def remote_size(repo, file):
    req = urllib.request.Request(HF.format(repo=repo, file=file), method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("x-linked-size") or r.headers.get("Content-Length") or 0)


def _install(mid):
    m, job = BY_ID[mid], jobs[mid]
    try:
        files = [(r, f, l, remote_size(r, f)) for r, f, l in filter(None, m["files"])]
        job["total"] = sum(s for *_, s in files)
        base = 0
        for repo, file, local, size in files:
            if os.path.exists(local) and os.path.getsize(local) >= size:
                base += size
                continue
            os.makedirs(os.path.dirname(local), exist_ok=True)
            part = local + ".part"
            for _ in range(60):  # automatic resume after interruptions
                p = subprocess.Popen(["curl.exe" if os.name == "nt" else "curl", "-L", "--fail", "-s", "-C", "-",
                                      "--retry", "10", "--retry-delay", "3", "--retry-all-errors", "-o", part,
                                      HF.format(repo=repo, file=file)], creationflags=NOWIN)
                while p.poll() is None:
                    job["done"] = base + (os.path.getsize(part) if os.path.exists(part) else 0)
                    time.sleep(1)
                if os.path.exists(part) and os.path.getsize(part) >= size:
                    break
                time.sleep(3)
            if not os.path.exists(part) or os.path.getsize(part) < size:
                raise RuntimeError("téléchargement incomplet : " + file)
            os.replace(part, local)
            base += size
            job["done"] = base
        job["state"] = "ok"
    except Exception as e:
        job.update(state="error", error=str(e))


_jobs_lock = threading.Lock()


def install(mid):
    with _jobs_lock:  # two clicks must not start two downloads writing the same file
        if mid not in BY_ID or jobs.get(mid, {}).get("state") == "running":
            return
        jobs[mid] = {"done": 0, "total": int(BY_ID[mid]["size_gb"] * 2**30), "state": "running", "error": ""}
    threading.Thread(target=_install, args=(mid,), daemon=True).start()


def listing(settings):
    import syscheck
    info = syscheck.current()
    rec = set(filter(None, info.get("recommended", {}).values()))
    out = []
    for m in CATALOG:
        key = m["id"] if m["kind"] == "llm" else ("vid:" if m["kind"] == "video" else "img:") + m["ckpt"]
        meas = settings.get("speed", {}).get(key)
        inst = installed(m)  # checks up to 11 files: once per model
        vis = has_vision(m) if inst else m.get("vision", False)
        out.append({k: m.get(k) for k in ("id", "kind", "label", "quality", "size_gb", "desc", "repo", "ckpt")}
                   | {"style": m.get("style", ""), "installed": inst, "job": jobs.get(m["id"]), "vision": vis,
                      "visionMissing": inst and m.get("vision", False) and not vis,
                      "visionGb": 0.9 if m.get("vision") else 0,
                      "think": m.get("think", False), "custom": m.get("custom", False),
                      "cloud": m.get("providerLabel") if m.get("cloud") else None,
                      "recommended": m["id"] in rec, "qualityGuess": m.get("qualityGuess", False),
                      **indicators(m, meas, info)})
    return out


# ---------------------------------------------------------------- models added from Hugging Face
CUSTOM_FILE = os.path.join(config.DATA, "custom_models.json")
THINKERS = ("qwen3", "gpt-oss", "gemma-4", "nemotron", "deepseek-r1", "magistral", "phi-4-reasoning")


def _params(name):
    """'Qwen3.5-35B-A3B' -> (35, 3); 'gemma-4-12b' -> (12, None)."""
    n = re.findall(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", name.replace("_", "-"))
    a = re.search(r"[-_]A(\d+(?:\.\d+)?)[bB]", name)
    total = max((float(x) for x in n), default=None)
    return total, float(a.group(1)) if a else None


def quality_guess(total_b):
    if not total_b:
        return 3
    return 2 if total_b < 6 else 3 if total_b < 15 else 4 if total_b < 60 else 5


def load_custom():
    for m in config.read_json(CUSTOM_FILE, []) or []:
        if isinstance(m, dict) and m.get("id"):
            m["files"] = [tuple(f) if f else None for f in m.get("files", [])]
            if m["id"] not in BY_ID and not blocked(m["id"], m.get("repo", "")):
                CATALOG.append(m)
                BY_ID[m["id"]] = m


def save_custom():
    config.write_json(CUSTOM_FILE, [m for m in CATALOG if m.get("custom")])


def hf_get(path):
    with urllib.request.urlopen("https://huggingface.co/api/" + path, timeout=30) as r:
        return json.load(r)


def hf_search(q, kind):
    from urllib.parse import quote
    if blocked(q):
        return []
    if kind == "image":
        url = f"models?search={quote(q)}&pipeline_tag=text-to-image&sort=downloads&limit=40&full=true"
    else:
        url = f"models?search={quote(q)}&filter=gguf&sort=downloads&limit=40&full=true"
    return [{"repo": r["id"], "downloads": r.get("downloads", 0), "likes": r.get("likes", 0)}
            for r in hf_get(url) if not blocked(r["id"], *r.get("tags", []))][:25]


def hf_files(repo, kind):
    if blocked(repo):
        raise RuntimeError("modèle non proposé dans Sparky")
    info = hf_get(f"models/{repo}")
    if blocked(*info.get("tags", [])):
        raise RuntimeError("modèle non proposé dans Sparky")
    tree = hf_get(f"models/{repo}/tree/main?recursive=true")
    files = [f for f in tree if f.get("type") == "file"]
    if kind == "image":  # single-file SDXL checkpoints
        out = [f for f in files if f["path"].endswith(".safetensors") and f.get("size", 0) > 1.5 * 2**30
               and "/" not in f["path"] and not blocked(f["path"])]
        return {"files": [{"path": f["path"], "size_gb": round(f["size"] / 2**30, 2)} for f in out], "mmproj": None}
    ggufs = [f for f in files if f["path"].endswith(".gguf") and not blocked(f["path"])]
    mm = [f for f in ggufs if "mmproj" in f["path"].lower()]
    mm.sort(key=lambda f: (0 if "F16" in f["path"] and "BF16" not in f["path"] else 1 if "BF16" in f["path"] else 2))
    models = [f for f in ggufs if "mmproj" not in f["path"].lower() and "mtp" not in f["path"].lower()
              and not re.search(r"-0000[2-9]-of-", f["path"])]
    total, active = _params(repo)
    out = []
    for f in sorted(models, key=lambda f: f["size"]):
        gb = f["size"] / 2**30
        fake = {"id": "", "kind": "llm", "size_gb": gb, "active_gb": gb * (active / total) if active and total else gb}
        out.append({"path": f["path"], "size_gb": round(gb, 2), **indicators(fake)})
    return {"files": out, "mmproj": mm[0]["path"] if mm else None,
            "mmproj_gb": round(mm[0]["size"] / 2**30, 2) if mm else 0, "quality": quality_guess(total)}


def hf_install(repo, path, kind, mmproj=None):
    if blocked(repo, path):
        raise RuntimeError("modèle non proposé dans Sparky")
    base = os.path.splitext(os.path.basename(path))[0]
    mid = "hf:" + base
    if mid not in BY_ID:
        info = hf_files(repo, kind)
        f = next(x for x in info["files"] if x["path"] == path)
        if kind == "image":
            m = ckpt(mid, base, 3, repo, path, f["size_gb"], desc=repo)
        else:
            total, active = _params(repo + " " + path)
            mm = mmproj if mmproj is not None else info["mmproj"]
            m = llm(mid, base, quality_guess(total), repo, path, None, f["size_gb"],
                    active_gb=f["size_gb"] * active / total if active and total else None,
                    think=any(t in repo.lower() for t in THINKERS), desc=repo)
            if mm:
                m["files"][1] = (repo, mm, os.path.join(P, "mmproj-" + base + ".gguf"))
            m["vision"] = bool(mm)
        m.update(custom=True, qualityGuess=True, repo=repo)
        CATALOG.append(m)
        BY_ID[mid] = m
        save_custom()
    install(mid)
    return mid


def delete(mid):
    """Deletes a model's files, except those another installed model still uses."""
    m = BY_ID.get(mid)
    if not m:
        return
    shared = {f[2] for o in CATALOG if o is not m and installed(o) for f in o["files"] if f}
    for f in [] if m.get("local") else m["files"]:  # models found on the PC (LM Studio, Ollama...) stay where they are
        if f and f[2] not in shared and os.path.exists(f[2]):
            os.remove(f[2])
    if m.get("custom"):
        CATALOG.remove(m)
        BY_ID.pop(mid)
        save_custom()


load_custom()
