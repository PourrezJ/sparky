"""Catalogue des modeles : installation (telechargement repris en cas de coupure), estimation de
vitesse sur cette machine (d'apres l'analyse systeme) et indicateurs (qualite, memoire, vision, reflexion).

Le catalogue ne propose que des modeles officiels ou largement utilises ; les variantes "uncensored",
"abliterated", NSFW, etc. sont exclues, y compris de la recherche Hugging Face.
"""
import json, os, re, subprocess, threading, time, urllib.request

import config

HF = "https://huggingface.co/{repo}/resolve/main/{file}"
M, P, CM = config.LLM_DIR, config.MMPROJ_DIR, config.COMFY_MODELS
REF_BW = 230.0  # bande passante (Go/s) de la machine du benchmark (RTX Spark) : ref_tps y a ete mesure


def _first(*paths):
    """Premier chemin existant (anciens noms de fichiers acceptes), sinon le premier (nom canonique)."""
    return next((p for p in paths if os.path.exists(p)), paths[0])


# brouillon MTP optionnel : double la vitesse de Qwen3.8 27B (mesure : 16 -> 33 mots/s)
MTP = _first(os.path.join(M, "mtp-Qwen3.8-27B-Q4_0.gguf"), os.path.join(os.path.dirname(M), "mtp", "mtp-Qwen3.8-27B-Q4_0.gguf"))
BLOCKED = ("uncensor", "abliterat", "heretic", "nsfw", "lustify", "pony", "unfilter", "unaligned", "jailbreak",
           "erotic", "porn", "hentai", "lewd", "nude", "xxx", "18+", "hauhau", "dolphin", "not-for-all-audiences")
NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def blocked(*texts):
    t = " ".join(str(x) for x in texts).lower()
    return any(w in t for w in BLOCKED)


def llm(id, label, quality, repo, file, mmproj, size_gb, active_gb=None, think=True, desc="", ref_tps=None, alt=()):
    """mmproj : fichier de vision dans le meme depot (None = modele qui ne voit pas les images).
    ref_tps : vitesse mesuree sur la machine du benchmark ; alt : autres noms locaux acceptes."""
    stem = repo.split("/")[-1].replace("-GGUF", "")
    main = _first(os.path.join(M, file), *[os.path.join(M, a) for a in alt])
    vis = _first(os.path.join(P, f"mmproj-{stem}-F16.gguf"), os.path.join(P, f"mmproj-{stem.replace('-it', '')}-F16.gguf"))
    return {"id": id, "kind": "llm", "label": label, "quality": quality, "repo": repo, "desc": desc,
            "files": [(repo, file, main), (repo, mmproj, vis) if mmproj else None], "ref_tps": ref_tps,
            "size_gb": size_gb, "active_gb": active_gb or size_gb, "think": think, "vision": mmproj is not None}


IPA = "h94/IP-Adapter"
IPADAPTER = [  # references personnage / style pour SDXL, partagees entre tous les modeles SDXL
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
    """FLUX.2 Klein 4B : 3 s par image au benchmark, comprend les phrases et retouche par instruction."""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10, creationflags=NOWIN).stdout.lower()
    except Exception:
        gpu = ""
    # encodeur en FP4 sur les GPU Blackwell (RTX 50, Spark), FP8 ailleurs
    te = "qwen_3_4b_fp4_mixed.safetensors" if any(k in gpu for k in ("spark", "gb10", "rtx 50", " 50")) else "qwen_3_4b_fp8_mixed.safetensors"
    files = [("black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors", os.path.join(CM, "diffusion_models", "flux-2-klein-4b-fp8.safetensors")),
             ("Comfy-Org/z_image_turbo", "split_files/text_encoders/" + te, os.path.join(CM, "text_encoders", te)),
             ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors", os.path.join(CM, "vae", "flux2-vae.safetensors"))]
    return {"id": "klein4b", "kind": "image", "engine": "klein", "label": "FLUX.2 Klein 4B", "quality": 4, "style": "photo",
            "files": files, "size_gb": 3.8 + (3.2 if "fp4" in te else 5.3) + 0.3, "ckpt": "klein4b", "lora": None,
            "repo": files[0][0], "est_secs": 3, "clip": te,
            "desc": "Images en ~3 s : rapide, texte lisible, retouche par instruction."}


WHISPER_FILES = ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json", "special_tokens_map.json", "model.safetensors")


def voice(id, label, quality, repo, size_gb, gpu_secs, cpu_secs, desc=""):
    """Dictee vocale hors ligne (Whisper, via transformers). *_secs : temps pour 10 s de parole."""
    d = os.path.join(config.VOICE_DIR, repo.split("/")[-1])
    return {"id": id, "kind": "voice", "label": label, "quality": quality, "repo": repo, "desc": desc,
            "files": [(repo, f, os.path.join(d, f)) for f in WHISPER_FILES], "dir": d, "size_gb": size_gb,
            "gpu_secs": gpu_secs, "cpu_secs": cpu_secs, "ckpt": id}


# vitesses (ref_tps) et qualites : benchmark local RTX Spark (oct. 2026), GPQA Diamond publie
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
    klein(),
    qwen_image("qwen21", "Qwen-Image 2.1", 5, desc="Le plus réaliste (~35 s par image), retouche par instruction."),
    ckpt("juggernaut", "Juggernaut XL v9", 4, "RunDiffusion/Juggernaut-XL-v9", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
         6.6, style="photo", desc="Photo réaliste (SDXL), léger."),
    ckpt("animagine", "Animagine XL 4.0", 4, "cagliostrolab/animagine-xl-4.0", "animagine-xl-4.0-opt.safetensors",
         6.5, style="illustration", desc="Illustration, anime, BD (SDXL)."),
]
BY_ID = {m["id"]: m for m in CATALOG}
jobs = {}  # id -> {"done": octets, "total": octets, "state": "running|ok|error", "error": str}


def _ok(f):
    return bool(f) and os.path.exists(f[2]) and not os.path.exists(f[2] + ".part")


def installed(m):
    """Un assistant est utilisable des que son modele est la ; la vision (mmproj) est un complement."""
    files = m["files"][:1] if m["kind"] == "llm" else m["files"]
    return all(_ok(f) for f in files if f)


def has_vision(m):
    return m["kind"] == "llm" and len(m["files"]) > 1 and _ok(m["files"][1])


def llm_args(m):
    a = [m["files"][0][2]] + (["--mmproj", m["files"][1][2]] if has_vision(m) else [])
    if m["id"] == "qwen38" and os.path.exists(MTP):
        a += ["--spec-type", "draft-mtp", "--spec-draft-model", MTP, "--spec-draft-n-max", "3"]
    return a


def indicators(m, measured=None, sysinfo=None):
    """speed : 1 lent / 2 correct / 3 rapide ; fit : 2 tient avec la generation d'images, 1 juste, 0 trop gros."""
    if sysinfo is None:
        import syscheck
        sysinfo = syscheck.current()
    a = sysinfo["assess"]
    mem, bw, img = a["model_mem_gb"], a["bandwidth"], a["image_gb"] if a["images"] else 0
    if m["kind"] == "llm":
        mtp = m["id"] == "qwen38" and os.path.exists(MTP)
        est = m["ref_tps"] * bw / REF_BW if m.get("ref_tps") else bw / m["active_gb"] * 0.85
        tps = measured or est * (2.05 if mtp else 1.0)
        speed = 3 if tps >= 40 else 2 if tps >= 20 else 1
        need = m["size_gb"] + 2.5  # modele + contexte
        fit = 2 if need + img <= mem else 1 if need <= mem else 0
        return {"speed": speed, "fit": fit, "tps": round(tps), "measured": bool(measured)}
    if m["kind"] == "voice":
        secs = measured or (m["gpu_secs"] if a["mode"] != "cpu" else m["cpu_secs"])
        return {"speed": 3 if secs <= 3 else 2 if secs <= 10 else 1, "fit": 2, "secs": round(secs, 1), "measured": bool(measured)}
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
            for _ in range(60):  # reprise automatique en cas de coupure
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


def install(mid):
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
        meas = settings.get("speed", {}).get(m["id"] if m["kind"] == "llm" else "img:" + m["ckpt"])
        out.append({k: m.get(k) for k in ("id", "kind", "label", "quality", "size_gb", "desc", "repo", "ckpt")}
                   | {"style": m.get("style", ""), "installed": installed(m), "job": jobs.get(m["id"]),
                      "vision": has_vision(m) if installed(m) else m.get("vision", False),
                      "visionMissing": installed(m) and m.get("vision", False) and not has_vision(m),
                      "visionGb": 0.9 if m.get("vision") else 0,
                      "think": m.get("think", False), "custom": m.get("custom", False),
                      "recommended": m["id"] in rec, "qualityGuess": m.get("qualityGuess", False),
                      **indicators(m, meas, info)})
    return out


# ---------------------------------------------------------------- modeles ajoutes depuis Hugging Face
CUSTOM_FILE = os.path.join(config.DATA, "custom_models.json")
THINKERS = ("qwen3", "gpt-oss", "gemma-4", "nemotron", "deepseek-r1", "magistral", "phi-4-reasoning")


def _params(name):
    """'Qwen3.5-35B-A3B' -> (35, 3) ; 'gemma-4-12b' -> (12, None)."""
    n = re.findall(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", name.replace("_", "-"))
    a = re.search(r"[-_]A(\d+(?:\.\d+)?)[bB]", name)
    total = max((float(x) for x in n), default=None)
    return total, float(a.group(1)) if a else None


def quality_guess(total_b):
    if not total_b:
        return 3
    return 2 if total_b < 6 else 3 if total_b < 15 else 4 if total_b < 60 else 5


def load_custom():
    if os.path.exists(CUSTOM_FILE):
        for m in json.load(open(CUSTOM_FILE, encoding="utf-8")):
            m["files"] = [tuple(f) if f else None for f in m["files"]]
            if m["id"] not in BY_ID and not blocked(m["id"], m.get("repo", "")):
                CATALOG.append(m)
                BY_ID[m["id"]] = m


def save_custom():
    os.makedirs(os.path.dirname(CUSTOM_FILE), exist_ok=True)
    json.dump([m for m in CATALOG if m.get("custom")], open(CUSTOM_FILE, "w", encoding="utf-8"), indent=1)


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
        raise RuntimeError("modèle non proposé dans Defend Local IA")
    info = hf_get(f"models/{repo}")
    if blocked(*info.get("tags", [])):
        raise RuntimeError("modèle non proposé dans Defend Local IA")
    tree = hf_get(f"models/{repo}/tree/main?recursive=true")
    files = [f for f in tree if f.get("type") == "file"]
    if kind == "image":  # checkpoints SDXL en un seul fichier
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
        raise RuntimeError("modèle non proposé dans Defend Local IA")
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
    """Supprime les fichiers d'un modele, sauf ceux qu'un autre modele installe utilise encore."""
    m = BY_ID.get(mid)
    if not m:
        return
    shared = {f[2] for o in CATALOG if o is not m and installed(o) for f in o["files"] if f}
    for f in m["files"]:
        if f and f[2] not in shared and os.path.exists(f[2]):
            os.remove(f[2])
    if m.get("custom"):
        CATALOG.remove(m)
        BY_ID.pop(mid)
        save_custom()


load_custom()
