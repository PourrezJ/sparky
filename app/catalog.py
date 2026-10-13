"""Catalogue des modeles : installation (telechargement repris en cas de coupure), estimation de
vitesse sur cette machine (d'apres l'analyse systeme) et indicateurs (qualite, memoire, vision, reflexion).

Le catalogue ne propose que des modeles officiels ou largement utilises ; les variantes "uncensored",
"abliterated", NSFW, etc. sont exclues, y compris de la recherche Hugging Face.
"""
import json, os, re, subprocess, threading, time, urllib.request

import config

HF = "https://huggingface.co/{repo}/resolve/main/{file}"
M, P, CM = config.LLM_DIR, config.MMPROJ_DIR, config.COMFY_MODELS
MTP = os.path.join(M, "mtp-Qwen3.8-27B-Q4_0.gguf")  # brouillon MTP optionnel (accelere Qwen3.8 27B)
BLOCKED = ("uncensor", "abliterat", "heretic", "nsfw", "lustify", "pony", "unfilter", "unaligned", "jailbreak",
           "erotic", "porn", "hentai", "lewd", "nude", "xxx", "18+", "hauhau", "dolphin", "not-for-all-audiences")
NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def blocked(*texts):
    t = " ".join(str(x) for x in texts).lower()
    return any(w in t for w in BLOCKED)


def llm(id, label, quality, repo, file, mmproj, size_gb, active_gb=None, think=True, desc=""):
    """mmproj : fichier de vision dans le meme depot (None = modele qui ne voit pas les images)."""
    stem = repo.split("/")[-1].replace("-GGUF", "")
    return {"id": id, "kind": "llm", "label": label, "quality": quality, "repo": repo, "desc": desc,
            "files": [(repo, file, os.path.join(M, file)),
                      (repo, mmproj, os.path.join(P, f"mmproj-{stem}-F16.gguf")) if mmproj else None],
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


U = "unsloth/"
CATALOG = [
    llm("qwen38", "Qwen3.8 27B", 5, U + "Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf", "mmproj-F16.gguf", 16.2,
        desc="Le plus intelligent : raisonnement, code, analyse d'images."),
    llm("gemma26", "Gemma 4 26B A4B", 4, U + "gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", "mmproj-F16.gguf", 16.9,
        active_gb=2.6, desc="Google. Mixture d'experts : très rapide et polyvalent."),
    llm("qwen35moe", "Qwen3.5 35B A3B", 4, U + "Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-Q4_K_M.gguf", "mmproj-F16.gguf", 21.3,
        active_gb=1.9, desc="Mixture d'experts : très rapide, demande beaucoup de mémoire."),
    llm("nemotron30", "Nemotron 3.5 Lightning 30B", 4, U + "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
        "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_M.gguf", None, 23.5, active_gb=2.2,
        desc="NVIDIA. Rapide, fort en raisonnement ; ne voit pas les images."),
    llm("gptoss20", "gpt-oss 20B", 4, U + "gpt-oss-20b-GGUF", "gpt-oss-20b-Q4_K_M.gguf", None, 10.8, active_gb=2.4,
        desc="OpenAI, modèle ouvert. Bon raisonnement ; ne voit pas les images."),
    llm("gemma12", "Gemma 4 12B", 3, U + "gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf", "mmproj-F16.gguf", 6.8,
        desc="Google. Léger et équilibré."),
    llm("qwen359", "Qwen3.5 9B", 3, U + "Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf", "mmproj-F16.gguf", 6.2,
        desc="Léger, bon en français."),
    llm("gemmae4b", "Gemma 4 E4B", 2, U + "gemma-4-e4b-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf", "mmproj-F16.gguf", 5.5,
        desc="Google. Pour les petites machines."),
    llm("qwen354", "Qwen3.5 4B", 2, U + "Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", "mmproj-F16.gguf", 3.2,
        desc="Le plus petit : fonctionne même sans GPU."),
    qwen_image("qwen21", "Qwen-Image 2.1", 5, desc="Le plus réaliste, comprend les phrases, retouche par instruction."),
    ckpt("juggernaut", "Juggernaut XL v9", 4, "RunDiffusion/Juggernaut-XL-v9", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
         6.6, style="photo", desc="Photo réaliste (SDXL), léger."),
    ckpt("animagine", "Animagine XL 4.0", 4, "cagliostrolab/animagine-xl-4.0", "animagine-xl-4.0-opt.safetensors",
         6.5, style="illustration", desc="Illustration, anime, BD (SDXL)."),
]
BY_ID = {m["id"]: m for m in CATALOG}
jobs = {}  # id -> {"done": octets, "total": octets, "state": "running|ok|error", "error": str}


def installed(m):
    return all(os.path.exists(f[2]) and not os.path.exists(f[2] + ".part") for f in m["files"] if f)


def llm_args(m):
    a = [m["files"][0][2]] + (["--mmproj", m["files"][1][2]] if len(m["files"]) > 1 and m["files"][1] else [])
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
        tps = measured or bw / m["active_gb"] * (1.8 if mtp else 1.0) * 0.85
        speed = 3 if tps >= 40 else 2 if tps >= 15 else 1
        need = m["size_gb"] + 2.5  # modele + contexte
        fit = 2 if need + img <= mem else 1 if need <= mem else 0
        return {"speed": speed, "fit": fit, "tps": round(tps), "measured": bool(measured)}
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
                      "vision": m.get("vision", True), "think": m.get("think", False), "custom": m.get("custom", False),
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
