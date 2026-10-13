"""Sparky's agent harness: local (vision) LLM loop <-> ComfyUI generation tools.

At each step the model answers with a JSON action (generate / edit / reply). After each
generation it receives the produced image (and the references used) to check it and
fix it itself if needed, up to MAX_STEPS steps per message.
"""
import base64, io, json, os, random, re, threading, time, urllib.error, urllib.request, uuid

from PIL import Image, ImageDraw

import config

COMFY = f"http://127.0.0.1:{config.COMFY_PORT}"
LLM = f"http://127.0.0.1:{config.LLM_PORT}"
COMFY_IN = config.COMFY_IN
COMFY_OUT = config.COMFY_OUT
LORA_DIR = os.path.join(config.COMFY_MODELS, "loras")
SDXL_LORAS = [("add-detail-xl.safetensors", 0.5), ("sdxl-face-detailer-lora.safetensors", 0.6)]  # optional
QWEN = {"unet": "qwen_image_2.1_int8_convrot.safetensors", "clip": "qwen3vl_8b_int8_convrot.safetensors",
        "vae": "qwen_image_2.1_vae_bf16.safetensors"}
DATA = config.DATA
# thinking level -> (effort passed to the model, thinking budget in tokens)
THINK = {"off": (None, 0), "low": ("low", 1024), "medium": ("medium", 4096), "high": ("high", 12000)}
ACTIVITY = {"llm": 0, "image": 0}  # number of requests in progress (for the UI monitor)


class busy:
    def __init__(self, what):
        self.what = what

    def __enter__(self):
        ACTIVITY[self.what] += 1

    def __exit__(self, *a):
        ACTIVITY[self.what] -= 1


SETTINGS_FILE = os.path.join(DATA, "settings.json")
SETTINGS = {"llm": None, "ckpt": None, "steps": 30, "cfg": 5.5, "max_steps": 6, "think": "off", "ui_lang": "en"}


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        SETTINGS.update(json.load(open(SETTINGS_FILE, encoding="utf-8")))


def save_settings():
    os.makedirs(DATA, exist_ok=True)
    json.dump(SETTINGS, open(SETTINGS_FILE, "w", encoding="utf-8"), indent=1)


RECENT_IMAGES = 10  # beyond this, the oldest images are turned into text (in batches of 6)
SIZES = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832)}
DEFAULT_WEIGHT = {"face": 0.9, "character": 0.45, "style": 0.6, "scene": 0.5}
DEFAULT_END = {"face": 0.7, "character": 0.6, "style": 1.0, "scene": 1.0}
ANNOUNCE = re.compile(
    r"\b(je vais (le |la |les |lui |l'|te |en |y )?(faire|refaire|g[eé]n[eé]rer|reg[eé]n[eé]rer|pousser|essayer|retenter|"
    r"modifier|corriger|cr[eé]er|relancer|ajouter|enlever|changer|mettre|tenter|retravailler|recommencer)"
    r"|je m'en occupe|je (re)?lance|je refais|je r?essaie"
    r"|i'?ll (try|make|generate|redo|push|fix|create|edit|change|add|remove|regenerate)|let me (try|redo|fix|generate))\b", re.I)
LOCATE = {"face": "the face of {t} (tight box around the head, hair included)",
          "character": "the head and upper body of {t} (hair and clothes, without furniture)"}
NEGATIVE = "blurry, low quality, deformed, bad anatomy, extra fingers, extra limbs, watermark, text"

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.md")
QWEN_NOTES = """

IMAGE MODEL = {label} — these rules REPLACE the SDXL-specific ones above (weights, end_at, face/character crops, denoise, region boxes, negative):
- It understands full sentences: write the prompt as one rich, natural English paragraph describing the finished image (subject, look, outfit, pose, place, light, camera/framing, mood). Exact text to render in the image goes in "double quotes".
- It is photorealistic by default and renders faces very well. For a drawing, say the style explicitly.
- REFERENCES: list images in refs (role/weight/end_at are ignored). Inside the prompt, call them "image 1", "image 2"... in the order of refs for generate. For edit, the edited image is ALWAYS "image 1" and refs follow as image 2, 3...  E.g. generate with refs [#1]: "The woman from image 1, same face, hair and outfit, now walking on a beach at sunset...".
- EDIT = INSTRUCTION. The prompt is an imperative instruction about image 1, and say what must stay unchanged: "Remove the beer bottle and the cigarettes from the table. Keep everything else exactly the same." / "Make the man in image 1 smile slightly, keep his face identical." Use region "all" (the model edits precisely by itself); use "mask" only if the user painted an area. Pose, framing, clothes, background: everything can be changed by instruction.
- negative and denoise are ignored.
- SAME CHARACTER/SCENE, DIFFERENT LOOK: to change the rendering STYLE of an existing image (comic book, watercolor, anime, real photo...) or the VIEW/POSE/FRAMING of its character (from behind, sitting, wide shot, close-up), use edit on that image with an instruction ("Turn this image into a 1940s comic book illustration with bold ink lines and flat colors, same character, same scene." / "Show the same woman from behind, walking away down the same alley."). Use generate + refs only for a really new scene or when combining several images.
- TEXT CHECK: when the image contains requested text, read it letter by letter (accents, spaces, punctuation) and compare with the request; any wrong or extra character = failure, fix it with an edit like: Replace the title text with exactly "Les Néons"."""


def system_prompt():
    """Re-read on every call: prompt.md can be tweaked without restarting."""
    m = image_model()
    info = f"{m['label']} ({m.get('style') or 'general'} style)"
    base = open(PROMPT_FILE, encoding="utf-8").read() + "\nCurrent image model: " + info + "."
    if m.get("engine") in INSTRUCT:
        base += QWEN_NOTES.replace("{label}", m["label"].upper())
    return base


# ---------------------------------------------------------------- utilities
def http_json(url, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> {e.code} : {e.read().decode(errors='replace')[:600]}")


def fit_1mp(im):
    w, h = im.size
    s = (1024 * 1024 / (w * h)) ** 0.5
    return im.resize((max(64, round(w * s / 8) * 8), max(64, round(h * s / 8) * 8)), Image.LANCZOS)


def box_px(box, w, h, pad=0.0):
    x1, y1, x2, y2 = [float(v) for v in box]
    if max(x1, y1, x2, y2) <= 1.0:  # tolerate 0-1 coordinates
        x1, y1, x2, y2 = x1 * 1000, y1 * 1000, x2 * 1000, y2 * 1000
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw, bh = x2 - x1, y2 - y1
    x1, x2, y1, y2 = x1 - bw * pad, x2 + bw * pad, y1 - bh * pad, y2 + bh * pad
    return (max(0, int(x1 * w / 1000)), max(0, int(y1 * h / 1000)),
            min(w, int(x2 * w / 1000)), min(h, int(y2 * h / 1000)))


def ref_crop(path, role, box):
    """Crops the reference (square face, body with margin) and pads it to a 512 square."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if box:
        if role == "face":
            x1, y1, x2, y2 = box_px(box, w, h)
            cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.4
            x1, y1, x2, y2 = cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
        else:
            x1, y1, x2, y2 = box_px(box, w, h, 0.06)
        if x2 - x1 > 16 and y2 - y1 > 16:
            im = im.crop((x1, y1, x2, y2))
    s = max(im.size)
    sq = Image.new("RGB", (s, s), (128, 128, 128))
    sq.paste(im, ((s - im.size[0]) // 2, (s - im.size[1]) // 2))
    return sq.resize((512, 512), Image.LANCZOS)


def comfy_file(im, prefix):
    name = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    im.save(os.path.join(COMFY_IN, name))
    return name


def image_model():
    """Catalog entry of the active image model (engine 'sdxl' or 'qwen21')."""
    import catalog
    m = next((c for c in catalog.CATALOG if c.get("ckpt") == SETTINGS["ckpt"]), None)
    return m or {"engine": "sdxl", "ckpt": SETTINGS["ckpt"], "label": SETTINGS["ckpt"] or "aucun"}


def build_qwen(prompt, seed, size=None, images=(), lora=None, steps=25):
    """Qwen-Image 2.1: generation (empty latent) or instruction-based edit (latent of image 1).
    images: file names in ComfyUI/input, referred to in the prompt as 'image 1', 'image 2'..."""
    g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN["unet"], "weight_dtype": "default"}},
         "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN["clip"], "type": "qwen_image", "device": "default"}},
         "4": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN["vae"]}}}
    model = ["1", 0]
    if lora:
        g["1b"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": lora, "strength_model": 1.0}}
        model = ["1b", 0]
    g["2"] = {"class_type": "QwenImage21Cache", "inputs": {"model": model, "device": "auto", "dtype": "default"}}
    enc = {"clip": ["3", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024, "vae": ["4", 0]}
    for i, name in enumerate(images[:10]):
        g[f"6{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        enc[f"images.image_{i + 1}"] = [f"6{i}", 0]
    g["5"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
    if size:  # new image (any images serve as references)
        g["7"] = {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
        latent = ["7", 0]
    else:  # edit: same framing as image 1
        latent = ["5", 2]
    g["8"] = {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["5", 0], "negative": ["5", 1],
              "latent_image": latent, "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
              "scheduler": "simple", "denoise": 1.0}}
    g["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}}
    g["20"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "sparky"}}
    return g


def build_klein(prompt, seed, size=None, images=(), steps=4, clip="qwen_3_4b_fp4_mixed.safetensors"):
    """FLUX.2 Klein 4B (distilled, 4 steps): generation, or instruction-based edit when size is None
    (same framing as image 1). Images are passed as latent references ('image 1', 'image 2'...)."""
    if size is None:
        w, h = Image.open(os.path.join(COMFY_IN, images[0])).size
        size = (max(256, w // 16 * 16), max(256, h // 16 * 16))
    g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b-fp8.safetensors", "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "flux2", "device": "default"}},
         "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
         "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}}}
    cond = ["4", 0]
    for i, name in enumerate(images[:4]):
        g[f"6{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[f"7{i}"] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"6{i}", 0], "vae": ["3", 0]}}
        g[f"8{i}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": cond, "latent": [f"7{i}", 0]}}
        cond = [f"8{i}", 0]
    g["5"] = {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": cond}}
    g["6"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["7"] = {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": size[0], "height": size[1]}}
    g["8"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["9"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
    g["10"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["5", 0], "sampler": ["6", 0],
                                                                 "sigmas": ["7", 0], "latent_image": ["9", 0]}}
    g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}}
    g["20"] = {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "sparky"}}
    return g


INSTRUCT = ("qwen21", "klein")  # engines that understand sentences and edit by instruction


def build_instruct(m, prompt, seed, size=None, images=(), steps=None):
    if m.get("engine") == "klein":
        return build_klein(prompt, seed, size=size, images=images, steps=steps or 4,
                           clip=m.get("clip", "qwen_3_4b_fp4_mixed.safetensors"))
    return build_qwen(prompt, seed, size=size, images=images, lora=m.get("lora"), steps=steps or 25)


def build_graph(prompt, negative, seed, size=None, src=None, mask=None, denoise=1.0, refs=()):
    g = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SETTINGS["ckpt"]}}}
    model, clip = ["1", 0], ["1", 1]
    for i, (name, strength) in enumerate(SDXL_LORAS):  # detail / face LoRAs (if present)
        if os.path.exists(os.path.join(LORA_DIR, name)):
            k = f"5{i}"
            g[k] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": name,
                                                          "strength_model": strength, "strength_clip": strength}}
            model, clip = [k, 0], [k, 1]
    g["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip}}
    g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip}}
    if refs:  # character / style / scene references via IP-Adapter
        g["30"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"}}
        g["31"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"}}
        g["32"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"}}
        for i, r in enumerate(refs):
            a, b = str(40 + 2 * i), str(41 + 2 * i)
            ip, wt = {"face": ("32", "linear"), "style": ("31", "style transfer"),
                      "scene": ("31", "composition")}.get(r["role"], ("31", "linear"))
            g[a] = {"class_type": "LoadImage", "inputs": {"image": r["file"]}}
            g[b] = {"class_type": "IPAdapterAdvanced", "inputs": {
                "model": model, "ipadapter": [ip, 0], "image": [a, 0], "weight": r["weight"], "weight_type": wt,
                "combine_embeds": "concat", "start_at": 0.0, "end_at": r.get("end_at", 1.0), "embeds_scaling": "V only",
                "clip_vision": ["30", 0]}}
            model = [b, 0]
    if src is None:
        g["4"] = {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
        lat, denoise = ["4", 0], 1.0
    else:
        g["4"] = {"class_type": "LoadImage", "inputs": {"image": src}}
        g["5"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}}
        lat = ["5", 0]
        if mask:
            g["6"] = {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}}
            g["7"] = {"class_type": "GrowMask", "inputs": {"mask": ["6", 0], "expand": 6, "tapered_corners": False}}
            if denoise >= 0.95:
                # full removal / replacement: the area is emptied before being redrawn,
                # otherwise SDXL starts from the original object and makes it reappear
                g["8"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
                    "pixels": ["4", 0], "vae": ["1", 2], "mask": ["7", 0], "grow_mask_by": 8}}
                denoise = 1.0
            else:  # modification: keep the area's structure (color, overall shape)
                g["8"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["5", 0], "mask": ["7", 0]}}
            lat = ["8", 0]
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": ["2", 0], "negative": ["3", 0], "latent_image": lat, "seed": seed,
        "steps": int(SETTINGS["steps"]), "cfg": float(SETTINGS["cfg"]), "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": denoise}}
    g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    out = ["11", 0]
    if mask:  # paste back only the modified area, with softened edges
        g["12"] = {"class_type": "MaskToImage", "inputs": {"mask": ["7", 0]}}
        g["13"] = {"class_type": "ImageBlur", "inputs": {"image": ["12", 0], "blur_radius": 12, "sigma": 6.0}}
        g["14"] = {"class_type": "ImageToMask", "inputs": {"image": ["13", 0], "channel": "red"}}
        g["15"] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": ["4", 0], "source": ["11", 0], "x": 0, "y": 0, "resize_source": False, "mask": ["14", 0]}}
        out = ["15", 0]
    g["20"] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": "sparky"}}
    return g


def record_speed(key, value):
    sp = SETTINGS.setdefault("speed", {})
    sp[key] = round(value if key not in sp else 0.7 * sp[key] + 0.3 * value, 1)
    save_settings()


def comfy_run(graph):
    if not SETTINGS.get("ckpt"):
        raise RuntimeError("aucun modèle d'image n'est installé : ouvre Paramètres > Bibliothèque de modèles")
    t0 = time.time()
    with busy("image"):
        out = _comfy_run(graph)
    record_speed("img:" + SETTINGS["ckpt"], time.time() - t0)
    return out


def _comfy_run(graph):
    pid = http_json(COMFY + "/prompt", {"prompt": graph})["prompt_id"]
    while True:
        time.sleep(1)
        h = http_json(f"{COMFY}/history/{pid}")
        if pid in h:
            e = h[pid]
            break
    if e["status"]["status_str"] != "success":
        errs = [m[1].get("exception_message", "") for m in e["status"].get("messages", []) if m[0] == "execution_error"]
        raise RuntimeError("ComfyUI : " + (errs[0] if errs else "echec"))
    im = [i for o in e["outputs"].values() for i in o.get("images", [])][0]
    return os.path.join(COMFY_OUT, im.get("subfolder", ""), im["filename"])


class _Pairs(list):
    pass


def _obj(x):
    if isinstance(x, _Pairs):
        return {k: _obj(v) for k, v in x}
    if isinstance(x, list):
        return [_obj(v) for v in x]
    return x


def split_actions(raw):
    """Some models pack several actions in ONE object ({"action":"artifact",...,"action":"reply",...}).
    Plain json.loads keeps only the last duplicate key (the artifact was lost): split them, in order."""
    pairs = json.loads(raw, object_pairs_hook=_Pairs)
    acts, cur = [], {}
    for k, v in pairs:
        if k in cur:  # a key seen twice starts the next action
            acts.append(cur)
            cur = {}
        cur[k] = _obj(v)
    acts.append(cur)
    return [a for a in acts if a.get("action")] or [_obj(pairs)]


PLACEHOLDER = re.compile(r"^\s*\[[^\]]*(voir ARTIFACTS|see ARTIFACTS|artifact|fichier)[^\]]*\]\s*$|^\s*(\.\.\.|…|TODO\.?|// ?TODO.*|# ?TODO.*)\s*$", re.I | re.S)


def check_content(path, content):
    """Refuses placeholders and stubs: the model must send the complete file (it gets a precise error and retries)."""
    code = re.sub(r"(?m)^\s*(//|#).*$", "", content).strip()  # without comments
    if PLACEHOLDER.match(content) or (str(path).endswith((".js", ".py", ".css", "javascript", "python", "css")) and len(code) < 40):
        raise RuntimeError(f"« {path} » est vide ou ne contient qu'un texte de remplacement. Renvoie ce fichier avec son "
                           "contenu COMPLET et fonctionnel (vrai code, pas de note ni de « ... »).")


def clean_path(p):
    """Relative path inside a project (no drive, no '..', forward slashes)."""
    parts = [x for x in re.split(r"[\\/]+", str(p or "").strip()) if x not in ("", ".", "..") and ":" not in x]
    return "/".join(re.sub(r"[^\w.\-]+", "_", x) for x in parts)[:120]


def json_closed(t, st):
    """Tracks the text as it streams: True once the first JSON object {...} is closed (st['pos'] = its end)."""
    i = st["pos"]
    while i < len(t):
        c = t[i]
        i += 1
        if st["str"]:
            if st["esc"]:
                st["esc"] = False
            elif c == "\\":
                st["esc"] = True
            elif c == '"':
                st["str"] = False
        elif c == '"' and st["depth"]:
            st["str"] = True
        elif c == "{":
            st["depth"] += 1
        elif c == "}" and st["depth"]:
            st["depth"] -= 1
            if not st["depth"]:
                st["pos"], st["done"] = i, True
                return True
    st["pos"] = i
    return False


def partial_json(buf):
    """Extracts thought / action / text from a JSON object being written (for live display)."""
    out = {}
    m = re.search(r'"action"\s*:\s*"(\w+)"', buf)
    if m:
        out["action"] = m.group(1)
    keys = ("thought", "text") + (("id", "project", "path", "title", "kind", "language", "content") if out.get("action") == "artifact" else ())
    for key in keys:
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)' % key, buf)
        if m:
            s = re.sub(r"\\(u[0-9a-fA-F]{0,3})?$", "", m.group(1))  # truncated escape sequence
            try:
                s = json.loads('"' + s + '"')
            except ValueError:
                s = s.replace("\\n", "\n").replace('\\"', '"')
            out[key] = s
    return out


LANG_WORDS = {
    "French": "je tu il elle on nous vous le la les un une des du de et est sont pas ne ça ca c'est merci "
              "pour avec sur dans qui que quoi moi toi oui non mets enlève ajoute fais peux veux tes mes ton "
              "génère genere refais cette ce cet très tres aussi mais bonjour salut coucou",
    "English": "the and you your is are was what can could please thanks thank with for it this that "
               "make add remove put give show me my hey hello hi how why which would like do does",
    "Spanish": "el los las que es y por gracias hola hablas también tambien usted puedes quiero una con para "
               "muy pero cómo como qué",
    "German": "der die das und ist ich du nicht mit bitte danke kannst ein eine auch wie was",
    "Italian": "il lo gli che è e per grazie ciao sono puoi vorrei una con anche come cosa",
}
LANG_SETS = {k: set(v.split()) for k, v in LANG_WORDS.items()}


def detect_lang(text, weak=False):
    """Message language (frequent-word heuristic). None if too short/ambiguous ('ok', 'cool', 'lol').
    weak=True accepts a single clue (useful at the start of a conversation, when no language is known yet)."""
    import re
    words = re.findall(r"[a-zà-ÿ']+", (text or "").lower())
    if len(words) < (1 if weak else 2):
        return None
    scores = {k: sum(w in s for w in words) for k, s in LANG_SETS.items()}
    if re.search(r"[éèêàùçœ]", text.lower()):
        scores["French"] += 1
    if re.search(r"[¿¡ñ]", text):
        scores["Spanish"] += 2
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    return best if ranked[0] >= (1 if weak else 2) and ranked[0] > ranked[1] else None


# ---------------------------------------------------------------- session / agent
class Session:
    def __init__(self, sid=None, ephemeral=False):
        self.id = sid or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        self.ephemeral = ephemeral
        self.dir = os.path.join(DATA, "_tmp" if ephemeral else "", self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.images = {}   # id -> {path, kind, desc}
        self.masks = {}    # image id -> path of the painted mask
        self.history = []  # [{role, text, images}]
        self.events = []   # to rebuild the UI
        self.n = 0
        self.lock = threading.Lock()
        self._durls = {}
        self.boxes = {}  # cache of locate results
        self._emit = self._ev = lambda e: None
        self._read = False
        self.lang = None  # conversation language (updated on each unambiguous message)
        self.artifacts = {}  # id -> {title, kind, language, versions: [content, ...]}
        self.projects = {}   # id -> {title, entry, files: {path: content}, version}: multi-file projects
        self._pending = []   # actions left over from a response that packed several of them

    # -- persistence
    def title(self):
        for e in self.events:
            if e.get("type") == "user":
                return (e.get("text") or "Images").strip().splitlines()[0][:60] if e.get("text") else "Images"
        return "Nouvelle conversation"

    def save(self):
        if self.ephemeral:
            return
        st = {k: getattr(self, k) for k in ("images", "masks", "history", "events", "n", "lang", "artifacts", "projects")}
        st["title"] = self.title()
        with open(os.path.join(self.dir, "session.json"), "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False)

    @staticmethod
    def list_all():
        out = []
        for d in os.listdir(DATA) if os.path.isdir(DATA) else []:
            p = os.path.join(DATA, d, "session.json")
            if os.path.exists(p):
                try:
                    st = json.load(open(p, encoding="utf-8"))
                except ValueError:
                    continue
                title = st.get("title") or next((e.get("text") or "Images" for e in st.get("events", [])
                                                 if e.get("type") == "user"), None)
                if title:
                    out.append({"id": d, "title": title, "updated": os.path.getmtime(p)})
        return sorted(out, key=lambda s: -s["updated"])

    def delete(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    @classmethod
    def load(cls, sid):
        s = cls(sid)
        p = os.path.join(s.dir, "session.json")
        if os.path.exists(p):
            st = json.load(open(p, encoding="utf-8"))
            s.images = {int(k): v for k, v in st["images"].items()}
            s.masks = {int(k): v for k, v in st["masks"].items()}
            s.history, s.events, s.n = st["history"], st["events"], st["n"]
            s.lang = st.get("lang")
            s.artifacts = st.get("artifacts", {})
            s.projects = st.get("projects", {})
        return s


    # -- images
    def add_image(self, im, kind, desc="", parent=None):
        self.n += 1
        p = os.path.join(self.dir, f"{self.n}.png")
        im.save(p)
        self.images[self.n] = {"path": p, "kind": kind, "desc": desc, "parent": parent}
        return self.n

    def data_url(self, iid, side=896):
        key = (iid, side)
        if key not in self._durls:
            im = Image.open(self.images[iid]["path"]).convert("RGB")
            s = side / max(im.size)  # also upscales small images (locate precision)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            b = io.BytesIO()
            im.save(b, "JPEG", quality=88)
            self._durls[key] = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
        return self._durls[key]

    def last_image(self):
        ids = [i for i, v in self.images.items() if v["kind"] in ("user", "gen")]
        return max(ids) if ids else None

    def catalog(self):
        return self._catalog() + self._artifacts() + (
            f"\nREPLY LANGUAGE: {self.lang}. Write \"thought\" and \"reply\" in {self.lang}." if self.lang else "")

    def _artifacts(self):
        """Current content of each artifact (the history only keeps a mention, to avoid duplicating it)."""
        if not self.artifacts and not self.projects:
            return ""
        out = ["\n\nARTIFACTS (current content, reuse the id to update):"]
        # the history keeps each action with its full content; a file is repeated here only when its current
        # version is no longer visible in the history window (trimmed) - avoids sending the same code twice
        visible = "\n".join(h["text"] for h in self.history[-40:] if h["role"] == "assistant")

        def body(content):
            return "(current version: the one in your last artifact action above)" if json.dumps(content, ensure_ascii=False)[1:-1] in visible else content
        for aid, a in self.artifacts.items():
            out.append(f'--- id "{aid}" · "{a["title"]}" · kind {a["kind"]} · v{len(a["versions"])}\n{body(a["versions"][-1])}')
        for pid, p in self.projects.items():
            out.append(f'--- PROJECT "{pid}" · "{p["title"]}" · entry {p.get("entry")} · files: {", ".join(p["files"])}')
            out += [f"### {path}\n{body(content)}" for path, content in p["files"].items()]
        return "\n".join(out)

    def missing_files(self, pids):
        """Local CSS/JS files linked by a project's HTML pages but never written."""
        miss = []
        for pid in pids:
            files = self.projects.get(pid, {}).get("files", {})
            for path, content in files.items():
                if not path.endswith((".html", ".htm")):
                    continue
                base = path.rsplit("/", 1)[0] + "/" if "/" in path else ""
                refs = re.findall(r"""(?:href|src)\s*=\s*["']([^"'#?:]+\.(?:css|js))["']""", content)
                for r in refs:
                    full = clean_path(base + r) if not r.startswith("/") else clean_path(r)
                    if full not in files and full not in miss:
                        miss.append(full)
        return miss

    def tool_project_file(self, a):
        """One file of a multi-file project (same "project" id, different "path")."""
        content = a.get("content") or ""
        path = clean_path(a.get("path"))
        if not path or not content.strip():
            raise RuntimeError("fichier de projet vide : envoie path et le contenu complet dans content")
        check_content(path, content)
        pid = re.sub(r"[^a-z0-9-]+", "-", str(a.get("project") or a.get("id") or a.get("title") or "projet").lower()).strip("-")[:40] or "projet"
        p = self.projects.setdefault(pid, {"title": a.get("title") or pid, "entry": None, "files": {}, "version": 0})
        if a.get("title") and not p["files"]:
            p["title"] = a["title"]
        p["files"][path] = content
        p["version"] += 1
        if a.get("entry"):
            p["entry"] = clean_path(a["entry"])
        names = list(p["files"])
        main = next((n for n in ("index.html", "main.py") if n in names), None)
        if main:  # the conventional entry always wins (models sometimes name a module as the entry)
            p["entry"] = main
        elif not p["entry"] or p["entry"] not in p["files"]:
            p["entry"] = next((n for n in ("app.py",) if n in names), None) or \
                next((n for n in names if n.endswith((".html", ".py"))), names[0])
        return pid, p, path

    def auto_artifacts(self, text, ev):
        """Safety net: when a model pastes an HTML page or long code into its reply instead of
        using the artifact action -> the block goes to the panel, the reply keeps the text."""
        def repl(mt):
            lang, code = (mt.group(1) or "").lower(), mt.group(2).rstrip("\n")
            if lang not in ("html", "svg") and code.count("\n") < 15:
                return mt.group(0)
            t = re.search(r"<title>(.*?)</title>", code, re.S | re.I)
            h = re.search(r"^#{1,3}\s+(.+)$", text, re.M)
            title = (t.group(1).strip() if t else h.group(1).strip("* ") if h else f"Code {lang or 'texte'}")[:60]
            kind = "html" if lang in ("html", "") and re.search(r"<(!doctype|html|body|div|script)", code, re.I) else \
                "svg" if lang == "svg" else "markdown" if lang in ("md", "markdown") else "code"
            aid, art = self.tool_artifact({"id": title, "title": title, "kind": kind, "language": lang or kind, "content": code})
            ev({"type": "artifact", "id": aid, "title": art["title"], "kind": art["kind"], "language": art["language"],
                "version": len(art["versions"]), "content": code})
            return ""
        return re.sub(r"```([\w+#.-]*)[^\n]*\n(.*?)```", repl, text, flags=re.S).strip() or "C'est prêt, regarde le panneau à droite."

    def tool_artifact(self, a):
        content = a.get("content") or ""
        if not content.strip():
            raise RuntimeError("artifact vide : envoie le contenu complet dans args.content")
        check_content(a.get("path") or a.get("language") or "", content)
        aid = re.sub(r"[^a-z0-9-]+", "-", str(a.get("id") or a.get("title") or "artifact").lower()).strip("-")[:40] or "artifact"
        kind = a.get("kind") if a.get("kind") in ("html", "code", "markdown", "svg") else "code"
        art = self.artifacts.setdefault(aid, {"title": a.get("title") or aid, "kind": kind,
                                              "language": a.get("language") or kind, "versions": []})
        art.update(title=a.get("title") or art["title"], kind=kind, language=a.get("language") or art["language"])
        art["versions"].append(content)
        return aid, art

    def _catalog(self):
        lab = {"user": "envoyee par l'utilisateur", "gen": "generee", "ref": "decoupe de reference"}
        rows = []
        for i, v in sorted(self.images.items()):
            extra = f", derivee de #{v['parent']}" if v.get("parent") else ""
            rows.append(f"#{i} ({lab[v['kind']]}{extra}) : {v['desc'][:220] or '-'}")
        last = self.last_image()
        return ("\n\nIMAGES DE LA SESSION :\n" + "\n".join(rows) +
                (f"\nImage la plus recente : #{last}" if last else "")) if rows else "\n\nAucune image pour l'instant."

    # -- LLM
    def llm_messages(self):
        # The start of the context must stay IDENTICAL from one turn to the next so llama.cpp reuses
        # its cache (otherwise it rereads everything, images included: 20-30 s). So the history is only trimmed
        # and images only dropped in large batches, rarely.
        n = len(self.history)
        hist = self.history[(((n - 40) // 16 + 1) * 16) if n > 40 else 0:]
        import catalog
        cur = catalog.BY_ID.get(SETTINGS["llm"])
        vision = catalog.has_vision(cur) if cur else True
        # a model without vision only receives the image descriptions
        imgs = [i for h in hist for i in h.get("images", [])]
        k = len(imgs)
        shown = imgs[(((k - RECENT_IMAGES) // 6 + 1) * 6) if k > RECENT_IMAGES else 0:] if vision else []
        msgs = [{"role": "system", "content": system_prompt()
                 + ("" if vision else "\nNOTE: you cannot see images in this session; rely on their descriptions.")}]
        for h in hist:
            if h["role"] == "assistant":
                msgs.append({"role": "assistant", "content": h["text"]})
                continue
            content = [{"type": "text", "text": h["text"]}]
            for i in h.get("images", []):
                if i in shown and i in self.images:
                    content += [{"type": "text", "text": f"Image #{i} :"},
                                {"type": "image_url", "image_url": {"url": self.data_url(i)}}]
                else:
                    content.append({"type": "text", "text": f"[image #{i}, plus affichee : {self.images.get(i, {}).get('desc', '')[:200]}]"})
            msgs.append({"role": "user", "content": content})
        # the image inventory changes every turn: put it at the END (cache intact)
        if msgs[-1]["role"] == "user":
            msgs[-1]["content"] = msgs[-1]["content"] + [{"type": "text", "text": self.catalog()}]
        return msgs

    def ask_llm(self, retry=True, think=None, dry=False):
        import catalog, zlib
        if think is None:  # level chosen in the model menu (if the model can think)
            think = SETTINGS.get("think", "off") if catalog.BY_ID.get(SETTINGS["llm"], {}).get("think") else "off"
        effort, budget = THINK.get(think, THINK["off"])
        kw = {"enable_thinking": bool(effort)} | ({"reasoning_effort": effort} if effort else {})
        body = {"messages": self.llm_messages(), "temperature": 0.6 if effort else 0.4, "max_tokens": 14000 + budget,
                "stream": True, "response_format": {"type": "json_object"}, "chat_template_kwargs": kw}
        if dry:  # llama.cpp anti-repetition, only after a loop (it can hurt legitimately repetitive code)
            body |= {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 6, "temperature": 0.7}
        req = urllib.request.Request(LLM + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t, rs, last, t0, t_rs, chk, looped = "", "", 0.0, time.time(), None, 0, False
        js = {"pos": 0, "depth": 0, "str": False, "esc": False, "done": False}
        with busy("llm"), urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[6:])
                    if chunk.get("timings", {}).get("predicted_per_second"):
                        record_speed(SETTINGS["llm"], chunk["timings"]["predicted_per_second"])
                    delta = chunk["choices"][0]["delta"]
                    t += delta.get("content") or ""
                    rs += delta.get("reasoning_content") or ""
                    if t and t_rs is None:  # end of thinking = start of the answer
                        t_rs = time.time() - t0
                except (ValueError, KeyError, IndexError):
                    continue
                if not self._read:
                    self._read = True
                    self._emit({"type": "read"})
                if budget and not t and len(rs) > budget * 4:  # ~4 characters per token: budget exceeded
                    break
                if json_closed(t, js):  # complete JSON object: the rest (whitespace, chatter) is useless
                    break
                out = t or rs
                if len(out) > 2500 and len(out) - chk > 400:  # repetition loop: the tail compresses extremely well
                    chk = len(out)
                    if len(zlib.compress(out[-1500:].encode())) < 160:
                        looped = True
                        break
                if time.time() - last > (0.12 if len(t) < 8000 else 0.35):
                    last = time.time()
                    if rs and not t:
                        self._emit({"type": "reasoning", "text": rs[-1500:], "secs": round(time.time() - t0)})
                    else:
                        self._emit({"type": "typing", **partial_json(t)})
        if looped and not dry:
            self._emit({"type": "status", "text": "le modèle tournait en boucle, nouvel essai", "reset": True})
            return self.ask_llm(retry, think, dry=True)
        if rs:  # keep in the conversation history (shown again on reopen)
            self._ev({"type": "reasoned", "secs": round(t_rs or time.time() - t0), "text": rs[-6000:]})
        if effort and not t.strip():  # thinking too long: direct answer
            self._emit({"type": "status", "text": "réflexion trop longue, réponse directe"})
            return self.ask_llm(retry, think="off")
        try:  # last raw output, for diagnostics
            open(os.path.join(DATA, "last_llm.txt"), "w", encoding="utf-8").write(t)
        except OSError:
            pass
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
        if js["done"]:
            t = t[:js["pos"]]
        m = re.search(r"\{.*\}", t, re.S)
        try:
            acts = split_actions(m.group(0))
            self._pending = acts[1:]  # several actions packed in one object: run them in order, without asking again
            return acts[0]
        except Exception:
            if retry:  # a single retry with a format reminder
                self.history.append({"role": "user", "images": [], "text":
                    "[FORMAT] Your last answer was not valid JSON. Answer again with ONE valid JSON object only."})
                try:
                    return self.ask_llm(retry=False)
                finally:
                    self.history.pop()
            return {"action": "reply", "args": {"text": t.strip() or "..."}}

    def fix_language(self, text):
        """Safeguard: if the reply isn't (entirely) in the conversation language, have it rewritten."""
        if not self.lang or not text:
            return text
        prose = re.sub(r"```.*?(```|$)|`[^`\n]*`", " ", text, flags=re.S)  # code has no language
        if len(prose.split()) < 4:
            return text
        if "```" in text:  # a rewrite could damage the code: keep it as is
            return text
        words = re.findall(r"[a-zà-ÿ']+", prose.lower())
        own = LANG_SETS.get(self.lang, set())
        foreign = {w for k, s in LANG_SETS.items() if k != self.lang for w in s} - own
        ratio = sum(w in foreign for w in words) / max(1, len(words))
        detected = detect_lang(text)
        if detected == self.lang and ratio < 0.06:
            return text
        self._emit({"type": "status", "text": "ajuste la langue"})
        body = {"messages": [{"role": "user", "content":
                    f"Rewrite the following message entirely in {self.lang}. Keep the meaning, tone, tutoiement, "
                    f"emojis and **bold**. Translate every word or expression that is in another language. "
                    f"Output only the rewritten message.\n\n{text}"}],
                "temperature": 0.2, "max_tokens": 900, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            with busy("llm"):
                out = http_json(LLM + "/v1/chat/completions", body)["choices"][0]["message"]["content"].strip()
            return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip() or text
        except Exception:
            return text

    def locate(self, iid, what):
        """Asks the vision model for the box (0-1000) of a specific element in the image."""
        key = f"{iid}|{what}"
        if key not in self.boxes:
            self._emit({"type": "status", "text": f"localise « {what} » sur #{iid}"})
            q = (f"Locate: {what}. Answer ONLY with JSON {{\"box\":[x1,y1,x2,y2]}}, coordinates normalized "
                 "from 0 to 1000 (0,0 = top-left corner, 1000,1000 = bottom-right corner). For a part of something "
                 "(neck, hand...) give its approximate box. If absent: {\"box\":null}.")
            # 1280 px: noticeably more precise than 896 for locating (tested)
            body = {"messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": self.data_url(iid, 1280)}}, {"type": "text", "text": q}]}],
                    "temperature": 0, "max_tokens": 120, "chat_template_kwargs": {"enable_thinking": False}}
            try:
                with busy("llm"):
                    t = http_json(LLM + "/v1/chat/completions", body)["choices"][0]["message"]["content"]
                box = json.loads(re.search(r"\{.*\}", t, re.S).group(0)).get("box")
                ok = isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)
                self.boxes[key] = box if ok else None
            except Exception:
                self.boxes[key] = None
        return self.boxes[key]

    # -- tools
    def prep_refs(self, refs):
        out, crops = [], []
        for r in (refs or [])[:4]:
            try:
                iid = int(str(r.get("image")).lstrip("#"))
            except (TypeError, ValueError):
                continue
            if iid not in self.images:
                continue
            role = r.get("role") if r.get("role") in DEFAULT_WEIGHT else "character"
            box = r.get("box")
            if role in LOCATE:
                box = self.locate(iid, LOCATE[role].format(t=r.get("target") or "the main person")) or box
            crop = ref_crop(self.images[iid]["path"], role, box)
            w = float(r.get("weight") or DEFAULT_WEIGHT[role])
            # the character is set in the first steps; the last ones let the checkpoint do
            # its rendering (otherwise a drawn reference's style "bleeds" onto a photo)
            end = float(r.get("end_at") or DEFAULT_END[role])
            out.append({"file": comfy_file(crop, "sparky_ref"), "role": role, "weight": max(0.0, min(1.5, w)),
                        "end_at": max(0.2, min(1.0, end))})
            crops.append(self.add_image(crop, "ref", f"{role} crop of #{iid}", parent=iid))
        return out, crops

    def ref_images(self, refs, exclude=()):
        """Qwen-Image: references are the whole images (the model understands 'the woman from image 2')."""
        out = []
        for r in refs or []:
            try:
                iid = int(str(r.get("image")).lstrip("#"))
            except (TypeError, ValueError):
                continue
            if iid in self.images and iid not in exclude and iid not in out:
                out.append(iid)
        return [comfy_file(fit_1mp(Image.open(self.images[i]["path"]).convert("RGB")), "sparky_ref") for i in out]

    def face_pass(self, iid, prompt, refs):
        """SDXL: redraws the main face at high resolution then pastes it back (sharp, detailed faces)."""
        if not SETTINGS.get("face_fix", True):
            return
        box = self.locate(iid, "the face of the main person")
        if not box:
            return
        im = Image.open(self.images[iid]["path"]).convert("RGB")
        W, H = im.size
        x1, y1, x2, y2 = box_px(box, W, H)
        if (y2 - y1) > 0.45 * H or (y2 - y1) < 24:  # face already large (close-up) or not found
            return
        self._emit({"type": "status", "text": "affine le visage"})
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.9
        x1, y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        x2, y2 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
        crop = im.crop((x1, y1, x2, y2))
        k = 1024 / max(crop.size)
        big = crop.resize((max(64, round(crop.width * k / 8) * 8), max(64, round(crop.height * k / 8) * 8)), Image.LANCZOS)
        face_refs = [r for r in refs if r["role"] == "face"]
        g = build_graph("detailed close-up of a face, sharp eyes, natural skin texture, " + prompt, NEGATIVE,
                        random.randrange(2 ** 31), src=comfy_file(big, "sparky_face"), denoise=0.38, refs=face_refs)
        fixed = Image.open(comfy_run(g)).convert("RGB").resize(crop.size, Image.LANCZOS)
        m = Image.new("L", crop.size, 0)
        ImageDraw.Draw(m).ellipse((crop.width * 0.08, crop.height * 0.06, crop.width * 0.92, crop.height * 0.94), fill=255)
        from PIL import ImageFilter
        im.paste(fixed, (x1, y1), m.filter(ImageFilter.GaussianBlur(max(4, crop.width // 14))))
        im.save(self.images[iid]["path"])
        self._durls = {k: v for k, v in self._durls.items() if k[0] != iid}

    def qwen_masked_edit(self, iid, src, prompt, refs, m):
        """Painted area + Qwen: Qwen doesn't see the mask, so we give it a close-up of the area
        (with context), it makes the change in the middle, and we paste it back with softened edges."""
        from PIL import ImageFilter
        mk = Image.open(self.masks[iid]).convert("L").resize(src.size)
        bb = mk.point(lambda v: 255 if v > 127 else 0).getbbox()
        if not bb:
            raise RuntimeError("la zone peinte est vide")
        W, H = src.size
        x1, y1, x2, y2 = bb
        side = max(x2 - x1, y2 - y1) * 1.7  # context around the area
        side = max(side, 0.3 * max(W, H))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cx1, cy1 = int(max(0, min(W - side, cx - side / 2))), int(max(0, min(H - side, cy - side / 2)))
        cx2, cy2 = int(min(W, cx1 + side)), int(min(H, cy1 + side))
        crop = src.crop((cx1, cy1, cx2, cy2))
        k = 1024 / max(crop.size)
        big = crop.resize((round(crop.width * k / 32) * 32, round(crop.height * k / 32) * 32), Image.LANCZOS)
        rel = mk.crop((cx1, cy1, cx2, cy2)).getbbox()
        where = ""
        if rel:  # position of the area in the close-up, in words
            rx, ry = (rel[0] + rel[2]) / 2 / crop.width, (rel[1] + rel[3]) / 2 / crop.height
            where = (("top" if ry < 0.35 else "bottom" if ry > 0.65 else "middle") + "-"
                     + ("left" if rx < 0.35 else "right" if rx > 0.65 else "center"))
        note = (f" (This image is a close-up crop of the area to modify: make the change in the {where} part of "
                "image 1, at a scale consistent with the surroundings, and keep the borders of the image unchanged.)")
        g = build_instruct(m, prompt + note, random.randrange(2 ** 31), images=[comfy_file(big, "sparky_zone")] + refs)
        out = Image.open(comfy_run(g)).convert("RGB").resize(crop.size, Image.LANCZOS)
        soft = mk.crop((cx1, cy1, cx2, cy2)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(max(6, crop.width // 40)))
        res = src.copy()
        res.paste(Image.composite(out, crop, soft), (cx1, cy1))
        return self.add_image(res, "gen", prompt, parent=iid)

    def tool_generate(self, a):
        size = SIZES.get(a.get("aspect"), SIZES["square"])
        seed = int(a.get("seed") or random.randrange(2 ** 31))
        m = image_model()
        if m.get("engine") in INSTRUCT:
            g = build_instruct(m, a.get("prompt", ""), seed, size=size, images=self.ref_images(a.get("refs")))
            return self.add_image(Image.open(comfy_run(g)).convert("RGB"), "gen", a.get("prompt", "")), []
        refs, crops = self.prep_refs(a.get("refs"))
        g = build_graph(a.get("prompt", ""), a.get("negative") or NEGATIVE, seed, size=size, refs=refs)
        new = self.add_image(Image.open(comfy_run(g)).convert("RGB"), "gen", a.get("prompt", ""))
        self.face_pass(new, a.get("prompt", ""), refs)
        return new, crops

    def tool_edit(self, a):
        try:
            iid = int(str(a.get("image")).lstrip("#"))
        except (TypeError, ValueError):
            iid = self.last_image()
        if iid not in self.images:
            raise RuntimeError(f"image #{iid} inconnue")
        src = fit_1mp(Image.open(self.images[iid]["path"]).convert("RGB"))
        region = a.get("region", "all")
        m = image_model()
        if m.get("engine") in INSTRUCT:
            refs = self.ref_images(a.get("refs"), exclude=(iid,))
            if region == "mask" and iid in self.masks:
                return self.qwen_masked_edit(iid, src, a.get("prompt", ""), refs, m), []
            # instruction-based edit: image 1 = image to modify, then the references
            names = [comfy_file(src, "sparky_src")] + refs
            g = build_instruct(m, a.get("prompt", ""), random.randrange(2 ** 31), images=names)
            out = Image.open(comfy_run(g)).convert("RGB")
            return self.add_image(out, "gen", a.get("prompt", ""), parent=iid), []
        mask = None
        if region == "mask":
            if iid not in self.masks:
                raise RuntimeError(f"aucune zone peinte sur l'image #{iid}")
            mask = Image.open(self.masks[iid]).convert("L").resize(src.size)
        elif isinstance(region, list) and region:
            items = region if isinstance(region[0], (list, str)) else [region]
            boxes = [self.locate(iid, it) if isinstance(it, str) else it for it in items]
            boxes = [b for b in boxes if b]
            if not boxes:
                raise RuntimeError("zone introuvable dans l'image : " + ", ".join(map(str, items)))
            mask = Image.new("L", src.size, 0)
            d = ImageDraw.Draw(mask)
            for b in boxes:
                x1, y1, x2, y2 = box_px(b, *src.size, pad=0.12)
                d.rounded_rectangle((x1, y1, x2, y2), radius=max(2, min(x2 - x1, y2 - y1) // 5), fill=255)
        refs, crops = self.prep_refs(a.get("refs"))
        den = float(a.get("denoise") or (0.9 if mask else 0.5))
        neg = a.get("negative") or NEGATIVE
        if mask and den >= 0.95 and re.search(r"\b(empty|bare|clean|clear|nothing|plain|uncluttered)\b", a.get("prompt", ""), re.I):
            # removal: the emptied area must stay empty (SDXL likes to put an object back)
            neg += ", object, item, clutter, bottle, glass, wine glass, cup, mug, vase, drink, food, leaves, flowers"
        g = build_graph(a.get("prompt", ""), neg, random.randrange(2 ** 31),
                        src=comfy_file(src, "sparky_src"),
                        mask=comfy_file(mask.convert("RGB"), "sparky_mask") if mask else None,
                        denoise=max(0.05, min(1.0, den)), refs=refs)
        new = self.add_image(Image.open(comfy_run(g)).convert("RGB"), "gen", a.get("prompt", ""), parent=iid)
        return new, crops

    # -- one conversation turn
    def turn(self, text, uploads, mask, mask_for, emit):
        def ev(e):
            e.setdefault("ts", time.time())
            self.events.append(e)
            emit(e)

        with self.lock:
            self._emit, self._ev, self._read = emit, ev, False
            self.lang = (detect_lang(text) or self.lang or detect_lang(text, weak=True)
                         or ("French" if SETTINGS.get("ui_lang") == "fr" else "English"))
            ids = [self.add_image(im, "user") for im in uploads]
            note = ""
            if mask is not None and mask_for in self.images:
                p = os.path.join(self.dir, f"mask_{mask_for}_{len(self.masks)}.png")
                mask.save(p)
                self.masks[mask_for] = p
                note += f"\n[L'utilisateur a peint une zone sur l'image #{mask_for} : pour la modifier, edit avec region \"mask\".]"
            if ids:
                note += "\n[Images jointes : " + ", ".join(f"#{i}" for i in ids) + "]"
            self.history.append({"role": "user", "text": (text or "(images)") + note, "images": ids})
            ev({"type": "user", "text": text, "images": ids, "maskFor": mask_for if mask is not None else None})
            try:
                max_steps, last_sig, nudges, tools_used = int(SETTINGS["max_steps"]), None, 0, False
                past_replies = {e["text"].strip() for e in self.events if e.get("type") == "reply"}
                self._pending, free, touched = [], 0, set()  # free: writing files does not use up the agent's attempts (max 20 files)
                for it in range(max_steps + 1 + 20):
                    step = it - free
                    if step > max_steps:
                        break
                    if step == max_steps:
                        self.history.append({"role": "user", "text": "[Plus d'etape disponible : reponds maintenant avec action reply.]", "images": []})
                    act = self._pending.pop(0) if self._pending else self.ask_llm()
                    self.history.append({"role": "assistant", "text": json.dumps(act, ensure_ascii=False)})
                    for k, v in (act.get("notes") or {}).items():
                        try:
                            if int(str(k).lstrip("#")) in self.images:
                                self.images[int(str(k).lstrip("#"))]["desc"] = str(v)
                        except ValueError:
                            pass
                    name, args = act.get("action"), act.get("args") or {}
                    if act.get("thought"):
                        ev({"type": "thought", "text": act["thought"]})
                    if name == "artifact" and (args.get("path") or args.get("project")) and step < max_steps:
                        try:
                            pid, p, path = self.tool_project_file(args)
                            tools_used = True
                            touched.add(pid)
                            free += 1 if free < 20 else 0
                            ev({"type": "project", "id": pid, "title": p["title"], "path": path, "content": p["files"][path],
                                "entry": p["entry"], "version": p["version"]})
                            self.history.append({"role": "user", "images": [], "text":
                                f"[PROJET] « {p['title']} » (project {pid}) : {path} enregistré. Fichiers : {', '.join(p['files'])} "
                                f"(entry {p['entry']}). S'il manque des fichiers référencés (CSS, JS, modules…), envoie le suivant avec "
                                "le même project. Sinon termine par une courte reply (sans recopier le code)."})
                        except Exception as e:  # incomplete file: the model gets the reason and rewrites it
                            self._emit({"type": "status", "text": "fichier incomplet, nouvel essai"})
                            self.history.append({"role": "user", "images": [], "text": f"[ERREUR PROJET] {e}"})
                        continue
                    if name == "artifact" and step < max_steps:
                        try:
                            aid, art = self.tool_artifact(args)
                            tools_used, n = True, len(art["versions"])
                            free += 1 if free < 20 else 0
                            ev({"type": "artifact", "id": aid, "title": art["title"], "kind": art["kind"],
                                "language": art["language"], "version": n, "content": art["versions"][-1]})
                            # the history keeps a simple mention: the up-to-date content is given at the end of the context
                            self.history.append({"role": "user", "images": [], "text":
                                f"[ARTIFACT] \"{art['title']}\" (id {aid}) v{n} est affiché à l'utilisateur dans le panneau à droite. "
                                "Termine maintenant par une courte reply (sans recopier le code)."})
                        except Exception as e:
                            ev({"type": "error", "text": str(e)})
                            self.history.append({"role": "user", "images": [], "text": f"[ERREUR ARTIFACT] {e}"})
                        continue
                    if name not in ("generate", "edit") or step == max_steps:
                        text = args.get("text") or act.get("text") or act.get("reply")
                        nudge = None
                        if not text:  # empty reply or malformed JSON
                            nudge = "[FORMAT] Your last answer was empty or malformed. Do what the user asked now: generate/edit, or reply with args.text."
                        elif step < max_steps and not tools_used and ANNOUNCE.search(text):
                            nudge = ("[ACTION] You announced an action without doing it. Do it NOW with generate or edit "
                                     "(do not describe it), or ask one precise question if something is truly unclear.")
                        elif step < max_steps and (miss := self.missing_files(touched)):
                            nudge = (f"[PROJET] Ces fichiers sont utilisés mais n'existent pas encore : {', '.join(miss)}. "
                                     "Envoie-les maintenant (même project, contenu complet), puis termine par une reply.")
                        elif text.strip() in past_replies:
                            nudge = "[REPEAT] This exact reply was already sent earlier. Write a new reply that describes THIS result honestly."
                        if nudge and nudges < 2:
                            nudges += 1
                            self.history.append({"role": "user", "images": [], "text": nudge})
                            continue
                        text = self.fix_language(text or act.get("thought") or "...")
                        if "```" in text:
                            short = self.auto_artifacts(text, ev)
                            if short != text:  # the code lives in the artifact: the history keeps no copy of it
                                self.history[-1]["text"] = json.dumps(act | {"args": {"text": short + " [code dans l'artifact, voir ARTIFACTS]"}}, ensure_ascii=False)
                                text = short
                        ev({"type": "reply", "text": text})
                        break
                    sig = json.dumps([name, args], sort_keys=True)
                    if sig == last_sig:
                        self.history.append({"role": "user", "images": [], "text":
                            "[REFUSED] Identical arguments to the previous attempt. Change something concrete, or reply."})
                        continue
                    last_sig = sig
                    ev({"type": "tool", "name": name, "args": args})
                    try:
                        tools_used = True
                        new, crops = (self.tool_generate if name == "generate" else self.tool_edit)(args)
                        ev({"type": "image", "id": new, "refs": crops})
                        left = max_steps - step - 1
                        parent = self.images[new].get("parent")
                        local = name == "edit" and isinstance(args.get("region"), list)
                        obs = (f"[RESULTAT] {name} a produit l'image #{new} (ci-dessous)"
                               + (f", avec les decoupes de reference {', '.join('#%d' % c for c in crops)}" if crops else "")
                               + f". Verifie-la par rapport a la demande{' et aux references' if crops else ''}. "
                               + (f"Compare la zone modifiee avec l'original #{parent} : dans \"thought\", liste ce qui se trouve "
                                  f"MAINTENANT a cet endroit. Un objet qui n'y etait pas (ou qui a juste change de forme) = echec. "
                                  if local and parent else "")
                               + f"Si c'est bon : reply. Sinon corrige et relance ({left} etape(s) restante(s)).")
                        self.history.append({"role": "user", "text": obs, "images": [new] + crops})
                    except Exception as e:
                        ev({"type": "error", "text": str(e)})
                        self.history.append({"role": "user", "text": f"[ERREUR OUTIL] {e}. Corrige les arguments ou explique le probleme a l'utilisateur.", "images": []})
            except Exception as e:
                ev({"type": "error", "text": f"assistant indisponible : {e}"})
            finally:
                ev({"type": "done"})
                self.save()
