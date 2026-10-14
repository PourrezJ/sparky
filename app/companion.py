"""Sparky Bot: a personal companion that remembers you, learns your style and grows with you.

- every reply is built from: persona + relationship level + style profile + the memories that matter
  right now (hybrid search) + a rolling summary of older conversations + follow-ups to bring up;
- after each exchange, a background pass extracts durable facts about the user (add / update / delete,
  like Mem0) and refines the style profile;
- when idle, the bot "reflects": merges duplicates, updates the summary, writes a journal entry and
  prepares what to ask next time (like OpenAI Dots' background work). Everything stays inspectable.
"""
import contextlib, json, re, threading, time
from datetime import datetime

import harness
import memory

S = memory.Store()
PERSONAS = {
    "warm": ("Warm & supportive", "kind, encouraging and attentive; celebrates small wins, notices how the user feels", "#8c7cf0"),
    "playful": ("Playful & witty", "funny, curious and a little cheeky; loves wordplay and light teasing, never mean", "#f0a35a"),
    "coach": ("Focused coach", "direct, practical and motivating; helps the user move forward on goals and projects", "#3ecf8e"),
    "calm": ("Calm & thoughtful", "calm, reflective and wise; asks good questions and gives perspective", "#5aa9ff"),
}
LEVELS = [(0, "Just met"), (15, "Getting to know you"), (50, "Friends"), (120, "Good friends"), (250, "Close friends")]
CONTEXT_MSGS = 24
# e5 cosines between short memories are compressed (they share the user's name): measured 0.94-0.96 for duplicates,
# 0.91-0.92 for the same fact updated, 0.88-0.90 for unrelated facts
DUP, SAME = 0.94, 0.905
state = {"busy": False, "last_ops": None, "last_activity": 0.0, "reflecting": False}
_lock = threading.Lock()


def profile():
    p = S.get("profile", {})
    p.setdefault("name", "Sparky")
    p.setdefault("persona", "warm")
    p.setdefault("custom", "")
    p.setdefault("memory_on", True)
    return p


def bond():
    b = S.get("bond", {"xp": 0, "days": [], "first": time.time(), "chats": 0})
    lvl = max(i for i, (xp, _) in enumerate(LEVELS) if b["xp"] >= xp)
    nxt = LEVELS[lvl + 1][0] if lvl + 1 < len(LEVELS) else None
    return b | {"level": lvl, "label": LEVELS[lvl][1], "next": nxt, "floor": LEVELS[lvl][0]}


def _gain(points):
    b = S.get("bond", {"xp": 0, "days": [], "first": time.time(), "chats": 0})
    day = time.strftime("%Y-%m-%d")
    if day not in b["days"]:
        b["days"] = (b["days"] + [day])[-400:]
        points += 3  # coming back on a new day matters more than chatting a lot
    b["xp"] += points
    b["chats"] = b.get("chats", 0) + 1
    S.set("bond", b)


def llm(messages, json_mode=False, max_tokens=900, temperature=0.7, stream=None):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    out, last = "", 0.0
    with harness.busy("llm"), contextlib.closing(harness.chat_deltas(body)) as deltas:  # local or cloud model
        for piece, _ in deltas:
            out += piece
            if stream and time.time() - last > 0.05:
                last = time.time()
                stream(re.sub(r"<think>.*?(</think>|$)", "", out, flags=re.S).lstrip())
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    if json_mode:
        m = re.search(r"\{.*\}", out, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except ValueError:
            return {}
    return out


def system_prompt(query):
    p, b = profile(), bond()
    title, traits, _ = PERSONAS.get(p["persona"], PERSONAS["warm"])
    mems = S.search(query, k=14) if p["memory_on"] else []
    style = S.get("style", {})
    summary = S.get("summary", "")
    checkin = S.get("checkin", "")
    days = max(1, int((time.time() - b.get("first", time.time())) / 86400) + 1)
    lines = [f'You are {p["name"]}, the user\'s personal AI companion. Personality: {traits}.'
             + (f' Extra personality notes from the user: {p["custom"]}.' if p["custom"] else ""),
             "You run 100% locally on the user's own PC. You remember them across conversations and get to know them over time.",
             f'Relationship stage: "{b["label"]}" ({b.get("chats", 0)} exchanges over {days} day(s)). The closer you are, the more '
             "personal, relaxed and familiar you can be (inside jokes, references to past talks). Stay a friend: never clingy, "
             "never romantic or sexual, never guilt-tripping.",
             f'Now: {datetime.now().strftime("%A %d %B %Y, %H:%M")}.']
    if mems:
        lines.append("What you know about the user (use it naturally when relevant; do not recite it; never invent new facts):")
        lines += [f"- {m['text']}" for m in mems]
    else:
        lines.append("You don't know much about the user yet: be curious, learn their name and what matters to them, one question at a time.")
    if style:
        lines.append("How the user likes to be talked to: " + "; ".join(f"{k}: {v}" for k, v in style.items() if v))
    if summary:
        lines.append("Summary of your earlier conversations: " + summary)
    if checkin:
        lines.append("Something you planned to follow up on (bring it up only if it fits): " + checkin)
    lines.append("Write like a friend texting: natural, short paragraphs, no headings, at most one question per message. "
                 f"Reply in {user_lang()} (the user's language). If the user asks what you remember, be transparent and "
                 "remind them they can see and edit everything in the Memory tab.")
    return "\n".join(lines), mems


def user_lang():
    """Language the user actually writes in (last messages), else the interface language."""
    recent = " ".join(m["text"] for m in S.messages(12) if m["role"] == "user")[-1500:]
    return harness.detect_lang(recent) or harness.detect_lang(recent, weak=True) or \
        ("French" if harness.SETTINGS.get("ui_lang") == "fr" else "English")


def history():
    return [{"role": m["role"], "content": m["text"]} for m in S.messages(CONTEXT_MSGS)]


def chat(text, emit):
    """One exchange: stream the reply, then learn from it in the background."""
    with _lock:
        state["busy"] = True
        try:
            S.add_message("user", text)
            sysp, mems = system_prompt(text)
            emit({"type": "context", "memories": mems})
            reply = llm([{"role": "system", "content": sysp}] + history(), max_tokens=900,
                        stream=lambda t: emit({"type": "token", "text": t}))
            mid = S.add_message("assistant", reply)
            _gain(1)
            emit({"type": "reply", "text": reply, "id": mid, "bond": bond()})
        finally:
            state["busy"] = False
            state["last_activity"] = time.time()
    if profile()["memory_on"]:
        threading.Thread(target=learn, args=(text, reply), daemon=True).start()
    return reply


EXTRACT = """You maintain the long-term memory of a personal AI companion. Read the latest exchange and decide what to remember about the USER.
Today is {today}.
Rules:
- Keep only durable, useful facts the USER stated about themselves: identity (name, age, job, city), preferences and dislikes, projects, people and pets in their life, goals, important events, health or diet constraints, how they like to be talked to.
- NEVER store anything that only the Assistant said or assumed. Ignore chit-chat, compliments, the assistant's suggestions, and temporary states ("I'm tired today" -> at most a mood).
- Write each memory as one short third-person sentence IN {lang}, with the user's name if known (e.g. "Anis has a cat named Pixel.").
- Dates: turn relative dates into absolute ones using today's date ("Tuesday" -> "Tuesday 20 October 2026"). When an event from memory has now happened, UPDATE it with the outcome (e.g. "Anis met an investor on 20 October 2026; it went well and a second meeting is planned.").
- ONE fact per memory. Never merge several facts into one memory, never rewrite a memory to add unrelated facts.
- If the new information changes, completes or contradicts an existing memory about the SAME fact, UPDATE that memory (use its id). Leave all other memories alone. Never add something already covered by an existing memory.
- If the user asks you to forget something, DELETE the matching memory ids.
- importance: 5 = core identity/strong constraint, 3 = useful, 1 = minor.
- style: when the user shows or asks how they want to be talked to, also fill style keys (tone, length, language, formality, humor, emojis, notes) with short values. {} if nothing new.
Return ONLY JSON: {"add":[{"text":"...","category":"identity|preference|project|people|goal|event|style|other","importance":3}],
"update":[{"id":0,"text":"...","importance":3}],"delete":[0],"style":{},"mood":"one word or empty"}"""


def learn(user_text, reply):
    try:
        related = S.search(user_text, k=10)
        mem_txt = "\n".join(f'[{m["id"]}] ({m["category"]}) {m["text"]}' for m in related) or "(none)"
        sysp = EXTRACT.replace("{today}", datetime.now().strftime("%A %d %B %Y")).replace("{lang}", user_lang())
        ops = llm([{"role": "system", "content": sysp},
                   {"role": "user", "content": f"EXISTING RELATED MEMORIES:\n{mem_txt}\n\nCURRENT STYLE PROFILE: "
                    f"{json.dumps(S.get('style', {}), ensure_ascii=False)}\n\nLATEST EXCHANGE\nUser: {user_text}\nAssistant: {reply}"}],
                  json_mode=True, max_tokens=700, temperature=0.2)
        done = {"add": [], "update": [], "delete": [], "style": {}, "mood": ops.get("mood") or ""}
        mems = {m["id"]: m for m in S.memories()}

        uw, rw = memory.words(user_text), memory.words(reply)

        def from_user(text):
            """Provenance check: a new memory must rest on the user's words, not only on the assistant's reply."""
            mw = memory.words(text) - memory.words(profile()["name"])
            in_user, in_reply = len(mw & uw), len(mw & rw - uw)
            return in_user >= 1 and in_user >= in_reply

        def add(text, category, importance):
            if not from_user(text):
                return
            near = S.nearest(text)
            if near and near[1] >= DUP:  # already known (same fact, other words): nothing to add
                return
            i = S.add_memory(text, category, importance)
            if i:
                done["add"].append({"id": i, "text": text})

        for a in ops.get("add") or []:
            if isinstance(a, dict) and a.get("text"):
                add(a["text"], a.get("category", "other"), int(a.get("importance") or 3))
        for u in ops.get("update") or []:
            if not (isinstance(u, dict) and u.get("id") in mems and u.get("text")):
                continue
            old, new = mems[u["id"]]["text"], u["text"].strip()
            if new == old:
                continue
            # guard-rails: an update must stay ONE fact about the SAME thing, otherwise the old memory is kept
            if len(new) > 2 * len(old) + 80 or new.count(". ") >= 2:
                continue  # several facts merged into one blob: refused
            if S.similarity(old, new) < SAME:
                add(new, mems[u["id"]]["category"], int(u.get("importance") or mems[u["id"]]["importance"]))
                continue  # different fact: stored as a new memory instead of overwriting
            S.update_memory(u["id"], new, importance=u.get("importance"))
            done["update"].append({"id": u["id"], "text": new})
        known = set(mems)
        for d in ops.get("delete") or []:
            if d in known:
                S.delete_memory(d)
                done["delete"].append(d)
        if isinstance(ops.get("style"), dict) and ops["style"]:
            st = S.get("style", {})
            st.update({k: str(v) for k, v in ops["style"].items() if v})
            S.set("style", st)
            done["style"] = ops["style"]
        if done["mood"]:
            S.set("mood", done["mood"])
        state["last_ops"] = done | {"ts": time.time()}
    except Exception as e:
        state["last_ops"] = {"error": str(e), "ts": time.time()}


REFLECT = """You are the reflective side of a personal AI companion named {name}. Between conversations you look back on what happened.
Today is {today}. Write the summary, the journal and the follow-up in {lang}.
Merge memories that say the same thing (keep the most complete, precise and recent version), delete memories that are not about
the user or that only repeat something the assistant said, and keep past events phrased in the past.
Given the recent messages, your previous summary and ALL current memories about the user, return ONLY JSON:
{{"summary":"updated summary of your relationship and conversations so far (max 6 sentences, factual)",
"journal":"a short diary entry in first person as {name} (2-3 sentences, warm, in the user's language) about today's conversations",
"checkin":"one thing to follow up on next time (e.g. ask how the job interview went), or empty",
"merge":[{{"ids":[1,2],"text":"one merged memory"}}],"delete":[ids of memories that are wrong, trivial or duplicated]}}"""


def reflect(force=False):
    """Background consolidation. Runs after a pause in the conversation (or on demand)."""
    if state["reflecting"] or state["busy"]:
        return False
    since = S.get("reflected_upto", 0)
    msgs = S.messages(80, after=since)
    if not msgs or (len(msgs) < 4 and not force):
        return False
    state["reflecting"] = True
    try:
        p = profile()
        mems = S.memories()
        r = llm([{"role": "system", "content": REFLECT.format(name=p["name"], today=datetime.now().strftime("%A %d %B %Y"), lang=user_lang())},
                 {"role": "user", "content": "PREVIOUS SUMMARY: " + (S.get("summary", "") or "(none)")
                  + "\n\nMEMORIES:\n" + "\n".join(f'[{m["id"]}] {m["text"]}' for m in mems[:120])
                  + "\n\nRECENT MESSAGES:\n" + "\n".join(f'{m["role"]}: {m["text"][:600]}' for m in msgs)}],
                json_mode=True, max_tokens=1200, temperature=0.4)
        known = {m["id"] for m in mems}
        texts = {m["id"]: m["text"] for m in mems}
        for g in r.get("merge") or []:
            ids = [i for i in g.get("ids", []) if i in known]
            # only merge memories that really talk about the same thing, into one fact
            ids = [ids[0]] + [i for i in ids[1:] if S.similarity(texts[ids[0]], texts[i]) >= SAME] if ids else []
            if len(ids) >= 2 and g.get("text") and g["text"].count(". ") < 2:
                S.update_memory(ids[0], g["text"])
                for i in ids[1:]:
                    S.delete_memory(i)
                    known.discard(i)
        for i in r.get("delete") or []:
            if i in known:
                S.delete_memory(i)
        if r.get("summary"):
            S.set("summary", r["summary"])
        if r.get("journal"):
            S.add_journal(r["journal"])
        S.set("checkin", r.get("checkin") or "")
        S.set("reflected_upto", msgs[-1]["id"])
        S.set("last_reflection", time.time())
        return True
    finally:
        state["reflecting"] = False


def greeting(emit=None):
    """When the user comes back after a while, the companion opens the conversation (using what it knows)."""
    last = S.messages(1)
    if last and time.time() - last[-1]["ts"] < 3 * 3600:
        return None
    sysp, _ = system_prompt("greeting how are you news follow up")
    first = not last
    ask = ("This is your very first conversation: introduce yourself in 2 short sentences (your name, that you run on their PC, "
           "remember what matters to them and get to know them over time) and ask their name. Use the user's interface language: {lang}."
           if first else "The user just came back after a while. Greet them warmly and personally in 1-2 short sentences, "
           "using what you know (and the follow-up topic if there is one). Write in {lang}.")
    lang = user_lang()
    text = llm([{"role": "system", "content": sysp}, {"role": "user", "content": "[" + ask.format(lang=lang) + "]"}],
               max_tokens=200, stream=(lambda t: emit({"type": "token", "text": t})) if emit else None)
    S.add_message("assistant", text)
    if not first:
        S.set("checkin", "")
    return text


def idle_loop():
    """Reflects ~90 s after the last exchange, at most every 10 minutes."""
    while True:
        time.sleep(20)
        try:
            if time.time() - state["last_activity"] > 90 and time.time() - S.get("last_reflection", 0) > 600:
                reflect()
            S.reindex()  # vectors for memories saved before the embedding model was there
        except Exception as e:
            print("companion reflect:", e)


def snapshot():
    p, b = profile(), bond()
    return {"profile": p, "persona": PERSONAS.get(p["persona"], PERSONAS["warm"]), "personas": {k: v[0] for k, v in PERSONAS.items()},
            "bond": b, "style": S.get("style", {}), "mood": S.get("mood", ""), "memories": S.memories(),
            "journal": S.journal(), "summary": S.get("summary", ""), "checkin": S.get("checkin", ""),
            "messages": S.messages(60), "busy": state["busy"], "reflecting": state["reflecting"],
            "last_ops": state["last_ops"], "embeddings": memory.EMB.ready()}


threading.Thread(target=idle_loop, daemon=True).start()
