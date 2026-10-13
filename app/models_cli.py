"""Download models from the command line (used by install.ps1).

  python app\\models_cli.py --list             catalog and status
  python app\\models_cli.py --recommended      install the models recommended for this machine
  python app\\models_cli.py qwen359 klein4b    install specific models
  add --lang=fr for French output
"""
import sys, time

import catalog
import syscheck

FR = "--lang=fr" in sys.argv
T = (lambda fr, en: fr if FR else en)
GB = T("Go", "GB")


def bar(pc, width=28):
    full = int(pc / 100 * width)
    return "█" * full + "░" * (width - full)


def install(ids):
    for mid in ids:
        m = catalog.BY_ID.get(mid)
        if not m:
            print("   " + T("modele inconnu : ", "unknown model: ") + mid)
            continue
        if catalog.installed(m):
            print(f"   ✓ {m['label']} " + T("(deja installe)", "(already installed)"))
            continue
        catalog.install(mid)
        while catalog.jobs[mid]["state"] == "running":
            j = catalog.jobs[mid]
            pc = j["done"] / j["total"] * 100 if j["total"] else 0
            print(f"\r   {m['label'][:26]:26} {bar(pc)} {pc:5.1f} %  {j['done'] / 2**30:5.1f}/{j['total'] / 2**30:.1f} {GB} ",
                  end="", flush=True)
            time.sleep(0.5)
        j = catalog.jobs[mid]
        ok = j["state"] == "ok"
        mark = "✓" if ok else "✗"
        print(f"\r   {mark} {m['label'][:26]:26} {bar(100 if ok else 0)} "
              + (f"{m['size_gb']:.1f} {GB}" if ok else T("ECHEC : ", "FAILED: ") + j["error"]) + " " * 12)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--lang")]
    if not args or "--list" in args:
        for m in catalog.listing({}):
            print(f"  {m['id']:14} {m['kind']:5} {m['size_gb']:5.1f} {GB}  "
                  f"{T('installe', 'installed') if m['installed'] else '':10}{T('recommande', 'recommended') if m['recommended'] else ''}  {m['label']}")
    elif "--recommended" in args:
        install([i for i in syscheck.current()["recommended"].values() if i])
    else:
        install(args)
