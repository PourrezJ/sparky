"""Telechargement des modeles en ligne de commande (utilise par install.ps1).

  python app\\models_cli.py --list             catalogue et etat
  python app\\models_cli.py --recommended      installe les modeles recommandes pour cette machine
  python app\\models_cli.py qwen359 juggernaut installe des modeles precis
"""
import sys, time

import catalog
import syscheck


def install(ids):
    for mid in ids:
        m = catalog.BY_ID.get(mid)
        if not m:
            print("  modele inconnu :", mid)
            continue
        if catalog.installed(m):
            print(f"  {m['label']} : deja installe")
            continue
        print(f"  {m['label']} ({m['size_gb']:.1f} Go)")
        catalog.install(mid)
        while catalog.jobs[mid]["state"] == "running":
            j = catalog.jobs[mid]
            pc = j["done"] / j["total"] * 100 if j["total"] else 0
            print(f"\r    {pc:5.1f} %  {j['done'] / 2**30:6.1f} / {j['total'] / 2**30:.1f} Go", end="", flush=True)
            time.sleep(1)
        j = catalog.jobs[mid]
        print("\r    " + ("termine" + " " * 30 if j["state"] == "ok" else "ECHEC : " + j["error"]))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or "--list" in args:
        info = syscheck.current()
        for m in catalog.listing({}):
            print(f"  {m['id']:12} {m['kind']:5} {m['size_gb']:5.1f} Go  {'installe' if m['installed'] else '':9}"
                  f"{'recommande' if m['recommended'] else ''}  {m['label']}")
    elif "--recommended" in args:
        install([i for i in syscheck.current()["recommended"].values() if i])
    else:
        install(args)
