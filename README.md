<p align="center"><img src="app/logo.png" width="96" alt="Defend Local IA"></p>

<h1 align="center">Defend Local IA</h1>

<p align="center"><b>Ton assistant IA privé, 100 % local.</b><br>
Discute, analyse des images, génère et retouche des visuels — sans qu'aucune donnée ne quitte ta machine.</p>

---

## Ce que ça fait

- **Chat comme ChatGPT**, en local, avec historique des conversations et **conversations éphémères** (rien n'est enregistré).
- **Vision** : envoie des photos, captures ou documents scannés, l'assistant les voit et les analyse.
- **Création et retouche d'images** par simple demande ; l'agent vérifie lui-même le résultat et corrige si besoin.
- **Niveau de réflexion réglable** (désactivée, légère, normale, approfondie) directement depuis le menu du modèle.
- **Analyse du système** à l'installation, relançable dans *Paramètres > Système* : détecte GPU, mémoire et disque, puis **recommande les modèles adaptés** à ta machine.
- **Bibliothèque de modèles** : installation en un clic, reprise automatique des téléchargements, indicateurs de vitesse, qualité et mémoire *pour ta machine*.
- **Que des modèles officiels** (Qwen, Gemma, gpt-oss, Nemotron, Qwen-Image, SDXL…). Les variantes « uncensored », « abliterated » ou NSFW sont exclues du catalogue et de la recherche Hugging Face.

## Installation (Windows 10/11)

**En une ligne**, dans PowerShell :

```powershell
irm https://raw.githubusercontent.com/anisayari/defend-local-ia/main/install.ps1 | iex
```

**Ou** télécharge le projet (bouton *Code > Download ZIP*), décompresse-le et double-clique sur `install.bat`.

L'installeur :

1. analyse la machine (GPU NVIDIA, mémoire, disque) ;
2. installe Python 3.12 si besoin (via winget), puis PyTorch adapté au GPU ;
3. installe le moteur d'images [ComfyUI](https://github.com/comfyanonymous/ComfyUI) et le moteur de l'assistant [llama.cpp](https://github.com/ggml-org/llama.cpp) (version CUDA si GPU NVIDIA, x64 ou ARM64) ;
4. affiche les modèles recommandés et propose de les télécharger ;
5. crée un raccourci **Defend Local IA** sur le Bureau et dans le menu Démarrer.

Ensuite : double-clic sur le raccourci (ou `start.bat`). L'interface s'ouvre sur <http://127.0.0.1:7860>. Pour tout arrêter : `stop.bat`.

### Configuration conseillée

| Machine | Ce qui tourne bien |
|---|---|
| GPU NVIDIA 24 Go+ ou mémoire unifiée 32 Go+ | Assistants 27–35B, Qwen-Image 2.1 |
| GPU NVIDIA 12–16 Go | Assistants 9–20B, images SDXL |
| GPU NVIDIA 6–8 Go | Assistants 4–9B, images SDXL (plus lent) |
| Sans GPU NVIDIA | Petits assistants (4B) sur le processeur, pas de génération d'images |

L'analyse du système choisit automatiquement parmi ces options.

## Modèles du catalogue

| Assistant | Taille | Vision | Réflexion |
|---|---|---|---|
| Qwen3.8 27B | 16 Go | ✓ | ✓ |
| Gemma 4 26B A4B | 17 Go | ✓ | ✓ |
| Qwen3.5 35B A3B | 21 Go | ✓ | ✓ |
| Nemotron 3.5 Lightning 30B | 24 Go | | ✓ |
| gpt-oss 20B | 11 Go | | ✓ |
| Gemma 4 12B | 7 Go | ✓ | ✓ |
| Qwen3.5 9B | 6 Go | ✓ | ✓ |
| Gemma 4 E4B | 6 Go | ✓ | ✓ |
| Qwen3.5 4B | 3 Go | ✓ | ✓ |

| Images | Taille | Style |
|---|---|---|
| Qwen-Image 2.1 | 16 Go | photo, texte, retouche par instruction |
| Juggernaut XL v9 | 10 Go* | photo (SDXL) |
| Animagine XL 4.0 | 10 Go* | illustration (SDXL) |

\* inclut les fichiers IP-Adapter (≈ 4 Go, partagés entre les modèles SDXL) qui permettent de réutiliser un personnage ou un style.

Chaque modèle reste soumis à sa propre licence (Apache 2.0, Gemma Terms of Use, NVIDIA Open Model License, etc.) : vérifie-la sur sa page Hugging Face avant un usage commercial.

## Installation existante (ComfyUI / llama.cpp déjà présents)

Crée un fichier `config.json` à la racine du projet pour réutiliser tes dossiers :

```json
{
  "python": "D:\\ai\\venv\\Scripts\\python.exe",
  "comfy_dir": "D:\\ai\\ComfyUI",
  "llama_server": "D:\\ai\\llama\\llama-server.exe",
  "llm_dir": "D:\\ai\\models",
  "mmproj_dir": "D:\\ai\\mmproj"
}
```

Autres clés possibles : `data_dir`, `app_port` (7860), `llm_port` (8080), `comfy_port` (8188).

## En ligne de commande

```powershell
runtime\venv\Scripts\python.exe app\syscheck.py                  # analyse du système
runtime\venv\Scripts\python.exe app\models_cli.py --list         # catalogue
runtime\venv\Scripts\python.exe app\models_cli.py --recommended  # installe les modèles conseillés
runtime\venv\Scripts\python.exe app\models_cli.py qwen359        # installe un modèle précis
```

## Mise à jour / désinstallation

- **Mettre à jour** : relance la commande d'installation (ou `git pull` dans le dossier).
- **Désinstaller** : `stop.bat`, puis supprime le dossier d'installation (`%LOCALAPPDATA%\DefendLocalIA` par défaut) et les raccourcis.

## Vie privée

Tout s'exécute sur ta machine : les conversations, les images et les modèles restent dans le dossier d'installation (`data\`, `models\`). Les seules connexions sortantes servent à télécharger les outils et les modèles (GitHub, PyPI, Hugging Face), et à la recherche de modèles dans la bibliothèque.

## Structure

```
app/
  server.py     serveur web + gestion des modèles
  harness.py    agent : boucle assistant <-> outils d'image
  catalog.py    catalogue, téléchargements, indicateurs
  syscheck.py   analyse du système et recommandations
  index.html    interface
install.ps1     installation
start.ps1 / stop.ps1
```

## Licence

Code sous licence [MIT](LICENSE). ComfyUI (GPL-3.0) et llama.cpp (MIT) sont téléchargés séparément à l'installation.
