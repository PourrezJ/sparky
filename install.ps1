# Defend Local IA - installation (Windows 10/11, x64 ou ARM64)
#
#   Depuis le dossier du projet :  powershell -ExecutionPolicy Bypass -File install.ps1
#   En une ligne :                 irm https://raw.githubusercontent.com/anisayari/defend-local-ia/main/install.ps1 | iex
#
# Options (variables d'environnement pour l'installation en une ligne) :
#   -Dir <dossier>   / $env:DLIA_DIR      dossier d'installation (defaut : %LOCALAPPDATA%\DefendLocalIA)
#   -NoModels        / $env:DLIA_NOMODELS ne telecharge pas les modeles recommandes
#   -Yes             / $env:DLIA_YES      repond oui a toutes les questions
#   -NoLaunch                             ne lance pas l'application a la fin
param([string]$Dir = "", [switch]$NoModels, [switch]$Yes, [switch]$NoLaunch)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest est 10x plus lent avec la barre de progression
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Repo = 'https://github.com/anisayari/defend-local-ia'
if (-not $Dir) { $Dir = if ($env:DLIA_DIR) { $env:DLIA_DIR } else { Join-Path $env:LOCALAPPDATA 'DefendLocalIA' } }
if ($env:DLIA_NOMODELS) { $NoModels = $true }
if ($env:DLIA_YES) { $Yes = $true }

function Step($t) { Write-Host ""; Write-Host "==> $t" -ForegroundColor Cyan }
function Info($t) { Write-Host "    $t" }
function Warn($t) { Write-Host "    ! $t" -ForegroundColor Yellow }
function Ask($q) {
  if ($Yes) { return $true }
  $r = Read-Host "    $q (O/n)"
  return ($r -eq '' -or $r -match '^(o|oui|y|yes)$')
}
function Download($url, $out) {
  for ($i = 0; $i -lt 5; $i++) {
    try { Invoke-WebRequest $url -OutFile $out -UseBasicParsing; return } catch { Start-Sleep 3 }
  }
  throw "Telechargement impossible : $url"
}

Write-Host ""
Write-Host "  Defend Local IA" -ForegroundColor Magenta
Write-Host "  Assistant IA prive, 100 % local" -ForegroundColor DarkGray

# ------------------------------------------------------------------ 1. code du projet
$here = if ($PSScriptRoot) { $PSScriptRoot } else { '' }
if ($here -and (Test-Path (Join-Path $here 'app\server.py'))) {
  $Root = $here
} else {
  Step "Telechargement de Defend Local IA dans $Dir"
  if (Test-Path (Join-Path $Dir 'app\server.py')) {
    if (Get-Command git -ErrorAction SilentlyContinue) { git -C $Dir pull --ff-only | Out-Null; Info "mis a jour" }
  } elseif (Get-Command git -ErrorAction SilentlyContinue) {
    git clone --depth 1 "$Repo.git" $Dir
  } else {
    $zip = Join-Path $env:TEMP 'dlia.zip'
    Download "$Repo/archive/refs/heads/main.zip" $zip
    $tmp = Join-Path $env:TEMP 'dlia_src'
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive $zip $tmp -Force
    New-Item -ItemType Directory -Force $Dir | Out-Null
    Copy-Item (Join-Path (Get-ChildItem $tmp -Directory | Select-Object -First 1).FullName '*') $Dir -Recurse -Force
  }
  $Root = $Dir
}
$Runtime = Join-Path $Root 'runtime'
New-Item -ItemType Directory -Force $Runtime, (Join-Path $Root 'data') | Out-Null

# ------------------------------------------------------------------ 2. machine
Step "Analyse rapide de la machine"
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') { 'arm64' } else { 'x64' }
$nvidia = $null
try {
  $line = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  if ($line) { $p = $line.Split(','); $nvidia = @{ name = $p[0].Trim(); vram = [int]$p[1].Trim(); driver = [double]($p[2].Trim().Split('.')[0]) } }
} catch {}
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Info "Architecture : $arch   Memoire : $ram Go"
if ($nvidia) { Info ("GPU NVIDIA   : {0} ({1:N0} Go)" -f $nvidia.name, ($nvidia.vram / 1024)) }
else { Warn "Pas de GPU NVIDIA detecte : l'assistant tournera sur le processeur (plus lent) et la generation d'images est deconseillee." }
$free = [math]::Round((Get-PSDrive ($Root.Substring(0,1))).Free / 1GB)
Info "Espace libre : $free Go"
if ($free -lt 25) { Warn "Moins de 25 Go libres : il faut environ 8 Go pour les outils + 4 a 40 Go par modele." }

# ------------------------------------------------------------------ 3. Python
Step "Python"
$py = $null
foreach ($c in @(@('py', '-3.12'), @('py', '-3.11'), @('python'))) {
  try {
    $v = & $c[0] $c[1..9] -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
    if ($v -match '^3\.(1[0-3])$') { $py = $c; break }
  } catch {}
}
if (-not $py) {
  Info "Python 3.12 absent : installation via winget"
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')
  $py = @('py', '-3.12')
}
$venv = Join-Path $Runtime 'venv'
$vpy = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $vpy)) { & $py[0] $py[1..9] -m venv $venv }
& $vpy -m pip install --upgrade pip --quiet
Info ("Python : " + (& $vpy --version))

# ------------------------------------------------------------------ 4. PyTorch
Step "PyTorch (calcul GPU pour les images)"
$hasTorch = (& $vpy -c "import importlib.util;print(importlib.util.find_spec('torch') is not None)") -eq 'True'
if (-not $hasTorch) {
  if ($nvidia -and $arch -eq 'arm64') {
    & $vpy -m pip install --pre --only-binary=:all: --extra-index-url https://pypi.nvidia.com/nvtorch_oot_nightly torch torchvision torchaudio
  } elseif ($nvidia) {
    & $vpy -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  } else {
    & $vpy -m pip install torch torchvision torchaudio
  }
}
& $vpy -c "import torch;print('    torch', torch.__version__, '| GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'non')"

# ------------------------------------------------------------------ 5. ComfyUI (generation d'images)
Step "ComfyUI (moteur d'images)"
$comfy = Join-Path $Runtime 'ComfyUI'
function GetRepo($url, $dest, $branch) {
  if (Test-Path (Join-Path $dest '.git')) { git -C $dest pull --ff-only | Out-Null; return }
  if (Test-Path $dest) { return }
  if (Get-Command git -ErrorAction SilentlyContinue) { git clone --depth 1 $url $dest; return }
  $zip = Join-Path $env:TEMP ((Split-Path $dest -Leaf) + '.zip')
  Download "$url/archive/refs/heads/$branch.zip" $zip
  $tmp = Join-Path $env:TEMP ('x_' + (Split-Path $dest -Leaf))
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  Expand-Archive $zip $tmp -Force
  Move-Item (Get-ChildItem $tmp -Directory | Select-Object -First 1).FullName $dest
}
GetRepo 'https://github.com/comfyanonymous/ComfyUI' $comfy 'master'
GetRepo 'https://github.com/cubiq/ComfyUI_IPAdapter_plus' (Join-Path $comfy 'custom_nodes\ComfyUI_IPAdapter_plus') 'main'
& $vpy -m pip install -r (Join-Path $comfy 'requirements.txt') --quiet
& $vpy -m pip install psutil pillow --quiet
Info "ok"

# ------------------------------------------------------------------ 6. llama.cpp (assistant)
Step "llama.cpp (moteur de l'assistant)"
$llama = Join-Path $Runtime 'llama'
if (-not (Test-Path (Join-Path $llama 'llama-server.exe'))) {
  $rels = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15' -Headers @{ 'User-Agent' = 'defend-local-ia' }
  # CUDA 13 demande un pilote NVIDIA recent (>= 580) ; sinon CUDA 12.4 (x64) ; sans NVIDIA : version processeur
  $want = @()
  if ($nvidia -and $nvidia.driver -ge 580) { $want += "bin-win-cuda-13\.[\d.]+-$arch\.zip$" }
  if ($nvidia) { $want += "bin-win-cuda-12\.[\d.]+-$arch\.zip$" }
  $want += "bin-win-cpu-$arch\.zip$"
  $pick = $null
  foreach ($w in $want) {
    foreach ($r in $rels) {
      $a = $r.assets | Where-Object { $_.name -match "^llama-.*$w" } | Select-Object -First 1
      if ($a) { $pick = @{ rel = $r; asset = $a }; break }
    }
    if ($pick) { break }
  }
  if (-not $pick) { throw "Aucune version de llama.cpp trouvee pour Windows $arch" }
  Info ("version " + $pick.rel.tag_name + " : " + $pick.asset.name)
  $zip = Join-Path $env:TEMP 'llama.zip'
  Download $pick.asset.browser_download_url $zip
  New-Item -ItemType Directory -Force $llama | Out-Null
  Expand-Archive $zip $llama -Force
  if ($pick.asset.name -match 'cuda-([\d.]+)-') {   # bibliotheques CUDA fournies a part
    $cv = $Matches[1]
    $rt = $pick.rel.assets | Where-Object { $_.name -eq "cudart-llama-bin-win-cuda-$cv-$arch.zip" } | Select-Object -First 1
    if ($rt) { Download $rt.browser_download_url $zip; Expand-Archive $zip $llama -Force }
  }
  # certaines archives ont un sous-dossier : on remonte les fichiers
  $exe = Get-ChildItem $llama -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
  if ($exe -and $exe.DirectoryName -ne $llama) { Get-ChildItem $exe.DirectoryName | Move-Item -Destination $llama -Force }
}
Info "ok"

# ------------------------------------------------------------------ 7. analyse complete + modeles
Step "Analyse du systeme et recommandations"
Push-Location $Root
& $vpy app\syscheck.py
if (-not $NoModels) {
  if (Ask "Telecharger maintenant les modeles recommandes ?") { & $vpy app\models_cli.py --recommended }
  else { Info "Tu pourras les installer depuis l'interface (Parametres > Bibliotheque de modeles)." }
}
Pop-Location

# ------------------------------------------------------------------ 8. raccourcis
Step "Raccourcis"
$ws = New-Object -ComObject WScript.Shell
foreach ($folder in @([Environment]::GetFolderPath('Desktop'), (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'))) {
  $lnk = $ws.CreateShortcut((Join-Path $folder 'Defend Local IA.lnk'))
  $lnk.TargetPath = 'powershell.exe'
  $lnk.Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Root\start.ps1`""
  $lnk.WorkingDirectory = $Root
  $lnk.IconLocation = "$Root\app\logo.ico"
  $lnk.Description = 'Defend Local IA - assistant IA prive, 100 % local'
  $lnk.Save()
}
Info "Bureau et menu Demarrer : Defend Local IA"

Write-Host ""
Write-Host "  Installation terminee." -ForegroundColor Green
Write-Host "  Lancer : double-clic sur 'Defend Local IA' (ou start.bat dans $Root)"
Write-Host ""
if (-not $NoLaunch -and (Ask "Lancer Defend Local IA maintenant ?")) { & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'start.ps1') }
