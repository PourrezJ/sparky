# Lance Defend Local IA (serveur + ComfyUI + assistant) et ouvre l'interface dans le navigateur.
$Root = $PSScriptRoot
$cfg = @{}
if (Test-Path "$Root\config.json") { (Get-Content "$Root\config.json" -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $cfg[$_.Name] = $_.Value } }
$port = if ($cfg.app_port) { $cfg.app_port } else { 7860 }
$py = if ($cfg.python) { $cfg.python } else { "$Root\runtime\venv\Scripts\python.exe" }
if (-not (Test-Path $py)) {
  Add-Type -AssemblyName PresentationFramework
  [System.Windows.MessageBox]::Show("Python introuvable ($py). Lance d'abord install.ps1.", 'Defend Local IA') | Out-Null
  exit 1
}
function Up { try { (New-Object Net.Sockets.TcpClient('127.0.0.1', $port)).Close(); $true } catch { $false } }
if (-not (Up)) {
  New-Item -ItemType Directory -Force "$Root\data" | Out-Null
  Start-Process $py -ArgumentList "app\server.py", "--port", $port -WorkingDirectory $Root -WindowStyle Hidden `
    -RedirectStandardOutput "$Root\data\server.log" -RedirectStandardError "$Root\data\server.err.log"
  for ($i = 0; $i -lt 60 -and -not (Up); $i++) { Start-Sleep -Milliseconds 500 }
}
Start-Process "http://127.0.0.1:$port"
