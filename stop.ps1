# Stops Sparky: UI, assistant (llama-server) and image engine (ComfyUI).
$Root = $PSScriptRoot
$cfg = @{}
if (Test-Path "$Root\config.json") { (Get-Content "$Root\config.json" -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $cfg[$_.Name] = $_.Value } }
$ports = @(($cfg.app_port, 7860 | Where-Object { $_ })[0], ($cfg.llm_port, 8080 | Where-Object { $_ })[0], ($cfg.comfy_port, 8188 | Where-Object { $_ })[0])
foreach ($p in $ports) {
  Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
  }
}
Write-Host "Sparky stopped."
