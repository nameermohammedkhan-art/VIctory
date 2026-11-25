# Clean Hyphen launcher for normal play (no scripted presses, no frame capture).
# Keeps VICTORY_INDIRECT_WINDOW and VICTORY_HOIST_CLAMPS unset (defaulting to
# 0 per-draw clamps without dropping character heads or splitting hair/walls).
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Cache = "$env:LOCALAPPDATA\ztrx\cache",
    [string]$Data = "$env:LOCALAPPDATA\ztrx\data",
    [string]$Exe = ""
)
if (-not $Exe) {
    $root = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
    $Exe = Join-Path $root "target\release\victory.exe"
}
foreach ($v in @(
    "VICTORY_KEYS", "VICTORY_CAPTURE", "VICTORY_CAPTURE_WORLD", "VICTORY_CAPTURE_WORLD_EVERY",
    "VICTORY_CAPTURE_FRAMES", "VICTORY_DUMP_TARGETS", "VICTORY_REFUSED_DIR",
    "VICTORY_INDIRECT_WINDOW", "VICTORY_MERGE_GAP", "VICTORY_MERGE_RUNS",
    "VICTORY_LATE_WAIT_LISTS", "VICTORY_CARD_REHASH_RUNS", "VICTORY_JOIN_COPIES",
    "VICTORY_FILL_WAIT", "VICTORY_CARD_FETCH_INFLIGHT", "VICTORY_HOIST_CLAMPS"
)) {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force $Cache, $Data | Out-Null
$env:VICTORY_CACHE_DIR = $Cache
$env:VICTORY_DATA_DIR = $Data
$env:VICTORY_ASYNC_SUBMIT = "1"
$env:VICTORY_QUEUE2 = "1"
$env:VICTORY_COPY_PLAIN = "1"
$env:VICTORY_TEXTURE_NO_WAIT = "1"
$env:VICTORY_LEVELS_ON_DEMAND = "1"
if (-not $env:VICTORY_DXIL) {
    $localDxil = Join-Path $env:USERPROFILE ".local\dxc\bin\x64\dxil.dll"
    if (Test-Path $localDxil) { $env:VICTORY_DXIL = $localDxil }
}
& $Exe run $Game --execute --console series-x 2>&1 | Out-File "$env:TEMP\ztrx.log" -Encoding utf8
Write-Host "Exited with $LASTEXITCODE; log: $env:TEMP\ztrx.log"
