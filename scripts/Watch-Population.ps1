param(
    [ValidateRange(1, 50000)][int]$Ticks = 4096,
    [ValidateRange(1, 600)][int]$WallSeconds = 120
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runsPath = Join-Path $projectRoot 'runs'
$evidenceBytes = (Get-ChildItem -LiteralPath $runsPath -Directory -Filter 'population-watch-*' |
    Get-ChildItem -File -Recurse | Measure-Object -Property Length -Sum).Sum
if ($evidenceBytes -ge 4GB) { throw 'Population observation evidence reached 4 GiB. Archive reviewed runs before adding more; nothing was deleted.' }
$pythonExecutable = Join-Path $env:USERPROFILE 'Projects/laya-lab/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'The existing PyTorch environment is missing.' }
$recordPath = Join-Path $projectRoot 'runs/server-process.json'
if (-not (Test-Path -LiteralPath $recordPath)) { throw 'Start the main world before using this continuation wrapper.' }
$record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
if ($record.publicDemo) { throw 'Stop sharing before running a population observation.' }
$receiptPath = Join-Path $projectRoot 'runs/checkpoint-status.json'
$requestedAt = [datetime]::UtcNow
$outputPath = Join-Path $projectRoot ('runs/population-watch-' + [datetime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))
try {
    & (Join-Path $PSScriptRoot 'Stop-World.ps1')
    if ((Test-Path -LiteralPath $recordPath) -or -not (Test-Path -LiteralPath $receiptPath) -or
        (Get-Item -LiteralPath $receiptPath).LastWriteTimeUtc -lt $requestedAt) {
        throw 'Fresh shutdown checkpoint could not be verified; no observation was started.'
    }
    & $pythonExecutable (Join-Path $projectRoot 'experiments/population_watch.py') --output $outputPath --ticks $Ticks --wall-seconds $WallSeconds --continue-live
    if ($LASTEXITCODE -ne 0) { throw 'Observation failed. The saved evidence and original checkpoint are retained.' }
} finally {
    & (Join-Path $PSScriptRoot 'Start-World.ps1') -Port ([int]$record.port) -NoBrowser
}
