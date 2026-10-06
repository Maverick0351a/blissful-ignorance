$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$serverScript = Join-Path $projectRoot 'server.py'
$runsPath = Join-Path $projectRoot 'runs'
$metadataPath = Join-Path $runsPath 'server-process.json'
$pidPath = Join-Path $runsPath 'server.pid'
if (-not (Test-Path -LiteralPath $metadataPath)) { Write-Host 'No launcher-owned server is recorded.'; return }
$processRecord = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if ([IO.Path]::GetFullPath($processRecord.script) -ne $serverScript) { throw 'The recorded script belongs to another folder. No process was stopped.' }
$worldProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$processRecord.pid)" -ErrorAction SilentlyContinue
if (-not $worldProcess) {
    Remove-Item -LiteralPath $metadataPath -Force
    if (Test-Path -LiteralPath $pidPath) { Remove-Item -LiteralPath $pidPath -Force }
    Write-Host 'The recorded server has already stopped.'
    return
}
$scriptPattern = '(?:^|\s|"|\x27)' + [regex]::Escape($serverScript) + '(?:\s|"|\x27|$)'
if (-not $worldProcess.ExecutablePath -or [IO.Path]::GetFullPath($worldProcess.ExecutablePath) -ne [IO.Path]::GetFullPath($processRecord.executable) -or $worldProcess.CommandLine -notmatch $scriptPattern) {
    throw 'Process ownership could not be verified. No process was stopped.'
}
$stopPath = Join-Path $runsPath "stop-$($processRecord.pid).request"
$stopRequestedAt = [datetime]::UtcNow
Set-Content -LiteralPath $stopPath -Value 'stop' -Encoding ASCII
for ($attempt = 0; $attempt -lt 100; $attempt++) {
    if (-not (Get-Process -Id ([int]$processRecord.pid) -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Milliseconds 100
}
$stillRunning = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$processRecord.pid)" -ErrorAction SilentlyContinue
if ($stillRunning) {
    if (-not $stillRunning.ExecutablePath -or [IO.Path]::GetFullPath($stillRunning.ExecutablePath) -ne [IO.Path]::GetFullPath($processRecord.executable) -or $stillRunning.CommandLine -notmatch $scriptPattern) { throw 'Process identity changed while stopping. No process was forcibly stopped.' }
    Stop-Process -Id ([int]$processRecord.pid) -Force
    Write-Warning 'The owned server required a forced stop. The last periodic autosave remains available; the final shutdown save is unverified.'
} else {
    $autosavePath = Join-Path $runsPath 'autosave.json'
    $checkpointPath = Join-Path $runsPath 'population.pt'
    $receiptPath = Join-Path $runsPath 'checkpoint-status.json'
    $finalSaveVerified = $false
    if ((Test-Path -LiteralPath $checkpointPath) -and (Get-Item -LiteralPath $checkpointPath).LastWriteTimeUtc -ge $stopRequestedAt -and (Test-Path -LiteralPath $receiptPath) -and (Get-Item -LiteralPath $receiptPath).LastWriteTimeUtc -ge $stopRequestedAt) {
        try {
            $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
            $finalSaveVerified = $receipt.checkpoint -eq 'population.pt' -and $receipt.bytes -eq (Get-Item -LiteralPath $checkpointPath).Length -and $null -ne $receipt.tick -and $receipt.residents.Count -gt 0
        } catch { }
    } elseif ((Test-Path -LiteralPath $autosavePath) -and (Get-Item -LiteralPath $autosavePath).LastWriteTimeUtc -ge $stopRequestedAt) {
        try {
            $savedWorld = Get-Content -LiteralPath $autosavePath -Raw | ConvertFrom-Json
            $finalSaveVerified = $savedWorld.schema -eq 1 -and $null -ne $savedWorld.tick
        } catch { }
    }
    if ($finalSaveVerified) { Write-Host 'Blissful Ignorance stopped and saved its latest world and resident controllers.' }
    else { Write-Warning 'Blissful Ignorance stopped. The final shutdown save could not be verified; check runs/server.stderr.log.' }
}
Remove-Item -LiteralPath $metadataPath -Force
if (Test-Path -LiteralPath $pidPath) { Remove-Item -LiteralPath $pidPath -Force }
if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath -Force }
