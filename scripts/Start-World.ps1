param(
    [ValidateRange(1, 65535)][int]$Port = 8788,
    [string]$Python,
    [switch]$NoBrowser,
    [switch]$PublicDemo
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$serverScript = Join-Path $projectRoot 'server.py'
$runsPath = Join-Path $projectRoot 'runs'
$metadataPath = Join-Path $runsPath 'server-process.json'
$pidPath = Join-Path $runsPath 'server.pid'
$worldUrl = "http://127.0.0.1:$Port/"
New-Item -ItemType Directory -Path $runsPath -Force | Out-Null

function Test-OwnedProcess($Record) {
    if (-not $Record -or [IO.Path]::GetFullPath($Record.script) -ne $serverScript) { return $false }
    $worldProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$Record.pid)" -ErrorAction SilentlyContinue
    if (-not $worldProcess -or -not $worldProcess.ExecutablePath) { return $false }
    if ([IO.Path]::GetFullPath($worldProcess.ExecutablePath) -ne [IO.Path]::GetFullPath($Record.executable)) { return $false }
    return $worldProcess.CommandLine -match ('(?:^|\s|"|\x27)' + [regex]::Escape($serverScript) + '(?:\s|"|\x27|$)')
}

if (Test-Path -LiteralPath $metadataPath) {
    $existingRecord = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if (Test-OwnedProcess $existingRecord) {
        if ($existingRecord.port -ne $Port -or [bool]$existingRecord.publicDemo -ne [bool]$PublicDemo) {
            throw 'An owned server is running with different options. Stop it with Stop-World.ps1 before changing its port or sharing mode.'
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$($existingRecord.port)/health" -TimeoutSec 2
        } catch { throw 'An owned server process is still starting or unresponsive. Stop it with Stop-World.ps1 before restarting.' }
        if ($health.service -eq 'godhood-trials' -and $health.pid -eq $existingRecord.pid) {
            if ($health.controllerMode -ne 'independent-learning') { throw 'The running server uses legacy scripted residents. Stop it with Stop-World.ps1, then restart to migrate the saved valley.' }
            $worldUrl = "http://127.0.0.1:$($existingRecord.port)/"
            Write-Host "Blissful Ignorance is already running: $worldUrl"
            if (-not $NoBrowser) { Start-Process $worldUrl }
            return
        }
        throw 'The recorded process and HTTP server do not match. No process was changed.'
    }
}

try {
    $unexpectedHealth = Invoke-RestMethod -Uri ($worldUrl + 'health') -TimeoutSec 1
    throw "Port $Port already has a server. Choose a different -Port or stop its owning process."
} catch {
    if ($_.Exception.Message -like 'Port * already has a server*') { throw }
}

$pythonExecutable = $null
foreach ($candidate in @($Python, (Join-Path $HOME 'Projects/laya-lab/.venv/Scripts/python.exe'), 'py', 'python', 'python3')) {
    if (-not $candidate) { continue }
    if (-not (Get-Command $candidate -ErrorAction SilentlyContinue)) { continue }
    $candidateArgs = @()
    if ($candidate -eq 'py') { $candidateArgs += '-3' }
    $candidateArgs += @('-c', 'import sys, torch; assert sys.version_info >= (3,11), "Python 3.11+ required"; print(sys.executable)')
    try {
        $detectedPath = & $candidate @candidateArgs 2>$null
        if ($LASTEXITCODE -eq 0 -and $detectedPath -and (Test-Path -LiteralPath ([string]$detectedPath))) {
            $pythonExecutable = [IO.Path]::GetFullPath([string]$detectedPath)
            break
        }
    } catch { continue }
}
if (-not $pythonExecutable) { throw 'Independent learners require an existing Python 3.11+ environment with PyTorch. Pass -Python with its executable. Nothing is downloaded and scripted residents are not substituted.' }
$arguments = @('"' + $serverScript + '"', '--port', [string]$Port)
if ($PublicDemo) { $arguments += @('--public-demo', '--host', '0.0.0.0') }
$worldProcess = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runsPath 'server.stdout.log') -RedirectStandardError (Join-Path $runsPath 'server.stderr.log')
$processRecord = @{ pid = $worldProcess.Id; executable = $pythonExecutable; script = $serverScript; port = $Port; publicDemo = [bool]$PublicDemo }
$processRecord | ConvertTo-Json | Set-Content -LiteralPath $metadataPath -Encoding UTF8
[string]$worldProcess.Id | Set-Content -LiteralPath $pidPath -Encoding ASCII
$ready = $false
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    if ($worldProcess.HasExited) { break }
    try {
        $health = Invoke-RestMethod -Uri ($worldUrl + 'health') -TimeoutSec 1
        $servingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$health.pid)" -ErrorAction SilentlyContinue
        if ($health.service -eq 'godhood-trials' -and ($health.pid -eq $worldProcess.Id -or $servingProcess.ParentProcessId -eq $worldProcess.Id) -and $health.publicDemo -eq [bool]$PublicDemo -and $health.controllerMode -eq 'independent-learning' -and $servingProcess.CommandLine.Contains($serverScript)) {
            $processRecord.pid = [int]$health.pid
            $processRecord.executable = $servingProcess.ExecutablePath
            $processRecord | ConvertTo-Json | Set-Content -LiteralPath $metadataPath -Encoding UTF8
            [string]$health.pid | Set-Content -LiteralPath $pidPath -Encoding ASCII
            $ready = $true; break
        }
    } catch { }
    Start-Sleep -Milliseconds 150
    $worldProcess.Refresh()
}
if (-not $ready) { throw "Server startup could not be verified. Read runs/server.stderr.log; use Stop-World.ps1 if the owned process is still running." }
Write-Host "Blissful Ignorance is ready: $worldUrl"
Write-Host "$($health.learners) independent model-controlled residents; starts paused."
if ($PublicDemo) { Write-Host 'Shared demo: LAN viewing enabled; all game controls are read-only.' }
if (-not $NoBrowser) { Start-Process $worldUrl }
