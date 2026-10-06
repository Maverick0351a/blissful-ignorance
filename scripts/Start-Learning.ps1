param([string]$Python = (Join-Path $HOME 'Projects/laya-lab/.venv/Scripts/python.exe'),[switch]$Paused)
$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$directory = Join-Path $project 'runs/learning-population'
$script = Join-Path $project 'experiments/living_population.py'
$recordPath = Join-Path $directory 'process.json'
if (Test-Path -LiteralPath $recordPath) {
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    $running = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$record.pid)" -ErrorAction SilentlyContinue
    if ($running -and $running.CommandLine -like '*living_population.py*') {
        Write-Host 'Experimental population is already running: http://127.0.0.1:8789/'
        return
    }
}
if (-not (Test-Path -LiteralPath $Python)) { throw 'Pass -Python with an existing Python environment containing PyTorch. Nothing will be downloaded.' }
New-Item -ItemType Directory -Path $directory -Force | Out-Null
$arguments=@('"'+$script+'"','--directory','"'+$directory+'"','--serve','--port','8789')
if($Paused){$arguments+='--paused'}
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $directory 'stdout.log') -RedirectStandardError (Join-Path $directory 'stderr.log')
for ($i=0; $i -lt 60; $i++) {
    if ($process.HasExited) { throw 'Startup failed. Read runs/learning-population/stderr.log.' }
    try {
        $health = Invoke-RestMethod http://127.0.0.1:8789/health -TimeoutSec 1
        $serving = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$health.pid)" -ErrorAction SilentlyContinue
        if (($health.pid -eq $process.Id -or $serving.ParentProcessId -eq $process.Id) -and $serving.CommandLine.Contains($script)) { Write-Host 'Experimental population ready: http://127.0.0.1:8789/'; return }
    } catch { }
    Start-Sleep -Milliseconds 150
    $process.Refresh()
}
throw 'Startup is not yet verified; inspect the logs before retrying.'
