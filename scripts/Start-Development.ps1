param([string]$Python=(Join-Path $HOME 'Projects/laya-lab/.venv/Scripts/python.exe'),
      [ValidateSet('near_food','depletion','hazards','farming','cooperation','scarcity')][string]$Stage='near_food',
      [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$')][string]$RunName='sequence-population',
      [ValidateRange(1024,65535)][int]$Port=8790)
$ErrorActionPreference='Stop'
$project=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$directory=Join-Path (Join-Path $project 'runs') $RunName
$script=Join-Path $project 'experiments/living_population.py'
$recordPath=Join-Path $directory 'process.json'
if(Test-Path -LiteralPath $recordPath){
    $record=Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    $running=Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$record.pid)" -ErrorAction SilentlyContinue
    if($running -and $running.CommandLine.Contains($script) -and $running.CommandLine.Contains($directory)){
        Write-Host "Development world already running: http://127.0.0.1:$($record.port)/"
        return
    }
}
if(-not(Test-Path -LiteralPath $Python)){throw 'Select an existing PyTorch Python environment. Nothing is downloaded.'}
New-Item -ItemType Directory -Path $directory -Force | Out-Null
$process=Start-Process -FilePath $Python -ArgumentList @('"'+$script+'"','--directory','"'+$directory+'"','--backend','recurrent-ppo','--stage',$Stage,'--serve','--paused','--port',"$Port") -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $directory 'stdout.log') -RedirectStandardError (Join-Path $directory 'stderr.log')
for($i=0;$i -lt 60;$i++){
    $process.Refresh()
    if($process.HasExited){throw "Startup failed; read runs/$RunName/stderr.log."}
    try{
        $health=Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 1
        $serving=Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$health.pid)" -ErrorAction SilentlyContinue
        if(($health.pid -eq $process.Id -or $serving.ParentProcessId -eq $process.Id) -and $serving.CommandLine.Contains($script) -and $serving.CommandLine.Contains($directory)){
            Write-Host "Development world ready: http://127.0.0.1:$Port/"
            return
        }
    }catch{}
    Start-Sleep -Milliseconds 150
}
throw 'Startup is not verified; inspect the logs before retrying.'
