$ErrorActionPreference='Stop'
$directory=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../runs/learning-population'))
$record=Get-Content -LiteralPath (Join-Path $directory 'process.json') -Raw | ConvertFrom-Json
$running=Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$record.pid)" -ErrorAction SilentlyContinue
if (-not $running -or $running.CommandLine -notlike '*living_population.py*') { throw 'No matching experimental process; nothing was stopped.' }
Set-Content -LiteralPath (Join-Path $directory 'STOP') -Value 'Graceful shutdown requested' -Encoding ASCII
Write-Host 'Stop requested. The experimental server saves both brains and the world before exiting.'
