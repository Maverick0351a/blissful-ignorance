param([ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$')][string]$RunName='sequence-population')
$ErrorActionPreference='Stop'
$directory=[IO.Path]::GetFullPath((Join-Path (Join-Path $PSScriptRoot '../runs') $RunName))
$record=Get-Content -LiteralPath (Join-Path $directory 'process.json') -Raw | ConvertFrom-Json
$running=Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$record.pid)" -ErrorAction SilentlyContinue
if(-not $running -or $running.CommandLine -notlike '*living_population.py*' -or -not $running.CommandLine.Contains($directory)){throw 'No matching development process.'}
Set-Content -LiteralPath (Join-Path $directory 'STOP') -Value 'Graceful shutdown requested' -Encoding ASCII
Write-Host 'Stop requested; the server saves both individual brains and the world.'
