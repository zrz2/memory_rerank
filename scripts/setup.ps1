$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache/pip'
$env:HF_HOME = Join-Path $projectRoot '.cache/huggingface'
$env:TEMP = Join-Path $projectRoot '.cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    python -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& $pythonPath -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'torch installation failed' }
& $pythonPath -m pip install -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'dependency installation failed' }
Write-Output "Environment ready: $pythonPath"
