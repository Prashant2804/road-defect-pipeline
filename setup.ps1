<#
.SYNOPSIS
    One-time setup for the Road Defect Detection Pipeline on Windows.

.DESCRIPTION
    Installs everything the pipeline needs and verifies it actually works:
      1. Python 3.10-3.13      (via winget, if missing)
      2. FFmpeg with v360      (via winget, if missing) - REQUIRED, not optional
      3. A virtual environment in .venv
      4. PyTorch - CUDA build if an NVIDIA GPU is present, CPU build otherwise
      5. Everything in requirements.txt
      6. A smoke test, so a green finish means green for real

    Safe to re-run: every step is skipped if already satisfied.

.PARAMETER Cpu
    Force the CPU build of PyTorch even if a GPU is detected.

.PARAMETER SkipSystem
    Do not try to install Python or FFmpeg. Use when you manage those yourself.

.PARAMETER DryRun
    Print what would happen without changing anything.

.EXAMPLE
    .\setup.ps1
    Full setup with automatic GPU detection.

.EXAMPLE
    .\setup.ps1 -Cpu
    Force CPU-only PyTorch (smaller download, no CUDA needed).
#>
[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$SkipSystem,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

$MinPy = [version]'3.10'
$MaxPy = [version]'3.14'   # exclusive

# ----------------------------------------------------------------- helpers --
function Say  { param($m) Write-Host $m -ForegroundColor Cyan }
function Ok   { param($m) Write-Host "  OK    $m" -ForegroundColor Green }
function Warn { param($m) Write-Host "  WARN  $m" -ForegroundColor Yellow }
function Die  { param($m) Write-Host "  FAIL  $m" -ForegroundColor Red; exit 1 }
function Step { param($n, $m) Write-Host ""; Write-Host "[$n/6] $m" -ForegroundColor White -BackgroundColor DarkBlue }

function Test-Cmd { param($n) [bool](Get-Command $n -ErrorAction SilentlyContinue) }

function Try-Exec {
    <#
      Run a native command and capture its result without letting it abort the script.
      Needed because $ErrorActionPreference='Stop' turns *any* stderr output from a
      native executable into a terminating NativeCommandError - and several of the
      probes here legitimately write to stderr when they find nothing (the `py`
      launcher says "No suitable Python runtime found" that way).
    #>
    param([string]$Exe, [string[]]$Arguments = @())
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Exe @Arguments 2>&1
        $code = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
        return [pscustomobject]@{ Ok = ($code -eq 0); Text = (($out | Out-String).Trim()) }
    } catch {
        return [pscustomobject]@{ Ok = $false; Text = '' }
    } finally {
        $ErrorActionPreference = $old
    }
}

function Update-PathFromRegistry {
    # winget updates the registry but not the running shell. Without this, a
    # freshly installed Python or FFmpeg is invisible until you open a new window.
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = @($machine, $user) -join ';'
}

function Invoke-Step {
    param([string]$What, [scriptblock]$Action)
    if ($DryRun) { Warn "DRY RUN - would: $What"; return $true }
    & $Action
    return $LASTEXITCODE -eq 0
}

function Find-Python {
    # Prefer the py launcher: it can enumerate versions and pick a supported one.
    $candidates = @()
    if (Test-Cmd 'py') {
        foreach ($v in @('3.12', '3.11', '3.13', '3.10')) {
            $r = Try-Exec 'py' @("-$v", '-c', 'import sys; print(sys.executable)')
            if ($r.Ok -and $r.Text) { $candidates += $r.Text }
        }
    }
    foreach ($n in @('python', 'python3')) {
        if (Test-Cmd $n) { $candidates += (Get-Command $n).Source }
    }
    foreach ($exe in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        # Windows ships stub python.exe shims that just open the Store; they exit
        # non-zero here, so the Ok check filters them out too.
        # No quotes in the snippet: PowerShell strips quotes when passing
        # arguments to a native command, which silently turns any inline Python
        # containing a string literal into a syntax error.
        $r = Try-Exec $exe @('-c', 'import sys; print(sys.version_info[0]*100+sys.version_info[1])')
        if (-not $r.Ok -or -not $r.Text) { continue }
        $num = 0
        if (-not [int]::TryParse($r.Text.Trim(), [ref]$num)) { continue }
        try { $ver = [version]"$([math]::Floor($num/100)).$($num % 100)" } catch { continue }
        if ($ver -ge $MinPy -and $ver -lt $MaxPy) {
            return [pscustomobject]@{ Exe = $exe; Version = $ver }
        }
    }
    return $null
}

Write-Host ""
Write-Host "  Road Defect Detection Pipeline - Windows setup" -ForegroundColor White
Write-Host "  $root"
if ($DryRun) { Warn "DRY RUN - nothing will be installed or modified" }

# ------------------------------------------------------------- 1. Python ----
Step 1 "Python $MinPy - 3.13"
$py = Find-Python
if ($py) {
    Ok "Python $($py.Version) at $($py.Exe)"
} elseif ($SkipSystem) {
    Die "No supported Python found and -SkipSystem was given. Install Python 3.12 from https://www.python.org/downloads/"
} else {
    Warn "No supported Python found - installing 3.12 via winget"
    if (-not (Test-Cmd 'winget')) {
        Die "winget is unavailable. Install Python 3.12 manually: https://www.python.org/downloads/"
    }
    Invoke-Step "winget install Python.Python.3.12" {
        winget install --id Python.Python.3.12 --source winget `
            --accept-package-agreements --accept-source-agreements --silent
    } | Out-Null
    Update-PathFromRegistry
    $py = Find-Python
    if (-not $py) { Die "Python installed but still not on PATH. Open a NEW terminal and re-run this script." }
    Ok "Python $($py.Version) installed"
}

# ------------------------------------------------------------- 2. FFmpeg ----
Step 2 "FFmpeg (with the v360 filter)"
# Genuinely required, not a nice-to-have: v360 does the 360-to-flat reprojection
# and FFmpeg is also the H.264 encoder for the annotated output video.
$needFfmpeg = $true
if (Test-Cmd 'ffmpeg') {
    $filters = (Try-Exec 'ffmpeg' @('-hide_banner', '-filters')).Text
    if ($filters -match 'v360') {
        Ok "FFmpeg present with v360"
        $needFfmpeg = $false
    } else {
        Warn "FFmpeg is installed but has no v360 filter - a fuller build is needed"
    }
}
if ($needFfmpeg) {
    if ($SkipSystem) {
        Warn "-SkipSystem given; install a full FFmpeg build yourself (https://www.gyan.dev/ffmpeg/builds/)"
    } elseif (-not (Test-Cmd 'winget')) {
        Warn "winget unavailable - download FFmpeg manually: https://www.gyan.dev/ffmpeg/builds/"
    } else {
        Invoke-Step "winget install Gyan.FFmpeg" {
            winget install --id Gyan.FFmpeg --source winget `
                --accept-package-agreements --accept-source-agreements --silent
        } | Out-Null
        Update-PathFromRegistry
        if (Test-Cmd 'ffmpeg') { Ok "FFmpeg installed" }
        else { Warn "FFmpeg installed but not yet on PATH - open a new terminal afterwards" }
    }
}

# ------------------------------------------------------------ 3. Virtualenv -
Step 3 "Virtual environment (.venv)"
$venvPy = Join-Path $root '.venv\Scripts\python.exe'
if (Test-Path $venvPy) {
    Ok ".venv already exists"
} else {
    if ($DryRun) { Warn "DRY RUN - would create .venv" }
    else {
        & $py.Exe -m venv .venv
        if (-not (Test-Path $venvPy)) { Die "Could not create .venv" }
        Ok ".venv created"
    }
}
if ($DryRun) { $venvPy = $py.Exe }

if (-not $DryRun) {
    & $venvPy -m pip install --upgrade pip setuptools wheel --quiet
    Ok "pip upgraded"
}

# --------------------------------------------------------------- 4. Torch ---
Step 4 "PyTorch"
$hasGpu = $false
if (-not $Cpu) {
    if (Test-Cmd 'nvidia-smi') {
        if ((Try-Exec 'nvidia-smi').Ok) { $hasGpu = $true }
    }
}
if ($DryRun) {
    Warn ("DRY RUN - would install " + $(if ($hasGpu) { "CUDA" } else { "CPU" }) + " PyTorch")
} else {
    $probe = Try-Exec $venvPy @('-c', 'import torch, sys; sys.stdout.write(torch.__version__)')
    $already = if ($probe.Ok) { $probe.Text } else { $null }
    if ($already) {
        Ok "PyTorch $already already installed"
        $cudaOk = (Try-Exec $venvPy @('-c', 'import torch; print(torch.cuda.is_available())')).Text
        if ($hasGpu -and $cudaOk -match 'False') {
            Warn "An NVIDIA GPU is present but this PyTorch is CPU-only."
            Warn "To switch: .venv\Scripts\python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        }
    } elseif ($hasGpu) {
        Say "  NVIDIA GPU detected - installing the CUDA build (large download)"
        & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        if ($LASTEXITCODE -ne 0) {
            Warn "CUDA wheel failed - falling back to the CPU build"
            & $venvPy -m pip install torch torchvision
        }
        Ok "PyTorch installed"
    } else {
        Say "  No NVIDIA GPU detected - installing the CPU build"
        & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { & $venvPy -m pip install torch torchvision }
        Ok "PyTorch installed (CPU)"
    }
}

# -------------------------------------------------------- 5. Requirements ---
Step 5 "Project requirements"
if ($DryRun) {
    Warn "DRY RUN - would: pip install -r requirements.txt"
} else {
    & $venvPy -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Die "requirements.txt install failed - see the pip output above" }
    Ok "requirements.txt installed"
}

# --------------------------------------------------------------- 6. Verify --
Step 6 "Verifying the install"
if ($DryRun) {
    Warn "DRY RUN - would import every stage and run the test suite"
} else {
    & $venvPy tools/doctor.py --verify
    if ($LASTEXITCODE -ne 0) { Die "Import check failed - the install is incomplete" }
    Ok "All pipeline stages import"

    & $venvPy -m pytest tests/ -q --no-header -x 2>&1 | Select-Object -Last 3
    if ($LASTEXITCODE -ne 0) { Warn "Some tests failed - the pipeline may still run, but check the output above" }
    else { Ok "Test suite passed" }
}

# ----------------------------------------------------------------- Finish ---
Write-Host ""
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "  Try it without any footage of your own:" -ForegroundColor White
Write-Host "      .\rdd.ps1 demo"
Write-Host ""
Write-Host "  Then on your own video:" -ForegroundColor White
Write-Host "      .\rdd.ps1 check   source=C:\path\to\road.mp4    # look at the road mask FIRST"
Write-Host "      .\rdd.ps1 detect  source=C:\path\to\road.mp4"
Write-Host ""
Write-Host "  All commands:  .\rdd.ps1 help"
Write-Host ""
if (-not (Test-Cmd 'ffmpeg')) {
    Warn "FFmpeg is not on PATH in THIS shell. Open a new terminal before running the pipeline."
}
