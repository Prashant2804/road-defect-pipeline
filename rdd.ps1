<#
.SYNOPSIS
    Road Defect Detection Pipeline - simple task runner (Windows).

.DESCRIPTION
    Ultralytics/Roboflow-style command line:

        .\rdd.ps1 <task> key=value key=value ...

    Runs inside .venv automatically - no activation needed.
    Run  .\rdd.ps1 help  for the full list.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Task = 'help',
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

# Prefer .venv, but accept any interpreter that already has the dependencies —
# plenty of people manage their own conda/system environment and should not be
# forced into a second one.
$venvPy = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    $fallback = $null
    if ($env:RDD_PYTHON) { $fallback = $env:RDD_PYTHON }
    else {
        foreach ($n in @('python', 'python3')) {
            $c = Get-Command $n -ErrorAction SilentlyContinue
            if (-not $c) { continue }
            $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
            & $c.Source -c 'import ultralytics, cv2, torch' 2>&1 | Out-Null
            $good = ($LASTEXITCODE -eq 0)
            $ErrorActionPreference = $old
            if ($good) { $fallback = $c.Source; break }
        }
    }
    if ($fallback) {
        Write-Host "  (no .venv - using $fallback, which already has the dependencies)" -ForegroundColor DarkYellow
        $venvPy = $fallback
    } else {
        Write-Host ""
        Write-Host "  No usable Python environment - run setup first:" -ForegroundColor Red
        Write-Host "      .\setup.ps1"
        Write-Host ""
        exit 1
    }
}

# ----------------------------------------------------- parse key=value args --
# Anything with a dot in the key (roadseg.backend=sam) is passed straight through
# to the pipeline as a config override, so every knob in config.yaml is reachable
# from the command line without editing files.
$opt = @{}
$overrides = @()
foreach ($a in ($Rest | Where-Object { $_ })) {
    if ($a -notmatch '=') {
        Write-Host "  Bad argument '$a' - expected key=value" -ForegroundColor Red
        exit 1
    }
    $k, $v = $a -split '=', 2
    $k = $k.Trim()
    if ($k -match '\.') { $overrides += @('--set', "$k=$v") } else { $opt[$k.ToLower()] = $v }
}

function Opt { param($name, $default = $null) if ($opt.ContainsKey($name)) { $opt[$name] } else { $default } }

function Need {
    param($name, $hint)
    $v = Opt $name
    if (-not $v) { Write-Host "  Missing required '$name='  e.g. $hint" -ForegroundColor Red; exit 1 }
    $v
}

function Run {
    param([string[]]$PipelineArgs)
    $all = @('run.py') + $PipelineArgs + $overrides
    Write-Host ("  > python " + ($all -join ' ')) -ForegroundColor DarkGray
    & $venvPy @all
    exit $LASTEXITCODE
}

# Shared optional flags for the video tasks.
function CommonArgs {
    $a = @()
    $view = Opt 'view'
    if ($view) { $a += @('--view', $view) }
    $device = Opt 'device'
    if ($device) { $a += @('--device', $device) }
    $out = Opt 'project'
    if ($out) { $a += @('--output', $out) }
    $name = Opt 'name'
    if ($name) { $a += @('--set', "run.name=$name") }
    $conf = Opt 'conf'
    if ($conf) { $a += @('--set', "inference.conf=$conf") }
    $imgsz = Opt 'imgsz'
    if ($imgsz) { $a += @('--set', "inference.imgsz=$imgsz") }
    $preset = Opt 'preset'
    if ($preset) { $a += @('--preset', $preset) }
    return $a
}

switch ($Task.ToLower()) {

    # ---------------------------------------------------------------- detect --
    { $_ -in 'detect', 'predict', 'run' } {
        $src = Need 'source' '.\rdd.ps1 detect source=C:\videos\road.mp4'
        $a = @('--input', $src) + (CommonArgs)
        $w = Opt 'weights'
        if ($w) { $a += @('--set', "inference.weights=$w") }
        Run $a
    }

    # ------------------------------------------------------------------ check --
    # Look at the road mask before trusting anything downstream.
    { $_ -in 'check', 'roadseg', 'preview' } {
        $src = Need 'source' '.\rdd.ps1 check source=C:\videos\road.mp4'
        $a = @('roadseg', '--input', $src, '--n', (Opt 'n' '8')) + (CommonArgs)
        Run $a
    }

    # --------------------------------------------------------------- validity --
    { $_ -in 'validity', 'coverage' } {
        $src = Need 'source' '.\rdd.ps1 validity source=C:\videos\road.mp4'
        $a = @('validity', '--input', $src, '--every', (Opt 'every' '10'),
               '--stride', (Opt 'stride' '3')) + (CommonArgs)
        if ((Opt 'traffic' 'false') -eq 'false') { $a += '--no-traffic' }
        $j = Opt 'json'
        if ($j) { $a += @('--json', $j) }
        Run $a
    }

    # ---------------------------------------------------------------- quality --
    { $_ -in 'quality' } {
        $src = Need 'source' '.\rdd.ps1 quality source=C:\videos\road.mp4'
        $a = @('quality', '--input', $src) + (CommonArgs)
        $csv = Opt 'csv'
        if ($csv) { $a += @('--csv', $csv) }
        Run $a
    }

    # ------------------------------------------------------------- preprocess --
    { $_ -in 'preprocess', 'prep' } {
        $src = Need 'source' '.\rdd.ps1 preprocess source=C:\videos\road.mp4'
        Run (@('preprocess', '--input', $src) + (CommonArgs))
    }

    # ----------------------------------------------------------------- label --
    { $_ -in 'label', 'annotate' } {
        $frames = Opt 'frames' 'data/rectified'
        Run @('annotate', '--frames', $frames)
    }

    # ----------------------------------------------------------------- train --
    { $_ -in 'train' } {
        $data = Opt 'data' 'data/labels'
        $a = @('train', '--labels', $data)
        $device = Opt 'device'
        if ($device) { $a += @('--device', $device) }
        foreach ($p in @(@('epochs', 'model.train.epochs'),
                         @('batch', 'model.train.batch'),
                         @('imgsz', 'model.train.imgsz'),
                         @('patience', 'model.train.patience'))) {
            $v = Opt $p[0]
            if ($v) { $a += @('--set', "$($p[1])=$v") }
        }
        Run $a
    }

    # ------------------------------------------------------------------- val --
    # Measures precision per unique defect and writes calibration.yaml.
    { $_ -in 'val', 'validate', 'evaluate', 'eval' } {
        $defects = Opt 'defects'
        if (-not $defects) {
            $name = Opt 'name' 'default'
            $defects = Join-Path (Opt 'project' 'out') "$name\defects.csv"
        }
        $truth = Need 'truth' 'truth=ground_truth.csv  (columns: class,first_frame,last_frame)'
        $a = @('evaluate', '--defects', $defects, '--truth', $truth)
        $s = Opt 'summary'
        if (-not $s) {
            $cand = Join-Path (Split-Path $defects -Parent) 'summary.json'
            if (Test-Path $cand) { $s = $cand }
        }
        if ($s) { $a += @('--summary', $s) }
        $o = Opt 'out'
        if ($o) { $a += @('--out', $o) }
        $t = Opt 'target'
        if ($t) { $a += @('--set', "eval.target_precision=$t") }
        Run $a
    }

    # ------------------------------------------------------------------ demo --
    # End-to-end on generated footage, so the install can be exercised with no
    # footage of your own.
    { $_ -in 'demo' } {
        Write-Host ""
        Write-Host "  Generating synthetic road footage with known ground truth..." -ForegroundColor Cyan
        & $venvPy tools/make_synthetic_road.py --out data/raw --frames 90 --only car
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host ""
        Write-Host "  Road + surface masks (check these look right):" -ForegroundColor Cyan
        & $venvPy run.py roadseg --input data/raw/synthetic_road_car.mp4 --view car_flat --n 4
        Write-Host ""
        Write-Host "  Full pipeline..." -ForegroundColor Cyan
        & $venvPy run.py --input data/raw/synthetic_road_car.mp4 --view car_flat `
            --set run.name=demo --set validity.traffic.enabled=false
        Write-Host ""
        Write-Host "  Artifacts in out\demo\  - open out\demo\report.html" -ForegroundColor Green
        exit $LASTEXITCODE
    }

    # ---------------------------------------------------------------- doctor --
    { $_ -in 'doctor', 'env' } {
        & $venvPy tools/doctor.py
        exit $LASTEXITCODE
    }

    # ------------------------------------------------------------------ test --
    { $_ -in 'test', 'tests' } {
        & $venvPy -m pytest tests/ -q
        exit $LASTEXITCODE
    }

    # ------------------------------------------------------------- synthetic --
    { $_ -in 'synthetic', 'gen' } {
        $a = @('tools/make_synthetic_road.py', '--out', (Opt 'out' 'data/raw'),
               '--frames', (Opt 'frames' '120'))
        if ((Opt 'scenarios' 'true') -ne 'false') { $a += '--scenarios' }
        if ((Opt 'degraded' 'true') -ne 'false') { $a += '--degraded' }
        & $venvPy @a
        exit $LASTEXITCODE
    }

    # ------------------------------------------------------------------ help --
    default {
        if ($Task -notin @('help', '-h', '--help', '/?')) {
            Write-Host "  Unknown task '$Task'" -ForegroundColor Red
        }
        @"

  Road Defect Detection Pipeline

  USAGE
      .\rdd.ps1 <task> key=value ...

  TASKS
      demo                     end-to-end on generated footage (start here)
      doctor                   check the environment is sane
      check    source=VIDEO    road + surface mask previews  <- DO THIS FIRST
      validity source=VIDEO    which frames are assessable, and why not
      quality  source=VIDEO    sharpness / exposure / noise report
      detect   source=VIDEO    full pipeline -> annotated video + report
      preprocess source=VIDEO  reproject + sample frames for labelling
      label    frames=DIR      pick the most useful frames to label first
      train    data=DIR        fine-tune on your labels
      val      truth=GT.csv    measure precision, calibrate thresholds
      synthetic                generate test footage
      test                     run the test suite

  COMMON KEYS
      source=PATH              input video
      view=car_flat            car_flat | car_360 | drone_nadir
      weights=best.pt          trained model
      conf=0.35                confidence threshold
      imgsz=1280               inference size
      device=cpu               cpu | cuda | cuda:0
      preset=fast              fast (~3x) | turbo (~8x) | accurate
      name=myrun               output goes to out\<name>\
      project=out              output root

  ANY config.yaml KEY works as an override - dotted keys pass straight through:
      .\rdd.ps1 detect source=road.mp4 geometry.camera.height_m=1.5
      .\rdd.ps1 detect source=road.mp4 roadseg.backend=sam surface.mud.min_warmer_z=3.0

  TYPICAL FIRST RUN
      .\rdd.ps1 doctor
      .\rdd.ps1 demo
      .\rdd.ps1 check  source=C:\videos\road.mp4 view=car_flat
      .\rdd.ps1 detect source=C:\videos\road.mp4 view=car_flat

  MEASURE THE ROAD IN METRES (needed for m2 areas and IRC severity)
      .\rdd.ps1 detect source=road.mp4 geometry.camera.height_m=1.35 geometry.camera.h_fov_deg=95

"@ | Write-Host
        exit 0
    }
}
