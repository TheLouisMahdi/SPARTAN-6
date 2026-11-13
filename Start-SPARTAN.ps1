# Run from the repository root: .\Start-SPARTAN.ps1 -Mode all -Port COM26 -Camera 0
# This script never edits RTL or touches persistent FPGA flash.
[CmdletBinding()]
param(
    [ValidateSet('check', 'build', 'program', 'run', 'all')]
    [string]$Mode = 'check',
    [string]$Port = '',
    [ValidateRange(0, 16)]
    [int]$Camera = 0
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = $PSScriptRoot
Set-Location $root

$runner = Join-Path $root 'PC\run.cmd'
$python = Join-Path $root 'RESULTS\runtime\Scripts\python.exe'
$requirements = Join-Path $root 'PC\requirements.txt'
$buildReport = Join-Path $root 'FPGA\result.json'
$bitstream = Join-Path $root 'FPGA\eye_top.bit'

foreach ($file in @($runner, $requirements)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing repository file: $file"
    }
}

function Invoke-Checked([string[]]$Arguments) {
    & $runner @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): PC\run.cmd $($Arguments -join ' ')"
    }
}

function Confirm-Runtime {
    $needsInstall = $false
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        if ($Mode -notin @('all', 'build')) {
            throw 'Missing RESULTS/runtime Python. Use -Mode all to create it.'
        }
        & py -3.11 -m venv (Join-Path $root 'RESULTS\runtime')
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 virtual environment creation failed' }
        $needsInstall = $true
    }
    if (-not $needsInstall) {
        & $python -c 'import cv2, mediapipe, numpy, serial'
        if ($LASTEXITCODE -ne 0) { $needsInstall = $true }
        if (-not $needsInstall) {
            & $python -m pip check
            if ($LASTEXITCODE -ne 0) { $needsInstall = $true }
        }
    }
    if ($needsInstall) {
        if ($Mode -notin @('all', 'build')) {
            throw 'The existing Python environment is inconsistent. Use -Mode all to repair only RESULTS/runtime.'
        }
        & $python -m pip install -r $requirements
        if ($LASTEXITCODE -ne 0) { throw 'Installing compatible dependencies failed' }
        & $python -c 'import cv2, mediapipe, numpy, serial'
        if ($LASTEXITCODE -ne 0) { throw 'Python imports still fail' }
        & $python -m pip check
        if ($LASTEXITCODE -ne 0) { throw 'Python dependency check still fails' }
    }
}

function Confirm-FaceLandmarker {
    $file = Join-Path $root 'RESULTS\models\face_landmarker.task'
    $digest = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        if ($Mode -ne 'all') { throw "Missing camera model: $file" }
        New-Item -ItemType Directory -Path (Split-Path -Parent $file) -Force | Out-Null
        $tmp = "$file.download"
        if (Test-Path -LiteralPath $tmp) { throw "Existing partial download: $tmp" }
        try {
            Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile $tmp
            if ((Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash -ine $digest) {
                throw 'Official face landmarker checksum mismatch'
            }
            Move-Item -LiteralPath $tmp -Destination $file
        } finally {
            if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp }
        }
    }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ine $digest) {
        throw 'Face Landmarker hash is not the pinned version'
    }
}

function Get-UartPort {
    $script = @'
import json
from serial.tools.list_ports import comports
print(json.dumps([dict(device=p.device, vid=p.vid, pid=p.pid, description=p.description) for p in comports()]))
'@
    $json = & $python -c $script
    if ($LASTEXITCODE -ne 0) { throw 'Serial port discovery failed' }
    $available = @(ConvertFrom-Json -InputObject ($json | Out-String))
    $available | ForEach-Object {
        Write-Host "SERIAL $($_.device): $($_.description) VID=$($_.vid) PID=$($_.pid)"
    }
    if ($Port) {
        if ($Port -notin @($available | ForEach-Object { $_.device })) {
            throw "Requested UART port $Port was not detected"
        }
        return $Port
    }
    $candidates = @($available | Where-Object { $_.vid -eq 0x10c4 -and $_.pid -eq 0xea60 })
    if ($candidates.Count -ne 1) {
        throw "Found $($candidates.Count) CP2102 UART adapters. Supply -Port COMxx explicitly."
    }
    return [string]$candidates[0].device
}

function Test-PretrainedFirmware([string]$Device) {
    $script = @'
import json
import sys
sys.path.insert(0, 'PC')
from fpga_host import FPGAClient
client = None
try:
    client = FPGAClient(sys.argv[1], baud=1000000, timeout=2.0)
    identity = client.ping()
    print('FPGA_PING ' + json.dumps(identity, sort_keys=True))
    if not (identity['method_id'] == 4 and identity['model_loaded'] and
            identity['fingerprint'] == 0x4daa1000 and identity['binocular_timing']):
        raise RuntimeError('Wrong FPGA firmware/model identity for pretrained UART')
except Exception as exc:
    print('FPGA_NOT_READY: ' + str(exc))
    sys.exit(1)
finally:
    if client is not None:
        client.close()
'@
    & $python -c $script $Device | Out-Host
    return ($LASTEXITCODE -eq 0)
}

function Test-VerifiedBuild {
    if (-not (Test-Path -LiteralPath $buildReport -PathType Leaf) -or
        -not (Test-Path -LiteralPath $bitstream -PathType Leaf)) { return $false }
    try {
        $report = Get-Content -LiteralPath $buildReport -Raw | ConvertFrom-Json -AsHashtable
        if ($report.status -ne 'PASS' -or $report.timing_met -ne $true) { return $false }
        if ((Get-FileHash -LiteralPath $bitstream -Algorithm SHA256).Hash -ine $report.bit_sha256) {
            return $false
        }
        foreach ($relative in $report.source_sha256.Keys) {
            $source = Join-Path $root (Join-Path 'FPGA' $relative)
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { return $false }
            if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ine $report.source_sha256[$relative]) {
                return $false
            }
        }
        return $true
    } catch {
        return $false
    }
}

function Build-Pretrained {
    Write-Host 'BUILD: canonical pretrained P2 (ISE 14.7)'
    Invoke-Checked -Arguments @('build', '--network', 'pretrained', '--pe-count', '2')
    if (-not (Test-VerifiedBuild)) { throw 'Build evidence is missing or inconsistent' }
    Write-Host 'BUILD_PASS'
}

function Program-Pretrained {
    if (-not (Test-VerifiedBuild)) {
        throw 'No verified current BIT/source/timing report. Run -Mode build first.'
    }
    Write-Host 'PROGRAM: volatile JTAG only; the existing programmer validates XC6SLX9 IDCODE'
    Invoke-Checked -Arguments @('program')
    Write-Host 'PROGRAM_PASS'
}

Confirm-Runtime

if ($Mode -eq 'build') {
    Build-Pretrained
    return
}

$device = Get-UartPort

if ($Mode -eq 'program') {
    Program-Pretrained
    if (-not (Test-PretrainedFirmware $device)) {
        throw 'UART still cannot identify FPGA after JTAG programming. Inspect TX/RX/GND, COM and board power.'
    }
    return
}

$ready = Test-PretrainedFirmware $device
if ($Mode -eq 'check') {
    if (-not $ready) { throw 'Pretrained FPGA UART is not ready. Inspect hardware or use -Mode all.' }
    Write-Host 'CHECK_PASS'
    return
}

if ($Mode -eq 'all' -and -not $ready) {
    if (-not (Test-VerifiedBuild)) { Build-Pretrained }
    Program-Pretrained
    $ready = Test-PretrainedFirmware $device
}
if (-not $ready) {
    throw 'FPGA PING failed. The camera will not open; CPU fallback is forbidden.'
}

foreach ($file in @(
    'RESULTS\models\open-closed-eye-0001\model.onnx',
    'RESULTS\models\face_landmarker.task'
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $file) -PathType Leaf)) {
        throw "Missing model asset: $file"
    }
}

Confirm-FaceLandmarker
Write-Host "LIVE_START FPGA=pretrained UART=$device CAMERA=$Camera"
Write-Host 'Press Q or Esc in the camera window to exit.'
Invoke-Checked -Arguments @('fpga', '--network', 'pretrained', '--port', $device,
    '--camera', [string]$Camera, '--pipeline', 'sequential')
