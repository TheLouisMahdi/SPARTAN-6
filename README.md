# SPARTAN-6 Compact Eye NPU

An electrical-engineering graduation project for the AX309 / XC6SLX9.
The physically validated pretrained implementation below remains the reference.
A trained TinyEye P2 alternative has passed physical inference checks on
registered inputs. Real webcam acceptance remains user initiated and untested.
See [compact implementation and acceptance gates](docs/COMPACT_EYE_NPU.md).

```powershell
.\run.cmd demo --check
```

This command checks deployment prerequisites without opening a camera or
programming hardware. It can read device identity. Normal `demo` starts only with registered, verified
firmware, model, and rollback artifacts. Real camera acceptance is pending.

A Verilog accelerator for the original Open Model Zoo **open-closed-eye-0001**
eye-state CNN on an **AX309 / Xilinx Spartan-6 XC6SLX9**, clocked at 50 MHz.
The project studies output-channel parallelism and verifies the complete pretrained
classifier against independent CPU references. No HLS, retraining or
data-dependent quantization calibration is used by this path.

## Architecture and scope

The CPU captures frames, runs MediaPipe landmarks, extracts BGR eye ROIs and
prepares 3x32x32 inputs. One UART owner uploads the two eyes sequentially. The
FPGA performs normalization, all four convolutions, pooling and global MAX.
The host correlates replies and computes/displays temporal results.

P2 has two output-channel MAC lanes sharing an activation; P1 has one lane.
This is channel-level parallelism, not simultaneous left/right eyes. The live
application already has a capture thread and parallel CPU eye workers. FPGA
requests block their caller in the original diagnostic mode. Compare mode
overlaps the CPU reference with the same frame's FPGA request, then waits.

Explicit `--pipeline sequential` and `--pipeline overlap` modes now share bounded
preparation, FPGA and result stages. Preparation can run ahead of the serial
owner; the result stage owns CPU temporal state and the main thread owns OpenCV
GUI. Traces distinguish request overlap from conservative kernel-overlap bounds.
See the [architecture and ownership map](docs/REPOSITORY.md) and current
[acceptance results](docs/VALIDATION_REPORT.md).

The current optimization evaluation is **offline only**. It uses registered CEW
inputs and explicitly synthetic schedules; it never opens a camera. See the
[micro-batch feasibility and protocol decision](docs/STREAMING_MICROBATCH.md).
The canonical UART RAW image remains the default.

The opt-in **JTAG USER1 CNN variant is physically verified**: **30.336 pairs/s
versus 7.519 over UART**, a **4.035x** improvement on identical complete workloads.
All 4846 CEW eyes matched fixed-point logits exactly; the kernel remains 14.70956 ms
per pair. The final design uses 1385 slices / 29 BRAM and meets 50 MHz. Independent
30 Hz input passed two short trials; sustained 50 Hz failed with explicit rejects.
See [architecture, protocol, measurements and reproduction](docs/JTAG_TRANSPORT.md).

## Validated results

| Result | Value | Provenance |
|---|---:|---|
| P2 CNN cycles per eye | 367,739 / 7.35478 ms at 50 MHz | Physical board cycle counter |
| P1 / P2 cycles per eye | 731,676 / 367,739; ratio 1.98966 | RTL simulation; P1 not physically benchmarked |
| CPU/FPGA fixed-point equality | 4,846 CEW images; zero logit differences | Physical UART inference |
| CEW accuracy / Macro-F1 | 76.228% / 75.753% | Post-inspection evaluation after class-order correction |
| Laptop FP32 / fixed mean, two eyes | 0.510 / 3.082 ms | Host measurements |
| FPGA CNN mean, two eyes | 14.70956 ms | Physical cycle counter; excludes UART |

The FPGA is slower than this laptop even before UART transfer. CNN time is not
camera FPS or end-to-end throughput. CEW static-image results do not establish
unseen-person generalization or fatigue diagnosis. [Validation details](docs/VALIDATION_REPORT.md)
separate physical measurements, RTL simulation and implementation reports.

New **ROI_REPLAY_ONLY** comparison (100 identical scheduled pairs at 10 Hz,
75 inferred pairs per mode): sequential **7.3535 pairs/s**, overlap **7.2615 pairs/s**,
ratio **0.9875**. This single comparison shows no throughput gain. CPU preparation
overlapped older FPGA request intervals for **6.9131 ms** in total; the nominal
50 MHz conservative kernel-overlap lower bound was **0.1753 ms** (MODELLED_ESTIMATE,
not directly synchronized FPGA timestamps). It excludes camera/MediaPipe work.
The 300 fresh integer comparisons had zero mismatches; they reuse CEW inputs and
are compatibility checks, not a new independent recognition evaluation.

## Model and arithmetic

The [publisher model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/open-closed-eye-0001)
is retained with its [original license](assets/models/open-closed-eye-0001/LICENSE).
The 46,164-byte ONNX has SHA256
`4daa100034482525a26c9afb9297c16580a531189e66e3d2b2ac7d32becfd593`;
its pinned official SHA384 is enforced before inference/export.

Weights are INT16, activations INT32 and accumulation INT48. Native ONNX/ROM/UART
channels are **[CLOSED, OPEN]**; application APIs expose **[OPEN, CLOSED]**.
This correction changes interpretation, not weights or RTL arithmetic. The
[deployment specification](docs/PRETRAINED_DEPLOYMENT.md) documents layers,
rounding, protocol, wiring and deterministic ROM generation.

## Setup

Use Windows, Python **3.11**, and the pinned dependencies:

```powershell
py -3.11 -m venv assets/runtime
assets/runtime/Scripts/python.exe -m pip install -r requirements.txt
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile assets/models/face_landmarker.task
.\run.cmd --help
.\run.cmd live --status
```

Launchers use this environment directly. OpenCV DNN runs ONNX; PyTorch remains
for optional TinyEye research. Offline RTL checks use the tested Icarus Verilog
13.0 Windows distribution under `assets/tools/iverilog/ucrt64/` (keep its `bin/`
and `lib/` layout). Pretrained verification currently uses that explicit path;
installing executables on PATH alone is insufficient for those commands.
Implementation requires Xilinx **ISE 14.7** for the target. Programming uses
the existing Vivado/XSDB SVF workflow and a compatible JTAG cable. See the
[operating guide](guid.md). Camera modes require the local
`assets/models/face_landmarker.task` asset. It is downloaded separately and is not
redistributed in the source release. Expected SHA256:
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

## Verification without a board

```powershell
assets/runtime/Scripts/python.exe -m compileall -q python
.\test_all.bat
.\run.cmd pretrained status
.\run.cmd pretrained verify --offline --pe-count 1 --output assets/runs/p1_new
.\run.cmd pretrained verify --offline --pe-count 2 --output assets/runs/p2_new
```

Use a new output directory for every run. `test_all.bat` runs Python, CNN arithmetic,
pin-level UART/controller and temporal RTL regressions without contacting a board.
CPU camera demonstration: `.\run.cmd live --precision fixed`. FP32 remains the
default; camera demonstrations are not accuracy measurements.

## Streaming demonstration and reports

Use registered offline eye inputs and the positively identified board port.
Give every experiment a new output directory; do not overwrite previous runs.

```powershell
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
.\run.cmd fpga --pipeline sequential --port COM26 --headless --warmup-seconds 0 --replay-manifest assets/datasets/cew/manifest.csv --replay-count 100 --replay-fps 10 --report "assets/runs/sequential_$tag/session.json"
.\run.cmd fpga --pipeline overlap --port COM26 --headless --warmup-seconds 0 --replay-manifest assets/datasets/cew/manifest.csv --replay-count 100 --replay-fps 10 --report "assets/runs/overlap_$tag/session.json"
```

`--headless` disables all window calls in explicit pipeline modes. `--verify`
compares returned logits on the CPU; use it for correctness, not primary timing.
The default command without `--pipeline` retains the synchronous hardware-timer
diagnostic behavior. Streaming mode labels its CPU temporal source explicitly.

The same bounded pipeline can replay a registered CEW source at identical pacing:

```powershell
.\run.cmd fpga --pipeline overlap --port COM26 --headless --warmup-seconds 0 --replay-manifest assets/datasets/cew/manifest.csv --replay-count 100 --replay-fps 10 --report "assets/runs/roi_overlap_$tag/session.json"
.\run.cmd experiment pipeline-compare --runs "assets/runs/sequential_$tag" "assets/runs/overlap_$tag" --output "assets/runs/comparison_$tag"
.\run.cmd session-report "assets/runs/overlap_$tag"
Start-Process "assets/runs/overlap_$tag/report.html"
```

A comparison without completed CNN samples returns NOT_AVAILABLE, never a
zero-sample hardware speedup. Each pipeline run records original IDs, capture and
stage timestamps, drops, kernel cycles, serial integrity, memory samples and a
real timestamp timeline in `trace.jsonl`, `frames.csv`, `session.json`,
`summary.json` and `report.html`. ROI replay is labelled ROI_REPLAY_ONLY;
capture-source drops and preparation drops are reported separately. The report
denominator includes drain/clean shutdown, excludes the declared warmup and
does not confuse capture FPS with successful inference FPS.

## Build, program and physical checks

```powershell
.\run.cmd pretrained prepare
.\run.cmd build --network pretrained --pe-count 2
# Explicit hardware operations, only when programming/testing is intended:
.\run.cmd program
.\run.cmd experiment golden --mode compare --port COM3 --output assets/runs/board_goldens_new
.\run.cmd fpga --port COM3
```

Replace COM3 with the known board port. `prepare` verifies/downloads the original
weights and regenerates coefficients. The sole canonical ISE project is
`fpga/eye_top.xise`: PRETRAINED=1, PE_COUNT=2, 50 MHz, 1,000,000 baud.
Build checks timing and artifact hashes; program validates the selected build
before volatile JTAG configuration. FPGA tests require the matching build.
Do not program an archived bitstream solely because an old report says PASS.

CEW and authorized video experiments require locally acquired data and explicit
registration. See [dataset setup](docs/DATASET_SETUP.md); images, private recordings,
environments and raw experimental reports are excluded from Git.

## Repository

| Directory | Role |
|---|---|
| `fpga/rtl/`, `fpga/generated/` | RTL and required coefficient ROMs |
| `fpga/` scripts, project, UCF | Build/program workflow and constraints |
| `python/` | Application, CPU references, UART host and experiments/reports |
| `research_ext/` | Regression tests, RTL benches and research protocol |
| `docs/`, `guid.md`, `REPO_MAP.md` | Specifications, operating guide and navigation |
| `assets/models/` | Pinned model, license and explicit research manifests |
| other `assets/` content | Local-only data, tools, environments and evidence |

AEEE2, grayscale/Sobel encoders and TinyEye remain **optional research**, separate
from the validated pretrained path. Their model slots remain untrained until
a valid checkpoint is supplied. See [research protocol](docs/RESEARCH_PROTOCOL.md).

## Limitations and credits

Binary eye scores are not calibrated aperture/PERCLOS80 measurements. Sparse
observations and UART latency can miss fast closures. The ocular indicator is
an unvalidated engineering heuristic, not a clinical fatigue diagnosis. No
power result or future pipeline speedup is claimed.

Credit belongs to Open Model Zoo/OpenVINO for the pretrained CNN, MediaPipe for
landmarks, and the CEW authors for the separately acquired evaluation data.
Preserve the bundled license and respect dataset redistribution terms.
No project-wide source-code license has yet been declared; third-party licenses
do not license this repository's own code. [Third-party details](docs/THIRD_PARTY.md).
