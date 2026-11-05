# Operating guide

Run commands from the repository root in PowerShell. See [README](README.md)
for Python 3.11 environment setup and [deployment](docs/PRETRAINED_DEPLOYMENT.md)
for arithmetic, UART, pins and clocks. Historical local reports are not required
to understand or run the source.

## Offline status and regression

```powershell
.\run.cmd --help
.\run.cmd live --status
.\run.cmd pretrained status
assets/runtime/Scripts/python.exe -m compileall -q python
.\test_all.bat
```

`test_all.bat` invokes Python unittest discovery, pretrained arithmetic, TinyEye
arithmetic, TinyEye/pretrained pin-level UART/controller simulation and temporal
RTL simulation. It never programs hardware. The opt-in synthetic training fixture
is skipped by this runner; existing research code/tests are retained.

The tested Icarus Verilog 13.0 Windows distribution is under
`assets/tools/iverilog/ucrt64/`; retain its complete `bin/` and `lib/` layout.
Pretrained verification currently uses that explicit directory. Some TinyEye
helpers also support PATH lookup; PATH alone does not satisfy pretrained checks.
The tools stay local and must not be copied into Git. Current pretrained checks use Icarus;
TinyEye layer simulation also supports the existing ISE/ISim path.

The tested local package came from
[MSYS2 Icarus Verilog](https://packages.msys2.org/packages/mingw-w64-ucrt-x86_64-iverilog),
with its bzip2, readline and termcap runtime dependencies. Preserve the packaged
VPI modules and DLLs as well as executables when arranging that local layout.
The existing runners prepend the tool's `bin/` to their subprocess PATH.

Run both pretrained lane configurations with deterministic fixtures:

```powershell
.\run.cmd pretrained verify --offline --pe-count 1 --output assets/runs/p1_new
.\run.cmd pretrained verify --offline --pe-count 2 --output assets/runs/p2_new
.\run.cmd pretrained verify-uart --output assets/runs/uart_new
```

`verify-uart` is pin-level RTL simulation, not a physical serial transaction.
Every output directory must be new. The offline flag avoids downloading example
images; the original pinned ONNX and required ROMs must be present.

## Model preparation and implementation

```powershell
.\run.cmd pretrained prepare
.\run.cmd build --network pretrained --pe-count 2
```

Preparation enforces the original ONNX checksum before deterministic ROM export.
The canonical build uses `fpga/eye_top.xise`, `board_pins.ucf`, PRETRAINED=1,
PE_COUNT=2, CLOCK_HZ=50000000 and BAUD=1000000. It generates BIT/SVF and reports
under `fpga/`, then checks hashes and 50 MHz timing. These generated products
remain local; required `fpga/generated/` coefficients remain versioned.

ISE 14.7 is required. The unified build currently expects the conventional Xilinx
14.7 ISE_DS installation (see `python/rtl_evidence.py`). The lower-level `fpga/build.cmd`
accepts `ISE_SETTINGS`. Do not change model/clock/constraints merely to fit
another installation. Research builds require an explicit fresh `--output`.

## Explicit hardware operations

These commands are for a later deliberate hardware session, not repository cleanup:

```powershell
.\run.cmd program
.\run.cmd experiment golden --mode compare --port COM3 --output assets/runs/board_new
.\run.cmd fpga --port COM3
```

Replace COM3 with the known board port; do not probe unrelated serial devices.
The existing program workflow validates build/source/BIT hashes, checks JTAG
identity and transports SVF through XSDB. Its current tool lookup expects an
existing Vivado/XSDB 2023.2 installation in the conventional Xilinx location;
see `python/rtl_evidence.py::program_variant` and `fpga/program.cmd` before use on
another workstation. Cleanup does not alter this working programming workflow.
Volatile JTAG configuration is lost after power-off; no Flash operation is implied.

## CPU and registered-data experiments

```powershell
.\run.cmd live --precision fixed
.\run.cmd experiment cew --mode cpu --cpu-precision fixed --manifest assets/datasets/cew/manifest.csv --output assets/runs/cew_cpu_new
.\run.cmd experiment cew --mode compare --manifest assets/datasets/cew/manifest.csv --port COM3 --output assets/runs/cew_compare_new
.\run.cmd session-report assets/runs/cew_compare_new
```

The first command opens a camera; experiment commands consume registered local
data. The compare command contacts the selected FPGA. Dataset registration and
access requirements are in [DATASET_SETUP](docs/DATASET_SETUP.md). No data is
downloaded implicitly by these experiments. Canonical output is OPEN/CLOSED.

Optional AEEE2/TinyEye execution requires an explicitly trained compatible model.
It is separate from pretrained inference; see the [research protocol](docs/RESEARCH_PROTOCOL.md).
Explicit sequential/overlapped pipeline commands and reporting are documented
in the [README](README.md); ownership, blocking and queue policies are mapped
in [REPOSITORY](docs/REPOSITORY.md).
