# SPARTAN-6

Bachelor's electrical engineering project implementing eye-state CNN inference on an ALINX AX309 FPGA, with Python camera acquisition, eye cropping, host inference, UART control and temporal eye-closure analysis. The default network is the pretrained `open-closed-eye-0001`; TinyEye training and AEEE encoding are separate, explicitly selected research paths.

## Hardware and environment

Spartan-6 XC6SLX9-2FTG256, Xilinx ISE 14.7, 50 MHz clock, two parallel processing elements, pretrained CNN enabled, UART 1,000,000 baud. The canonical top is `fpga/rtl/eye_top.v`; `fpga/eye_top.xise` and `board_pins.ucf` define the project and board constraints.

## Processing architecture

Python owns camera capture, MediaPipe facial landmarks, eye ROI extraction and temporal analysis. The CPU reference uses OpenCV FP32. For FPGA inference the host sends BGR 32 x 32 crops through the CRC-protected UART protocol; `command_controller` sequences the CNN and returns scores and timing. The pretrained datapath uses INT16 weights, INT32 activations and INT48 accumulation. No CPU/FPGA concurrency or application accuracy is claimed. TinyEye retains serial/parallel INT8 datapaths, runtime model loading and grayscale/Sobel/AEEE encoders; its shipped model is untrained.

## Source layout

- `fpga/`: canonical ISE project, board constraints, modular RTL, initialized model memories and build/JTAG scripts.
- `assets/models/eye_model.json`: untrained TinyEye configuration; downloaded models remain local.
- `python/`: application, capture, references, transport, ocular metrics and research tooling.
- `research_ext/`: experiment configuration, unit tests and RTL testbenches.

## Setup

Windows, Python 3.11 and the pinned packages in `requirements.txt` are required. Use an activated environment; `run.cmd` uses `python` on PATH, or the executable in `PYTHON`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force assets/models | Out-Null
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile assets/models/face_landmarker.task
.\run.cmd pretrained prepare
```

Face Landmarker SHA256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`. The pretrained preparation command checks the official ONNX SHA384, regenerates the included fixed-point ROMs/configuration, and fetches its license. Model downloads need network access. See [upstream model metadata](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/open-closed-eye-0001/model.yml), [MediaPipe documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) and `THIRD_PARTY_NOTICES.md`. Original project licensing is unresolved; see `LICENSE`.

## Build and program

Install licensed ISE 14.7 at `C:\Xilinx\14.7\ISE_DS` for the application workflow. Direct `fpga/build.cmd` also accepts `ISE_SETTINGS` pointing to its settings64.bat. Synthesis and implementation use ISE, not Vivado.

```powershell
.\run.cmd build --network pretrained
.\run.cmd program
```

Programming uses Xilinx XSDB/cable transport and an ISE-generated SVF. `XILINX_VIVADO_BIN` can override the default `C:\Xilinx\Vivado\2023.2\bin` for the validated application programming command. Direct `fpga/program.cmd` accepts `XSDB` (default `xsdb.bat` on PATH). A connected XC6SLX9 and working cable server are required. JTAG configuration is volatile. Application programming checks build hashes before programming.

## Application

```powershell
.\run.cmd live --camera 0
.\run.cmd fpga --port COM3 --camera 0
.\run.cmd --help
```

Choose the actual camera and serial port. `live` defaults to CPU pretrained inference. Select `--network tiny` explicitly for research, with trained research artifacts; personal data and trained TinyEye checkpoints are not included.

## Verification

Install Icarus Verilog with `iverilog.exe` and `vvp.exe` on PATH. Run setup first. Output directories below must be new.

```powershell
python -B -m unittest discover -s research_ext -p 'test_*.py'
.\run.cmd pretrained verify --output assets/runs/omz_check
.\run.cmd pretrained verify-uart --output assets/runs/uart_check
.\run.cmd rtl system --output assets/runs/system_check
python -B research_ext/verify_timing.py simulate --output assets/runs/timing_check
```

## Verified limitations

Publication checks: 66 unit tests completed (65 passed, one skipped) in the pinned runtime; all 18 application modules imported. Pretrained arithmetic, pretrained and TinyEye UART simulations passed. Timing simulation passed 18 cases / 91 samples. XISE dependencies and UCF port references were checked. Model regeneration reproduced the shipped files byte for byte.

Clean ISE 14.7 synthesis and implementation completed: 4,176 LUTs, 2,745 registers, 29 RAMB16BWERs and 10 DSP48A1s. Post-route analysis met the 20 ns clock constraint with a 19.431 ns minimum period (51.464 MHz), zero setup/hold timing errors. This validates the declared constraints; external I/O timing budgets are not specified in the UCF.

Camera/board end-to-end operation, recognition accuracy on users, clinical validity and physical timing have not been verified for this publication. Simulation, ISE implementation and physical operation are separate checks. No datasets, recordings, environments, bitstreams or build reports are distributed. Historical Git-baseline comparison requires a separately supplied prior revision; the clean source release contains no earlier history. Research reproduction requires separately licensed datasets and training runs.
