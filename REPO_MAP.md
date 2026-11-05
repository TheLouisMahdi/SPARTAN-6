# Repository Map

Python eye-state inference + AX309 Spartan-6 hardware. Default: pretrained
`open-closed-eye-0001`; TinyEye/AEEE research requires explicit selection.
Paths below are repository-relative.

## Entry Points
- `run.cmd` → CLI launcher; uses `assets/runtime/Scripts/python.exe`.
- `run_cpu.bat`, `run_fpga.bat`, `test_all.bat` → existing experiment/regression shortcuts.
- `python/app.py` → command dispatch; replay, prediction, build/program orchestration.
- `python/live_pipeline.py` → live CPU/FPGA inference; model selection and timing synchronization.
- `python/experiment.py` → registered-data CPU/FPGA comparison and HTML/JSON reports; same-frame reference overlap.
- `python/stream_pipeline.py` → bounded preparation, exclusive FPGA owner and ordered result workers.
- `python/pipeline_report.py` → stage timeline, measured/request and bounded/kernel overlap, fair-scope comparisons.

## Core
- `python/pretrained_eye.py` → official model preparation; FP32 and fixed-point references, RTL weight export.
- `python/capture.py` → camera ownership, landmarks, eye ROI extraction and collection.
- `python/blink_timing.py` → binocular closure timing reference and transport contract.
- `python/eye_state.py` → thresholds, calibration, stable states, blink events and evaluation.
- `python/ocular_metrics.py` → validity-aware closure metrics and video timestamps.
- `python/encoders/` → grayscale, Sobel and AEEE encodings; shared ROI/packing contracts.
- `python/integer_model.py` → TinyEye INT8 arithmetic, quantization and inference reference.

## Data and Research
- `python/dataset.py` → canonical ROI storage; manifests, locking and integrity checks.
- `python/data_tools.py` → dataset import, splits and encoded bundles.
- `python/public_data.py` → licensed source registry adapters and subject splits.
- `python/research_model.py` → TinyEye network definition.
- `python/research.py` → offline training, personal fitting and encoder ablations.
- `python/evidence.py` → public-data experiments and evidence reporting.

## Hardware Interfaces and RTL
- `python/fpga_host.py` → serial client; UART packets, CRC and model transfer.
- `fpga/rtl/eye_top.v` → hardware entry; reset, transport, CNN selection and board integration.
- `fpga/rtl/command_controller.v` → command FSM; input loading, CNN sequencing and replies.
- `fpga/rtl/uart_rx.v`, `uart_tx.v` → byte transport.
- `fpga/rtl/packet_rx.v`, `packet_tx.v`, `crc16_function.vh` → packet framing and integrity.
- `fpga/rtl/omz_cnn.v` → pretrained CNN datapath.
- `fpga/rtl/eye_cnn.v`, `eye_cnn_parallel.v` → TinyEye serial/parallel datapaths.
- `fpga/rtl/cnn_addresses.v`, `model_store.v`, `activation_store.v` → TinyEye addressing and memories.
- `fpga/rtl/mac_int8.v`, `requantize_q24.v` → TinyEye arithmetic primitives.
- `fpga/rtl/blink_event.v`, `binocular_timer.v` → hardware blink and closure timing.
- `fpga/rtl/activity_status.v`, `latency_meter.v`, `key_filter.v`, `board_display.v` → board monitoring and UI.

Grouped RTL filenames share the first file's directory. Data persists in files;
there is no database or web API layer.

## Configuration and Build
- `requirements.txt` → pinned Python 3.11 dependencies.
- `research_ext/protocol.json` → experiment, metric and hardware settings.
- `research_ext/public_sources.example.json` → public dataset registry template.
- `fpga/board_pins.ucf` → board pins and clock constraints.
- `fpga/eye_top.xise` → canonical Xilinx ISE project.
- `python/rtl_evidence.py` → RTL checks, isolated/canonical builds and validated programming.
- `fpga/build.cmd`, `create_project.tcl`, `build.tcl` → ISE 14.7 project setup and bitstream build.
- `fpga/generate_svf.cmd` → iMPACT bitstream-to-SVF export.
- `fpga/program.cmd`, `program.tcl`, `play_ise_svf.tcl` → XSDB/JTAG SVF programming.
- `.gitignore` → artifact exclusions.

Grouped build filenames share `fpga/`. Prefer `run.cmd build` / `run.cmd program`
for the validated workflow.

## Tests and Verification
- `research_ext/test_*.py` → unittest suites: pretrained/live inference, timing, evidence and build contracts.
- `research_ext/tb_omz.v`, `tb_cnn.v` → pretrained/TinyEye arithmetic benches.
- `research_ext/tb_system.v`, `top_equivalence.vh` → UART/controller integration and top equivalence checks.
- `research_ext/tb_timing.v`, `verify_timing.py` → timing traces; Python, RTL and board comparison.
- `python/pretrained_verify.py` → pretrained ONNX/fixed-point/RTL and UART verification.
- `python/system_sim.py` → pin-level UART integration simulation driver.

Grouped bench filenames share `research_ext/`; test fixtures/results are generated.

## Navigation Docs
- `README.md`, `guid.md` → overview and execution guide.
- `docs/PRETRAINED_DEPLOYMENT.md` → default model, arithmetic and CPU/FPGA partition.
- `docs/RTL_ARCHITECTURE.md` → integration hierarchy and retained TinyEye architecture.
- `docs/DATASET_SETUP.md`, `RESEARCH_PROTOCOL.md` → data setup and experiment procedure.
- `docs/VALIDATION_REPORT.md`, `PARALLELISM_BASELINE.md` → validation evidence and parallelism baseline.
- `docs/REPOSITORY.md` → file classification and actual concurrency/blocking map.
- `docs/THIRD_PARTY.md` → retained attribution and licensing boundaries.
- `docs/LITERATURE_REVIEW.md` → research background.

Grouped documentation filenames share `docs/`.

## Dependency Flow
- Live: `run.cmd` → `app.py` → `live_pipeline.py` → `capture.py` + `pretrained_eye.py` + `blink_timing.py`.
- FPGA: `live_pipeline.py` → `fpga_host.py` → UART/packets → `command_controller.v` → CNN → reply/timing.
- Research: `dataset.py` / `public_data.py` → `data_tools.py` + `encoders/` → `research.py` → `research_model.py` + `integer_model.py`.
- Build: `app.py` → `rtl_evidence.py` → ISE Tcl → bitstream/SVF → JTAG.

## Ignore During Source Navigation
- Local `assets/` data, runtime, tools, datasets, runs and logs are ignored. Pinned model/license and research manifests are retained exceptions.
- `fpga/generated/` → required exported weights/configuration; retained in Git and deterministically regenerated through model tooling.
- FPGA `*.bit`, `*.svf`, `*.syr`, `*.twr`, `*.mrp`, `*.log`, `result.json`, `rtl_screenshots/` → build/evidence artifacts.
- `fpga/output/`, `_ngo/`, `_xmsgs/`, `iseconfig/`, `xlnx_auto_*/`, `xst/` → ISE intermediates.
- `.git/`, `__pycache__/`, `*.pyc`, virtual environments, caches and temporary outputs.
