# Pretrained CNN on AX309

The default model is Open Model Zoo `open-closed-eye-0001`. No weight training,
fine tuning, distillation or data-dependent quantization calibration is required
by this deployment. Real-world accuracy still requires evaluation.

Current physical validation is summarized in [VALIDATION_REPORT.md](VALIDATION_REPORT.md):
4,846 CEW images matched the integer reference exactly; P2 measured 367,739
cycles per eye at 50 MHz. CEW accuracy is 76.228%, Macro-F1 75.753%, after
class-order correction. This is post-inspection evaluation, not a fresh independent
test. Earlier implementation records below retain their historical dates/scopes.

Official source: https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/open-closed-eye-0001

The original 46164-byte ONNX has SHA256
`4daa100034482525a26c9afb9297c16580a531189e66e3d2b2ac7d32becfd593`.
Its official SHA384 is enforced before any inference/export. Apache 2.0 license
is retained beside the model. No new Python dependency is installed: OpenCV DNN runs ONNX.

## Exact network and partition

| Operation | Output C,H,W | Implementation |
|---|---|---|
| BGR crop, resize | 3,32,32 | PC; original channel order; lossless U8 transfer |
| Normalize `(pixel-127)/255` | 3,32,32 | CPU in CPU mode; FPGA in FPGA mode |
| Conv3x3 valid, 3 to 10 | 10,30,30 | FPGA two output-channel MAC lanes |
| MaxPool2 stride2, then ReLU | 10,15,15 | FPGA; pool fused with convolution |
| Conv3x3 valid, 10 to 20 | 20,13,13 | FPGA; unused odd pool border omitted |
| MaxPool2 stride2, then ReLU | 20,6,6 | FPGA; floor rounding |
| Conv3x3 valid, 20 to 50 | 50,4,4 | FPGA; **no ReLU after this convolution** |
| Conv1x1, 50 to 2 | 2,4,4 | FPGA; zero bias in the original model |
| Global MaxPool4x4 | 2 | FPGA; **maximum, not average** |
| Softmax | 2 | Omitted for decision; argmax of pre-softmax logits is identical |

The pinned ONNX/ROM outputs native channels `[CLOSED, OPEN]`, consistent with
Intel's [reference eye-state estimator](https://github.com/openvinotoolkit/open_model_zoo/blob/2c9b5c563ab3a202d085a42e2bbfc48f77961f4b/demos/gaze_estimation_demo/cpp/src/eye_state_estimator.cpp).
Its model README lists the opposite order. `PretrainedEye.predict` and
`FPGAClient.infer_pretrained` reorder native channels to the public
`[OPEN, CLOSED]` contract. Layer traces and UART bytes retain native order;
weights, arithmetic, and the bitstream are unchanged by this host correction.

PC owns the camera, MediaPipe facial landmarks, original-color crop extraction,
resizing to 32x32, transport, display and a capture-timestamp timing reference.
The existing 2:1 eye-region geometry is retained; BGR pixels are taken directly
from the original crop, never from the legacy gray 16x32 thumbnail.
Independent CPU workers own separate OpenCV networks; one shared network is
never called concurrently. CPU mode uses original FP32 weights by default.
`--precision fixed` selects the exact FPGA integer arithmetic on the CPU.

FPGA runs the entire classifier and the existing bilateral timer, LEDs and
six-digit display. The two eyes run sequentially on one reusable accelerator;
two output channels are computed in parallel. This is not MediaPipe on FPGA.
P1 uses one lane and is retained for an arithmetic/performance comparison.

## Numerical representation

There are 11170 convolution weights, 80 original biases and two explicit zero
final-layer biases. Coefficients are signed INT16; activations signed INT32;
products and accumulators signed INT48. Conservative recursive L1 bounds over
all legal input bytes establish accumulator/activation range. They are not
estimated from test images or fitted to a person.

| Layer | Weight fractional bits | Input fractional bits | Output fractional bits | Right shift |
|---|---:|---:|---:|---:|
| Conv1 | 14 | 16 | 16 | 14 |
| Conv2 | 14 | 16 | 16 | 14 |
| Conv3 | 15 | 16 | 16 | 15 |
| Conv4 | 13 | 16 | 14 | 15 |

Every rounding operation is `floor(value + 0.5)` in the corresponding scaled
integer domain, including negatives. Final logits are Q14. Quantization changes
numerical values slightly; the original parameters are never optimized again.
The FPGA omits Softmax because only the two-score ordering is consumed. A tied
score becomes UNAVAILABLE in the host pipeline. Neither logits nor Softmax
scores establish calibrated eye-state confidence or eye-opening fraction.

Weight ROM is split into three 4096-word banks with two read ports.
Bias ROM uses distributed LUTs to avoid depending on 9K BRAM initialization;
see the official [XCN11014 notice](https://docs.amd.com/v/u/en-US/xcn11014).
The intermediate `ise_p2_banked` build passed timing but used one initialized
9K block for biases, so it was superseded by `ise_p2_final`, with **zero** 9K blocks. This avoids
ISE rounding a single 11170-word ROM up to 16384 words. Activation memories are
reused between layers and implemented with block RAM. All activation locations
read by inference are written first; no initialized activation contents are assumed.

## Protocol and wiring

Board constraints and pins are unchanged. Framing/version/CRC/sequence checks
are unchanged. Firmware is explicitly built with `PRETRAINED=1`, method ID 4,
fingerprint `4daa1000`; old TinyEye weight uploads are rejected.

| Request | Payload | Reply after zero status byte |
|---|---|---|
| 0x01 | empty | existing 12-byte AEEE ping; method4, ROM-loaded bit, fingerprint |
| 0x20 | LE16 offset + 1..1024 input bytes | empty ACK |
| 0x21 | empty | LE32 signed native CLOSED, OPEN; LE32 cycles, fingerprint |
| 0x14 | existing capture ms, validity, two U16 states | existing bilateral timing reply |
| 0x11 / 0x13 | empty | invalidate continuity / reset counters |

Each eye contains 3072 bytes in C,H,W order, B then G then R. Offset zero starts
a new eye, later chunks must be contiguous and within 3072 bytes. Inference
requires all bytes and consumes readiness, so the previous eye cannot be rerun
accidentally. Three CRC-checked input packets and one inference request are used
per eye. Model identity is checked before camera startup and on every CNN reply.

Default pretrained firmware and host use **1000000 baud**, 8N1. The board remains
clocked at 50MHz. Changing baud requires matching host and firmware settings.
Physical inference and CRC/sequence integrity at this rate were subsequently
verified; see the current validation summary. This is not an indefinite-uptime
reliability proof or validation of every USB-UART adapter.
At this rate 6144 input bytes alone take at least 61.44ms per eye pair. Packet
overheads, USB scheduling, capture, localization and inference add time. The old
115200 rate would take at least 533ms for the same bytes, exceeding the existing
250ms live-observation freshness budget. Do not treat raw CNN speed as video FPS.

The existing timer requires completed, overlapping bilateral closures. Winks,
missing observations and stale frames do not create accepted bilateral events.
Timing uses camera-read timestamps, not receipt time or number of frames. Fast
closures can be missed when sampling/transport is too slow. Duration resolution
is limited by observation intervals; this model does not estimate continuous aperture.

LED0 is subject-left closed, LED3 subject-right closed; the middle LEDs show
activity. The six digits retain: three closure-duration ms digits, dash,
left state, right state. A power cycle loses JTAG configuration; pretrained
weight ROMs return automatically when the matching BIT is programmed again.

## Historical implementation verification and limits

Pre-pipeline evidence is under `assets/runs/omz_20261030`. Arithmetic checks compare all
intermediate fixed-point values and final logits against an independent Python
integer reference, with four synthetic and three official example images.
Original ONNX intermediate outputs are compared separately. These seven inputs
are engineering fixtures, not an accuracy dataset.

The first monolithic-ROM ISE build failed with 33 RAMB16 blocks for 32 available;
that failure is retained as `ise_p2`. The corrected `ise_p2_final` build passed
synthesis/map/place-route/BIT generation and the actual 50MHz timing constraint:

| Checked result | Value |
|---|---:|
| Occupied slices | 1314 / 1430 |
| Slice LUTs / registers | 4194 / 2743 |
| DSP48A1 | 10 / 16 |
| RAMB16 / RAMB8 | 29 / 0 |
| Static Fmax | 51.573 MHz |
| Configured clock | 50 MHz |
| P1 simulated cycles per eye | 2001852 |
| P2 simulated cycles per eye | 1002827 |
| P1/P2 compute-cycle ratio | 1.9962 |

Both lane variants passed all 26390 intermediate comparisons and seven final
logit pairs. All seven decisions matched the original FP32 network; the maximum
absolute final-score error after rescaling was below 0.00273 on these fixtures.
P2 CNN-only computation is 20.057ms per eye at the configured 50MHz; this is
cycle-derived, not physical-board/video timing. The pin-level UART suite passed
24 requests and 23 independently CRC-checked replies, rejecting one deliberately
corrupted packet. It also checked two bilateral cycles, one long closure, wink
rejection, incomplete/out-of-order upload rejection, no stale input reuse, LEDs
and all six scanned digits.

The pipelined update is checked separately under `assets/runs/omz_pipeline_20261030`.
It overlaps synchronous reads, multiplication and accumulation, accepting one
operand pair per lane per clock within each dot product, with two drain clocks.
The model, weights, integer arithmetic and UART protocol are unchanged. Both P1
and P2 passed the same 26390 intermediate comparisons and seven final logit pairs,
plus pipeline reset/flush, exact MAC-count and steady-state no-bubble checks.
P1 now takes 731676 cycles and P2 367739 cycles per eye. At 50 MHz, P2 is 7.355 ms,
a 2.727-fold compute-cycle improvement over the pre-pipeline P2. These are
simulation-derived values, not measured camera throughput. The updated P2 UART
suite also passed all 24 requests and 23 CRC-checked replies.
The updated `ise_p2` build passed BIT generation and 50 MHz timing: 1303 slices,
4207 LUTs, 2745 registers, 10 DSP48A1, 29 RAMB16 and zero RAMB8; static Fmax is
51.427 MHz. Compared with the pre-pipeline build, DSP and BRAM counts are unchanged,
with 13 additional LUTs and two registers. This historical implementation-only
record preceded the modular canonical build and later physical validation.
Consolidated checks remain locally in
`assets/runs/omz_pipeline_20261030/verification.json`, outside the source distribution.

Only a successful current build with met timing is selected for programming.
See individual `result.json`, `compile.log`, `simulation.log`, map and timing reports.
Those historical checks did not test physical-board operation or UART throughput.
Later physical measurements are listed in the current validation summary. Camera
accuracy, power and clinical fatigue validity are not established by these fixtures.

## Compatibility and files

TinyEye INT8 and AEEE research remain explicit: `live --network tiny` and
`build --network tiny --output PATH`. That model's manifest is still untrained. Arbitrary
two-bit encoding of this pretrained BGR model is not supported or silently applied.
HLS is not used; the generated Verilog core is implemented with ISE 14.7.
Open `fpga/eye_top.xise` for manual ISE use. This is the only official ISE project.
Its modular top defaults to `PRETRAINED=1`, `PE_COUNT=2`, `CLOCK_HZ=50000000` and
`BAUD=1000000`. `run.cmd build` cleans and rebuilds this project, writes reports,
BIT/SVF and `result.json` under `fpga/`, and checks timing before reporting PASS.
`run.cmd program` checks the canonical source and BIT hashes before programming.
Explicit research variants require `build --output PATH`; TinyEye benches select
`PRETRAINED=0` explicitly. Historical reports, bitstreams and simulation fixtures
remain under `assets/runs/`; removed projects and build copies are recoverable
from the Git baseline commit preceding consolidation.

Direct implementation files, relative to the project root:

```text
python/pretrained_eye.py
python/pretrained_verify.py
python/live_pipeline.py
python/fpga_host.py
python/rtl_evidence.py
python/app.py
fpga/rtl/omz_cnn.v
fpga/rtl/eye_top.v
fpga/rtl/command_controller.v
fpga/generated/omz_config.vh
fpga/generated/omz_biases.hex
fpga/generated/omz_weights.hex
fpga/generated/omz_weights0.hex
fpga/generated/omz_weights1.hex
fpga/generated/omz_weights2.hex
fpga/create_project.tcl
fpga/eye_top.xise
fpga/result.json
research_ext/tb_omz.v
research_ext/test_pretrained.py
research_ext/test_build_workflow.py
assets/models/open-closed-eye-0001/model.onnx
assets/models/open-closed-eye-0001/deployment.json
assets/models/open-closed-eye-0001/LICENSE
README.md
guid.md
docs/PRETRAINED_DEPLOYMENT.md
docs/RTL_ARCHITECTURE.md
docs/VALIDATION_REPORT.md
```
