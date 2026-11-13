# Validation record

Current scope is offline physical ROI inference and bounded streaming evaluation.
The micro-batch optimization is governed by the resource and correctness gates in
[STREAMING_MICROBATCH.md](STREAMING_MICROBATCH.md). Publication is a separate action.

This document summarizes existing evidence. Repository cleanup performs offline
regressions only; it does not program, reset or contact the connected board.
Simulation, implementation reports and physical measurements remain distinct.

## Current pretrained system

- Model: original `open-closed-eye-0001`, SHA256
  `4daa100034482525a26c9afb9297c16580a531189e66e3d2b2ac7d32becfd593`.
- Native channels: `[CLOSED, OPEN]`; canonical application order: `[OPEN, CLOSED]`.
  Host APIs reorder once; UART bytes, weights and integer arithmetic are unchanged.
- Target: AX309, XC6SLX9; 50 MHz; pretrained P2; 1,000,000-baud UART.
- Physical-validation evidence: local record `physical_validation_20261105_ffc1e3c4`.
  Raw reports, logs, hashes and datasets remain on disk and are excluded from Git.
- Verified physical bitstream SHA256:
  `2f75636570e8ff79ca2d6c2584a629affbef26a3c289846a212553e72e6aea80`.

| Finding | Value | Provenance |
|---|---:|---|
| P2 kernel, per eye | 367,739 cycles / 7.35478 ms | PHYSICAL_MEASUREMENT; 50 MHz counter |
| P2 kernel, two eyes | 735,478 cycles / 14.70956 ms | PHYSICAL_MEASUREMENT; excludes UART |
| P1 / P2, per eye | 731,676 / 367,739 cycles | RTL_SIMULATION on identical current-model fixtures |
| P1-to-P2 cycle ratio | 1.989661 | RTL_SIMULATION; P1 not physically tested or fitted in this validation |
| CPU/FPGA integer equality | 4,846 CEW images; maximum absolute difference 0 | PHYSICAL_MEASUREMENT with Python integer reference |
| CEW accuracy / Macro-F1 | 76.2278168% / 75.7531956% | Corrected post-inspection evaluation |
| Laptop FP32 / fixed, two-eye mean | 0.510458 / 3.081513 ms | HOST_MEASUREMENT; retain original timing scopes |
| UART roundtrip / upload, two-eye mean | 132.6364 / 106.0798 ms | PHYSICAL_MEASUREMENT; upload is part of roundtrip |
| FPGA end-to-end, two-eye mean | 134.6169 ms | PHYSICAL_MEASUREMENT; a different scope from kernel time |

The FPGA kernel is approximately 4.77x slower than the measured laptop fixed-point
reference, before UART. No laptop speedup is claimed. UART roundtrip, kernel and
end-to-end values are nested/different timing scopes; do not add them blindly.

The corrected confusion matrix is `[[2186, 276], [876, 1508]]` in OPEN/CLOSED order.
OPEN recall is 88.7896%; CLOSED recall is 63.2550%. The corrected accuracy follows
inspection of the output-channel semantics; this is **not a fresh independent test**.
Static CEW images do not validate fatigue, blink timing, continuous aperture or
PERCLOS80. No authorized video evaluation or clinical diagnosis is claimed.

## Physical verification retained locally

The original validation records include four physical golden inputs, smoke and
extended tests, repeatability, the full CEW FPGA/compare runs, JTAG identity,
volatile programming/DONE checks and UART CRC/sequence/timeout diagnostics.
The full 4,846-image equality result has no normal-path integer mismatch.
The intentionally injected zero-deadline PING timeout and its successful recovery
are recorded separately from normal-path integrity results.

## Implementation resources and historical milestones

The current canonical P2 implementation report records 1,318/1,430 occupied slices,
4,204 LUTs, 2,745 registers, 10/16 DSP48A1, 29/32 RAMB16 and zero RAMB8.
Reported static Fmax is 51.464 MHz; configured operation remains 50 MHz.
These are ISE implementation/static-timing results, not a clock-frequency sweep
or power measurement. No new implementation build is needed for documentation cleanup.

Historical local records retain the pre-pipeline P2 result of 1,002,827 cycles,
the pipelined 367,739-cycle result, monolithic-ROM capacity failures, banked-ROM
corrections and modular command-controller equivalence checks. Earlier builds
have different resource counts and must not be substituted for the current one.
TinyEye arithmetic (72 P1/P2 cases), pin-level controller comparisons and temporal
RTL cases remain useful research/regression evidence. No historical TinyEye score
is substituted for the pretrained CEW result.

## Reproducing checks

See [the operating guide](../guid.md) for offline Python, pretrained P1/P2,
pin-level UART/controller and temporal simulation commands. A hardware run requires
the matching verified pretrained build, a deliberately selected board port and
separately acquired registered data. Offline regressions do not require CEW images.
Use fresh output directories. Reports generated under `assets/runs/` stay local.

The build/program source chain, required ROMs and original model license remain
versioned. BIT/SVF files and raw build reports are generated locally. Build hashes
are checked again before any explicit programming request.

## Heterogeneous streaming acceptance, 2026-11-06

The new bounded stages preserve the original model, class mapping, ROI geometry,
50 MHz clock, UART packets and P2 RTL. Native Python threads own preparation,
serial inference and result processing; the UI stays on the main thread. Explicit
sequential/overlap modes share the same algorithms. See [architecture](REPOSITORY.md).

### Physical configuration and numerical gates

Initial PING timed out. Read-only JTAG identification found XC6SLX9 IDCODE
`24001093`; configuration inspection showed DONE low and USERCODE zero. The
existing verified BIT/SVF was then loaded once through volatile JTAG. DONE and
USERCODE passed, fingerprint `4daa1000` matched, and normal inference resumed.
No SPI Flash, pin, voltage, model or RTL change was made. This initial timeout is
retained separately from zero normal-path timeouts in subsequent measurements.

- Existing physical 4,846-image equality and corrected CEW results remain unchanged.
- Fresh deterministic hardware goldens: four eyes, including a neutral/low-contrast
  vector; exact CPU/FPGA equality.
- Fresh pipeline compatibility: **300 eye comparisons, 186 unique registered CEW
  sources**, covering open/closed left/right folders; **zero integer mismatches**.
  Reused/post-inspection data does not create a new recognition-accuracy test.
- Sequential/overlap share **57 common inferred pair IDs** with equal input tensor
  hashes and logits. Actual dropped/submitted IDs are retained in local traces.
- Physical onboard timer diagnostic: **18 synthetic engineering cases / 91 samples**,
  no temporal mismatches or bad packets; explicit 1,000,000 baud. This is not
  camera blink-detection accuracy and is separate from streaming CPU timing.

### Fair deterministic post-ROI comparison

Both modes used the same fixed 100-pair SHA-ordered source selection and an
independent 10 Hz producer, the same board/model/bitstream and no timed CPU
verification. Both inferred 75 pairs. Sequential dropped 25 capture-queue frames;
overlap dropped 25 preparation-queue frames. Different accepted subsets are
disclosed, not silently treated as identical decoded samples.

| Metric | Sequential | Overlap | Provenance |
|---|---:|---:|---|
| Successful pair throughput | 7.35349/s | 7.26153/s | PHYSICAL_MEASUREMENT |
| Throughput ratio | reference | 0.987495 | Derived from measured throughput |
| CPU prep mean / p95 | 0.16684 / 0.28143 ms | 0.070261 / 0.125 ms | HOST_MEASUREMENT; post-ROI only |
| FPGA pair kernel mean / p95 | 14.70956 / 14.70956 ms | same | PHYSICAL_MEASUREMENT; cycle-derived |
| UART roundtrip mean / p95 | 133.37762 / 139.51085 ms | 134.91141 / 141.18613 ms | PHYSICAL_MEASUREMENT; includes kernel |
| End-to-end mean / p95 | 180.10644 / 223.50023 ms | 187.06803 / 231.80480 ms | Capture to result publication |
| CPU prep vs older request overlap | 0 ms | 6.9131 ms total | Recorded host-clock intersections |
| Overlap fraction of request wall time | 0 | 0.068274% | Explicit measured-interval denominator |
| Conservative kernel-overlap lower bound | 0 ms | 0.1753 ms total | MODELLED_ESTIMATE; nominal 50 MHz envelope bound |

The pipeline genuinely overlaps post-ROI work with FPGA transactions. This trial
does **not** show a throughput improvement; UART/host roundtrip dominates the
very small post-ROI CPU work, and desktop/USB scheduling variation remains.
This is one paired ROI benchmark, not a statistically rigorous speedup estimate
or a measurement of the complete camera/MediaPipe path.

### Acceptance interpretation

Final source regression: **119 Python tests run, 118 passed, one pre-existing
opt-in training skip**. All five RTL stages passed: pretrained arithmetic,
72-case TinyEye P1/P2 arithmetic, TinyEye and pretrained pin-level UART, and
91-sample temporal simulation. The isolated source-only publication checkout
also passed the same 119-test suite after provisioning the documented ignored
Face Landmarker/Icarus prerequisites. No private application module was required.
An initial publication-check timeout test correctly reported a missing Icarus
dependency before that local prerequisite was provisioned; it was not hidden
by deleting the test. No new ISE implementation was needed because RTL,
coefficients, constraints and the verified BIT were unchanged.

Repository integrity, deterministic worker tests, numerical correctness, physical
P2 inference and controlled post-ROI comparison are supported by evidence.
Physical request overlap and a labelled kernel-overlap bound are supported for
ROI_REPLAY_ONLY. The current optimization scope uses offline inputs exclusively.
Historical runtime records remain unchanged in their original ignored run folders;
they are not acceptance evidence for the new streaming optimization.
