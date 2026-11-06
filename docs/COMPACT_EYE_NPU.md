# SPARTAN-6 Compact Eye NPU

The project preserves the verified pretrained P2 implementation and develops a
compact TinyEye alternative on a separate branch. Neither a successful model
training run nor RTL simulation proves a physical throughput or webcam result.

## Compact tensor and transport contract

TinyEye uses 1x16x32 grayscale inputs, four layers and 1,130 learned parameters.
Grayscale2 divides unsigned ROI values by 64. Four ordinal codes pack into one
byte in pixel order, most significant pair first. Each eye occupies 128 bytes;
the FPGA expands codes to 0, 85, 170 and 255 before the existing integer core.
Weights are INT8, biases and accumulators INT32, and requantization uses the
existing Q24 half-up rounding and saturation. Output order is OPEN,CLOSED.

Compact USER1 has signature JTY1, protocol version 2, command 2, and a 146-byte
payload. Its fields are frame ID (u32), capture timestamp (u64 nanoseconds), eye
side (u8), model fingerprint (u32), encoder ID (u8, 1 for Grayscale2), and 128
packed bytes. Integer fields are little endian. The scan has a 41-byte reply
prefix, eight-byte header, payload, CRC16 and one zero tail: 198 bytes per eye.
The existing JTN1 BGR protocol remains distinct and unchanged.

The asynchronous FIFO remains 16x9 bits. Symbol expansion stalls FIFO reads
for four system cycles. One shared P2 core and one outstanding eye credit
bound inference memory. Sequence wrap, request CRC, frame, eye, timestamp,
method and model identity are checked. Replies echo the request CRC; there is
no independent reply CRC. Do not describe this as end-to-end reply CRC protection.
The board-clock ownership latch outlives a TAP reset while the core is busy.
UART packet ownership protects coefficient upload and input processing.

## Frozen local training evidence

The approved local CEW manifest produced a deterministic photo-group split:
3,153 training, 729 validation and 964 test eyes. Crops from the same original
photo stay in the same partition. Source hashes were checked and exact duplicate
images rejected. Subject identities are unavailable: this is within-CEW held-out
photo evaluation, not an unseen-subject generalization claim. The source images
are not distributed with this repository.

One seed (42), 12 epochs and validation-only checkpoint selection were used.
Quantization calibration uses training data only. Frozen INT8 checkpoint SHA256:
`353bd6cae1ddadc0ce31a4b12b0803854605866214ad28f45ed659dd8300240f`.
Fingerprint: `353bd6ca`. Frozen split SHA256:
`3534a3fa15a370472d1348d5bb4ea468fd5e26a4adecbccd91b95f12a376d753`.

| Test result | Value |
|---|---:|
| Accuracy | 76.1411% |
| Macro-F1 | 0.7611795 |
| OPEN recall | 71.5447% |
| CLOSED recall | 80.9322% |
| Confusion matrix, true rows / predicted columns OPEN,CLOSED | [[352,140],[90,382]] |

These values describe this new local model and split only. They are not the
previous pretrained-model evaluation, live camera accuracy, or FPGA agreement.
The trained checkpoint remains local pending redistribution review. Physical
checks passed on all 964 held-out eyes with zero integer mismatches; the existing
default model manifest is not silently replaced. The checked demo catalog selects
the explicit trained compact slot.

## Physical implementation and measurements

The serial-requantization dual-transport candidate failed placement. An optional
DSP implementation preserves the exact unsigned product, Q24 rounding,
saturation and busy-cycle schedule. Its 560 arithmetic/boundary cases match the
serial reference, and the trained core passed packed-input simulation and
physical integer checks. The canonical pretrained RTL and ROMs are unchanged.

The DSP candidate uses 1,403/1,430 slices, 5,087 LUTs, 3,834 registers, 10 DSPs,
three BRAM18 and two BRAM9 blocks. ISE 14.7 place-and-route meets both clock and
CDC constraints: minimum board period 19.432 ns, 0.568 ns margin at 50 MHz.
BIT SHA256: `f83f91ca793fabfa358a1d89dece9cbe0632a4fa6697596c6340970b896fc803`.
The image uses volatile JTAG configuration with USER code `4A545931`.

| Physical test on the new compact model | Result |
|---|---:|
| UART, two unpaced 64-pair trials | 72.586 / 71.427 completed pairs/s |
| JTAG, two unpaced 64-pair trials | 89.949 / 90.274 completed pairs/s |
| Kernel mean on the same 64 pairs | 4.16672 ms/pair, summed eye counters |
| JTAG 50 Hz for 30 seconds | 1,500 arrived, 1,500 completed, 0 rejected |
| Sustained JTAG end-to-end p95 | 15.1060 ms |
| Sustained JTAG maximum queue | 2 of 8 |
| Physical held-out equality | 964 eyes, 0 integer mismatches |

Bounded Tcl-side result polling removes host turnarounds while preserving the
198-byte wire contract and host identity checks. JTAG is faster in the latest
matched trials; earlier per-scan trials were slower and are retained locally. The UART controller pair counter also includes packed-input expansion;
kernel comparisons use `left_cycles + right_cycles`. Input-dependent ReLU paths
make compact kernel cycles vary. Neither an ideal wire-time estimate nor RTL
simulation is a measured transport latency.

On this laptop, NumPy INT8 inference over the same 64 pairs, repeated three times
after five warmups, averaged 0.340796 ms/pair. Packing averaged 0.031343 ms/pair.
This desktop measurement is distinct from FPGA kernel, UART/JTAG roundtrip and
an embedded CPU benchmark. No new embedded-CPU result or energy measurement is
claimed. These physical tests use registered eye images, not a live camera.

The board initially exposed an unrelated default USER code and did not respond
to the expected UART protocol. A verified reference configuration was restored
before the experiment. After compact tests, explicit rollback and its UART model
fingerprint were verified. Flash, EEPROM and cable firmware were not written.

## Reproduction without devices

Use the project runtime and approved local dataset. Choose new output directories:

```powershell
.\assets\runtime\Scripts\python.exe -B research_ext/train_compact.py --dataset assets/datasets/cew --output assets/runs/compact_training_new
.\assets\runtime\Scripts\python.exe -B research_ext/tiny_evaluation.py simulate --dsp --checkpoint assets/runs/compact_training_new/compact_model/int8_parameters.npz --output assets/runs/compact_sim_new
.\assets\runtime\Scripts\python.exe -B research_ext/tiny_evaluation.py build --dsp --output assets/runs/compact_build_new
.\test_all.bat
```

The training recipe reuses the existing research implementation. The simulator
checks actual trained coefficients, packed symbol order, logits, sequencing,
metadata, malformed requests and deferred pin-level UART model upload. The ISE
14.7 build uses the existing pins, 50 MHz and UART 1 Mbaud; it performs no physical
programming. Timing and resource success must be read from its own result JSON.
No candidate may replace a verified firmware image merely because synthesis ran.

## Demo startup and current limits

`run.cmd demo --check` supports `--model auto|tiny|pretrained` and
`--transport auto|jtag|uart`. It inventories CP2102 ports without opening them
and checks a local `assets/models/demo_deployment.json` catalog. Each image entry
binds BIT, SVF, rollback SVF, model and physical evidence to SHA256 digests.
Missing or changed artifacts fail closed. Face Landmarker hash, package versions
and report-directory permissions are also checked. Read-only JTAG discovery
requires the exact XC6SLX9 identity; UART PING reports current firmware where
accessible. A missing target is NOT_AVAILABLE and inaccessible firmware is
NOT_VERIFIED. The check never opens a camera, creates a landmarker or programs a device.

Normal demo programming is volatile and uses the existing explicit SVF path.
It verifies target configuration and model identity before camera startup and
restores the registered reference image on close. Startup failure closes owned
resources. CPU predictions are never substituted for hardware output. A matching
UART control channel remains necessary for model/timer control even with USER1
inference. Real webcam acceptance remains exclusively user initiated.

The normal demo supports both accepted images. Automatic selection prefers the
physically accepted compact JTAG configuration; explicit UART and pretrained
selections remain available. TinyEye coefficients are uploaded through UART,
read back, and checked against a synthetic hardware inference before camera startup.
The bounded scheduler now routes both model contracts with one transport owner,
fixed queue sizes (compact preparation capacity four), original frame/capture metadata, stale-result handling and
capture-timed binocular events. It does not claim simultaneous eye execution or live camera 50 Hz performance. Scheduler tests use synthetic frames and mocked
transports; their results are not hardware measurements.

The original capture, ROI geometry and display path are reused. A future live
test must check image quality, both-class performance, frame association and
blink timing. No webcam or live landmark session was run during development.

## Saved checkpoint at user-requested stop

Development and additional testing stopped after the current physical pipeline
trial. See [machine-readable results](CHECKPOINT_RESULTS.json). Both final
30-second offline pipeline trials completed and published all 1,500 pairs with
zero dropped, stale or communication-error results. JTAG measured 50.005849
pairs/s (p95 31.2734 ms); UART measured 49.998580 pairs/s (p95 30.0199 ms).
Under the strict rule that a measured rate below 50 fails the target, this UART
trial is a 50 Hz throughput failure despite completing all requested inputs.

Earlier trials include placement failures, scheduling stalls, stale results and
explicit drops. They have not been erased or relabeled. The final implementation
uses bounded asynchronous log writers so filesystem I/O does not hold inference
owners or the shared pipeline event lock. Replay fixtures are hash-checked and
preloaded, with fixed synthetic capture timestamps and actual dispatch timestamps
recorded separately. These are offline replay tests, not webcam measurements.

The earlier six-stage regression passed. The latest targeted suite passed 24
tests before the final startup-cleanup adjustment. A full regression rerun after
all final changes, the independent 10/25/30/60 Hz matrix, and real webcam
acceptance remain unfinished at the user's stop request. This commit is a saved
engineering checkpoint and does not declare final project acceptance.
