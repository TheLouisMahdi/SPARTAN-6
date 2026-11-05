# Current pretrained architecture

The default application and `run.cmd build` now target the official pretrained
`open-closed-eye-0001` network, not TinyEye. Its complete layer graph, INT16/INT32/INT48
arithmetic, UART framing and CPU/FPGA partition are in [PRETRAINED_DEPLOYMENT.md](PRETRAINED_DEPLOYMENT.md).
The TinyEye implementation described below remains available with `--network tiny`.
It is not tensor-compatible with the pretrained model and its old measurements do not qualify the new core.

The canonical top is an integration module. It owns the power-on reset counter,
connects UART/packet transport, selects the CNN with elaboration-time parameters,
and connects the existing monitoring and display blocks. `command_controller.v`
owns the command FSM, synchronous payload reads, contiguous input chunks, reply
buffer and CNN start/done sequencing. No clocked boundary or pipeline stage was
added by this extraction. The numeric CNN datapaths and generated ROMs are unchanged.

```text
eye_top
  receiver / transmitter                 uart_rx / uart_tx
  packets_in / packets_out               packet_rx / packet_tx
  commands                               command_controller
  pretrained_core.cnn                    omz_cnn (official PRETRAINED=1, P2)
  buttons / activity / latency / display  existing board modules
  left_events / right_events / timing    blink_event / binocular_timer
```

Explicit `PRETRAINED=0` research builds select `serial_core.cnn` (`eye_cnn`)
or `parallel_core.cnn` (`eye_cnn_parallel`). Those host commands and tests remain
supported. Source files stay in the existing flat RTL directory so the ISE and
simulation workflows share one source set. Input loading remains in the command
FSM because its byte reads and writes share the packet RAM handshake.

# Retained TinyEye P2 RTL and P1 comparison

Historical TinyEye integration: `assets/runs/rtl_integration_20261030`. With `PRETRAINED=0`, activation memories
infer BRAM and `model_loaded` starts false. Embedded storage is zeroed and carries no trained model.
Runtime UART loading is the only activation path. The model shape is retained as a deployment contract;
the rejected checkpoint is not reused. No HLS is used.

The current P2 plus binocular timer passes ISE implementation at 50 MHz: 1375/1430 slices,
4580 LUTs, 3118 slice registers, 2 DSP48A1, 3 RAMB16BWER and 2 RAMB8BWER.
Reported Fmax is 51.666 MHz. These are tool results, not a new physical-board qualification.

Target: ALINX AX309, XC6SLX9-2FTG256C, oscillator/constraint 50MHz, 20ns at T8.
`board_pins.ucf`, UART version 1, CRC16, method IDs, fingerprint format and model image remain unchanged.

```text
PC video -> MediaPipe landmarks -> canonical 32x16 eye -> encoder on PC
 -> UART 115200 8N1 -> packet_rx/CRC -> input activation RAM
 -> Conv1 / ReLU+Q24 / MaxPool -> Conv2 / ReLU+Q24 / MaxPool
 -> FC1 / ReLU+Q24 -> FC2 INT32 logits -> packet_tx/CRC
 -> PC metrics; capture-timed FPGA binocular timer and existing LEDs/display
```

This is FPGA CNN inference, not a complete FPGA camera/MediaPipe/AEEE vision pipeline.
The PC owns decode, ROI, encoding and offline statistical analysis. Left/right eyes run sequentially.

| Operation | Logical output C,H,W | Actual implementation |
|---|---|---|
| input | 1,16,32 | U8 magnitude of affine INT8, zero point -128 |
| valid Conv3x3 1->4 | 4,14,30 | signed INT8 weights, INT32 accumulator |
| ReLU/Q24, floor MaxPool2 | 4,7,15 | four cells per pooled output; scalar max |
| valid Conv3x3 4->8 | 8,5,13 | unused final odd row/column need not be computed |
| ReLU/Q24, floor MaxPool2 | 8,2,6 | flatten C,H,W to 96 |
| FC96->8, ReLU/Q24 | 8 | same dot-product arithmetic |
| FC8->2 | 2 | two INT32 logits, ties OPEN |

1130 trainable values comprise 1108 INT8 weights and 22 INT32 biases. Three Q24 multipliers
give a 1208-byte model image. Nominal full-layer MAC count is 34624 per eye; fused floor pooling
omits unused Conv2 borders, so the RTL commits 29728 useful products per eye. Neither number
is a measured wall-clock throughput.

## Modules and scheduling

`eye_top.v` selects `PE_COUNT`, default 2. P1 continues to instantiate untouched `eye_cnn.v`.
P2 selects `eye_cnn_parallel.v`. Unsupported PE counts fail elaboration rather than silently
falling back. P4 is not implemented/claimed. The older CNN-focused P2 used 1238/1430 slices;
the current integrated timer build uses 1375/1430 slices.

P1 uses one `mac_int8`. P2 has two physically distinct instances; output channels (0,1), (2,3),
etc. share one activation operand and read different weights from replicated model memories.
Both update independent INT32 accumulators in DOT_MAC. DOT_REQ/WAIT provide synchronous-read
latency, so this is not two new committed products on every FPGA clock. Bias reads are parallel.
One `cnn_addresses` instance supplies the first channel's weight address. The other channel
adds the fixed layer stride (9/36/96/8), while both MACs receive the same activation address.
Each lane has its own unchanged `requantize_q24`; completion is latched independently because
nonpositive accumulators complete earlier. Pooling maxima are per lane. Two output writes are
serialized through the existing activation write port; feature addresses preserve C,H,W order.

| Memory | P1 ports/use | P2 ports/use |
|---|---|---|
| model_store 2048x8 | host write + one synchronous read | two copies; broadcast host write, one independent read per MAC |
| activation A/B 512x8 each | synchronous BRAM; host/writeback + read | same BRAM banks; broadcast one read operand; serialize writeback |
| packet RX | retained UART packet buffer | unchanged |

Normal debug addresses remain: model at 0..1207, retained FC1 bank B at 1280..1535
(the existing low-byte addressing exposes 256 bytes), retained pool2 bank A at 1536..2047.
Diagnostic-only builds in the evidence run expose snapshots in unused model padding; they
are not deliverable performance variants or changes to the retained packet ABI.

## Arithmetic, verification and limits

`mac_int8`: nonnegative U8 activation times signed INT8 weight produces signed 17-bit product,
sign-extended into an INT32 sum. `model_image` rejects coefficient sets whose worst-case
accumulator bound exceeds INT32. No runtime overflow/stall counter is claimed.
`requantize_q24`: ReLU, serial unsigned shift/add product, add 2^23, shift right 24,
clip to 0..255. FC2 bypasses requantization. The Python integer reference is unchanged.

The self-checking bench loads real 1208-byte synthetic images, covers four encoders, six input
patterns and three arithmetic modes, compares every committed Pool1/Pool2/FC1 activation,
both final logits, reset while busy, done pulse, model boundary and debug reads.
It counts 29728 P1 product commits and 14864 simultaneous two-lane commits per eye.

Synthesis/place-route and RTL simulation are distinct evidence. Earlier P2 candidates passed
simulation but failed on the physical board. The shared-address implementation passes the
72-case simulation and 72-pair board suite, including exact retained layer readback.
Those physical-board results belong to the earlier memory implementation. The current BRAM integration
passes 72 layer cases and pin-level top/UART simulation, but requires a new board check.
The earlier mismatch and diagnostic evidence remain in the run; a vendor synthesis defect
has not been established. See the [validation report](VALIDATION_REPORT.md) for accepted evidence.

Report compute milliseconds as `cycles / 50000000 * 1000`. Static timing Fmax is derived
from the reported minimum internal clock period; the actual board still runs at 50MHz.
UART, USB scheduling, decode, MediaPipe and CPU encoding are separate. A faster compute core
does not establish an equal end-to-end speedup. UART wire floors are theoretical, not measured
latencies. No power meter or CPU/GPU workload comparison has been performed.

## Original project requirements and implementation

| Requirement | Concrete implementation |
|---|---|
| RTL CNN accelerator | `fpga/rtl/eye_cnn_parallel.v`, selected by `eye_top.v` |
| Convolution | `cnn_addresses.v` and two `mac_int8.v` instances; independent output-channel accumulators |
| ReLU | `requantize_q24.v`, nonpositive sums map to zero magnitude |
| Pooling | per-lane maximum over four computed convolution cells, stored in activation RAM |
| Parallel computation | two weight reads, one broadcast activation, two simultaneous useful MAC commits |
| FC/inference | the same integer datapath executes FC96->8 and FC8->2; no training runs on FPGA |
| Host integration | explicit model slot, laptop ROI/encoding, CRC-framed UART, optional independent CPU verification |
| Evidence | per-layer 72-case suite, pin-level protocol suite and ISE implementation reports |

ReLU and pooling are logical layers even though they are fused with requantization/writeback to avoid
unnecessary full feature-map storage. Left/right eyes remain sequential on the board; parallel lanes
work on channels within each eye. Laptop workers process the two independent eyes concurrently in CPU mode.

ISE emits a 9K BRAM initialization warning. These activation RAMs have no required initial contents:
all consumed inputs/intermediate activations are written before being read. The self-checking simulation
starts their arrays undefined and checks all layer outputs. Configuration-time initialization of 9K
RAM is not relied upon; see [AMD XCN11014](https://docs.amd.com/v/u/en-US/xcn11014).
