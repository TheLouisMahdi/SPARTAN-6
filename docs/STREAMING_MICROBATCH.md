# Offline streaming micro-batch feasibility

The current firmware remains the verified pretrained P2 implementation. The
experimental codec is a reference format, **not supported by that firmware**.
No camera is accessed by this evaluation. Existing capture functionality remains.

## Reproduction

Use the project's Python 3.11 environment and registered, hash-verified CEW data:

```powershell
assets/runtime/Scripts/python.exe -B research_ext/streaming_evaluation.py assets/runs/microbatch_NEW
assets/runtime/Scripts/python.exe -B -m unittest research_ext.test_stream_codec
# Contacts only the positively identified board; firmware must already be verified:
assets/runtime/Scripts/python.exe -B research_ext/streaming_evaluation.py assets/runs/microbatch_NEW --physical
assets/runtime/Scripts/python.exe -B research_ext/streaming_evaluation.py assets/runs/microbatch_NEW --full-cew
.\test_all.bat
```

The physical runner verifies the pinned model fingerprint and the registered
CP210x serial identity. It never programs hardware. Volatile JTAG restoration of
the separately verified baseline is a distinct, recorded operation.
For a complete local report, retain the read-only GitHub languages response as
`github_languages.json`, the memory-gate synthesis evidence and the full CEW
result in the experiment directory, then run the evaluator with `--report`.

## Architecture and ownership

The offline producer runs independently of the serial consumer. Its queue holds
at most eight pairs plus one pair in the consumer. Paced overload rejects new
samples explicitly; unpaced saturation uses bounded backpressure and completes
all samples. One thread owns the UART. Logical batches 1/2/4/8 roll immediately;
the host does not wait for a complete group or allocate FPGA batch buffers.

Source identities start at 0xfffffff0 to exercise uint32 wrap. A monotonically
increasing host index disambiguates order. Original synthetic capture times and
batch IDs remain in the host journal: legacy packets carry packet sequence IDs,
not these frame metadata. There is no claim of a new hardware result stream.
Rejected indices are emitted before the next accepted result, with a final tail
flush. This needs no unbounded in-memory rejection list. Every arrived index has
one terminal journal record. Completed results are released immediately.

The finite timing model explores periods 10/16.67/20/33.33/40/100 ms and logical
batches 1/2/4/8. Its deadline is 20 ms and its total capacity is nine including the
active pair. A complete four-frame group at 50 Hz still takes 60 ms to fill;
deadline flushing does not make that full group available earlier. The model
serializes phases and does not assume transfer/compute overlap.

## Experimental reference format

All integers are little endian. Header `<BBBBIQIIHH` contains version, eye,
codec, key flag, frame ID, capture ns, batch ID, reference ID, decoded length,
encoded length. A CRC32 follows the payload and covers header plus payload.
Overhead is 32 bytes per eye. This is **not a protocol negotiation**.

Codecs are RAW=0, zero/literal RLE=1, previous-same-eye XOR plus zero/literal
RLE=2, and repeated-byte/literal hybrid=3. A token's low seven bits specify
length minus one; bit seven selects a run. Hybrid runs include one repeated
value byte; zero runs imply zero. Literals contain their bytes. Counts are
bounded to 128. Decoded size must be exactly 3072 before reference commit.

An ineffective candidate falls back to RAW. Independent records are explicit
keys; deltas require a matching, immediately preceding same-eye reference.
Duplicates and older identities are rejected with modular ordering. Any invalid
record clears references; explicit independent keys restore synchronization.
These are reference-codec guarantees, not implemented hardware guarantees.

CEW consists of independent photographs and is tested as static keys, never as
temporal video. Synthetic identical, sparse changes, all-pixel increments,
translation, alternating states, random, uniform and maximum-difference sources
are reported separately. Their compression is not real-world video performance.

## Memory gate and decision

The routed baseline uses 29/32 RAMB16BWER, 1318/1430 slices, 4204 LUTs, 2745
registers and 10 DSPs. Its minimum period is 19.431 ns, slack +0.569 ns at 50 MHz.
CNN activation A is overwritten by layer outputs; it is not persistent reference
storage. The 2048-byte packet buffer is owned by the receiver until controller
acceptance and cannot simultaneously receive another packet.

The investigated overlapped architecture uses three explicitly banked 2048x8
reference memories and one 3072-byte transactional decoded staging eye. ISE 14.7
XST actually inferred **4 RAMB16BWER + 1 RAMB8BWER**, with 21 LUTs and three
bank registers for the memory probe alone. Added to the baseline, the four full
blocks already exceed the device's 32-block limit. Decoder, metadata and protocol
logic would add more resources. It fails the memory gate before integration or
place-and-route, so no candidate bitstream is produced or programmed.

The probe is retained only in the ignored experiment directory, where it was
used for synthesis and memory simulation; it is not unused production Verilog.
This is a rejection of this overlap/staging architecture, not a mathematical
proof that every possible serialized decoder is impossible. A two-pass decoder
using the existing packet buffer for small compressed records could avoid full
staging, but needs a separate transactional protocol, resource/timing/physical
validation and a different handling path for oversized RAW keys. It is not
implemented or counted as a performance gain here. External SDRAM is not added.

Intra-frame coding saves little on CEW. Temporal zero coding helps identical and
sparse synthetic changes, but full-pixel changes, motion and random inputs fall
back to RAW. The selected overlap design fails the resource gate, so the new
protocol and production RTL are deliberately not installed. RAW remains selected.

## Physical evidence and limits

See `assets/runs/microbatch_20261106_01/` for baseline/compression/model/streaming/
comparison JSON, per-run arrival and ordered result journals, ISE memory-gate
sources/logs, rollback and final HTML. Historical 4846-eye physical equivalence
evidence and the baseline bitstream are preserved.

Legacy RAW uses 6388 measured bidirectional wire bytes per pair (including ACKs
and framing). The conservative 100000-byte/s aggregate budget gives 15.65 pairs/s
before processing and host/USB delay. UART has independent TX/RX directions;
the input-only 100000/6252 bound is 16.00 pairs/s. Neither permits 50 Hz. The
measured full transaction and bounded-load results, rather than source emission
rate or best-case codec ratio, determine actual throughput.

Only configuration A is implemented physically. B/C/D/E/F are unavailable; no
before/after improvement or new 50 Hz capability is claimed. Python language
share is secondary; no language padding, model change or test removal is used.

### Recorded outcome, 2026-11-06

The 57 legacy physical trials completed 1541 pairs with exact fixed-point logits.
A fresh full CEW pass checked all 4846 eyes (1192 closed left, 1192 closed right,
1231 open left, 1231 open right), also with zero integer mismatches. Normal trial
and full-CEW CRC, timeout and sequence-error counters were zero. The separate
initial configuration diagnostic and volatile baseline restoration are retained.

At an independent 50 Hz source, 3000 arrived, 469 were accepted/completed and
2531 were explicitly rejected. Queue occupancy stayed bounded at eight waiting
pairs. The run lasted 61.214 s including drain, completed 7.662 pairs/s, and had
mean upload 104.519 ms, mean RTT 130.244 ms and p95 end-to-end 1212.000 ms.
Each pair still used 735478 kernel cycles and 6388 actual wire bytes. Unpaced
trials completed every submitted pair with backpressure, at 7.469–7.793 pairs/s.
These saturation rates are not a verified no-rejection source frequency.

All 125 Python tests ran (124 passed; one pre-existing optional training skip),
and all five original RTL stages passed. The standalone memory probe also passed
Icarus simulation. No new integrated ISE route was attempted after the memory
gate failed. The original BIT and all protected FPGA sources remain unchanged.

GitHub's read-only language response reports Python 86.173% for the published
baseline. Local source-byte estimates are separately recorded and do not meet
the below-80% target. Production RTL added: zero. The outcome is
`BASELINE_PRESERVED_NO_GAIN`; experimental sources remain uncommitted and no push
was performed. No completed-optimization commit is appropriate for a rejected
hardware candidate.
