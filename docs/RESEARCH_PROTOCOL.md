# Public evidence protocol v1

The machine-readable freeze is [protocol.json](../research_ext/protocol.json).
Its SHA256 is embedded in every new public fit. Historical personal runs,
their validation scores and previously observed results are development only.
No personal face, webcam or private checkpoint is used for new public results.

## Questions and decisions fixed before testing

1. Does the RTL compute the exact specified INT8 network, including fused pooling?
2. Do two simultaneous useful MAC operations reduce compute cycles at a documented resource cost?
3. Does frozen AEEE v0.1 improve heldout macro-F1 over grayscale2 at the same 128-byte/eye budget?
4. How do descriptive ocular indicators relate to independently labeled public clips?

No positive answer, novelty or superiority is assumed. A negative AEEE result is retained.
No accuracy, energy-efficiency, fatigue diagnosis or employment-performance claim follows from synthesis.

## Data and leakage controls

`evidence prepare` accepts explicit locally acquired sources with rights/version/source metadata.
RT-BENE disagreement 0.5 is excluded. Internal classes are OPEN=0 and CLOSED=1.
RT-BENE publisher-discarded subject 6 is excluded. MRL 0=closed and 1=open are mapped explicitly.
All eyes, frames and sessions of an identity share one partition. Allocation uses seed 42,
60/20/20 subjects, with at least one validation and test subject. This is a newly declared
split, not a claim to reproduce the publisher's original folds. External MRL is test-only
when its registry entry has `external_test: true`.

Exact canonical pixel duplicates, inconsistent duplicate labels, reused sample IDs,
cross-split subject/session/frame groups and duplicate registered videos are rejected.
`identity_map` must merge identities known to occur under different publisher IDs.
Unreported cross-dataset identities cannot be proven distinct; this limitation stays explicit.
Missing source files or rights approval cause an actionable error, never a synthetic replacement.

## Equal budgets and analysis

TinyEye architecture, optimizer, 12 epochs, batch size 1024, learning rate 0.001,
argmax rule and seeds **42,43,44,45,46** are common to all encoders.
Epoch selection uses validation macro-F1, earliest epoch on a tie. Quantization calibration
uses TRAIN only. No public threshold tuning or test-based model selection occurs.
All seeds are reported; no best-seed result is substituted.

Primary endpoint: the mean over five seeds of pooled heldout macro-F1(AEEE) minus
macro-F1(grayscale2). The paired percentile bootstrap resamples subjects 2000 times
using seed 611; each resample pools subject confusion matrices before computing F1.
The same sampled subjects are used for both methods and all seeds. Seeds are not
independent people. With fewer than two test subjects the interval is not computed;
with fewer than ten an explicit small-sample warning accompanies it.

Secondary outputs: CLOSED recall/F1, balanced accuracy, per-subject metrics,
grayscale8 and Sobel2 comparisons, encoder time, payload size and separate hardware timing.
`comparison.csv` from the retained training engine is supplemented by
`public_comparison.csv`, `metrics_per_subject.csv`, `paired_results.json` and honest conclusions.

## Controls and robustness

`--ablations` trains separate models for removal of central weighting, adaptive threshold,
continuity, orientation gate, polarity and strong/weak split. Each control has its own version
and retains a 128-byte two-bit payload. Removing strong split collapses all retained edges
to symbol 1; unused symbols are documented rather than implying equal information content.
The optional polarity-order control swaps symbols 2 and 3 and trains another model.
AEEE v0.1 itself is unchanged. Its ordinal mapping of signed categories to 170/255 is arbitrary;
no category ordering is selected by looking at external-test performance.

`evidence perturb` applies the frozen lighting, contrast, blur and top-row occlusion controls
only after model selection freezes. They are controlled engineering stressors, not proof of
real-world generalization. MRL IR differs from RGB in sensor and crop domain. Source metadata
retain glasses, lighting and sensor fields. Pose/eyeglasses annotation studies require actual
publisher metadata; no attribute labels are inferred or fabricated.

## Temporal and sleepiness scope

Replay uses media timestamps, not inference wall time. PTS mode probes presentation timestamps
with ffprobe and rejects duplicate/backward millisecond values or decoder-count mismatches.
Existing explicit CFR mode remains available. No face/crop means UNAVAILABLE.

Ocular summaries use nonoverlapping 30s/60s windows and the complete clip, with partial windows
marked. Binary CLOSED fraction is distinct from PERCLOS80. Calibrated aperture provenance is
required for PERCLOS80; otherwise null. Short blinks are included in the aperture fraction.
OPEN->CLOSED->OPEN must complete without an invalid interval or gap over 250ms.
Blink thresholds 30..600ms and long-closure threshold 500ms are experimental, not medical.
Rates are null below 30s of valid observation. Per-eye counts are not added as human blinks.

UTA-RLDD labels 0/5/10 describe the predominant whole clip. `features` aggregates whole-clip
valid-time CLOSED fractions and per-eye blink rates. Insufficient clips are excluded with a reason.
`sleepiness` uses a fixed train-standardized nearest-centroid baseline, fitting means, scales
and centroids on TRAIN only. Validation/test report three-class confusion matrices, macro-F1
and balanced accuracy. This internal subject split is not a published five-fold RLDD benchmark.
DROZY KSS is retained as source metadata; KSS/PVT correlation remains optional and unrun.
Yawning requires independent mouth data and is not implemented by the eye CNN.

## Completion rule

Actual external accuracy, ablation, robustness and sleepiness findings remain
`NOT_RUN_DATA_UNAVAILABLE` until lawful files are locally registered and the real protocol runs.
Synthetic arrays/videos test software and arithmetic only. Hardware results require retained
tool reports and board transcripts. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md).
