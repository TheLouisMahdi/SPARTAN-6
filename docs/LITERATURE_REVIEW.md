# Prior-art review of the proposed FPGA CNN paper

Search date: 2026-10-30. This is a targeted literature and novelty review, not an
exhaustive systematic review or a guarantee that no unindexed work exists.
No implementation, trained model, board configuration or research result was changed.

**Verdict:** The broad proposal, "deploy a pretrained CNN without retraining,
select numerical precision under accuracy/resource constraints, and improve FPGA
memory mapping," has substantial prior art. It should not be presented as a new
method on the strength of the current implementation. The exact combination of
open-closed-eye-0001, XC6SLX9, this RTL controller and binocular timer was not
identified in the examined scholarly sources. That observation does not establish novelty.

## Search scope and evidence quality

Searches covered indexed publisher pages at IEEE Xplore, ACM Digital Library,
Springer Nature and Elsevier/ScienceDirect, and author-deposited papers at arXiv
and university repositories. DBLP and FINN's official publications page were
used as discovery aids. There was no authenticated search inside subscription
Scopus or Web of Science. Paywalled/inaccessible full texts were not claimed as read.

Queries included:

```text
FPGA pretrained CNN analytical fixed point word length optimization no retraining memory banking
FPGA convolutional neural networks wordlength optimization analytical error bounds
site:ieeexplore.ieee.org CNN FPGA word length optimization precision
site:dl.acm.org FPGA CNN memory banking mixed precision design space exploration
Angel-Eye quantization retraining
neural network quantization interval FPGA retraining
Evolutionary Bin Packing for Memory-Efficient Dataflow Inference Acceleration on FPGA
Memory-Efficient Dataflow Inference for Deep CNNs on FPGA
FPGA CNN eye state blink detection Spartan 6 pretrained
Spartan-6 eye state neural network
open-closed-eye-0001 FPGA
open-closed-eye-0001 RTL Verilog Spartan
open-closed-eye-0001 XC6SLX9
CNN blink UART FPGA paper
A numerical workflow for post-training fixed-point CNN deployment on FPGA
```

False-positive hits were excluded: EEG-based eye-state classification is not
camera eye-state classification; cellular neural networks are not convolutional
neural networks; the word "Spartan" in an application name does not prove an FPGA
implementation. An OpenVINO release note containing both a model name and generic
FPGA support is not evidence that this particular model was deployed on that FPGA.

## Closest methods and implementation precedents

| Reference and verified source | Relevant existing contribution | Difference and implication for this project | Evidence inspected |
|---|---|---|---|
| **Data wordlength optimization for FPGA synthesis**, SIPS 2005, [IEEE / DOI 10.1109/SIPS.2005.1579941](https://ieeexplore.ieee.org/document/1579941/) | Couples wordlength selection to hardware synthesis under an accuracy constraint using analytical error propagation and operator costs. | General DSP rather than CNNs. Accuracy-constrained analytical wordlength selection itself is longstanding. | Indexed publisher abstract/metadata; direct page required bot verification. |
| **CascadeCNN**, FPL 2018, Kouris, Venieris and Bouganis, [author paper](https://arxiv.org/abs/1807.05053), [conference DOI](https://doi.org/10.1109/FPL.2018.00034) | Generates a low/high precision FPGA cascade from a pretrained CNN without retraining, using confidence evaluation and an error-tolerance/resource-aware design flow. | Not the same single-core design. The author paper uses a small labeled evaluation set; "no retraining/training data" must not be mistaken for "no data whatsoever." | Author abstract and author-PDF method excerpts. |
| **Fixed-point Quantization of Convolutional Neural Networks for Quantized Inference on Embedded Platforms**, Goyal et al., 2021, [author preprint](https://arxiv.org/abs/2102.02147) | Selects per-layer weights/biases/activation bitwidths and fractional offsets while limiting accuracy loss, without retraining. | Very close to the proposed numerical-precision question. Introduction explicitly excludes implementation on embedded hardware from its scope, so it is a numerical baseline, not a measured FPGA baseline. | Author PDF, particularly introduction pp. 1-2 and conclusions/future work pp. 21-22; not a claim that every page was reviewed. |
| **Low Precision Floating-point Arithmetic for High Performance FPGA-based CNN Acceleration**, Wu et al., [2020 author preprint](https://arxiv.org/abs/2003.03852), [corresponding ACM record](https://doi.org/10.1145/3474597) | Low-precision pretrained CNN inference and FPGA arithmetic without retraining. | Uses an 8-bit floating-point representation rather than our INT16 weights/INT32 activations. No-retraining FPGA deployment is already established. | Author abstract and indexed journal abstract. |
| **Evolutionary Bin Packing for Memory-Efficient Dataflow Inference Acceleration on FPGA**, GECCO 2020, Kroes et al., [author paper](https://arxiv.org/abs/2003.12449), [ACM DOI](https://doi.org/10.1145/3377930.3389808) | Optimizes mapping of CNN parameter memories to physical FPGA memory using constrained bin packing. | Considerably more general than splitting our ROM into three banks. Avoiding BRAM shape/depth waste is not, by itself, a new research claim. | Author abstract and primary conference DOI metadata. |
| **Memory-Efficient Dataflow Inference for Deep CNNs on FPGA**, FPT 2020, Petrica et al., [author paper](https://arxiv.org/abs/2011.07317) | Frequency-compensated memory packing trades memory utilization against throughput and enables deployment on smaller FPGA devices. | Different architecture and memory clocking, but closely related to the proposed memory/performance research theme. | Author abstract; venue corroborated by the [official FINN publications list](https://xilinx.github.io/finn/publications.html). |
| **On the RTL Implementation of FINN Matrix Vector Compute Unit**, Alam et al., 2022, [author paper](https://arxiv.org/abs/2201.11409) | Studies RTL implementations across design parameters and compares resources/timing against HLS. | Writing an RTL backend and measuring parallelism/resource tradeoffs is not a new general idea. Its hardware/workloads are not directly comparable to our whole system. | Author abstract; not a full-paper performance reproduction. |
| **SIRA: Scaled-Integer Range Analysis for Optimizing FPGA Dataflow Neural Network Accelerators**, Umuroglu et al., 2025, [author preprint](https://arxiv.org/abs/2508.21493), [full text, section 4.2](https://arxiv.org/html/2508.21493v1#S4.SS2) | Propagates scaled-integer intervals and exploits constant weights to reduce accumulator width and downstream operator resources safely. | Especially close to the proposed range/overflow argument. It starts with quantized networks and does more than our current conservative bounds. Listed as submitted to ACM TRETS in the inspected preprint; acceptance was not independently established. | Full-text range-analysis and accumulator-minimization sections, plus abstract. |
| **Accumulator-Aware Post-Training Quantization for Large Language Models**, Colbert et al., first submitted 2024, revised 2025, [author paper](https://arxiv.org/abs/2409.17092), [full text](https://arxiv.org/html/2409.17092v2) | AXE adds accumulator-overflow avoidance guarantees to PTQ, including multi-stage accumulation. | The inspected revision evaluates LLMs, not our eye CNN. It is methodological prior art for claiming overflow-aware PTQ, not a direct camera/FPGA benchmark. | Latest abstract and full-text introductory/method context. |
| **A numerical workflow for post-training fixed-point CNN deployment on FPGA**, Cruz, Robles and Bravo, Array 30 (2026), 100885, [publisher page](https://www.sciencedirect.com/science/article/pii/S2590005626002080), [DOI](https://doi.org/10.1016/j.array.2026.100885) | Reports a reproducible MATLAB/Vitis HLS fixed-point CNN workflow, co-simulation agreement and accuracy/latency/resource evaluation on Artix-7 using MNIST. | Recent and close to the general deployment/evidence-package framing. Different toolchain, target and application; abstract does not establish that its method is identical to ours. | Indexed publisher abstract/highlights. Full article and university repository access were unavailable; internal methods were not verified. |

## Eye/blink application precedents

| Reference | Verified scope | Implication |
|---|---|---|
| **Design and implementation of a real time and train less eye state recognition system**, Dehnavi and Eshghi, EURASIP JASP, 2012, [publisher full text](https://link.springer.com/article/10.1186/1687-6180-2012-30) | FPGA eye-state recognition based on image projections, without a trained neural classifier. | The application of FPGA eye-state recognition without a new training phase has a long history. This is not a CNN quantization baseline. |
| **E-BTS: A low-power Event-driven Blink Tracking System with hardware-software co-optimized design for real-time driver drowsiness detection**, Choi et al., Alexandria Engineering Journal 128 (2025), 867-877, [DOI](https://doi.org/10.1016/j.aej.2025.07.020), [author university copy](https://scholarworks.bwise.kr/cau/bitstream/2019.sw.cau/87405/1/E-BTS%20A%20low-power%20Event-driven%20Blink%20Tracking%20System%20with%20hardware-software%20co-optimized%20design%20for%20real-time%20driver%20drowsiness%20detection.pdf) | Event-camera blink tracking with dedicated hardware/software co-design and an FPGA-prototyped processor; it does not depend on a CNN. | Blink tracking and whole-system hardware/software co-design are not new applications. Different sensor/algorithm means its published FPS and accuracy cannot be compared directly with our RGB pipeline. |

The official [open-closed-eye-0001 documentation](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/open-closed-eye-0001)
is model provenance, not our contribution. Its published MRL result must not be
reported as measured accuracy of our hardware or as independent validation on unseen people.

## Consequences for our claim

| Proposed claim | Review verdict |
|---|---|
| First pretrained CNN on FPGA without retraining | Unsupported; clear counterexamples. |
| First per-layer fixed-point selection with bounded accuracy loss | Unsupported; Goyal et al. and broader wordlength literature overlap. |
| First analytical range/overflow-based FPGA optimization | Unsupported; SIRA and earlier range-analysis work overlap. |
| First memory mapping/banking to make CNNs fit a small FPGA | Unsupported; memory-packing literature overlaps. |
| First FPGA-based eye/blink monitoring | Unsupported; explicit application precedents. |
| First exact model/board/controller combination | No exact scholarly match identified, but absence from these searches is not proof; a new combination alone is weak methodological novelty. |
| Reproducible implementation and controlled engineering evaluation on XC6SLX9 | A defensible description of our work, subject to completing board/dataset measurements; not a claim of a new general algorithm. |

The current exporter uses **fixed** INT16 coefficient, INT32 activation and INT48
accumulator widths. It selects fractional scaling from conservative bounds; it
does not yet implement a wordlength optimizer. Its range bounds prevent numerical
overflow under the specified contract; they do not prove that every quantized
decision matches the FP32 model. The seven passing fixtures are not an accuracy study.

The CNN is pretrained and AEEE is not applied in its current deployment path.
The old TinyEye/AEEE training protocol cannot simply be relabeled as the
evaluation protocol for the new pretrained network.

## Recommended decision

Keep the current RTL implementation as the BSc engineering deliverable. Do not
describe the broad precision/memory/no-retraining theme as an established original
contribution, and do not promise a publication from the existing evidence alone.

For a research extension, first define a measurable improvement over a relevant
baseline and evaluate it with the same model, data, numerical contract, clock,
device and communication boundary. SIRA-style range minimization, a conventional
layerwise PTQ baseline and ordinary BRAM packing should be comparison methods,
not renamed contributions. Not all cited methods target the same computation
model or device; their published headline numbers are not apples-to-apples baselines.

A candidate narrower study is the relationship between compute/memory choices,
communication-induced sampling gaps and bilateral blink event/duration errors
under an explicit transport budget. This review does **not** establish that
candidate as novel. Evidence of a materially better method or a useful, carefully
bounded empirical result would still be needed before selecting a paper claim.

Confidence is high that the broad component claims have prior art. Confidence
that no exact end-to-end implementation exists is limited by access and indexing.
