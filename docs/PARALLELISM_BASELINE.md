# Selected baseline for the FPGA CNN parallelism study

Decision date: 2026-10-30. This document selects a research baseline; it does not
change RTL, model weights, the host application or the board configuration.

## Primary paper

**Chen Zhang et al., Optimizing FPGA-based Accelerator Design for Deep
Convolutional Neural Networks, ACM/SIGDA FPGA 2015.**

[Publication](https://doi.org/10.1145/2684746.2689060) |
[Author PDF](https://chenzhangsjtu.github.io/files/2015-FPGA.pdf)

Select this paper as the architectural starting point, not as the newest work.
It directly studies convolution loop parallelism, reuse and the balance between
computation and memory bandwidth. Its Tm and Tn parameters describe output-channel
and input-channel parallelism. Read sections 3.2, 3.3 and 3.4 first, followed by
3.5 and the implementation section. Its VC707 floating-point results are not a
performance baseline that can be compared directly with our fixed-point XC6SLX9.
Reimplement an explicitly documented adaptation on our device for a fair comparison.

## Citation and recency comparison

Counts below are OpenAlex `cited_by_count`, retrieved on 2026-10-30 for each
specified DOI record. They are not Google Scholar counts, do not sum separate
preprint records, and do not establish a worldwide ranking. Newness and citation
count are different selection criteria. This is a targeted candidate comparison,
not an exhaustive systematic review.

| Paper | Publication year | Citations | Role |
|---|---:|---:|---|
| [Zhang: Optimizing FPGA-based Accelerator Design](https://doi.org/10.1145/2684746.2689060) | 2015 | 2159 | Primary architectural baseline; highest count among these candidates |
| [Qiu: Going Deeper with Embedded FPGA Platform](https://doi.org/10.1145/2847263.2847265) | 2016 | 1268 | Embedded CNN deployment comparator |
| [FINN](https://doi.org/10.1145/3020078.3021744) | 2017 | 1189 | Binary-network accelerator comparator |
| [Shen: Maximizing CNN Accelerator Efficiency Through Resource Partitioning](https://doi.org/10.1145/3079856.3080221) | 2017 | 292 | Prior art to check before proposing resource partitioning |
| [FINN-R](https://doi.org/10.1145/3242897) | 2018 | 450 | Quantized inference and design-space exploration comparator |
| [NN2FPGA](https://doi.org/10.1109/TCAD.2024.3507570) | 2025 issue; online 2024 | 9 | Modern resource-allocation comparator |
| [HMSA](https://doi.org/10.1145/3759458) | 2025 | 1 | Recent mixed-precision systolic-array comparator |
| [An efficient open-source design and implementation framework for non-quantized CNNs on FPGAs](https://doi.org/10.1016/j.vlsi.2025.102625) | 2026 issue; online 2025 | 1 | Recent full-precision framework comparator |

Raw counts, retrieval times and source API URLs:
local record `assets/runs/literature_20261030/parallelism_citations.json`
(historical evidence, not distributed with source).

NN2FPGA uses HLS static dataflow and binary integer programming to allocate
resources across layers, with particular attention to skip connections. Its
[author repository record](https://iris.polito.it/handle/11583/2994852) explains
the method. Use it to check modern prior art, not as a drop-in implementation for
ISE/Spartan-6. It has relatively few indexed citations so far.

The [2026 Integration author paper](https://ntampouratzis.com/papers/integration2025.pdf)
is newer than our primary baseline, but its full-precision Darknet/Vitis workflow
has a different numerical and implementation scope from this project.

## Mapping to the pre-pipeline RTL baseline

These observations concern the saved pre-pipeline source and execution evidence,
not results reported by the selected papers. The subsequent pipeline correction
is documented in [PRETRAINED_DEPLOYMENT.md](PRETRAINED_DEPLOYMENT.md).

- `fpga/rtl/omz_cnn.v` has two output-channel MAC lanes sharing an activation.
  In the selected paper's terminology, its arithmetic parallelism corresponds
  approximately to Tm=2, Tn=1. This does not mean the complete architecture is a
  reproduction of that paper.
- The controller repeats `DOT_REQ -> DOT_WAIT -> DOT_MAC`. Within a dot product,
  each lane commits a MAC once every three clocks, rather than every clock.
- Saved P2 simulation reports 1,002,827 cycles per eye, or 20.05654 ms at 50 MHz.
  Seven inputs pass integer-reference comparisons; these are arithmetic checks,
  not a recognition-accuracy benchmark.
- Saved ISE implementation uses 10/16 DSP48A1, 29/32 RAMB16 and 1314/1430 slices.
  It meets 50 MHz. These are implementation reports, not board measurements.
  Adding lanes may fail because of memory ports or placement even when DSPs remain.

Evidence: `assets/runs/omz_20261030/arithmetic_p2_final/result.json`,
`assets/runs/omz_20261030/arithmetic_p2_final/simulation.log`, and
`assets/runs/omz_20261030/ise_p2_final/result.json`.

## Historical study proposal and remaining research

The pipelined P2 reference in step 1 was subsequently implemented and verified;
see the current validation summary. Further Tm/Tn exploration below remains a
research proposal, not the next repository-maintenance task or a measured result.

Research question: **with a fixed arithmetic and memory budget, which combination
of channel parallelism and memory scheduling minimizes inference latency?**

1. Preserve the current implementation and evidence as the engineering baseline.
   Establish a pipelined two-lane reference, targeting one new MAC per lane per
   clock in the steady state where dependencies and timing permit. Prove integer
   equivalence and run implementation before treating that target as achieved.
   Removing avoidable controller bubbles is established engineering, not novelty.
2. Compare output-channel and input-channel parallelism at equal MAC budgets,
   initially Tm=2/Tn=1 versus Tm=1/Tn=2. Account for odd channel counts, reduction
   logic, BRAM ports, data reuse and fill/drain overhead. Larger configurations
   are candidates only after demonstrating resource feasibility.
3. Keep model weights, preprocessing, integer arithmetic contract, clock target
   and UART settings unchanged. Report useful MACs/cycle, per-layer cycles,
   complete inference latency, achieved clock, LUT/FF/DSP/BRAM and exact integer
   agreement. Compare implemented results on the same device, not headline GOPS
   from unrelated devices or precisions.
4. Measure communication separately. At 1 Mbaud, 6144 input bytes for two eyes
   require at least 61.44 ms with 8N1 framing, excluding packet overhead. Two
   sequential P2 inferences add 40.11308 ms. Thus the current sequential protocol
   has a calculated lower bound of 101.55308 ms per pair before other work.
   This is not measured live FPS. Internal memory bandwidth and UART bandwidth
   must be modeled at their respective boundaries.
5. Check the proposed improvement against NN2FPGA and resource-partitioning work
   before finalizing originality. A new board/application combination, extra MAC
   lanes, pipelining, memory banking or per-layer allocation alone is insufficient
   evidence of a new method. See also [the prior-art review](LITERATURE_REVIEW.md).

The defensible objective now is a reproducible, resource-constrained RTL study.
A publication claim requires a specific methodological difference, controlled
comparisons and repeatable measured improvements. No speedup, novelty, accuracy
gain or publication outcome is asserted by this selection document.
