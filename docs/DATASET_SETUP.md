# Lawful local data setup

Datasets, registries and outputs live under ignored `assets/` or an explicitly
selected external directory. The current pretrained CEW evaluation used separately
acquired local data; data acquisition is not performed implicitly by experiments.
Historical external research packages are not required to run the pretrained path.

## Pretrained CEW evaluation

Obtain CEW through its
[official research source](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html).
Keep access links, extracted images and local registries outside Git. For example,
extract authorized images under `assets/datasets/cew/raw/` and create
`raw/provenance.json` containing:

- `approved_local_use: true` only after verifying permission for the intended use;
- `source_url` equal to the official URL above;
- an actual `rights_note` and verified publisher `label_provenance`;
- `label_directories` mapping each verified class-folder path to an object with
  `label` (OPEN/CLOSED) and `source_type` (eye_patch/face_original/face_warped).

Do not guess labels or identities from filenames. Alternatively supply
`--annotations FILE.csv` with independently verified `source_path,label,source_type,subject_id`.
Paths in that annotation file are relative to the extracted source root.

```powershell
.\run.cmd dataset register-cew --input assets/datasets/cew/raw --output assets/datasets/cew/manifest.csv
.\run.cmd experiment cew --mode cpu --cpu-precision fixed --manifest assets/datasets/cew/manifest.csv --output assets/runs/cew_cpu_new
.\run.cmd experiment cew --mode compare --manifest assets/datasets/cew/manifest.csv --port COM3 --output assets/runs/cew_compare_new
```

Registration writes a new CSV and provenance beside it; existing files are never
overwritten. Replace COM3 with the known board port only for deliberate hardware
testing. Eye patches use the existing BGR resize directly; full-face sources
require MediaPipe ROI localization. CPU mode needs no board. These experiments
do not train weights. Corrected CEW figures are post-inspection results.

Authorized UTA-RLDD video experiments use the existing `schema_version: 1,
datasets: [...]` registry and `videos.csv` fields documented below. The pretrained
entrypoint is `run.cmd experiment video --registry PATH --video-id ID --mode cpu`;
add a fresh `--output` directory. No authorized video result is implied by this recipe.

## Optional TinyEye research: obtain and register sources

The remaining preparation, training, split and ablation recipes belong to the
explicit TinyEye research path. They are not prerequisites for pretrained inference.

Do: copy `research_ext/public_sources.example.json` to a local `sources.json`, edit paths/version,
record the actual permission/terms and set `approved_local_use` only after lawful acquisition.
Remove entries for datasets you do not have. This is for source provenance, not a download bypass.

| Source | Official origin | Local files and label scope |
|---|---|---|
| RT-BENE | [Zenodo](https://zenodo.org/records/3685316) | `rt_bene_subjects.csv`, selected `sNNN_blink_labels.csv`, `sNNN_noglasses_eyes.tar`; eye labels 0.0=open,1.0=blink,0.5=disagreement |
| MRL | [Publisher](https://mrl.cs.vsb.cz/eyedataset.html) | extracted original PNGs; filename fields subject,image,gender,glasses,eye,reflection,lighting,sensor; eye 0=closed,1=open |
| UTA-RLDD | [University dataset page](https://sites.google.com/view/utarldd/home) | authorized local videos plus `videos.csv`; clip labels 0=alert,5=low-vigilant,10=drowsy |
| DROZY | [University repository](https://orbi.uliege.be/handle/2268/191620) | original videos, timestamps and KSS annotation mapping; read the deposited license agreement first |

RT-BENE is CC BY-NC-SA 4.0: respect attribution, noncommercial and applicable sharing terms.
MRL download availability does not establish unrestricted redistribution rights.
UTA-RLDD restricts which participants' faces can appear in publications; this pipeline retains
metadata only by default. Access forms/login for other datasets are not bypassed.

RT-BENE uses the existing importer: crop 3 rows at each margin of 36x60 patches, grayscale,
AREA resize to 32x16. MRL uses deterministic centered 2:1 crop and AREA resize. Both are
documented domain adapters, not the live landmark crop. At least three development identities
with both eye classes in each resulting split are required; more are needed for useful inference.

## Stage 2: freeze and encode

Run from the SPARTAN-6 folder in PowerShell. Choose new output directories every time.

```powershell
.\run.cmd evidence prepare --registry D:\licensed_data\sources.json --output assets\data\public_v1
.\run.cmd evidence encode --data-dir assets\data\public_v1 --output assets\data\public_encoded_v1
.\run.cmd evidence fit --data-dir assets\data\public_v1 --encoded assets\data\public_encoded_v1 --output assets\runs\public_v1
.\run.cmd evidence evaluate --data-dir assets\data\public_v1 --encoded assets\data\public_encoded_v1 --output assets\runs\public_v1
```

It can: canonicalize, hash, reject leakage, train the same network at the frozen budget and
report all five seeds plus paired subject-bootstrap intervals. To run controls, add `--ablations`
to encode/fit/evaluate and use separate `public_ablation_*` directories. Test is reporting-only.

```powershell
.\run.cmd evidence perturb --data-dir assets\data\public_v1 --run-dir assets\runs\public_v1 --bundle assets\data\public_encoded_v1\bundle.json --output assets\runs\public_v1\robustness.csv
```

## Stage 3: register and replay videos

Add a UTA-RLDD or DROZY entry to the registry with the same mandatory fields as the example,
`root` pointing to the lawful local video directory and `videos_csv: "videos.csv"`.
`videos.csv` must have:

```text
subject_id,session_id,video_id,path,sha256,label_scope,source_label
```

`path` is relative to that dataset root. Set `label_scope` to `video`. Record original clip/KSS
labels from the dataset, never CNN predictions. `sha256` must match `Get-FileHash -Algorithm SHA256`.
The CSV itself is hashed for annotation provenance. Register at least three video subjects for
the fixed subject split. Keep the source registry frozen when replaying multiple clips.

```powershell
.\run.cmd evidence replay --registry D:\licensed_data\sources.json --video-id UTA_subject01_alert --output assets\runs\uta_replay01
```

This uses the existing headless MediaPipe replay and real PTS. No frames are saved. For public
CNN/FPGA results additionally select `--backend int8` or `--backend fpga` and explicitly provide
`--run-dir assets\runs\public_v1 --bundle assets\data\public_encoded_v1\bundle.json --method aeee2 --seed 42`.
Historical personal models are refused by this public wrapper. COM26 remains the retained FPGA default.

```powershell
.\run.cmd evidence features --replays assets\runs\uta_replay01 assets\runs\uta_replay02 assets\runs\uta_replay03 --output assets\runs\uta_features.csv
.\run.cmd evidence sleepiness --features assets\runs\uta_features.csv --output assets\runs\uta_sleepiness.json
```

Actual training needs all three sleepiness classes in TRAIN and heldout subjects. The example
command structure alone does not supply enough real clips. Outputs do not diagnose fatigue.
Pure ocular analysis can also read the reference package CSV with `evidence ocular --frames ... --output ...`.

## Stage 4: engineering verification and hardware

```powershell
.\assets\runtime\Scripts\python.exe -B -m unittest discover -s research_ext -p "test_*.py" -v
.\run.cmd rtl simulate --simulator iverilog --output assets\runs\new_rtl_check
.\run.cmd rtl build --pe-count 1 --output assets\runs\new_ise_p1
.\run.cmd rtl build --pe-count 2 --output assets\runs\new_ise_p2
```

The local Icarus binary is under `assets/tools/iverilog/ucrt64/bin`; provenance and upstream
licenses are retained there. ISE 14.7 builds XC6SLX9. The retained XSDB/hw_server is only cable
transport for ISE-generated SVF; Vivado is not used to synthesize Spartan-6.
The accepted P2 build passes physical-board parity; see [VALIDATION_REPORT.md](VALIDATION_REPORT.md).
Do not accept a newly modified variant based on synthesis alone.

```powershell
.\run.cmd rtl program --output assets\runs\new_ise_p1
.\run.cmd rtl board --port COM26 --output assets\runs\new_board_p1
```

Programming changes volatile FPGA configuration; the original `fpga/eye_top.bit` is never
overwritten by these commands. The board suite loads synthetic coefficients, validates both
eyes and layer readback, logs raw UART chunks and restores the prior model held in RAM.
The current `run.cmd build` defaults to canonical pretrained P2 under `fpga/`;
`run.cmd program` validates that canonical build unless `--build-dir` is supplied.
Live `host`/`fpga` default to pretrained; TinyEye requires explicit model selection.
Neither path silently selects an archived personal checkpoint.
The frozen public research commands and explicit archived replay interface are retained.
