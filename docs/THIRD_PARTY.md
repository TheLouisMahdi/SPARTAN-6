# Third-party assets and attribution

- **open-closed-eye-0001:** original Open Model Zoo/OpenVINO pretrained CNN;
  [publisher source](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/open-closed-eye-0001).
  The original license is retained at
  [assets/models/open-closed-eye-0001/LICENSE](../assets/models/open-closed-eye-0001/LICENSE).
  The ONNX is checksum pinned; exporting integer ROMs does not retrain the model.
- **MediaPipe:** facial landmarks and the supplied `face_landmarker.task` asset.
  [Upstream Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
  Retain upstream package/model notices and applicable distribution terms; the CNN
  license above is not a license for this separate asset.
- **CEW:** separately acquired Closed Eyes in the Wild evaluation data;
  [publisher page](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html).
  Raw images, manifests with local paths and access-related records stay local.
  Dataset access does not establish unrestricted redistribution permission.
- **Other research datasets and software:** see [dataset setup](DATASET_SETUP.md),
  [research protocol](RESEARCH_PROTOCOL.md), [literature review](LITERATURE_REVIEW.md)
  and pinned Python dependencies. Their own terms remain applicable.
- **Icarus Verilog / MSYS2 and Xilinx tools:** simulation and implementation tools
  are separately installed local dependencies, not redistributed in this source
  tree. Keep their upstream licenses/notices with their local installations.

No project-wide license for the repository's original source has been declared.
The owner must choose one explicitly before offering that code under a particular
license. This cleanup does not invent or substitute a license.
