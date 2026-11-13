# Third-party model material

The `FPGA/generated/omz_*` files are fixed-point conversions of Intel's
`open-closed-eye-0001` model (Copyright (C) 2018-2021 Intel Corporation).
The conversions change weights to INT16, biases to INT48, bank the ROM, and
record arithmetic shifts. The upstream Apache 2.0 license is in LICENSE.openvino.
Model provenance and SHA384 are pinned in PC/pretrained_eye.py.
Upstream model specification: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/open-closed-eye-0001/model.yml
Upstream license: https://github.com/opencv/openvino_training_extensions/blob/develop/LICENSE

The MediaPipe Face Landmarker model is downloaded separately from Google;
it is not redistributed here. Follow the upstream model terms:
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
No training datasets, personal recordings, or Python/Xilinx runtimes are included.
