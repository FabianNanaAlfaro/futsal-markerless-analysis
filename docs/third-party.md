# External software

The package calls external pose-estimation systems through optional adapters. Their software, models, pretrained parameters and trademarks remain governed by the terms of their own projects.

| Software | Use in this project | Project source |
| --- | --- | --- |
| Sports2D | 2D pose and tracking route | [Official repository](https://github.com/davidpagnon/Sports2D) |
| DeepLabCut | DLC model loading, inference and training utilities | [Official repository](https://github.com/DeepLabCut/DeepLabCut) |
| MediaPipe Pose | 33-landmark pose estimation | [Official documentation](https://mediapipe.readthedocs.io/en/latest/solutions/pose.html) |
| PyTorch | DLC model execution | [Official documentation](https://pytorch.org/get-started/locally/) |
| SPM1D | Optional one-dimensional repeated-measures statistics | [Official documentation](https://spm1d.org/doc/) |

The repository distributes its own five trained model bundles. No third-party video software, source recordings, participant annotations or camera calibration files are bundled. The applicable terms for project code and model artifacts are described in [`NOTICE.md`](../NOTICE.md).
