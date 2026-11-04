# DeepLabCut models

The five selected PyTorch checkpoints from the thesis benchmark are available as separate GitHub Release downloads. Each archive contains the selected pose checkpoint, the shared SSDLite detector, portable DeepLabCut project settings, inference configuration and a SHA-256 manifest. No training videos, frame images, labels or camera calibrations are included.

## Selected models

| Model | DLC shuffle | Selected epoch | Selection-validation RMSE (px) | RMSE at p-cutoff 0.6 (px) | Checkpoint SHA-256 | Download |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| ResNet-101 | 105 | 25 | 12.55 | 12.25 | `7245E79B6C39E5045BF25C36E4A1D1D2AD6EF7410B204B301DC401073F541332` | [ZIP](https://github.com/FabianNanaAlfaro/futsal-markerless-analysis/releases/download/v0.2.0/ResNet-101.zip) |
| HRNet-W48 | 104 | 35 | 12.49 | 12.26 | `ADEC05378B8246241559FC25306FC2FF3FFE488A5D0F381D6BB5D7958AB09319` | [ZIP](https://github.com/FabianNanaAlfaro/futsal-markerless-analysis/releases/download/v0.2.0/HRNet-W48.zip) |
| RTMPose-S | 101 | 15 | 13.82 | 13.64 | `842003A5530EA74515D3381B5E753F6A39ECE511ADD1DFC5B08944E79BE9A6C5` | [ZIP](https://github.com/FabianNanaAlfaro/futsal-markerless-analysis/releases/download/v0.2.0/RTMPose-S.zip) |
| RTMPose-M | 102 | 28 | 15.54 | 14.58 | `CF21F72B2D9E5D0BF76EB0BE8207E5B37E55B1947E51EB67B0B9F1220414FD01` | [ZIP](https://github.com/FabianNanaAlfaro/futsal-markerless-analysis/releases/download/v0.2.0/RTMPose-M.zip) |
| RTMPose-X | 103 | 4 | 15.62 | 15.83 | `4B39C184CCA4144158206012513F05BA04C144CB2308DC5B2D3DA6E30B336AC4` | [ZIP](https://github.com/FabianNanaAlfaro/futsal-markerless-analysis/releases/download/v0.2.0/RTMPose-X.zip) |

All five models estimate eight bilateral lower-limb points: right and left hip, knee, ankle and foot. The model uses the thesis labels `caderaDer`, `caderaIzq`, `rodillaDer`, `rodillaIzq`, `tobilloDer`, `tobilloIzq`, `pieDer` and `pieIzq`. The package maps these names to its common hip/knee/ankle/foot schema.

## Install and run

Create a dedicated Python 3.10–3.12 environment. Install a PyTorch build that matches the computer's CPU or CUDA setup, then install this package's DLC extra. See the [official DeepLabCut installation guide](https://deeplabcut.github.io/DeepLabCut/docs/installation.html) for supported combinations.

```powershell
python -m venv .venv-dlc
.\.venv-dlc\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/FabianNanaAlfaro/futsal-markerless-analysis.git
cd futsal-markerless-analysis
python -m pip install ".[dlc]"
```

Download and extract one selected model. The example uses ResNet-101:

```powershell
Invoke-WebRequest `
  -Uri https://github.com/FabianNanaAlfaro/futsal-markerless-analysis/releases/download/v0.2.0/ResNet-101.zip `
  -OutFile ResNet-101.zip
Expand-Archive .\ResNet-101.zip -DestinationPath .\models\ResNet-101
futsal-markerless pose-dlc `
  --model ResNet-101 --model-bundle .\models\ResNet-101 `
  --video .\study\camera-1.mp4 --camera CAM_1 --output .\predictions\CAM_1
```

Run the command separately for each synchronized camera view, using a distinct camera label and a new or empty output directory. The adapter checks both checkpoint hashes, fixes pose and detector snapshot selection to `-1`, saves native DeepLabCut output and writes `<video>_tracks_2d.csv` in the common schema. The canonical CSV can be passed to `futsal-markerless run` together with the second camera's tracks and researcher-supplied calibration.

The versions used to verify the model bundles were DeepLabCut 3.0.1, PyTorch 2.5.1+cu121 and torchvision 0.20.1+cu121. Use the [official PyTorch selector](https://pytorch.org/get-started/locally/) to choose a CPU or CUDA wheel compatible with the machine. The `dlc` extra does not install a CUDA runtime or select a GPU automatically. The model project does not record the original DeepLabCut, PyTorch or torchvision training versions; each bundle records the verified inference environment separately.

## Training and evaluation record

The benchmark contains 3,181 labels: 2,324 training labels and 857 validation labels. Labels were sampled every four frames from 240 Hz videos across the sequences. This is the annotation interval used to build the training set. It is not the inference stride, the processing frame rate or the ten-frame strict 3D evaluation subset.

The partition was subject-level, with one split and one exploratory run per configuration. The validation partition was used to select checkpoints and is not an independent test set. The reported values describe this study's selection partition and do not establish a general model ranking. ResNet-101 was selected for the integrated analysis using the thesis criterion.

Training and architecture settings are preserved in each archive's `train/pytorch_config.yaml` and `test/pose_cfg.yaml`. Checkpoint metadata records the selected epoch, model shuffle, detector hash, source project settings and validation metrics. The shared detector SHA-256 is `4AA1810F85075A70177DB64FB7C3D21C65F39B95B8CEAC9190D068693F39ED63`.

The packages omit the annotated training dataset and do not provide a turnkey fine-tuning dataset. Fine-tuning requires new labeled examples and a compatible DeepLabCut training project. The selected checkpoints can be used as initial parameters when the corresponding architecture and eight-point output head are retained.

## Citations

- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), [Mathis et al. (2018)](https://doi.org/10.1038/s41593-018-0209-y) and [Nath et al. (2019), 3D protocol](https://doi.org/10.1038/s41596-019-0176-0).
- Cite the thesis and this repository when these trained parameters or the associated methods are used. Model artifacts are covered by the model terms in [`NOTICE.md`](../NOTICE.md); DeepLabCut and dependencies retain their own licenses.
