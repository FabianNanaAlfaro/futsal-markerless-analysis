# DeepLabCut benchmark

The thesis trained and compared five DeepLabCut PyTorch configurations. This repository provides their benchmark records, sanitized project and model YAML files in [`../configs/deeplabcut/benchmark/`](../configs/deeplabcut/benchmark/), and inference metadata; it does not distribute the trained pose checkpoints or the shared SSDLite detector. To run inference, provide an authorized, compatible local model bundle with the `--model-bundle` option.

## Selected configurations

| Model | Pose architecture | DLC shuffle | Selected epoch | Selection-validation RMSE (px) | RMSE at p-cutoff 0.6 (px) | mAP | mAR | Checkpoint SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet-101 | ResNet-101 | 105 | 25 | 12.55 | 12.25 | 84.88 | 90.75 | `7245E79B6C39E5045BF25C36E4A1D1D2AD6EF7410B204B301DC401073F541332` |
| HRNet-W48 | HRNet-W48 | 104 | 35 | 12.49 | 12.26 | 86.75 | 90.91 | `ADEC05378B8246241559FC25306FC2FF3FFE488A5D0F381D6BB5D7958AB09319` |
| RTMPose-S | RTMPose-S | 101 | 15 | 13.82 | 13.64 | 77.40 | 85.93 | `842003A5530EA74515D3381B5E753F6A39ECE511ADD1DFC5B08944E79BE9A6C5` |
| RTMPose-M | RTMPose-M | 102 | 28 | 15.54 | 14.58 | 66.23 | 76.00 | `CF21F72B2D9E5D0BF76EB0BE8207E5B37E55B1947E51EB67B0B9F1220414FD01` |
| RTMPose-X | RTMPose-X | 103 | 4 | 15.62 | 15.83 | 61.29 | 75.06 | `4B39C184CCA4144158206012513F05BA04C144CB2308DC5B2D3DA6E30B336AC4` |

The benchmark used one exploratory run per configuration with the shared SSDLite person detector. ResNet-101 was selected for the integrated comparison using the thesis criterion. These selection-partition results do not establish a general architecture ranking.

All configurations estimate eight bilateral lower-limb points: right and left hip, knee, ankle and foot. The study labels were `caderaDer`, `caderaIzq`, `rodillaDer`, `rodillaIzq`, `tobilloDer`, `tobilloIzq`, `pieDer` and `pieIzq`. The package maps them to its common hip/knee/ankle/foot schema.

## Run inference with a local model bundle

Create a dedicated Python 3.10–3.12 environment. Install a PyTorch build that matches the computer's CPU or CUDA setup, then install this package's DLC extra. See the [official DeepLabCut installation guide](https://deeplabcut.github.io/DeepLabCut/docs/installation.html) for supported combinations. The researcher must supply the model bundle locally; it is not available from this repository or its Releases.

```powershell
python -m venv .venv-dlc
.\.venv-dlc\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/FabianNanaAlfaro/futsal-markerless-analysis.git
cd futsal-markerless-analysis
python -m pip install ".[dlc]"
```

Run the selected model on a video:

```powershell
futsal-markerless pose-dlc `
  --model ResNet-101 --model-bundle .\study-models\ResNet-101 `
  --video .\study\camera-1.mp4 --camera CAM_1 --output .\predictions\CAM_1
```

Run the command separately for each synchronized camera view, using a distinct camera label and a new or empty output directory. The adapter checks the pose checkpoint hash against the thesis registry and supplied bundle metadata, and the detector hash against the bundle metadata. It resolves both registered checkpoint filenames to the corresponding DeepLabCut snapshot indexes, retains native DeepLabCut output, and writes `<video>_tracks_2d.csv` in the common schema. The canonical CSV can be passed to `futsal-markerless run` with the second camera's tracks and researcher-supplied calibration.

The verified inference environment used DeepLabCut 3.0.1, PyTorch 2.5.1+cu121 and torchvision 0.20.1+cu121. Use the [official PyTorch selector](https://pytorch.org/get-started/locally/) to choose a CPU or CUDA wheel compatible with the machine. The `dlc` extra does not install a CUDA runtime or select a GPU automatically. The original DeepLabCut, PyTorch and torchvision training versions were not recorded in the source project files.

## Training and evaluation record

The benchmark contains 3,181 labels: 2,324 training labels and 857 validation labels. Labels were sampled every four frames from 240 Hz videos across the complete sequences. This interval describes training-set annotation only. It is not the inference stride, processing frame rate or the ten-frame strict 3D evaluation subset.

The partition was participant-level, with one split and one exploratory run per configuration. The validation partition was used to select checkpoints and is not an independent test set. The model registry preserves the architecture, shuffle, selected checkpoint identifier and hash, epoch, validation metrics, p-cutoff, shared detector identifier and hash, batch size, and dynamic-cropping setting. The CLI selects the pose and detector files by their registered names rather than relying on a fixed snapshot-list position.

The annotated training dataset, original configuration files and trained checkpoint weights are not distributed. Fine-tuning requires authorized local model files, new labeled examples and a compatible DeepLabCut training project. A checkpoint can be used as initial parameters only when its architecture and eight-point output head are retained.

## Citations

- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), [Mathis et al. (2018)](https://doi.org/10.1038/s41593-018-0209-y) and [Nath et al. (2019), 3D protocol](https://doi.org/10.1038/s41596-019-0176-0).
- Cite the thesis and this repository when the benchmark record or associated methods are used. External model artifacts and DeepLabCut dependencies retain their own terms; see [`NOTICE.md`](../NOTICE.md).
