# RTMPose-M DeepLabCut model record

This record preserves the portable DeepLabCut project and architecture settings for the thesis-trained RTMPose-M lower-limb pose model.

## Files

- config.yaml: portable DLC project settings with study video paths removed.
- dlc-models-pytorch/: architecture/training configuration and inference configuration.
- model_metadata.json: selected epoch, checkpoint hashes, benchmark metadata and source project settings.

The public repository configuration record does not include pose or detector weights, training videos, participant annotations or calibration files. The training-set label interval was one frame in four at 240 Hz; it describes annotation sampling only. When this script is run without `--metadata-only`, checkpoint files are copied from the supplied local source project into a local bundle.

## Inference

Inference requires a compatible local bundle containing the corresponding checkpoint files, plus the dlc extra in a Python 3.10–3.12 environment with a compatible PyTorch CPU or CUDA installation. Then run:

    futsal-markerless pose-dlc --model "RTMPose-M" --model-bundle ./RTMPose-M --video input.mp4 --camera CAM_1 --output predictions

The source project YAML records -1 as its default snapshot index. The CLI resolves the registered pose and detector checkpoint filenames to the current DeepLabCut snapshot indexes and verifies their hashes before inference. Use a new or empty output directory so DLC does not reuse previous predictions. DLC outputs preserve its native files and the command also writes a canonical <video>_tracks_2d.csv. Optional device and batch-size settings are available in the CLI.

## Artifact terms

This repository does not assign license terms to trained checkpoint artifacts. Confirm their applicable terms before using or redistributing local model files. DeepLabCut and its dependencies retain their own licenses. Do not redistribute study videos, annotations or calibration records.
