# Validation scope

The tests verify the package's numerical operations and file flow with known, synthetic inputs. They cover 2D homologation, DLT reconstruction, temporal filtering, missing values, angle signs, velocities, normalization, metrics and the end-to-end demo.

The five DLC bundles are generated from the original selected checkpoints and their shared detector. Their hashes are checked before inference. Model inference needs the optional DeepLabCut/PyTorch environment and hardware with enough memory for the selected architecture; RTMPose-X, HRNet-W48 and ResNet-101 are the largest bundles. Model performance on new studies depends on video quality, viewpoint, population and annotation conventions.

The thesis did not use an external optical 3D system, assess inter-rater or intra-rater tracking reliability, or benchmark processing time. Its DLC validation partition was used for model selection. These design limits remain in force when interpreting any new output.

The package implements general linear DLT from researcher-supplied calibrations. It has not been shown numerically identical to DVIDEOW's legacy `.CAL` parser and reconstruction. Validate a new calibration and camera mapping against known points or reference outputs before using reconstructed values scientifically.

The runnable demo is synthetic. It tests execution and mathematical conventions but is not evidence of performance on the thesis videos. The thesis outcome values are not regenerated or altered by the demo.
