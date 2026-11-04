# Scientific conventions

The thesis is the method specification. This page records the conventions needed to use the reusable implementation with new videos and calibrations.

## Pose routes

| Route | Study method | Functional points passed to the common pipeline |
| --- | --- | --- |
| Reference | Manual frame-by-frame DVIDEOW tracking and camera calibration | Kicking-side hip, knee, ankle and foot |
| Sports2D | YOLOX detector, Halpe-26 body-with-feet pose model, greatest-displacement ordering and DeepSORT; a manual-selection rerun was used when tracking failed | Kicking-side hip, knee, ankle and foot |
| MediaPipe Pose | v0.10.21, fixed camera/side ROI, expanded crop mapped back to source pixels, segmentation off | Kicking-side hip, knee, ankle and Foot Index from the 33-landmark output |
| DeepLabCut | PyTorch, shared SSDLite detector and five pose configurations | Bilateral hip, knee, ankle and foot |

MediaPipe side and camera-specific ROI are explicit analysis inputs. The thesis route retains the four landmarks on the declared kicking side.

## 2D preprocessing for automated tracks

The common automated 2D curation is applied before DLT:

1. Calculate coordinate velocity and apply a rolling Hampel median/MAD filter.
2. Mark the complete point missing when either coordinate is an outlier.
3. Interpolate interior gaps no longer than five frames per axis using PCHIP, with linear fallback when fewer than three local supports are available.
4. Apply a light Savitzky–Golay smoother with a five-sample window and second-order polynomial.
5. Retain long or edge gaps as missing.

The five-frame half-window, three-MAD threshold, and Savitzky–Golay window/order are recovered from the experimental MediaPipe processing script; the thesis describes the algorithms and maximum gap, but not every parameter. DVIDEOW manual coordinates bypass this automatic 2D curation.

## Reconstruction and coordinates

Use the camera-specific calibration supplied by the investigator. The study uses two camera views per kicking side and 11 DLT coefficients per camera. The package also accepts 3×4 projection matrices, which makes the numerical core independent of legacy calibration file formats.

The world axes are X anterior–posterior, Y mediolateral and Z vertical. For the sagittal analysis, +X is anterior and +Z is superior. The package reports calibration units unchanged. It does not parse DVIDEOW `.CAL` files or assert numerical equivalence to DVIDEOW without a direct paired calibration check.

## Common temporal filter and phase

The shared filter is a first-order, zero-phase Butterworth low-pass at 18 Hz with 240 Hz sampling, followed by robust LOWESS with span 0.10. Finite segments are filtered independently; the package does not extrapolate missing intervals. The reusable API allows sampling frequency and filter parameters to be supplied for another acquisition system.

The analysis window begins before ball impact and ends at impact. The user supplies both frame indices. Curves are mapped to 101 points (0–100%); derivatives are computed before phase normalization using physical frame timing.

## Angles and velocities

All joint angles are computed from 3D landmarks projected to X–Z:

- Knee: oriented angle relative to full extension; 0° at extension, positive flexion, negative hyperextension.
- Ankle: oriented angle relative to the neutral perpendicular shank/foot reference; 0° neutral, positive dorsiflexion, negative plantarflexion.
- Speed: Euclidean norm of the three velocity components at each functional point.
- Angular velocity: time derivative of the unwrapped joint-angle signal, in degrees per second.

The implementation follows the intended sign conventions stated in the thesis. For the ankle angle, the proximal shank vector is oriented ankle-to-knee so that the neutral reference is −90° before the 90° shift to zero.

## Bilateral and method metrics

Dominant (D) and nondominant (ND) labels are supplied by the study manifest. The package uses the thesis’s signed speed index:

```text
IS = 200 × (D − ND) / (D + ND)
```

Knee/ankle angles and angular velocities use signed `D − ND` differences. A zero denominator returns missing.

Spatial agreement includes Euclidean MAE/RMSE, centered RMSE, P95 error, per-axis bias, CCC, NME and PCK. Temporal agreement includes Pearson/Spearman association and peak magnitude/phase errors. Bland–Altman confidence intervals resample participant clusters so paired trials remain together. Optional SPM1D uses one-way repeated-measures ANOVA and paired comparisons on complete, balanced trajectories. Apply Holm or BH correction to the family of p-values as specified by the analysis plan.

## DeepLabCut training and evaluation partitions

The study labelled eight lower-limb points on 3,181 frames: 2,324 training and 857 selection-validation annotations. Labels were sampled at one frame in four from 240 Hz videos across the complete sequences. The split was participant-level, with one partition and one run per architecture. This sampling describes label construction only; it is not a pose-inference stride.

The thesis’s strict spatial subset contains ten frames immediately before impact. The 101-point phase curves are a separate temporal representation. The validation partition was also used to select the checkpoint and architecture; it is not an independent external test set. See [`deeplabcut-benchmark.md`](deeplabcut-benchmark.md) for selected checkpoints and reported metrics.
