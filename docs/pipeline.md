# Processing workflow

```text
synchronized camera videos
  ├─ manual frame tracking in DVIDEOW (reference)
  ├─ Sports2D 2D pose estimates
  ├─ MediaPipe Pose 2D landmarks
  └─ DeepLabCut 2D landmarks (five trained configurations)
          ↓
common 2D schema → markerless 2D curation → camera-specific DLT
          ↓
common 3D temporal filter → positions / speeds / sagittal angles
          ↓
pre-impact-to-impact curves → bilateral and method comparisons
```

## 1. Estimate or digitize image coordinates

Each route estimates or digitizes coordinates in image pixels. The manual DVIDEOW track remains the study reference. Sports2D, MediaPipe and DeepLabCut are separate inference routes; their outputs are not merged before evaluation.

For the thesis, Sports2D used YOLOX with a Halpe-26 body-with-feet pose model. It ordered candidates by greatest displacement and used DeepSORT tracking; unusable tracks received a separate run with manual target selection. MediaPipe Pose 0.10.21 generated 33 landmarks from a fixed ROI per camera and side; only the kicking-side hip, knee, ankle and Foot Index were retained. The DLC benchmark trained five models on bilateral lower-limb points.

MediaPipe laterality is supplied explicitly by the researcher. This package does not select MediaPipe side or frame offset by minimizing error against DVIDEOW.

## 2. Homologate and curate 2D coordinates

The common CSV schema is:

```text
frame_index,camera,landmark,x_px,y_px,confidence
```

The four point names are `hip`, `knee`, `ankle` and `foot`. Recognized route labels are mapped to that vocabulary. If a label encodes side, the unselected side is discarded. Labels without laterality require the user-selected side.

Automated 2D coordinates receive a point-level validity mask based on a Hampel filter on coordinate velocity, using a five-frame half-window and a three-MAD threshold. An outlier in either coordinate invalidates both coordinates for that point. Interior gaps up to five frames are filled axis by axis with PCHIP, with linear interpolation when local support is insufficient. Long gaps and leading/trailing gaps remain missing. A five-sample, second-order Savitzky–Golay filter smooths finite segments. These numerical settings follow the experimental method script; the thesis describes the operations but does not list every numerical setting.

`--source manual` skips this automated 2D curation. This keeps the DVIDEOW reference path distinguishable from pose-estimator post-processing.

## 3. Reconstruct 3D with supplied camera calibration

The researcher supplies either a 3×4 projection matrix or 11 classical DLT coefficients for each camera. Each point and frame is triangulated from at least two matching views. The package exports coordinates, the cameras used, reprojection errors and a conditioning diagnostic. It does not impose a universal reprojection cutoff.

The coordinate system follows the thesis: X anterior–posterior, Y mediolateral and Z vertical. Units come from the supplied calibration. Two cameras are required for a reconstruction; use the camera pair assigned to the kicking side in the study design.

This is a general linear DLT implementation. Numerical agreement with DVIDEOW’s legacy reconstruction must be checked against the investigator’s DVIDEOW calibration and reference outputs before interpreting it as an exact software reproduction.

## 4. Apply common 3D temporal processing

The thesis applies a first-order Butterworth low-pass filter at 18 Hz (240 Hz sampling) followed by robust LOWESS with a 0.10 span to the four routes. The implementation accepts other sampling rates and filter settings through the reusable function API. Missing intervals are filtered as separate finite runs and are not filled across long gaps.

Positions and angular velocities are derived at the physical sampling rate. The declared window start and ball-impact frame are then mapped to 101 phase points from 0 to 100%. Missing intervals remain missing in the phase-normalized export.

The researcher supplies the start and impact frame for each trial so the temporal window is explicit and consistent across camera views.

## 5. Derive biomechanical variables

For hip, knee, ankle and foot, the pipeline exports 3D positions, vector velocities and speed magnitudes. Sagittal joint angles use X–Z projections. Knee extension is 0° and flexion is positive; ankle neutral is 0°, dorsiflexion is positive and plantarflexion is negative. Angular velocity is computed from the unwrapped angle signal using the physical time step.

For paired normalized trials, speed symmetry uses the signed index `200 × (D − ND) / (D + ND)`. Angle and angular-velocity metrics are signed dominant-minus-nondominant differences. Zero speed denominators are marked missing.

## 6. Evaluate methods

The library includes paired coordinate agreement (Euclidean MAE, RMSE, centered RMSE, P95, per-axis bias, CCC, and optional NME/PCK), trajectory Pearson and Spearman association, peak magnitude/phase error, cluster-bootstrap Bland–Altman limits, and Holm/BH p-value adjustment. Optional SPM1D support provides a repeated-measures omnibus test and paired automated-versus-reference tests for complete balanced curves.

The thesis’s strict spatial evaluation used ten frames immediately before impact. That evaluation subset is distinct from the every-fourth-frame sampling used to construct the DLC training labels and from the 101-point temporal representation.
