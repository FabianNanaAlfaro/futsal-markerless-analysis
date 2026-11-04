# Python API and command-line reference

The reusable implementation is in `src/futsal_markerless_analysis/`. The command-line interface calls the same functions exported by the package.

## Python API

The central function accepts canonical 2D observations and calibration matrices:

```python
from futsal_markerless_analysis.pipeline import process_trial, read_tracks_csv, write_result
import json

tracks = read_tracks_csv("camera_pair_tracks.csv")
calibration = json.load(open("calibration.json", encoding="utf-8"))
result = process_trial(
    tracks,
    calibration,
    selected_side="right",
    source="markerless",
    sampling_hz=240.0,
    window_start_frame=180,
    impact_frame=240,
    cameras=["CAM_1", "CAM_2"],
)
write_result(result, "trial-output")
```

For DVIDEOW manual coordinates, set `source="manual"` to skip the markerless 2D Hampel/interpolation/smoothing stage. Supply the same camera pair and verified calibration for the selected kicking side. The common 3D filter runs for both sources unless `apply_common_filter=False` is supplied.

## Core modules

| Module | Use |
| --- | --- |
| `schema` | Map estimator-specific labels and laterality to hip, knee, ankle and foot. |
| `geometry` | Load 3×4 projection matrices or 11 DLT coefficients and triangulate with reprojection diagnostics. |
| `temporal` | Curate automated 2D tracks, filter finite 3D segments and normalize a declared frame window. |
| `kinematics` | Derive 3D velocities, speed magnitudes, signed knee/ankle angles and bilateral metrics. |
| `metrics` | Coordinate and trajectory agreement, peak errors, participant-cluster Bland–Altman intervals, p-value adjustments and optional SPM1D analysis. |
| `pipeline` | Coordinate CSV loading, 2D homologation, reconstruction, kinematics and file export. |
| `adapters` | Call MediaPipe, Sports2D or DeepLabCut where the optional software is installed. |

## Common 2D CSV

Use one row per frame, camera and landmark:

```text
frame_index,camera,landmark,x_px,y_px,confidence
0,CAM_1,hip,412.5,203.1,0.97
0,CAM_1,knee,419.2,282.6,0.91
```

`frame_index` is a zero-based integer. Coordinates are in pixels. `confidence` is optional; empty coordinates remain missing. Labels may use canonical names, common MediaPipe/Sports2D labels or the Spanish labels emitted by the thesis DLC models. When a label encodes laterality, the selected side determines which rows are used.

Calibration JSON contains a `projection_matrices` object whose camera values are either 3×4 arrays or 11-coefficient DLT arrays. The calibration units define the output 3D units. A calibration must match the image coordinates, camera identity and frame pairing supplied in the 2D file.

## CLI groups

Run `futsal-markerless --help` and `futsal-markerless <command> --help` to see options for `demo`, `pose-mediapipe`, `pose-sports2d`, `pose-dlc`, `run`, `bilateral`, `agreement` and `models`. Installation and runnable commands are in the [README](../README.md).
