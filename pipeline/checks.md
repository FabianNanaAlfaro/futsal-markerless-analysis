# Input checks for a study run

Before interpreting a reconstruction, verify:

- Each view belongs to the same trial and has the correct camera label.
- Frame indexes use the same origin and synchronization convention across views.
- Pixel coordinates refer to the original image dimensions used during calibration.
- Calibration matrices match those camera views and produce the expected world-axis directions and units.
- The selected side, window-start frame and impact frame match the study record.
- The 2D export preserves missing coordinates and confidence values.
- Reprojection residuals and finite reconstructed points are reviewed before kinematic interpretation.

The package exports the input tracks, 3D kinematics, phase-normalized curves, reprojection diagnostics and run metadata. Use a paired reference and the evaluation functions in `futsal_markerless_analysis.metrics` when agreement is the analysis question.
