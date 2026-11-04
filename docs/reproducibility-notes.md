# Reproducibility notes

The thesis defines the analysis reported in this repository. The experimental files provide implementation details where the thesis leaves a parameter unspecified. These distinctions affect how a new study should interpret or adapt the package.

## Method details recovered from the experimental files

- Automated 2D curation uses a five-frame Hampel half-window, a three-MAD cutoff, interpolation for gaps up to five frames, and a five-sample second-order Savitzky–Golay smoother. The thesis describes the processing sequence and maximum interpolated gap but does not list every value.
- The Sports2D route in the thesis uses DeepSORT. The executable configuration uses Sports2D 0.8.20 defaults for unspecified tracker settings, the bundled MobileNet appearance embedder, CPU inference and ONNX Runtime. These choices make the example runnable but are not settings reported by the thesis.
- The benchmark files contain the five selected pose checkpoints and the common SSDLite detector. The bundle builder verifies the selected checkpoint hashes and removes paths to the private experiment folder.

## Differences between thesis wording and experimental code

- An experimental MediaPipe path used DVIDEOW error to choose point ordering and frame offset. The thesis does not describe reference-optimized selection as part of the MediaPipe method, and such selection can inflate agreement. The adapter requires the researcher to specify side, ROI and frame range, then exports the model predictions directly.
- The thesis describes the speed symmetry index as non-negative in one passage, while its displayed equation and results interpretation are signed. The implementation follows the equation `200 × (D − ND) / (D + ND)` and preserves direction.
- The ankle-angle prose describes segment directions that conflict with the stated neutral angle and dorsiflexion sign. The implementation orients the shank from ankle to knee, uses the neutral perpendicular reference, and reports zero at neutral, positive dorsiflexion and negative plantarflexion.
- The thesis describes DVIDEOW camera calibration and reconstruction. The package implements standard triangulation from supplied 3×4 matrices or 11-coefficient DLT calibrations; it does not parse legacy `.CAL` files. Numerical equivalence to DVIDEOW requires a paired validation against the investigator's calibration and outputs.

## Evaluation scope

The every-fourth-frame interval at 240 Hz describes DLC training-label construction. It is separate from full-sequence inference and from the ten pre-impact frames in the strict 3D evaluation subset. The five-model comparison used one participant-level partition and one run per architecture; its validation subset was also used for model selection. Treat the values as selection-partition results, not an independent test or a universal architecture ranking.

The thesis reports no external optical 3D reference, inter-rater or intra-rater tracking study, or runtime benchmark. The repository does not add new scientific outcome values. Synthetic data verify code behavior; they do not replace a real-data reproduction.
