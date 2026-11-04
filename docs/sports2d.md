# Sports2D

In the thesis workflow, Sports2D supplies a 2D pose route based on YOLOX and the Halpe-26 body-with-feet model. Candidate selection used greatest displacement and the selected person was tracked with DeepSORT. When tracking was unusable, a separate rerun used manual target selection.

The thesis does not report all package-specific thresholds or the DeepSORT appearance embedder, inference backend or device. The checked-in TOML provides an executable starting point for Sports2D 0.8.20. The executable example uses the bundled MobileNet embedder, CPU inference and ONNX Runtime. These are software configuration choices, not thesis-reported settings. The CPU-tested pair is PyTorch 2.5.1 and torchvision 0.20.1; a CUDA installation needs compatible builds of both. The TOML only records options supported by Sports2D 0.8.20. Unreported values are inherited from that version's default configuration.

## Run a view

In a new environment, install the tested CPU PyTorch pair and the optional package extra. For a CUDA run, select matching PyTorch and torchvision wheels from the official PyTorch selector. Edit a copy of [`sports2d.thesis.toml`](../configs/sports2d.thesis.toml) for the camera and study. Its `visible_side` setting describes the view orientation; the CLI `--side` argument selects the anatomical side written to the common tracks. Then run:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[sports2d]"
futsal-markerless pose-sports2d `
  --video .\study\camera-1.mp4 `
  --config .\config\sports2d-camera-1.toml `
  --side right --camera CAM_1 `
  --output .\pose-output\camera-1
```

The adapter calls Sports2D's Python API, retains its native TRC and angle exports, and writes `<video>_tracks_2d.csv` in the common schema. The CSV contains image-pixel coordinates, original frame indices, camera label, source landmark names and blank confidence values because Sports2D TRC exports do not include per-point likelihoods. Use a new or empty output directory. Supply each synchronized camera view separately, with its camera label and selected anatomical side, before DLT reconstruction. Use two synchronized camera views and researcher-supplied calibration for 3D reconstruction; a single-view Sports2D scale estimate is a separate output.

Sports2D can filter or interpolate its own 2D output. Record those settings and avoid applying them twice without checking their effect. The shared markerless 2D curation in this package is a separate downstream operation.

## Sources

- [Sports2D source code and configuration](https://github.com/davidpagnon/Sports2D)
- [Pagnon and Kim (2024), Journal of Open Source Software](https://doi.org/10.21105/joss.06849)
- [HALPE-FullBody keypoint definition](https://github.com/Fang-Haoshu/Halpe-FullBody)
