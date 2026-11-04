# Study context

The thesis evaluates markerless algorithms for analyzing lower-limb motion and bilateral asymmetry during the futsal instep kick. Its study compares frame-by-frame manual DVIDEOW tracking with Sports2D, MediaPipe Pose and five DeepLabCut configurations. Paired camera views and researcher-supplied calibration support 3D reconstruction, temporal processing and kinematic comparisons.

The analysis includes 13 selected players and 78 kicking trials. The thesis evaluates 2D landmark agreement, 3D reconstruction, temporal waveform similarity and bilateral variables as distinct questions. No single score ranks a method for every downstream use.

The five-model DeepLabCut comparison uses one participant-level split and one exploratory run per configuration. Its validation partition was used during model selection. It is therefore a selection benchmark rather than an independent test of generalization. See [`methods.md`](methods.md), [`deeplabcut-benchmark.md`](deeplabcut-benchmark.md) and [`limitations.md`](limitations.md) for the method and its limits.
