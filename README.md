# MS-Net
Mutual Semantic Refinement for Boundary-Aware Unified Optic Disc and Cup Segmentation

BIBM 2026 DLB2H Workshop.

- Train on REFUGE: `trainer.py`
- Train on GAMMA: `train_gamma.py`
- Test on REFUGE or DRISHTI-GS: `test.py`

This code is built upon the official implementation of [RFAU-CNxt](https://github.com/SiddMallick/RFAUCNxt-official).
1. We adopt an ROI-localization pipeline inspired by RFAU-CNxt. A YOLO model is trained on 80 REFUGE training images and used to extract ROIs from the original images.
2. The processed ROIs are saved as the training and testing datasets.
3. All comparison models are reproduced, which explains differences from the results reported in their original papers.
4. `Ours*` uses depthwise separable convolutions by setting the `groups` parameter in `model/msnet/msnet.py`; without this setting, the model corresponds to `Ours`.


For independent evaluation, the network is implemented in `model/msnet/msnet.py` and can be directly integrated into other training pipelines.
