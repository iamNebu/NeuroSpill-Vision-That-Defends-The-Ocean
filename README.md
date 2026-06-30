# 🌊 NeuroSpill — The Vision That Defends The Ocean

NeuroSpill is a lightweight deep-learning system for detecting oil spills in **Synthetic Aperture Radar (SAR)** satellite imagery, built from scratch and trained on the Zenodo Sentinel-1 SAR oil spill dataset. It segments raw SAR scenes pixel-by-pixel to flag oil-covered ocean surface, estimate spill severity, and present everything through an interactive Streamlit dashboard.

## How it works

SAR satellites image the ocean surface regardless of cloud cover or daylight, but oil slicks, calm-water "lookalikes" (e.g. algae, wind shadows), and open water all look visually similar in raw radar backscatter. NeuroSpill tackles this with a custom **Attention U-Net++** segmentation model trained to tell real oil spills apart from both clean water and lookalike clutter.

The pipeline has three stages:

1. **Preprocessing** — raw two-band Sentinel-1 SAR scenes (VV and VH polarizations) are converted into a 3-channel feature stack (`VV`, `VH`, and the `VV/VH` ratio), normalized, and cut into 256×256 patches.
2. **Training** — the Attention U-Net++ model is trained on a balanced mix of oil-spill, lookalike, and open-water patches using a combined Dice + edge-aware BCE loss, with heavy augmentation to simulate sensor noise and real-world SAR variability.
3. **Inference (UI)** — a Streamlit app loads the trained model, runs patch-wise prediction over an uploaded full-size SAR `.tif` image, stitches the results back together, and reports detection, coverage area, severity, and (optionally) accuracy metrics against a ground-truth mask.

## Repository structure

```
.
├── Literature_Surveys/      # Reference papers used to design the model and preprocessing approach
├── PREPROCESS/               # Notebooks that convert raw Zenodo SAR .tif scenes into training patches
│   ├── oil_pre_process.ipynb            # Extracts oil-spill patches (filtered by minimum oil pixel count)
│   └── pre_process_no_oil_look_alike.ipynb  # Extracts background patches (lookalikes + clean open water)
├── MODEL_TRAINING/
│   └── m3.ipynb              # Defines the Attention U-Net++ model and runs the full training loop
├── Neurospill/                # The deployable application
│   ├── app.py                 # Streamlit dashboard for inference and reporting
│   ├── model.py                # Attention U-Net++ architecture (shared with training)
│   ├── m3_t2_best_oil_spill_model.pth  # Pretrained model weights
│   └── README.txt              # One-line run instructions
└── User Interface/             # Screenshots of the dashboard
```

## Model architecture

`model.py` implements an **Attention U-Net++**: a U-Net-style encoder-decoder with four feature scales (64 → 128 → 256 → 512 channels), where each decoder stage uses an **attention gate** to weight the encoder's skip-connection features before concatenation, helping the model focus on subtle, irregular spill boundaries rather than uniform background ocean.

- **Input:** 3-channel SAR feature stack (VV, VH, VV/VH ratio), 256×256 patches
- **Output:** single-channel logit map → sigmoid → binary oil/no-oil mask
- **Loss:** Dice loss + edge-aware BCE loss (extra weight on the outer 15 pixels of each patch to sharpen boundary predictions)
- **Optimizer:** Adam (lr 5e-5) with `ReduceLROnPlateau` scheduling
- **Training:** 50 epochs, batch size 4, mixed-precision (AMP), with the best checkpoint (by validation IoU) saved to `m3_t2_best_oil_spill_model.pth`

## Dataset

Trained on the **Sentinel-1 SAR Oil Spill Detection dataset** hosted on Zenodo, which provides three image classes per scene:

- Oil spill images + masks
- Lookalike images (non-oil features that resemble spills, e.g. biogenic slicks, wind shadows)
- No-oil (open ocean) images

The preprocessing notebooks tile each 2048×2048 scene into 256×256 patches, keep only oil patches with a meaningful spill area (≥1500 oil pixels), and sample background patches from both the lookalike and no-oil categories so the model learns to reject false positives, not just detect oil.

## Getting started

### Requirements

```
torch
torchvision
streamlit
numpy
opencv-python
rasterio
pandas
albumentations   # only needed for re-training (MODEL_TRAINING/m3.ipynb)
```

Install with:

```bash
pip install torch torchvision streamlit numpy opencv-python rasterio pandas albumentations
```

### Run the inference dashboard

```bash
cd Neurospill
streamlit run app.py
```

This launches a browser dashboard where you can:

1. Upload a SAR image (`.tif`, expected to contain VV and VH bands)
2. Optionally upload a ground-truth mask `.tif` for validation
3. Adjust the confidence threshold via the sidebar slider
4. View detection status, spill coverage %, severity level (NONE / LOW / MEDIUM / HIGH), and — if a ground truth was supplied — pixel-level accuracy, precision, recall, F1, IoU, and a confusion matrix
5. Inspect visual outputs: raw SAR, predicted mask, ground truth (if provided), and a red-highlighted overlay on the original image

The model runs inference patch-by-patch (256×256 tiles) across the full image, only keeping patches whose predicted oil area exceeds a noise floor, then reassembles the full-resolution mask and applies morphological cleanup before computing severity.

### Re-training the model

1. Download the Zenodo Sentinel-1 SAR oil spill dataset (Parts I and II) and update the input paths in the `PREPROCESS/` notebooks.
2. Run `PREPROCESS/oil_pre_process.ipynb` to generate oil-spill patches, then `PREPROCESS/pre_process_no_oil_look_alike.ipynb` (run twice — once for lookalikes, once for no-oil images) to generate background patches.
3. Run `MODEL_TRAINING/m3.ipynb`, pointing `OIL_DIR` and `BACK_DIR` at the preprocessed patch folders, to train the Attention U-Net++ model from scratch and produce a new `.pth` checkpoint.
4. Copy the resulting checkpoint into `Neurospill/` to use it in the dashboard.

## Literature survey

`Literature_Surveys/survey_nebu/` contains the research papers used to inform the model design and preprocessing pipeline, covering prior SAR oil-spill detection approaches (Attention U-Net variants, dual-stream U-Nets, state-space models, classical classifiers, and SAR water/land separation techniques).

## License

Released under the [GPL-3.0 License](LICENSE).
