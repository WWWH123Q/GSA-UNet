# GSA-UNet

### Global-Local Spatiotemporal Aggregation Network for Physics-Guided Precipitation Nowcasting

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C" alt="PyTorch 2.1+">
  <img src="https://img.shields.io/badge/Task-Precipitation%20Nowcasting-2E8B57" alt="Precipitation nowcasting">
  <img src="https://img.shields.io/badge/Parameters-1.14M-4C78A8" alt="1.14M parameters">
  <img src="https://img.shields.io/badge/FLOPs-2.63G-F2C14E" alt="2.63 GFLOPs">
</p>

## 【Release Status】

This repository is being prepared to accompany the GSA-UNet manuscript. The README and architecture figure follow the current paper, while the public code tree is still being synchronized with the final LightGLSP, CCM-SIMVP, MSAF, and PGRL implementation. Verify the implementation before using the repository to reproduce the reported results.

## 【Overview】

Precipitation nowcasting must represent nonlinear motion, multi-scale evolution, sparse high-intensity structures, and imbalanced intensity distributions. These properties often cause blurred forecasts and systematic underestimation of intense precipitation.

**GSA-UNet** is a compact encoder-decoder network that combines global-local spatial perception, scale-aware feature reconstruction, multi-scale fusion, and physics-guided optimization. The model is evaluated on cloud-content, radar-reflectivity, and reanalysis-derived precipitation fields.

## 【Highlights】

- **LightGLSP** expands the effective receptive field while preserving local precipitation structures.
- **CCM-SIMVP** adapts hierarchical skip features and refines spatial and temporally encoded representations without recurrent propagation.
- **MSAF** combines six decoder scales using spatially varying attention weights.
- **PGRL** integrates intensity-aware asymmetric regression with spatial, field-level, and temporal consistency terms.
- The complete model contains **1.14M parameters** and requires **2.63G FLOPs** for an input of `1 × 5 × 256 × 256`.
- Experiments cover **LAPS**, **Shanghai radar**, and **ERA5-Land-FS**.

## 【Architecture】

<p align="center">
  <img src="assets/gsa_unet_architecture.jpg" width="100%" alt="Overall architecture of GSA-UNet">
</p>

<p align="center"><b>Figure 1.</b> Overall architecture of GSA-UNet.</p>

GSA-UNet follows a U-shaped encoder-decoder design. Shallow encoder stages retain local boundaries, while deeper stages use LightGLSP for broader spatial perception. Each skip pathway applies CCM followed by a SimVP-inspired refinement block. The decoder reconstructs future fields progressively, and MSAF combines representations from six spatial scales before prediction.

### Lightweight Global Spatial Perception

LightGLSP performs pseudo-sequence spatial mixing on flattened features and restores local two-dimensional structure through depth-wise convolution. This design enlarges the receptive field without introducing full self-attention.

### Cross-Scale Skip Refinement

CCM aligns scale-dependent encoder features before fusion. The following SIMVP block applies gated channel projection, local spatial enhancement, and non-recurrent mixing of temporally encoded features.

### Multi-Scale Attentive Fusion

MSAF upsamples six decoder representations to the target resolution and predicts spatially varying scale weights. The weighted features are then fused to generate the future meteorological sequence.

### Physics-Guided Rainfall Loss

PGRL combines four terms:

1. intensity-aware asymmetric regression;
2. spatial-curvature regularization;
3. field-level statistical consistency;
4. temporal-evolution consistency.

The asymmetric term places greater emphasis on underestimated high-value targets, while the remaining terms act as weak physically motivated regularizers.

## 【Datasets】

| Dataset | Target field | Interval | Input → Output | Forecast horizon | Evaluation thresholds |
|---|---|---:|---:|---:|---|
| LAPS | Cloud content | 1 h | 5 → 3 | 3 h | 0.1, 0.3, 0.5, 0.7, 0.8 |
| Shanghai | Radar reflectivity | 6 min | 5 → 20 | 120 min | 20, 30, 35, 40 dBZ |
| ERA5-Land-FS | Reanalysis precipitation | 1 h | 5 → 3 | 3 h | 0.1, 0.3, 0.5, 0.7, 0.8 |

LAPS contains 696 cloud-content events over East China. The Shanghai dataset contains 1,534 radar events collected from 2015 to 2018. ERA5-Land-FS contains two flood seasons of hourly precipitation fields over East China.

The datasets are not distributed with this repository. Users must obtain them from their respective providers and follow the corresponding data-use requirements.

## 【Key Results】

The table reports representative high-threshold results from the manuscript. SSIM and RMSE are averaged over all test samples and lead times.

| Dataset | High threshold | CSI ↑ | HSS ↑ | SSIM ↑ | RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| LAPS | 0.8 | **0.5717** | **0.3817** | **0.8899** | **0.0579** |
| Shanghai | 40 dBZ | **0.3601** | **0.5098** | **0.8285** | **5.44** |
| ERA5-Land-FS | 0.8 | **0.4469** | **0.2842** | **0.6327** | **0.4670** |

The most consistent gains occur at higher cloud-content, rain-rate, and radar-reflectivity thresholds. On Shanghai, relative CSI improvements over Mamba-UNet reach **7.65% at 40 dBZ** under the reported evaluation protocol.

### Ablation Evidence

On LAPS at threshold 0.8, the complete framework achieves a CSI of `0.5717` and an HSS of `0.3817`. Removing PGRL reduces these values to `0.4056` and `0.2786`, respectively. Using intensity-aware asymmetric regression without the three consistency terms produces a CSI of `0.4972` and an HSS of `0.3295`.

These results indicate that the architecture and PGRL make complementary contributions to high-value target prediction.

## 【Efficiency】

| Model | Parameters (M) ↓ | Model size (MB) ↓ | FLOPs (G) ↓ |
|---|---:|---:|---:|
| SimNowcasting | 3.46 | 13.84 | 14.58 |
| Mamba-UNet | 19.42 | 77.68 | 19.23 |
| TransUNet | 105.33 | 421.32 | 64.50 |
| **GSA-UNet** | **1.14** | **4.56** | **2.63** |

Complexity is measured using an input tensor of shape `1 × 5 × 256 × 256`.

## 【Installation】

```bash
git clone https://github.com/WWWH123Q/GSA-UNet.git
cd GSA-UNet

pip install -r requirements.txt
```

## 【Data Format】

The current data loader accepts an HDF5 file containing a `vil` array with shape:

```text
[time, height, width]
```

The default configuration uses five historical frames to predict three future frames and splits the time series chronologically using a ratio of `0.8 : 0.1 : 0.1`.

Dataset-specific preprocessing and the Shanghai `5 → 20` configuration must be aligned with the final manuscript implementation before reproducing the paper results.

## 【Training】

```bash
python train.py \
  --data-path /path/to/merged_data.h5 \
  --output-dir results/gsa_unet \
  --amp
```

Useful options:

```text
--epochs
--batch-size
--num-workers
--device
--data-scale
--resume
--amp
```

Use `--data-scale 255` when stored values must be normalized from `[0, 255]` to `[0, 1]`.

## 【Evaluation】

```bash
python test.py \
  --checkpoint results/gsa_unet/best_gsa_unet.pth \
  --data-path /path/to/merged_data.h5
```

The evaluation pipeline reports CSI, HSS, RMSE, probability of detection, false-alarm ratio, and accuracy at the configured thresholds.

## 【Project Structure】

```text
GSA-UNet/
├── assets/
│   └── gsa_unet_architecture.jpg
├── models/
│   └── gsa_unet.py
├── config.py
├── datasets.py
├── engine.py
├── losses.py
├── metrics.py
├── train.py
├── test.py
├── utils.py
└── requirements.txt
```


## 【Citation】

If this work is useful for your research, please cite:

```bibtex
@article{wang2026gsaunet,
  title  = {GSA-UNet: A Global-Local Spatiotemporal Aggregation Network for Physics-Guided Precipitation Nowcasting},
  author = {Wang, Sihan and Huang, Xiaohui and Wang, Fu and Yang, Xiaofei and Ban, Yifang},
  year   = {2026},
  note   = {Manuscript under review}
}
```

Publication information and the BibTeX entry will be updated after acceptance.

