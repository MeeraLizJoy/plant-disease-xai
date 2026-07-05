# Interpretable Deep Learning for Plant Disease Detection

Official code for the paper:

> **Interpretable Deep Learning for Plant Disease Detection: A Comparative Study with Explainability Metrics**  
> Meera Liz Joy, Vipin Jain, Manisha V., Piyush Kumar Sinha, Neetesh Kumar Gupta  
> *2025 International Conference on Engineering Innovations and Technologies (ICoEIT), IEEE*  
> DOI: [10.1109/ICoEIT.2025.01](https://doi.org/10.1109/ICoEIT.2025.01)

---

## What this paper does

We compare four CNN architectures (InceptionV3, DenseNet121, Xception, MobileNetV2) on tomato leaf disease classification using the PlantVillage dataset, evaluated on **both accuracy and interpretability**. We introduce a novel **Explainability Score** — the average pairwise Jaccard similarity of Grad-CAM heatmaps within each disease class — to quantitatively measure how consistently a model attends to disease-relevant regions.

**Key finding:** MobileNetV2 achieves the highest accuracy (98.47%) but Xception and DenseNet121 achieve the highest Explainability Scores, revealing an accuracy–interpretability trade-off.

---

## Results (Table V & VI of paper)

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Explainability Score |
|---|---|---|---|---|---|---|
| InceptionV3 | 97.47% | 97.49% | 97.47% | 97.46% | 0.9994 | 0.25 |
| DenseNet121 | 98.06% | 98.10% | 98.06% | 98.07% | 0.9997 | 0.34 |
| **Xception** | 97.60% | 97.66% | 97.60% | 97.60% | 0.9996 | **0.39** |
| **MobileNetV2** | **98.47%** | **98.51%** | **98.47%** | **98.48%** | **0.9998** | 0.29 |

---

## Dataset

**PlantVillage — Tomato subset** (10 classes, 22,929 images)  
Download from Kaggle: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Expected directory structure:
```
newplantdisease_subset/
├── train/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   └── ... (10 classes)
└── valid/
    ├── Tomato___Bacterial_spot/
    └── ... (10 classes)
```

---

## Installation

```bash
git clone https://github.com/[yourhandle]/plant-disease-xai-cnn
cd plant-disease-xai-cnn
pip install -r requirements.txt
```

---

## Usage

**Train all four models:**
```bash
python train.py --data_dir /path/to/newplantdisease_subset
```

**Train a single model:**
```bash
python train.py --data_dir /path/to/data --model mobilenetv2
```

**Train + compute Explainability Score:**
```bash
python train.py --data_dir /path/to/data --model xception --xai --n_samples 10
```

**Evaluate a saved model:**
```bash
python train.py --data_dir /path/to/data --eval_only --model_path outputs/mobilenetv2_final.keras --xai
```

---

## Novel contribution: Explainability Score

The `explainability_score()` function in `train.py` implements Equation 6 of the paper. You can use it independently on any set of heatmaps:

```python
from train import explainability_score, get_gradcam_heatmap

# Generate heatmaps for images of the same disease class
heatmaps = [get_gradcam_heatmap(model, img, "out_relu") for img in class_images]

# Compute score (higher = more consistent, disease-relevant attention)
score = explainability_score(heatmaps, threshold=0.5)
print(f"Explainability Score: {score:.4f}")
```

---

## Fine-tuning configuration (Table III of paper)

| Model | Unfrozen Layers | Initial LR | Fine-tune LR | Fine-tune Epochs | Dropout |
|---|---|---|---|---|---|
| InceptionV3 | Last 50 | 0.0005 | 0.0001 | 15 | 0.6 |
| DenseNet121 | Last 12 | 0.0005 | 0.0002 | 10 | 0.5 |
| Xception | Last 15 | 0.0005 | 0.0002 | 10 | 0.5 |
| MobileNetV2 | All layers | 0.0005 | 0.0001 | 10 | 0.5 |

---

## Repository structure

```
plant-disease-xai-cnn/
├── train.py                 # Main training + XAI script
├── requirements.txt
├── README.md
└── experiments/
    └── hsv_ablation.ipynb   # HSV preprocessing experiment (not in final paper)
```

---

## Citation

If you use this code or the Explainability Score metric, please cite:

```bibtex
@inproceedings{joy2025interpretable,
  title     = {Interpretable Deep Learning for Plant Disease Detection:
               A Comparative Study with Explainability Metrics},
  author    = {Joy, Meera Liz and Jain, Vipin and Manisha, V. and
               Sinha, Piyush Kumar and Gupta, Neetesh Kumar},
  booktitle = {2025 International Conference on Engineering Innovations
               and Technologies (ICoEIT)},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/ICoEIT.2025.01}
}
```

---

## License

MIT License. See `LICENSE` for details.  
Note: PlantVillage dataset has its own license — download it directly from Kaggle.
