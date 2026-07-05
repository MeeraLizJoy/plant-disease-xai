"""
Interpretable Deep Learning for Plant Disease Detection
========================================================
Reproduces the results from:

    Joy, M. L., Jain, V., Manisha V., Sinha, P. K., & Gupta, N. K. (2025).
    "Interpretable Deep Learning for Plant Disease Detection: A Comparative
    Study with Explainability Metrics."
    2025 International Conference on Engineering Innovations and Technologies
    (ICoEIT). IEEE. DOI: 10.1109/ICoEIT.2025.01

Models evaluated: InceptionV3, DenseNet121, Xception, MobileNetV2
Dataset: PlantVillage tomato subset (10 classes, 22,929 images)
Novel metric: Explainability Score (Eq. 6) — pairwise Jaccard consistency

Usage:
    python train.py --data_dir /path/to/newplantdisease_subset
    python train.py --data_dir /path/to/data --model mobilenetv2 --epochs 30
    python train.py --data_dir /path/to/data --eval_only --model_path saved_model.keras
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import InceptionV3, DenseNet121, Xception, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import cv2

# ── Constants (matching paper Section III-A) ──────────────────────────────────
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
NUM_CLASSES  = 10
INIT_LR      = 0.0005
FINETUNE_LR  = 0.0002
INIT_EPOCHS  = 30
FINETUNE_EPOCHS = 10

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Fine-tuning config per model (Table III of paper)
FINETUNE_CONFIG = {
    "inceptionv3":  {"unfreeze_last_n": 50,  "dropout": 0.6, "finetune_lr": 0.0001, "finetune_epochs": 15},
    "densenet121":  {"unfreeze_last_n": 12,  "dropout": 0.5, "finetune_lr": 0.0002, "finetune_epochs": 10},
    "xception":     {"unfreeze_last_n": 15,  "dropout": 0.5, "finetune_lr": 0.0002, "finetune_epochs": 10},
    "mobilenetv2":  {"unfreeze_last_n": None, "dropout": 0.5, "finetune_lr": 0.0001, "finetune_epochs": 10},
}


# ── Data loading ──────────────────────────────────────────────────────────────

def build_generators(data_dir: str):
    """Build train and validation generators with augmentation (Section III-A)."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,       # ±20° per paper
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,          # ±20% per paper
        horizontal_flip=True,
        fill_mode="nearest",
    )
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    valid_gen = valid_datagen.flow_from_directory(
        os.path.join(data_dir, "valid"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, valid_gen


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(model_name: str) -> Model:
    """
    Build a transfer-learning model with frozen base + custom head.
    Architecture matches Table III of the paper.
    """
    backbone_map = {
        "inceptionv3": InceptionV3,
        "densenet121":  DenseNet121,
        "xception":     Xception,
        "mobilenetv2":  MobileNetV2,
    }
    if model_name not in backbone_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(backbone_map)}")

    cfg = FINETUNE_CONFIG[model_name]
    base = backbone_map[model_name](
        weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)
    )
    base.trainable = False  # Phase 1: freeze all

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(cfg["dropout"])(x)
    out = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    return model, base


# ── Training ──────────────────────────────────────────────────────────────────

def train(model_name: str, data_dir: str, output_dir: str):
    """Two-phase training: frozen base → selective fine-tuning."""
    os.makedirs(output_dir, exist_ok=True)
    train_gen, valid_gen = build_generators(data_dir)
    model, base = build_model(model_name)
    cfg = FINETUNE_CONFIG[model_name]

    # ── Phase 1: train head only ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: Training head — {model_name}")
    print(f"{'='*60}")

    model.compile(
        optimizer=Adam(learning_rate=INIT_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_p1 = [
        ModelCheckpoint(
            os.path.join(output_dir, f"{model_name}_best.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max", verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=INIT_EPOCHS,
        callbacks=callbacks_p1,
    )

    # ── Phase 2: fine-tune last N layers ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: Fine-tuning — {model_name}")
    print(f"{'='*60}")

    if cfg["unfreeze_last_n"] is None:
        base.trainable = True   # MobileNetV2: unfreeze all
    else:
        for layer in base.layers[: -cfg["unfreeze_last_n"]]:
            layer.trainable = False
        for layer in base.layers[-cfg["unfreeze_last_n"] :]:
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=cfg["finetune_lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_p2 = [
        ModelCheckpoint(
            os.path.join(output_dir, f"{model_name}_finetuned.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max", verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ]
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=cfg["finetune_epochs"],
        callbacks=callbacks_p2,
    )

    model.save(os.path.join(output_dir, f"{model_name}_final.keras"))
    print(f"\nSaved: {output_dir}/{model_name}_final.keras")
    return model, valid_gen


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, valid_gen):
    """Print accuracy, macro-F1, and macro-AUC (Table V of paper)."""
    valid_gen.reset()
    y_pred_prob = model.predict(valid_gen, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = valid_gen.classes

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=6))

    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auc = roc_auc_score(y_true_bin, y_pred_prob, average="macro", multi_class="ovr")
    print(f"Macro-average AUC: {auc:.6f}")
    return y_pred_prob, y_pred, y_true


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def get_gradcam_heatmap(model, img_array: np.ndarray, last_conv_layer_name: str) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a single image.
    Implements gradient-weighted class activation mapping (Selvaraju et al., 2020).

    Args:
        model: Trained Keras model.
        img_array: Preprocessed image, shape (1, H, W, 3), values in [0, 1].
        last_conv_layer_name: Name of the final convolutional layer.

    Returns:
        heatmap: Normalised heatmap, shape (H, W), values in [0, 1].
    """
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_class = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_class]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_heatmap(img_path: str, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlaid = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlaid


# ── Explainability Score (Novel metric — Equation 6 of paper) ─────────────────

def jaccard_index(mask_a: np.ndarray, mask_b: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Jaccard Index (IoU) between two binary masks.
    Equation 5 of the paper:  |A ∩ B| / |A ∪ B|

    Args:
        mask_a: First mask, float array, values in [0, 1].
        mask_b: Second mask, float array, values in [0, 1].
        threshold: Binarisation threshold.

    Returns:
        Jaccard Index in [0, 1].
    """
    a = (mask_a >= threshold).astype(bool)
    b = (mask_b >= threshold).astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def explainability_score(heatmaps: list[np.ndarray], threshold: float = 0.5) -> float:
    """
    Compute the Explainability Score for a set of Grad-CAM heatmaps
    from the same disease class.

    This is the average pairwise Jaccard similarity between all heatmap
    pairs within the class. Higher scores indicate more consistent,
    disease-relevant model attention.

    Equation 6 of the paper:
        ES = (2 / n(n-1)) * Σ_i Σ_{j>i} |H_i ∩ H_j| / |H_i ∪ H_j|

    Args:
        heatmaps: List of heatmaps (np.ndarray, shape H×W), all same class.
        threshold: Binarisation threshold for mask creation.

    Returns:
        Explainability Score in [0, 1].

    Example:
        >>> heatmaps = [generate_heatmap(img) for img in class_images]
        >>> score = explainability_score(heatmaps)
        >>> print(f"Explainability Score: {score:.4f}")
    """
    n = len(heatmaps)
    if n < 2:
        raise ValueError("Need at least 2 heatmaps to compute Explainability Score.")

    total = 0.0
    count = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            total += jaccard_index(heatmaps[i], heatmaps[j], threshold)
            count += 1

    return total / count  # equivalent to (2 / n(n-1)) * Σ jaccard


def compute_explainability_scores_per_class(
    model,
    valid_gen,
    last_conv_layer_name: str,
    n_samples_per_class: int = 10,
) -> dict[str, float]:
    """
    Compute Explainability Score for every disease class.
    Reproduces Table VI of the paper.

    Args:
        model: Trained Keras model.
        valid_gen: Validation generator (shuffle=False).
        last_conv_layer_name: Name of the final conv layer for Grad-CAM.
        n_samples_per_class: How many images per class to sample.

    Returns:
        Dictionary mapping class name → Explainability Score.
    """
    # Group image paths by class
    class_to_paths: dict[int, list[str]] = {i: [] for i in range(NUM_CLASSES)}
    for path, label in zip(valid_gen.filenames, valid_gen.classes):
        full_path = os.path.join(valid_gen.directory, path)
        class_to_paths[label].append(full_path)

    scores = {}
    for class_idx, paths in class_to_paths.items():
        sampled = paths[:n_samples_per_class]
        heatmaps = []
        for img_path in sampled:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            hm = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
            hm_resized = cv2.resize(hm, IMAGE_SIZE)
            heatmaps.append(hm_resized)

        if len(heatmaps) >= 2:
            score = explainability_score(heatmaps)
            scores[CLASS_NAMES[class_idx]] = score
            print(f"  {CLASS_NAMES[class_idx]}: {score:.4f}")

    mean_score = np.mean(list(scores.values()))
    print(f"\nMean Explainability Score: {mean_score:.4f}")
    return scores


# ── Last conv layer names (required for Grad-CAM) ────────────────────────────

LAST_CONV_LAYERS = {
    "inceptionv3": "mixed10",
    "densenet121":  "conv5_block16_concat",
    "xception":     "block14_sepconv2_act",
    "mobilenetv2":  "out_relu",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate CNN models for plant disease detection with XAI."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to dataset root containing train/ and valid/ subdirectories.",
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["inceptionv3", "densenet121", "xception", "mobilenetv2", "all"],
        help="Which model to train. Default: all four models.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory to save trained models and results.",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training; load saved model and evaluate.",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to saved .keras model (required with --eval_only).",
    )
    parser.add_argument(
        "--xai", action="store_true",
        help="Compute Explainability Score after training/evaluation.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10,
        help="Images per class for Explainability Score. Default: 10.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models_to_run = (
        list(FINETUNE_CONFIG.keys()) if args.model == "all" else [args.model]
    )

    for model_name in models_to_run:
        print(f"\n{'#'*60}")
        print(f"  Model: {model_name.upper()}")
        print(f"{'#'*60}")

        if args.eval_only:
            if args.model_path is None:
                raise ValueError("--model_path required with --eval_only")
            model = tf.keras.models.load_model(args.model_path)
            _, valid_gen = build_generators(args.data_dir)
        else:
            model, valid_gen = train(model_name, args.data_dir, args.output_dir)

        evaluate(model, valid_gen)

        if args.xai:
            print(f"\n── Explainability Scores — {model_name} ────────────────")
            compute_explainability_scores_per_class(
                model, valid_gen,
                last_conv_layer_name=LAST_CONV_LAYERS[model_name],
                n_samples_per_class=args.n_samples,
            )


if __name__ == "__main__":
    main()
