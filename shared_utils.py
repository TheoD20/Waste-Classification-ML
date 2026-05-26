"""
Shared utility functions for the Waste Classification ML project.

This file contains only common code that all model notebooks can reuse:
- load the shared train/validation/test split CSVs
- evaluate predictions using the same metrics
- plot train/test confusion matrices side by side when available
- save metrics for the final comparison notebook
- plot deep learning training curves
- compute class weights for imbalanced training
- extract HOG features

Model-specific preprocessing should stay inside each model notebook.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from sklearn.utils.class_weight import compute_class_weight


# Project paths are resolved from this file location.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "garbage_classification"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def load_data():
    """
    Load the shared train/validation/test split CSVs.

    Expected files:
        data/splits/train.csv
        data/splits/val.csv
        data/splits/test.csv

    Expected CSV columns:
        path,label,class_name
    """
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    required_columns = {"path", "label", "class_name"}

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        missing_columns = required_columns - set(split_df.columns)

        if missing_columns:
            raise ValueError(
                f"{split_name}.csv is missing columns: {missing_columns}. "
                "Expected columns: path,label,class_name"
            )

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    class_list = (
        all_df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")["class_name"]
        .tolist()
    )

    train_paths = np.array([str(DATA_DIR / p) for p in train_df["path"]])
    val_paths = np.array([str(DATA_DIR / p) for p in val_df["path"]])
    test_paths = np.array([str(DATA_DIR / p) for p in test_df["path"]])

    train_labels = train_df["label"].to_numpy()
    val_labels = val_df["label"].to_numpy()
    test_labels = test_df["label"].to_numpy()

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_paths": train_paths,
        "val_paths": val_paths,
        "test_paths": test_paths,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "class_list": class_list,
    }


def eval_model(
    y_test,
    test_pred,
    class_names,
    model_name="Model",
    y_train=None,
    train_pred=None,
    train_time=None,
    inference_time=None,
    plot_matrix=True,
):
    """
    Evaluate model predictions using the shared project metrics.

    This function works for all models as long as predicted labels are provided.

    Required:
        y_test = true test labels
        test_pred = predicted test labels
        class_names = class names ordered by label index

    Optional:
        y_train = true training labels
        train_pred = predicted training labels

    If train predictions are provided, train and test confusion matrices are
    plotted side by side.
    """
    y_test = np.asarray(y_test)
    test_pred = np.asarray(test_pred)

    test_accuracy = accuracy_score(y_test, test_pred)

    test_macro_precision = precision_score(
        y_test, test_pred, average="macro", zero_division=0
    )
    test_macro_recall = recall_score(
        y_test, test_pred, average="macro", zero_division=0
    )
    test_macro_f1 = f1_score(
        y_test, test_pred, average="macro", zero_division=0
    )

    test_weighted_precision = precision_score(
        y_test, test_pred, average="weighted", zero_division=0
    )
    test_weighted_recall = recall_score(
        y_test, test_pred, average="weighted", zero_division=0
    )
    test_weighted_f1 = f1_score(
        y_test, test_pred, average="weighted", zero_division=0
    )

    train_accuracy = None
    train_macro_f1 = None

    has_train_results = y_train is not None and train_pred is not None

    if has_train_results:
        y_train = np.asarray(y_train)
        train_pred = np.asarray(train_pred)

        train_accuracy = accuracy_score(y_train, train_pred)
        train_macro_f1 = f1_score(
            y_train, train_pred, average="macro", zero_division=0
        )

    print(model_name)
    print("-" * len(model_name))

    if train_time is not None:
        print(f"Training time     : {train_time:.2f}s")

    if inference_time is not None:
        print(f"Inference time    : {inference_time:.3f}s")
        print(f"Time per sample   : {inference_time / len(y_test) * 1000:.3f}ms")

    print("-" * len(model_name))
    print(classification_report(
        y_test,
        test_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))

    if plot_matrix:
        if has_train_results:
            fig = plt.figure(figsize=[20, 8])

            ax = fig.add_subplot(1, 2, 1)
            ConfusionMatrixDisplay.from_predictions(
                y_train,
                train_pred,
                display_labels=class_names,
                normalize="true",
                ax=ax,
                xticks_rotation=45,
                colorbar=False,
            )
            ax.set_title(
                f"Training set - acc={train_accuracy:.3f}, macro F1={train_macro_f1:.3f}"
            )

            ax = fig.add_subplot(1, 2, 2)
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                test_pred,
                display_labels=class_names,
                normalize="true",
                ax=ax,
                xticks_rotation=45,
                colorbar=False,
            )
            ax.set_title(
                f"Test set - acc={test_accuracy:.3f}, macro F1={test_macro_f1:.3f}"
            )

            fig.suptitle(f"{model_name}: Normalised Confusion Matrices")
            plt.tight_layout()
            plt.show()

        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1)

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                test_pred,
                display_labels=class_names,
                normalize="true",
                ax=ax,
                xticks_rotation=45,
            )

            ax.set_title(f"{model_name} Test Confusion Matrix")
            plt.tight_layout()
            plt.show()

    return {
        "model_name": model_name,
        "train_accuracy": None if train_accuracy is None else float(train_accuracy),
        "train_macro_f1": None if train_macro_f1 is None else float(train_macro_f1),
        "test_accuracy": float(test_accuracy),
        "test_macro_precision": float(test_macro_precision),
        "test_macro_recall": float(test_macro_recall),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_precision": float(test_weighted_precision),
        "test_weighted_recall": float(test_weighted_recall),
        "test_weighted_f1": float(test_weighted_f1),
        "train_time": None if train_time is None else float(train_time),
        "inference_time": None if inference_time is None else float(inference_time),
    }


def save_metrics(metrics, output_path):
    """
    Save a metrics dictionary as JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Saved metrics to: {output_path}")


def plot_training_history(history, title="Training History"):
    """
    Plot training and validation accuracy/loss for Keras models.
    """
    history_dict = history.history

    if "accuracy" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(history_dict["accuracy"], label="training accuracy")

        if "val_accuracy" in history_dict:
            plt.plot(history_dict["val_accuracy"], label="validation accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{title}: Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if "loss" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(history_dict["loss"], label="training loss")

        if "val_loss" in history_dict:
            plt.plot(history_dict["val_loss"], label="validation loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title}: Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()


def get_class_weights(labels):
    """
    Compute class weights for imbalanced classification.

    Useful for Keras models:
        model.fit(..., class_weight=class_weights)
    """
    labels = np.asarray(labels)
    classes = np.sort(np.unique(labels))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    return {
        int(class_label): float(weight)
        for class_label, weight in zip(classes, weights)
    }

# Shared HOG settings for traditional ML models
HOG_IMAGE_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

FEATURE_CACHE_DIR = PROJECT_ROOT / "data" / "features"
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extract_hog_one(img_path):
    """
    Extract a HOG feature vector from one image.

    Used by traditional ML models such as:
    - Random Forest
    - PCA/LDA pipeline
    """
    img = Image.open(img_path).convert("RGB").resize(HOG_IMAGE_SIZE, Image.BILINEAR)
    grey = rgb2gray(np.array(img))

    return hog(
        grey,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def extract_hog_batch(paths, split_name):
    """
    Extract HOG features for a split of image paths.

    Features are cached in data/features/ to avoid recomputing them every run.
    """
    cache_name = (
        f"hog_{split_name}_"
        f"size{HOG_IMAGE_SIZE[0]}x{HOG_IMAGE_SIZE[1]}_"
        f"ori{HOG_ORIENTATIONS}_"
        f"ppc{HOG_PIXELS_PER_CELL[0]}x{HOG_PIXELS_PER_CELL[1]}_"
        f"cpb{HOG_CELLS_PER_BLOCK[0]}x{HOG_CELLS_PER_BLOCK[1]}.npz"
    )

    cache_path = FEATURE_CACHE_DIR / cache_name

    if cache_path.exists():
        print(f"loading cached features: {cache_path.name}")
        data = np.load(cache_path)
        return data["X"]

    print(f"extracting HOG for {len(paths)} {split_name} images...")

    features = []

    for i, path in enumerate(paths):
        features.append(extract_hog_one(path))

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(paths)}")

    X = np.stack(features).astype(np.float32)

    np.savez_compressed(cache_path, X=X)
    print(f"saved: {cache_path.name}  shape={X.shape}")

    return X