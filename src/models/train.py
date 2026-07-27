"""Train and persist one predictive-maintenance classifier per target."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Final, TypeAlias

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
FEATURE_MATRIX_PATH: Final[Path] = (
    PROJECT_ROOT / "data" / "processed" / "features.parquet"
)
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "cooler_condition",
    "valve_condition",
    "pump_condition",
    "accumulator_condition",
    "stable_flag",
)
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
F1_TIE_TOLERANCE: Final[float] = 1e-9

RF_HYPERPARAMETERS: Final[dict[str, int]] = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
XGB_BASE_HYPERPARAMETERS: Final[dict[str, int | float]] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

Classifier: TypeAlias = RandomForestClassifier | XGBClassifier
EncodedLabels: TypeAlias = NDArray[np.int64]
Metrics: TypeAlias = dict[str, float]
Hyperparameters: TypeAlias = dict[str, str | int | float]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Hold one fitted model and its training metadata."""

    algorithm: str
    model: Classifier
    metrics: Metrics
    training_time_seconds: float
    hyperparameters: Hyperparameters


def load_dataset(
    path: Path = FEATURE_MATRIX_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the engineered feature matrix.

    Returns:
        A tuple containing the numeric feature matrix and target table.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. Run the Phase 1 feature "
            "engineering pipeline first."
        )

    LOGGER.info("Loading feature matrix from %s", path)
    dataset = pd.read_parquet(path)
    if dataset.empty:
        raise ValueError(f"Feature matrix is empty: {path}")
    if dataset.columns.has_duplicates:
        raise ValueError("Feature matrix contains duplicate column names")

    missing_targets = [
        target for target in TARGET_COLUMNS if target not in dataset.columns
    ]
    if missing_targets:
        raise ValueError(
            f"Feature matrix is missing targets: {', '.join(missing_targets)}"
        )

    features = dataset.drop(columns=list(TARGET_COLUMNS))
    targets = dataset.loc[:, list(TARGET_COLUMNS)]
    if features.empty:
        raise ValueError("Feature matrix does not contain any model features")
    if len(features.select_dtypes(include="number").columns) != features.shape[1]:
        raise TypeError("All model feature columns must be numeric")
    if features.isna().any(axis=None) or targets.isna().any(axis=None):
        raise ValueError("Feature matrix contains missing values")

    LOGGER.info(
        "Loaded %d cycles, %d features, and %d targets",
        len(dataset),
        features.shape[1],
        targets.shape[1],
    )
    return features, targets


def split_data(
    features: pd.DataFrame,
    labels: EncodedLabels,
) -> tuple[pd.DataFrame, pd.DataFrame, EncodedLabels, EncodedLabels]:
    """Create a reproducible, stratified 80/20 train/test split."""
    class_counts = np.bincount(labels)
    if class_counts.size < 2:
        raise ValueError("A classification target must contain at least two classes")
    if np.any(class_counts < 2):
        raise ValueError("Every target class needs at least two samples to stratify")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return x_train, x_test, y_train, y_test


def build_models(
    class_count: int,
) -> dict[str, tuple[Classifier, Hyperparameters]]:
    """Construct the Random Forest and class-aware XGBoost baselines."""
    if class_count < 2:
        raise ValueError("class_count must be at least 2")

    xgb_objective = "binary:logistic" if class_count == 2 else "multi:softprob"
    xgb_eval_metric = "logloss" if class_count == 2 else "mlogloss"
    xgb_hyperparameters: Hyperparameters = {
        **XGB_BASE_HYPERPARAMETERS,
        "objective": xgb_objective,
        "eval_metric": xgb_eval_metric,
    }
    if class_count > 2:
        xgb_hyperparameters["num_class"] = class_count

    return {
        "Random Forest": (
            RandomForestClassifier(**RF_HYPERPARAMETERS),
            dict(RF_HYPERPARAMETERS),
        ),
        "XGBoost": (
            XGBClassifier(**xgb_hyperparameters),
            xgb_hyperparameters,
        ),
    }


def evaluate_model(
    model: Classifier,
    features: pd.DataFrame,
    labels: EncodedLabels,
) -> Metrics:
    """Evaluate a fitted classifier using the required classification metrics."""
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_weighted": float(
            precision_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


def _fit_and_evaluate(
    algorithm: str,
    model: Classifier,
    hyperparameters: Hyperparameters,
    x_train: pd.DataFrame,
    y_train: EncodedLabels,
    x_test: pd.DataFrame,
    y_test: EncodedLabels,
) -> TrainingResult:
    """Fit one classifier, time training, and evaluate the fitted model."""
    LOGGER.info("Training %s", algorithm)
    started_at = perf_counter()
    model.fit(x_train, y_train)
    training_time = perf_counter() - started_at
    metrics = evaluate_model(model, x_test, y_test)
    LOGGER.info(
        "%s F1: %.6f (trained in %.2f seconds)",
        algorithm,
        metrics["f1_weighted"],
        training_time,
    )
    return TrainingResult(
        algorithm=algorithm,
        model=model,
        metrics=metrics,
        training_time_seconds=training_time,
        hyperparameters=hyperparameters,
    )


def _select_winner(results: list[TrainingResult]) -> TrainingResult:
    """Select the highest-F1 model, preferring XGBoost for near ties."""
    if not results:
        raise ValueError("At least one training result is required")

    best_f1 = max(result.metrics["f1_weighted"] for result in results)
    tied_results = [
        result
        for result in results
        if best_f1 - result.metrics["f1_weighted"] <= F1_TIE_TOLERANCE
    ]
    return next(
        (result for result in tied_results if result.algorithm == "XGBoost"),
        tied_results[0],
    )


def save_model(
    result: TrainingResult,
    label_encoder: LabelEncoder,
    feature_columns: list[str],
    target_name: str,
    output_dir: Path,
) -> Path:
    """Persist the winning estimator with its inference schema and label mapping."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.joblib"
    artifact = {
        "model": result.model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "target_name": target_name,
        "algorithm": result.algorithm,
    }
    joblib.dump(artifact, output_path)
    LOGGER.info("Saved model artifact to %s", output_path)
    return output_path


def _result_metadata(result: TrainingResult) -> dict[str, object]:
    """Return metadata shared by winner and comparison artifacts."""
    return {
        "algorithm": result.algorithm,
        "training_time_seconds": result.training_time_seconds,
        "hyperparameters": result.hyperparameters,
    }


def save_metrics(
    result: TrainingResult,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> Path:
    """Persist winning-model metrics and training metadata as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"
    payload = {
        **_result_metadata(result),
        "metrics": result.metrics,
        "timestamp": datetime.now(UTC).isoformat(),
        "classes": [value.item() for value in label_encoder.classes_],
    }
    with output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")

    LOGGER.info("Saved metrics to %s", output_path)
    return output_path


def save_comparison(
    results: list[TrainingResult],
    winner: TrainingResult,
    output_dir: Path,
) -> Path:
    """Persist evaluation results for every candidate model as JSON."""
    if not results:
        raise ValueError("At least one training result is required")
    if all(result is not winner for result in results):
        raise ValueError("The selected winner must be present in training results")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison.json"
    payload = {
        "selected_model": winner.algorithm,
        "models": [
            {
                **_result_metadata(result),
                **result.metrics,
            }
            for result in results
        ],
    }
    with output_path.open("w", encoding="utf-8") as comparison_file:
        json.dump(payload, comparison_file, indent=2, sort_keys=True)
        comparison_file.write("\n")

    LOGGER.info("Saved model comparison to %s", output_path)
    return output_path


def train_target(
    target_name: str,
    features: pd.DataFrame,
    labels: pd.Series,
    models_dir: Path = MODELS_DIR,
) -> TrainingResult:
    """Train, compare, select, and persist classifiers for one target."""
    LOGGER.info("Training models for %s", target_name)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels).astype(np.int64, copy=False)
    x_train, x_test, y_train, y_test = split_data(features, encoded_labels)

    results = [
        _fit_and_evaluate(
            algorithm,
            model,
            hyperparameters,
            x_train,
            y_train,
            x_test,
            y_test,
        )
        for algorithm, (model, hyperparameters) in build_models(
            len(label_encoder.classes_)
        ).items()
    ]
    winner = _select_winner(results)
    LOGGER.info(
        "Selected %s for %s with weighted F1 %.6f",
        winner.algorithm,
        target_name,
        winner.metrics["f1_weighted"],
    )

    target_dir = models_dir / target_name
    save_model(
        winner,
        label_encoder,
        list(features.columns),
        target_name,
        target_dir,
    )
    save_metrics(winner, label_encoder, target_dir)
    save_comparison(results, winner, target_dir)
    return winner


def main() -> None:
    """Train and save the best classifier for each maintenance target."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    features, targets = load_dataset()
    for target_name in TARGET_COLUMNS:
        train_target(target_name, features, targets[target_name])
    LOGGER.info("Completed training for all %d targets", len(TARGET_COLUMNS))


if __name__ == "__main__":
    main()
