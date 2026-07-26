"""Build the cycle-level feature matrix for the hydraulic dataset."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, TypeAlias

import pandas as pd

from src.data.loader import load_hydraulic_dataset
from src.features.feature_engineering import (
    FeatureVector,
    Signal,
    extract_flow_features,
    extract_motor_power_features,
    extract_pressure_features,
    extract_temperature_features,
    extract_vibration_features,
    extract_virtual_features,
)

LOGGER = logging.getLogger(__name__)

FeatureExtractor: TypeAlias = Callable[[Signal], FeatureVector]
SensorFrames: TypeAlias = Mapping[str, pd.DataFrame]

OUTPUT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
)
TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "cooler_condition",
    "valve_condition",
    "pump_condition",
    "accumulator_condition",
    "stable_flag",
)
SENSOR_EXTRACTORS: Final[dict[str, FeatureExtractor]] = {
    "PS1": extract_pressure_features,
    "PS2": extract_pressure_features,
    "PS3": extract_pressure_features,
    "PS4": extract_pressure_features,
    "PS5": extract_pressure_features,
    "PS6": extract_pressure_features,
    "FS1": extract_flow_features,
    "FS2": extract_flow_features,
    "EPS1": extract_motor_power_features,
    "TS1": extract_temperature_features,
    "TS2": extract_temperature_features,
    "TS3": extract_temperature_features,
    "TS4": extract_temperature_features,
    "VS1": extract_vibration_features,
    "CE": extract_virtual_features,
    "CP": extract_virtual_features,
    "SE": extract_virtual_features,
}


def _validate_inputs(sensors: SensorFrames, profile: pd.DataFrame) -> None:
    """Validate that all required sensors and targets are present and aligned."""
    missing_sensors = [
        sensor_name for sensor_name in SENSOR_EXTRACTORS if sensor_name not in sensors
    ]
    if missing_sensors:
        raise KeyError(f"Missing required sensors: {', '.join(missing_sensors)}")

    missing_targets = [
        target_name for target_name in TARGET_COLUMNS if target_name not in profile
    ]
    if missing_targets:
        raise KeyError(f"Missing required targets: {', '.join(missing_targets)}")

    cycle_count = len(profile)
    mismatched_sensors = {
        sensor_name: len(sensors[sensor_name])
        for sensor_name in SENSOR_EXTRACTORS
        if len(sensors[sensor_name]) != cycle_count
    }
    if mismatched_sensors:
        details = ", ".join(
            f"{sensor_name}={count}"
            for sensor_name, count in mismatched_sensors.items()
        )
        raise ValueError(
            f"Sensor cycle counts must match profile count {cycle_count}: {details}"
        )


def _prefix_features(
    sensor_name: str,
    features: Mapping[str, float],
) -> FeatureVector:
    """Prefix feature names with their originating sensor name."""
    return {
        f"{sensor_name}_{feature_name}": feature_value
        for feature_name, feature_value in features.items()
    }


def extract_cycle_features(
    sensors: SensorFrames,
    cycle_index: int,
    *,
    sensor_extractors: Mapping[str, FeatureExtractor] = SENSOR_EXTRACTORS,
) -> FeatureVector:
    """Extract and combine features from every sensor for one operating cycle."""
    cycle_features: FeatureVector = {}
    for sensor_name, extractor in sensor_extractors.items():
        signal = (
            sensors[sensor_name]
            .iloc[cycle_index]
            .to_numpy(
                dtype="float64",
                copy=False,
            )
        )
        prefixed_features = _prefix_features(sensor_name, extractor(signal))
        duplicate_names = cycle_features.keys() & prefixed_features.keys()
        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Duplicate engineered feature names: {duplicates}")
        cycle_features.update(prefixed_features)
    return cycle_features


def build_feature_matrix(
    sensors: SensorFrames,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """Create one engineered feature row per operating cycle with targets."""
    _validate_inputs(sensors, profile)
    cycle_count = len(profile)
    LOGGER.info(
        "Engineering features for %d cycles across %d sensors",
        cycle_count,
        len(SENSOR_EXTRACTORS),
    )

    feature_rows: list[FeatureVector] = []
    for cycle_index in range(cycle_count):
        feature_rows.append(extract_cycle_features(sensors, cycle_index))
        if (cycle_index + 1) % 250 == 0 or cycle_index + 1 == cycle_count:
            LOGGER.info(
                "Engineered %d/%d operating cycles",
                cycle_index + 1,
                cycle_count,
            )

    features = pd.DataFrame.from_records(feature_rows)
    targets = profile.loc[:, list(TARGET_COLUMNS)].reset_index(drop=True)
    feature_matrix = pd.concat([features, targets], axis="columns")

    if feature_matrix.isna().any(axis=None):
        raise ValueError("The completed feature matrix contains missing values")
    return feature_matrix


def save_feature_matrix(
    feature_matrix: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Save a feature matrix as Parquet and return its resolved output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        feature_matrix.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Saving Parquet requires an installed engine such as pyarrow or "
            "fastparquet"
        ) from exc

    LOGGER.info("Saved feature matrix to %s", output_path)
    return output_path


def print_summary(
    feature_matrix: pd.DataFrame,
    target_columns: Sequence[str],
    output_path: Path,
) -> None:
    """Print a concise summary of the generated feature matrix."""
    target_count = len(target_columns)
    engineered_feature_count = feature_matrix.shape[1] - target_count
    print(f"Operating cycles: {feature_matrix.shape[0]}")
    print(f"Engineered feature columns: {engineered_feature_count}")
    print(f"Target columns: {target_count}")
    print(f"Final DataFrame shape: {feature_matrix.shape}")
    print(f"Output file: {output_path}")


def main() -> None:
    """Load data, engineer cycle-level features, and save them as Parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    sensors, profile = load_hydraulic_dataset()
    feature_matrix = build_feature_matrix(sensors, profile)
    output_path = save_feature_matrix(feature_matrix)
    print_summary(feature_matrix, TARGET_COLUMNS, output_path)


if __name__ == "__main__":
    main()
