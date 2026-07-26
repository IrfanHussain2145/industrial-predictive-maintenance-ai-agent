"""Load and validate the UCI Hydraulic Systems dataset.

Each row in a sensor file represents one operating cycle and each column is a
sample recorded during that cycle.  ``profile.txt`` contains the condition
labels for the same cycles.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_RAW_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "raw"
EXCLUDED_TEXT_FILES: Final[frozenset[str]] = frozenset(
    {"description.txt", "documentation.txt", "profile.txt"}
)
PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    "cooler_condition",
    "valve_condition",
    "pump_condition",
    "accumulator_condition",
    "stable_flag",
)


class DatasetValidationError(ValueError):
    """Raised when the dataset is incomplete or internally inconsistent."""


def discover_sensor_files(raw_data_dir: Path | str) -> list[Path]:
    """Return the sensor text files in *raw_data_dir*, sorted by filename.

    Metadata, documentation, and profile-label files are excluded
    case-insensitively.

    Args:
        raw_data_dir: Directory containing the extracted dataset files.

    Raises:
        FileNotFoundError: If *raw_data_dir* does not exist or is not a directory.
        DatasetValidationError: If no sensor files are found.
    """
    directory = Path(raw_data_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {directory}")

    sensor_files = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".txt"
            and path.name.casefold() not in EXCLUDED_TEXT_FILES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not sensor_files:
        raise DatasetValidationError(
            f"No sensor .txt files found in raw data directory: {directory}"
        )

    LOGGER.info("Discovered %d sensor files in %s", len(sensor_files), directory)
    return sensor_files


def _read_numeric_table(
    path: Path,
    *,
    usecols: Sequence[int] | None = None,
    column_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a headerless, whitespace-delimited numeric table."""
    try:
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            usecols=usecols,
            names=column_names,
            dtype=float,
            engine="c"
        )
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise DatasetValidationError(
            f"Unable to load numeric data from {path}"
        ) from exc

    if frame.empty:
        raise DatasetValidationError(f"Data file is empty: {path}")
    if frame.isna().any(axis=None):
        raise DatasetValidationError(f"Data file contains missing values: {path}")
    return frame


def load_sensor(path: Path | str) -> pd.DataFrame:
    """Load one sensor file as a cycle-by-sample DataFrame.

    Args:
        path: Path to a headerless, whitespace-delimited sensor file.

    Returns:
        A numeric DataFrame whose rows are operating cycles.
    """
    sensor_path = Path(path).expanduser()
    LOGGER.debug("Loading sensor %s from %s", sensor_path.stem, sensor_path)
    return _read_numeric_table(sensor_path)


def load_sensors(raw_data_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Discover and load all sensor files keyed by their filename stem."""
    sensors: dict[str, pd.DataFrame] = {}
    for path in discover_sensor_files(raw_data_dir):
        sensor_name = path.stem
        if sensor_name in sensors:
            raise DatasetValidationError(
                f"Duplicate sensor name after removing file extension: {sensor_name}"
            )
        sensors[sensor_name] = load_sensor(path)

    LOGGER.info("Loaded %d sensors", len(sensors))
    return sensors


def load_profile(raw_data_dir: Path | str) -> pd.DataFrame:
    """Load the four requested condition targets from ``profile.txt``.

    The source dataset also contains a fifth stability indicator. It is
    intentionally omitted because it is not one of the requested targets.
    Target values are returned as integers.
    """
    profile_path = Path(raw_data_dir).expanduser() / "profile.txt"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile file does not exist: {profile_path}")

    profile = _read_numeric_table(
        profile_path,
        usecols=range(len(PROFILE_COLUMNS)),
        column_names=PROFILE_COLUMNS,
    )
    if not all((profile[column] % 1 == 0).all() for column in PROFILE_COLUMNS):
        raise DatasetValidationError(
            f"Profile labels must contain integer values: {profile_path}"
        )

    profile = profile.astype("int64")
    LOGGER.info("Loaded %d profile label rows from %s", len(profile), profile_path)
    return profile


def validate_cycle_counts(
    sensors: Mapping[str, pd.DataFrame], profile: pd.DataFrame
) -> None:
    """Ensure every sensor contains exactly one row per profile cycle.

    Raises:
        DatasetValidationError: If sensors are absent or any row count differs.
    """
    if not sensors:
        raise DatasetValidationError("At least one sensor is required")
    if profile.empty:
        raise DatasetValidationError("Profile labels are empty")

    expected_cycles = len(profile)
    mismatches = {
        name: len(frame)
        for name, frame in sensors.items()
        if len(frame) != expected_cycles
    }
    if mismatches:
        details = ", ".join(
            f"{name}={cycle_count}" for name, cycle_count in sorted(mismatches.items())
        )
        raise DatasetValidationError(
            f"Sensor cycle counts do not match profile ({expected_cycles}): {details}"
        )

    LOGGER.info(
        "Validated %d operating cycles across %d sensors",
        expected_cycles,
        len(sensors),
    )


def load_hydraulic_dataset(
    raw_data_dir: Path | str = DEFAULT_RAW_DATA_DIR,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all sensors and profile targets, then validate cycle alignment."""
    directory = Path(raw_data_dir).expanduser()
    sensors = load_sensors(directory)
    profile = load_profile(directory)
    validate_cycle_counts(sensors, profile)
    return sensors, profile


def _format_summary(sensors: Mapping[str, pd.DataFrame], profile: pd.DataFrame) -> str:
    """Build a concise human-readable dataset summary."""
    sensor_shapes = ", ".join(
        f"{name}={frame.shape[0]}x{frame.shape[1]}"
        for name, frame in sorted(sensors.items())
    )
    targets = ", ".join(str(column) for column in profile.columns)
    return (
        f"Loaded {len(sensors)} sensors ({len(profile)} operating cycles)\n"
        f"Sensors [cycles x samples]: {sensor_shapes}\n"
        f"Target labels ({profile.shape[1]}): {targets}"
    )


def main() -> None:
    """Load the default dataset and print a concise summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    sensors, profile = load_hydraulic_dataset()
    print(_format_summary(sensors, profile))


if __name__ == "__main__":
    main()
