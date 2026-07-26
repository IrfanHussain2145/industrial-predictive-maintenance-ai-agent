"""Console-based exploration and validation for the hydraulic dataset.

This module is intentionally read-only: it reports on the in-memory DataFrames
returned by the project loader and does not transform or persist any data.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Final

import pandas as pd

from src.data.loader import load_hydraulic_dataset

LOGGER = logging.getLogger(__name__)

SensorFrames = Mapping[str, pd.DataFrame]
SECTION_WIDTH: Final[int] = 50

SENSOR_CATEGORIES: Final[dict[str, str]] = {
    "PS": "Pressure",
    "EPS": "Motor Power",
    "FS": "Flow",
    "TS": "Temperature",
    "VS": "Vibration",
    "CE": "Cooling Efficiency",
    "CP": "Cooling Power",
    "SE": "Efficiency Factor",
}

def _sensor_category(sensor_name: str) -> str:
    """Return the physical sensor category for a sensor name."""
    for prefix, category in SENSOR_CATEGORIES.items():
        if sensor_name.startswith(prefix):
            return category
    return "Unknown"

def _print_table(table: pd.DataFrame, *, index: bool = False) -> None:
    """Print a DataFrame as an aligned, index-free console table by default."""
    print(table.to_string(index=index))


def _sensor_dtype(frame: pd.DataFrame) -> str:
    """Return a concise representation of the dtypes used by a sensor."""
    dtypes = sorted({str(dtype) for dtype in frame.dtypes})
    return dtypes[0] if len(dtypes) == 1 else ", ".join(dtypes)


def summarize_sensor_inventory(sensors: SensorFrames) -> None:
    """Print shape, cycle count, sample count, and dtype for every sensor."""
    LOGGER.info("Summarizing inventory for %d sensors", len(sensors))
    rows = [
        {
            "Sensor": sensor_name,
            "Category": _sensor_category(sensor_name),
            "Shape": f"{frame.shape[0]} x {frame.shape[1]}",
            "Operating Cycles": frame.shape[0],
            "Samples / Cycle": frame.shape[1],
            "Dtype": _sensor_dtype(frame),
        }
        for sensor_name, frame in sorted(sensors.items())
    ]
    _print_table(pd.DataFrame(rows))


def check_data_quality(sensors: SensorFrames) -> None:
    """Report missing values and duplicate operating cycles for every sensor.

    ``Duplicate Operating Cycles`` counts repeat occurrences beyond the first
    copy. ``Total Duplicate Rows`` counts every row belonging to a duplicated
    group, including its first occurrence.
    """
    LOGGER.info("Checking data quality for %d sensors", len(sensors))
    rows: list[dict[str, int | str]] = []

    for sensor_name, frame in sorted(sensors.items()):
        duplicate_mask = frame.duplicated(keep=False)
        rows.append(
            {
                "Sensor": sensor_name,
                "Missing Values": int(frame.isna().sum().sum()),
                "Duplicate Operating Cycles": int(frame.duplicated(keep="first").sum()),
                "Total Duplicate Rows": int(duplicate_mask.sum()),
            }
        )

    quality = pd.DataFrame(rows)
    _print_table(quality)
    print()

    missing = quality.loc[quality["Missing Values"] > 0, ["Sensor", "Missing Values"]]
    duplicates = quality.loc[
        quality["Duplicate Operating Cycles"] > 0,
        ["Sensor", "Duplicate Operating Cycles", "Total Duplicate Rows"],
    ]

    if missing.empty:
        print("✓ No missing values detected.")
    else:
        affected = ", ".join(
            f"{row['Sensor']} ({row['Missing Values']})"
            for row in missing.to_dict(orient="records")
        )
        print(f"✗ Missing values detected in: {affected}.")

    if duplicates.empty:
        print("✓ No duplicate operating cycles detected.")
    else:
        affected = ", ".join(
            f"{row['Sensor']} "
            f"({row['Duplicate Operating Cycles']} repeated cycles, "
            f"{row['Total Duplicate Rows']} total rows)"
            for row in duplicates.to_dict(orient="records")
        )
        print(f"✗ Duplicates are likely legitimate identical recordings of data from sensors, found in: {affected}.")


def summarize_target_distributions(profile: pd.DataFrame) -> None:
    """Print sorted class counts and percentages for each target column."""
    LOGGER.info("Summarizing distributions for %d targets", profile.shape[1])
    for target_name in profile.columns:
        counts = profile[target_name].value_counts(dropna=False).sort_index()
        distribution = pd.DataFrame(
            {
                "Value": counts.index,
                "Count": counts.to_numpy(),
                "Percentage": [
                    f"{percentage:.2f}%"
                    for percentage in counts.to_numpy() / len(profile) * 100
                ],
            }
        )

        unique_values = ", ".join(str(value) for value in counts.index)
        print(f"{target_name} (unique values: {unique_values})")
        _print_table(
            distribution,
            index=False,
        )
        print()


def _overall_sensor_statistics(
    frame: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Calculate overall min, max, mean, and sample std without flattening data."""
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] != frame.shape[1]:
        raise TypeError("Sensor DataFrames must contain only numeric columns")

    column_counts = numeric.count()
    total_count = int(column_counts.sum())
    if total_count == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    column_means = numeric.mean()
    overall_mean = float((column_means * column_counts).sum() / total_count)

    # Pool within-column and between-column sums of squares. This avoids
    # materializing a potentially very large flattened Series.
    within_sum_squares = (numeric.var(ddof=1).fillna(0.0) * (column_counts - 1)).sum()
    between_sum_squares = (column_counts * (column_means - overall_mean).pow(2)).sum()
    variance = (
        float((within_sum_squares + between_sum_squares) / (total_count - 1))
        if total_count > 1
        else 0.0
    )

    return (
        float(numeric.min().min()),
        float(numeric.max().max()),
        overall_mean,
        variance**0.5,
    )


def summarize_sensor_statistics(sensors: SensorFrames) -> None:
    """Print overall numeric statistics across all values in each sensor."""
    LOGGER.info("Computing overall statistics for %d sensors", len(sensors))
    rows: list[dict[str, float | str]] = []
    for sensor_name, frame in sorted(sensors.items()):
        minimum, maximum, mean, standard_deviation = _overall_sensor_statistics(frame)
        rows.append(
            {
                "Sensor": sensor_name,
                "Min": minimum,
                "Max": maximum,
                "Mean": mean,
                "Std": standard_deviation,
            }
        )

    statistics = pd.DataFrame(rows)
    print(
        statistics.to_string(
            index=False,
            formatters={
                column: lambda value: f"{value:.4f}"
                for column in ("Min", "Max", "Mean", "Std")
            },
        )
    )


def _print_section_header(title: str) -> None:
    """Print a consistent console section header."""
    rule = "=" * SECTION_WIDTH
    print(f"\n{rule}\n{title}\n{rule}\n")


def main() -> None:
    """Load the hydraulic dataset and print exploration results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    sensors, profile = load_hydraulic_dataset()

    _print_section_header("Sensor Inventory")
    summarize_sensor_inventory(sensors)

    _print_section_header("Data Quality")
    check_data_quality(sensors)

    _print_section_header("Target Distributions")
    summarize_target_distributions(profile)

    _print_section_header("Sensor Statistics")
    summarize_sensor_statistics(sensors)


if __name__ == "__main__":
    main()
