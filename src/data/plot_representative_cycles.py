"""Plot representative operating cycles from the hydraulic dataset.

The module loads data exclusively through the project's dataset loader and
writes one PNG for the first operating cycle of each selected sensor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import matplotlib
import pandas as pd

from src.data.loader import load_hydraulic_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

FIGURES_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPRESENTATIVE_SENSORS: Final[tuple[str, ...]] = (
    "PS1",
    "FS1",
    "TS1",
    "EPS1",
    "VS1",
    "CE",
)
MAX_PLOT_POINTS: Final[int] = 600


def _bin_cycle(
    cycle: pd.Series,
    *,
    max_points: int = MAX_PLOT_POINTS,
) -> tuple[pd.Series, pd.Series]:
    """Mean-bin a cycle while retaining its original sample-number scale.

    Cycles at or below ``max_points`` are returned unchanged. Longer cycles
    are divided into consecutive, equally sized groups and represented by the
    mean sample number and mean sensor value in each group.
    """
    if max_points < 1:
        raise ValueError("max_points must be at least 1")
    if cycle.empty:
        raise ValueError("Cannot plot an empty operating cycle")

    values = cycle.reset_index(drop=True)
    sample_numbers = pd.Series(
        range(len(values)),
        index=values.index,
        dtype="float64",
    )
    if len(values) <= max_points:
        return sample_numbers, values

    bin_size = (len(values) + max_points - 1) // max_points
    bin_ids = sample_numbers.astype("int64") // bin_size
    return (
        sample_numbers.groupby(bin_ids, sort=True).mean(),
        values.groupby(bin_ids, sort=True).mean(),
    )


def plot_sensor_cycle(
    sensor_name: str,
    sensor_data: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot and save the first operating cycle for one sensor.

    Args:
        sensor_name: Sensor identifier used in the title and output filename.
        sensor_data: Cycle-by-sample sensor readings.
        output_dir: Directory in which the PNG should be saved.

    Returns:
        The path of the saved PNG.

    Raises:
        ValueError: If the sensor DataFrame has no operating cycles or samples.
    """
    if sensor_data.empty or sensor_data.shape[1] == 0:
        raise ValueError(f"Sensor {sensor_name} contains no plottable data")

    sample_numbers, values = _bin_cycle(sensor_data.iloc[0])
    output_path = output_dir / f"{sensor_name}_cycle0.png"

    figure, axis = plt.subplots(figsize=(10, 4))
    try:
        axis.plot(sample_numbers, values)
        axis.set_title(f"{sensor_name} - Representative Operating Cycle")
        axis.set_xlabel("Sample Number")
        axis.set_ylabel("Sensor Value")
        axis.grid(True)
        figure.tight_layout()
        figure.savefig(output_path, format="png")
    finally:
        plt.close(figure)

    LOGGER.info(
        "Saved %s using %d plotted points from %d samples",
        output_path,
        len(values),
        sensor_data.shape[1],
    )
    return output_path


def main() -> None:
    """Load the dataset and save representative plots for selected sensors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    sensors, profile = load_hydraulic_dataset()
    LOGGER.info(
        "Loaded %d sensors and %d operating cycles",
        len(sensors),
        len(profile),
    )

    missing_sensors = [
        sensor_name
        for sensor_name in REPRESENTATIVE_SENSORS
        if sensor_name not in sensors
    ]
    if missing_sensors:
        missing = ", ".join(missing_sensors)
        raise KeyError(f"Required sensors are missing from the dataset: {missing}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for sensor_name in REPRESENTATIVE_SENSORS:
        plot_sensor_cycle(sensor_name, sensors[sensor_name], FIGURES_DIR)


if __name__ == "__main__":
    main()
