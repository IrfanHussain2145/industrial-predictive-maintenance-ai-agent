"""Reusable feature extractors for one hydraulic-system operating cycle."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.stats import kurtosis, linregress, skew

Signal: TypeAlias = NDArray[np.float64]
FeatureVector: TypeAlias = dict[str, float]


def _validate_signal(signal: Signal, *, min_samples: int = 1) -> Signal:
    """Validate and return a floating-point view of a one-dimensional signal."""
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a NumPy array")
    if signal.ndim != 1:
        raise ValueError(
            f"signal must be one-dimensional; received shape {signal.shape}"
        )
    if signal.size < min_samples:
        raise ValueError(f"signal must contain at least {min_samples} samples")
    if not np.issubdtype(signal.dtype, np.number):
        raise TypeError("signal must contain numeric values")

    values = signal.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("signal must contain only finite values")
    return values


def _rms(signal: Signal) -> float:
    """Return the root mean square of a validated signal."""
    return float(np.sqrt(np.mean(np.square(signal))))


def _slope(signal: Signal) -> float:
    """Return the least-squares slope over zero-based sample positions."""
    sample_numbers = np.arange(signal.size, dtype=np.float64)
    return float(linregress(sample_numbers, signal).slope)


def _basic_features(signal: Signal) -> FeatureVector:
    """Return statistics shared by several sensor categories."""
    minimum = float(np.min(signal))
    maximum = float(np.max(signal))
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal, ddof=0)),
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "rms": _rms(signal),
    }


def _pressure_like_features(signal: Signal) -> FeatureVector:
    """Return features shared by pressure and motor-power signals."""
    values = _validate_signal(signal, min_samples=2)
    derivative = np.diff(values)
    features = _basic_features(values)
    features.update(
        {
            "derivative_mean": float(np.mean(derivative)),
            "derivative_max": float(np.max(derivative)),
        }
    )
    return features


def extract_pressure_features(signal: Signal) -> FeatureVector:
    """Extract time-domain features from one pressure operating cycle."""
    return _pressure_like_features(signal)


def extract_flow_features(signal: Signal) -> FeatureVector:
    """Extract statistical, peak, valley, and derivative flow features."""
    values = _validate_signal(signal, min_samples=2)
    derivative = np.diff(values)
    peak_indices, _ = find_peaks(values)
    valley_indices, _ = find_peaks(-values)

    features = _basic_features(values)
    features.update(
        {
            "peak_count": float(peak_indices.size),
            "max_peak": float(
                np.max(values[peak_indices]) if peak_indices.size else np.max(values)
            ),
            "min_valley": float(
                np.min(values[valley_indices])
                if valley_indices.size
                else np.min(values)
            ),
            "derivative_max": float(np.max(derivative)),
            "derivative_min": float(np.min(derivative)),
        }
    )
    return features


def extract_motor_power_features(signal: Signal) -> FeatureVector:
    """Extract time-domain features from one motor-power operating cycle."""
    return _pressure_like_features(signal)


def extract_temperature_features(signal: Signal) -> FeatureVector:
    """Extract level and trend features from one temperature cycle."""
    values = _validate_signal(signal, min_samples=2)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "slope": _slope(values),
        "initial_value": float(values[0]),
        "final_value": float(values[-1]),
    }


def extract_vibration_features(signal: Signal) -> FeatureVector:
    """Extract distribution and energy features from one vibration cycle."""
    values = _validate_signal(signal, min_samples=4)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    is_constant = bool(np.all(values == values[0]))

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "rms": _rms(values),
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "skewness": 0.0 if is_constant else float(skew(values, bias=False)),
        "kurtosis": (
            0.0 if is_constant else float(kurtosis(values, fisher=True, bias=False))
        ),
        "signal_energy": float(np.sum(np.square(values))),
    }


def extract_virtual_features(signal: Signal) -> FeatureVector:
    """Extract level and trend features from one virtual-sensor cycle."""
    values = _validate_signal(signal, min_samples=2)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    initial_value = float(values[0])
    final_value = float(values[-1])
    return {
        "mean": float(np.mean(values)),
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "slope": _slope(values),
        "initial_value": initial_value,
        "final_value": final_value,
        "net_change": final_value - initial_value,
    }
