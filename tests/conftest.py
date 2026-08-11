"""
conftest.py - shared fixtures for the SASA DSP test suite.

Every generator here produces a signal whose acoustic properties are known in
closed form, so the tests that consume them can assert against an analytic
oracle rather than against whatever the code currently returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The DSP modules are top-level (no package), so the repository root must be
# importable. pytest's default `prepend` import mode only inserts tests/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from calibration import P_REF  # noqa: E402


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def make_sine(
    freq: float,
    level_dB: float,
    fs: int,
    duration: float,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Sine wave with an EXACT RMS level in Pascals.

    A sine of amplitude A has RMS A/sqrt(2), so for a target level L:

        p_rms = p_ref * 10^(L/20)
        A     = sqrt(2) * p_ref * 10^(L/20)

    The RMS of the sampled array equals p_rms exactly (to float64) whenever the
    array spans a whole number of cycles, i.e. freq*duration is an integer:

        sum_{n=0}^{N-1} sin^2(2*pi*k*n/N) = N/2 - (1/2)*sum cos(4*pi*k*n/N)

    and the cosine sum is exactly zero for 0 < 2k < N. Tests that need an exact
    level choose freq*duration integral.
    """
    n = int(round(fs * duration))
    t = np.arange(n, dtype=np.float64) / fs
    amplitude = np.sqrt(2.0) * P_REF * (10.0 ** (level_dB / 20.0))
    return amplitude * np.sin(2.0 * np.pi * freq * t + phase)


def make_friedlander(P0: float, T: float, fs: int, duration: float) -> np.ndarray:
    """
    Friedlander (modified Friedlander) free-field blast wave:

        p(t) = P0 * (1 - t/T) * exp(-t/T)

    Exact analytic properties used as oracles by the metrics tests:

      * peak overpressure is P0, attained at t = 0
      * the positive phase ends exactly at t = T, so A-duration == T
      * specific impulse of the positive phase is

            int_0^T P0 (1 - t/T) e^{-t/T} dt
              = P0*T * int_0^1 (1-u) e^{-u} du
              = P0*T * [ (1 - e^-1) - (1 - 2 e^-1) ]
              = P0*T / e

      * the total impulse over [0, inf) is exactly zero
            int_0^inf (1-u) e^{-u} du = 1 - 1 = 0
        so the waveform carries no DC and a high-pass must not move its peak.
    """
    n = int(round(fs * duration))
    t = np.arange(n, dtype=np.float64) / fs
    return P0 * (1.0 - t / T) * np.exp(-t / T)


def make_decaying_sinusoid(
    amplitude: float,
    tau: float,
    freq: float,
    fs: int,
    n: int,
) -> np.ndarray:
    """
    A * exp(-t/tau) * sin(2*pi*f*t).

    Its analytic-signal envelope is A*exp(-t/tau), which crosses -20 dB of peak
    at exactly t = tau*ln(10). That is the oracle for the B-duration test.
    """
    t = np.arange(n, dtype=np.float64) / fs
    return amplitude * np.exp(-t / tau) * np.sin(2.0 * np.pi * freq * t)


def make_shot_train(
    fs: int,
    times,
    *,
    duration: float = 2.0,
    amplitude: float = 0.95,
    amplitude_jitter: float = 0.0,
    tau: float = 0.004,
    freq: float = 900.0,
    noise_rms: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """
    A string of impulsive events at the requested times.

    Each event is a decaying sinusoid, which is a reasonable stand-in for the
    band-limited ringdown a microphone actually records. The peak of the whole
    signal is `amplitude`, so a train built with amplitude ~1.0 is an
    UNCALIBRATED full-scale recording (ceiling ~94 dB re 20 uPa with
    Pa_per_FS = 1).
    """
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration))
    x = rng.normal(0.0, noise_rms, n) if noise_rms > 0 else np.zeros(n)

    for k, t0 in enumerate(times):
        i0 = int(round(t0 * fs))
        if i0 >= n:
            continue
        length = min(n - i0, int(round(0.25 * fs)))
        amp = amplitude * (1.0 - amplitude_jitter * (k % 3) / 2.0)
        x[i0:i0 + length] += make_decaying_sinusoid(amp, tau, freq, fs, length)
    return x


def make_white_noise(level_dB: float, fs: int, duration: float, seed: int = 0) -> np.ndarray:
    """
    Gaussian white noise rescaled to an EXACT RMS level in Pascals.

    p_rms = p_ref * 10^(L/20); the realised sample RMS is forced to that value,
    so broadband-level assertions have no sampling error.
    """
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration))
    x = rng.normal(0.0, 1.0, n)
    x -= x.mean()
    target = P_REF * (10.0 ** (level_dB / 20.0))
    return x * (target / float(np.sqrt(np.mean(x ** 2))))


def rms(x) -> float:
    """Root-mean-square of an array."""
    a = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(a ** 2)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine():
    return make_sine


@pytest.fixture
def friedlander():
    return make_friedlander


@pytest.fixture
def decaying_sinusoid():
    return make_decaying_sinusoid


@pytest.fixture
def shot_train():
    return make_shot_train


@pytest.fixture
def white_noise():
    return make_white_noise


# Sample rates the instrument is expected to support end to end.
SAMPLE_RATES = (44100, 48000, 96000, 192000)


@pytest.fixture(params=SAMPLE_RATES)
def any_sample_rate(request):
    return request.param
