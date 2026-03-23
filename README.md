<p align="center">
  <strong>S A S A</strong><br>
  <em>Shot Acoustic Spectral Analysis</em><br>
  <sub>by Ridgeback Defense</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/license-proprietary-red" alt="License">
</p>

---

**SASA** is a professional-grade acoustic analysis tool purpose-built for gunshot WAV recordings. It transforms raw audio into calibrated sound pressure level (SPL) measurements, spectrograms, 1/3-octave band analyses, and a full suite of ISO/IEC-standard acoustic metrics — all from a single command or web-based UI.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
  - [From Source (Python)](#from-source-python)
  - [Standalone App (macOS / Windows)](#standalone-app-macos--windows)
  - [Web UI](#web-ui)
- [Quick Start](#quick-start)
- [Calibration](#calibration)
- [The Analysis Pipeline](#the-analysis-pipeline)
  - [1. Audio Loading](#1-audio-loading)
  - [2. Calibration — Digital to Pascals](#2-calibration--digital-to-pascals)
  - [3. Shot Detection](#3-shot-detection)
  - [4. Frequency Weighting](#4-frequency-weighting)
  - [5. Short-Time Fourier Transform (STFT)](#5-short-time-fourier-transform-stft)
  - [6. 1/3-Octave Band Analysis](#6-13-octave-band-analysis)
  - [7. Acoustic Metrics](#7-acoustic-metrics)
  - [8. Visualization](#8-visualization)
- [Metrics Reference](#metrics-reference)
- [Command Line Options](#command-line-options)
- [Output Files](#output-files)
- [Module Reference](#module-reference)
- [Building from Source](#building-from-source)
- [Standards Compliance](#standards-compliance)
- [License](#license)

---

## Features

- **Calibrated SPL measurements** — converts digital samples to physical Pascals using microphone sensitivity data
- **Automatic shot detection** — finds gunshot events using RMS envelope peak-picking with configurable thresholds and refractory periods
- **Three frequency weightings** — Z (flat/unweighted), A (human hearing), C (peak measurements), all per IEC 61672-1
- **Three time weightings** — Fast (125 ms), Slow (1 s), Impulse (35 ms attack / 1500 ms decay)
- **STFT spectrograms** — calibrated time-frequency display with Hann windowing and 75% overlap
- **1/3-octave band analysis** — ISO 266 center frequencies with IEC 61260-1 bandpass filters
- **Gunshot-specific metrics** — rise time, B-duration, crest factor, spectral centroid, kurtosis
- **Per-shot and aggregate statistics** — CSV summaries, JSON metadata, multi-panel PNG figures
- **Interactive HTML plots** — zoomable/pannable spectrograms and waveforms via Plotly
- **Publication-quality dark-themed plots** — consistent tactical-style Matplotlib figures
- **Web UI** — browser-based interface with drag-and-drop file upload and real-time analysis progress
- **Cross-platform** — runs on macOS and Windows as a standalone app or from Python source

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SASA Pipeline                            │
│                                                                 │
│  WAV File ──→ Calibration ──→ Shot Detection ──→ Per-Shot       │
│                   │                                Analysis     │
│                   ▼                                   │         │
│            Pressure (Pa)                              ▼         │
│                   │              ┌─────────────────────────────┐│
│        ┌──────────┼──────────┐   │    Per-Shot Metrics:        ││
│        ▼          ▼          ▼   │    • Lpeak (Z/A/C)          ││
│    Z-Weight   A-Weight   C-Weight│    • LAE, LZE, LCE          ││
│        │          │          │   │    • LAFmax, LASmax         ││
│        ▼          ▼          ▼   │    • LAImax, LZImax         ││
│      STFT      STFT       STFT   │    • Rise time, B-duration  ││
│    (Z-spec)  (A-spec)   (C-spec) │    • Crest factor, Kurtosis ││
│        │          │          │   │    • Spectral centroid      ││
│        ▼          ▼          ▼   │    • 1/3-octave band SEL    ││
│   Spectrogram  Spectrogram   │   └─────────────────────────────┘│
│     Plots       Plots        │                                  │
│                              │                                  │
│   1/3-Octave Bands ──→ Heatmap + Bar Charts                     │
│                                                                 │
│   Time-Weighted Levels ──→ LAF/LAS/LZF/LZS Curves               │ 
│                                                                 │
│   Aggregate Stats ──→ CSV + JSON + Summary Figures              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### From Source (Python)

**Requirements:** Python 3.10 or newer.

```bash
# Clone the repository
git clone https://github.com/ridgeback-defense/SASA.git
cd SASA

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: video file support
pip install moviepy imageio-ffmpeg

# Optional: ISO 532-1 loudness
pip install mosqito
```

### Standalone App (macOS / Windows)

Download the latest release from the [Releases](https://github.com/ridgeback-defense/SASA/releases) page:

| Platform | Download | Notes |
|----------|----------|-------|
| **macOS** | `SASA-macOS.zip` | Unzip → double-click `SASA.app` |
| **Windows** | `SASA-Windows.zip` | Unzip → run `SASA.exe` |

Or build it yourself:

```bash
# macOS
chmod +x build_macos.sh
./build_macos.sh

# Windows
build_windows.bat
```

### Web UI

SASA includes a browser-based interface powered by a Node.js server that bridges to the Python backend.

```bash
# Install UI dependencies (requires Node.js)
cd ui && npm install && cd ..

# Launch the UI (opens browser automatically)
node ui/server.js
```

The UI runs at `http://localhost:3847` and provides drag-and-drop file upload, real-time analysis progress, and interactive result exploration.

---

## Quick Start

```bash
# Interactive file selection (opens native file picker)
python main.py

# Analyze a specific file
python main.py path/to/recording.wav

# With calibration
python main.py recording.wav --Pa-per-FS 50.0

# With microphone sensitivity
python main.py recording.wav --sensitivity-mV 10.0 --V-per-FS 1.0

# Load settings from a config file
python main.py recording.wav --config my_config.json
```

---

## Calibration

For physically meaningful dB SPL values, SASA needs to know how to convert digital sample values to pressure in Pascals. Without calibration, results are in "relative dB" — internally consistent but not traceable to absolute sound pressure levels.

### Method 1: Direct Factor (`Pa_per_FS`)

If you know how many Pascals correspond to a full-scale digital value of ±1.0:

```bash
python main.py recording.wav --Pa-per-FS 50.0
```

This applies the conversion:

```math
p(t) = x_{\mathrm{digital}}(t) \times \mathrm{Pa{\_}per{\_}FS}
```

### Method 2: Microphone Sensitivity

If you have the microphone datasheet:

```bash
python main.py recording.wav --sensitivity-mV 10.0 --V-per-FS 1.0
```

The conversion factor is derived as:

```math
\mathrm{Pa{\_}per{\_}FS} = \frac{V_{\mathrm{per{\_}FS}}}{S_{\mathrm{mic}}} = \frac{V_{\mathrm{per{\_}FS}}}{S_{\mathrm{mV/Pa}} \times 10^{-3}}
```

**Example:** A microphone with 10 mV/Pa sensitivity and a recorder where ±1.0 digital = 1 V:

```math
\mathrm{Pa{\_}per{\_}FS} = \frac{1.0}{0.01} = 100 \;\mathrm{Pa/FS}
```

This means a full-scale sample represents 100 Pa, which is about 154 dB SPL.

---

## The Analysis Pipeline

### 1. Audio Loading

**Module:** `WavLoader.py`

SASA loads WAV files using `libsndfile` (via the `soundfile` library) at the file's native sample rate — **no resampling is performed**. This preserves the full bandwidth of high-sample-rate recordings (e.g., 96 kHz or 192 kHz), which is critical for accurate high-frequency analysis of gunshot spectra.

- Files are loaded as 32-bit floating point (`float32`) normalized to [-1, 1]
- Multichannel files are averaged to mono for analysis
- For files longer than 10 minutes, SASA uses chunked loading to limit RAM usage

### 2. Calibration — Digital to Pascals

**Module:** `calibration.py`

The calibration module converts dimensionless digital samples into physical pressure units (Pascals), enabling dB SPL calculations referenced to the standard threshold of hearing.

**Core conversion:**

```math
p(t) = x_{\mathrm{digital}}(t) \times \mathrm{Pa{\_}per{\_}FS}
```

**Sound Pressure Level (SPL):**

SPL converts pressure amplitude to a logarithmic decibel scale referenced to the threshold of human hearing:

```math
L_p = 20 \cdot \log_{10}\!\left(\frac{|p|}{p_{\mathrm{ref}}}\right) \quad \mathrm{dB\;SPL}
```

where $p_{\mathrm{ref}} = 20\;\mu\mathrm{Pa} = 2 \times 10^{-5}\;\mathrm{Pa}$.

**Power-based SPL** (used for time-weighted levels where mean-square pressure is computed):

```math
L_p = 10 \cdot \log_{10}\!\left(\frac{\overline{p^2}}{p_{\mathrm{ref}}^2}\right) \quad \mathrm{dB\;SPL}
```

**Typical reference levels:**

| Pressure | dB SPL | Description |
|----------|--------|-------------|
| 20 µPa | 0 dB | Threshold of hearing |
| 1 Pa | 94 dB | Standard calibrator level |
| 20 Pa | 120 dB | Threshold of pain |
| 200 Pa | 140 dB | Near-field gunshot |
| 6,300 Pa | 170 dB | Muzzle blast (unprotected) |

### 3. Shot Detection

**Module:** `shot_detect.py`

SASA detects gunshot events using an RMS envelope peak-picking algorithm with a refractory period to prevent double-triggering on echoes or reflections.

**Algorithm:**

1. **Compute short-term RMS envelope** over ~1 ms sliding windows with 0.5 ms hop:

```math
e[n] = \sqrt{\frac{1}{W} \sum_{k=0}^{W-1} p^2[nH + k]}
```

where $W$ is the window length in samples and $H$ is the hop size.

2. **Threshold detection** — find envelope samples exceeding the configured threshold:

```math
e[n] > p_{\mathrm{ref}} \times 10^{\,L_{\mathrm{threshold}} / 20}
```

3. **Peak picking** — within each contiguous above-threshold region, find the sample with maximum envelope value.

4. **Refractory period** — suppress detections within a configurable dead time (default 200 ms) of a previous detection. This prevents echoes, reflections, and reverberant tails from being counted as separate shots.

5. **Peak refinement** — the approximate peak from the envelope is refined by searching for the exact maximum absolute pressure within ±500 samples of the envelope peak.

6. **Window extraction** — each shot is assigned a pre-trigger (default 50 ms) and post-trigger (default 200 ms) window for per-shot analysis.

### 4. Frequency Weighting

**Module:** `weighting.py`

Frequency weighting shapes the signal to match specific response curves. SASA implements three standard weightings defined by IEC 61672-1:2013:

#### Z-Weighting (Flat / Unweighted)

Z-weighting applies no frequency shaping — the signal passes through unchanged. This shows the raw physical acoustic energy at all frequencies.

```math
H_Z(f) = 1 \quad \text{(0 dB at all frequencies)}
```

#### A-Weighting (Human Hearing)

A-weighting approximates the sensitivity of human hearing at moderate levels. It attenuates low frequencies (where humans are less sensitive) and slightly boosts the 2–4 kHz region (where the ear is most sensitive due to ear canal resonance).

**Analytical formula (IEC 61672-1):**

```math
R_A(f) = \frac{12194.217^2 \cdot f^4}{(f^2 + 20.598997^2)\,\sqrt{(f^2 + 107.65265^2)(f^2 + 737.86223^2)}\,(f^2 + 12194.217^2)}
```

```math
A(f) = 20 \cdot \log_{10}\!\big(R_A(f)\big) + 2.0 \quad \mathrm{dB}
```

The +2.0 dB offset normalizes the curve to 0 dB at 1000 Hz.

**IIR filter implementation:**

The analog A-weighting filter has:
- 4 zeros at $s = 0$ (high-pass character)
- Double poles at $f_1 = 20.6$ Hz and $f_4 = 12{,}194$ Hz
- Single poles at $f_2 = 107.7$ Hz and $f_3 = 737.9$ Hz

SASA converts this to a digital IIR filter using the bilinear transform, then implements it as cascaded second-order sections (SOS) for numerical stability. The filter gain is normalized so that |H(1000 Hz)| = 0 dB.

**Approximate A-weighting corrections at standard frequencies:**

| Frequency | A-weighting |
|-----------|-------------|
| 31.5 Hz | -39.4 dB |
| 63 Hz | -26.2 dB |
| 125 Hz | -16.1 dB |
| 250 Hz | -8.6 dB |
| 500 Hz | -3.2 dB |
| 1000 Hz | 0.0 dB |
| 2000 Hz | +1.2 dB |
| 4000 Hz | +1.0 dB |
| 8000 Hz | -1.1 dB |
| 16000 Hz | -6.6 dB |

#### C-Weighting (Peak Measurements)

C-weighting is flatter than A-weighting, primarily attenuating only the very low and very high frequencies. It is used for peak sound pressure measurements in some standards.

```math
R_C(f) = \frac{12194.217^2 \cdot f^2}{(f^2 + 20.598997^2)(f^2 + 12194.217^2)}
```

```math
C(f) = 20 \cdot \log_{10}\!\big(R_C(f)\big) + 0.062 \quad \mathrm{dB}
```

#### Zero-Phase Filtering

For offline (post-hoc) analysis of recorded signals, SASA uses zero-phase filtering (`sosfiltfilt` — forward-backward filtering) rather than causal filtering. This eliminates:
- **Startup transients** — critical for short shot windows where the filter hasn't settled
- **Group delay** — peak location is preserved exactly
- **Phase distortion** — symmetric frequency response

### 5. Short-Time Fourier Transform (STFT)

**Module:** `STFT.py`

The STFT decomposes the pressure signal into a time-frequency representation, showing how spectral energy evolves over time — essential for visualizing the frequency content of gunshot impulses, muzzle blast, and reverberant tails.

**Mathematical formulation:**

Given a discrete signal $x[n]$, the STFT at frame $m$ and frequency bin $k$ is:

```math
X[m, k] = \sum_{n=0}^{N-1} x[mH + n] \cdot w[n] \cdot e^{-j2\pi kn/N}
```

where:
- $N$ = window size (`nperseg`, default 2048)
- $H$ = hop size ( $N - \mathrm{noverlap}$ , default $N/4 = 512$ )
- $w[n]$ = window function (Hann by default)
- $k$ = frequency bin index, $f_k = k \cdot f_s / N$

**Amplitude scaling:**

For calibrated amplitude output, SASA normalizes by the window sum and applies a factor of 2 for the one-sided spectrum:

```math
|X_{\mathrm{amp}}[m, k]| = \frac{2 \cdot |\mathrm{FFT}(x_m \cdot w)|_k}{\sum_{n=0}^{N-1} w[n]}
```

DC ( $k=0$ ) and Nyquist ( $k=N/2$ ) bins are corrected by dividing by 2 since they are not mirrored.

**Conversion to dB SPL:**

The amplitude is converted to RMS (dividing by $\sqrt{2}$ since each bin represents a sinusoidal component) and then to dB SPL:

```math
L[m, k] = 20 \cdot \log_{10}\!\left(\frac{|X_{\mathrm{amp}}[m, k]| \;/\; \sqrt{2}}{p_{\mathrm{ref}}}\right)
```

**Frequency weighting in the spectral domain:**

A-weighting can be applied directly to STFT magnitude by multiplying each frequency bin by the A-weighting curve:

```math
|X_A[m, k]| = |X[m, k]| \cdot 10^{A(f_k)/20}
```

**Supported window functions:**

| Window | Main-lobe width | Side-lobe level | Use case |
|--------|----------------|-----------------|----------|
| Hann | Moderate | -31 dB | General purpose (default) |
| Hamming | Moderate | -43 dB | Slightly less leakage |
| Blackman | Wide | -58 dB | Minimum leakage |
| Rectangular | Narrow | -13 dB | Maximum time resolution |

### 6. 1/3-Octave Band Analysis

**Module:** `bands.py`

1/3-octave band analysis divides the spectrum into logarithmically spaced frequency bands, each 1/3 of an octave wide. This representation matches how the human auditory system groups frequencies and is the basis for many noise standards.

**ISO 266:1997 center frequencies (Hz):**

```
20  25  31.5  40  50  63  80  100  125  160  200  250  315
400  500  630  800  1000  1250  1600  2000  2500  3150  4000
5000  6300  8000  10000  12500  16000  20000
```

**Band edge frequencies:**

For a 1/3-octave band centered at $f_c$ :

```math
f_{\mathrm{low}} = \frac{f_c}{2^{1/6}} \qquad f_{\mathrm{high}} = f_c \cdot 2^{1/6}
```

The ratio $2^{1/6} \approx 1.1225$ gives each band a relative bandwidth of about 23%.

**Bandpass filter design:**

Each band uses a 4th-order Butterworth bandpass filter (IIR, implemented as cascaded SOS). The 4th order meets IEC 61260-1:2014 Class 1 filter slope requirements.

**Time weighting:**

After bandpass filtering, the squared pressure in each band is exponentially averaged:

```math
y[n] = \alpha \cdot x^2[n] + (1 - \alpha) \cdot y[n-1]
```

where the smoothing coefficient depends on the time constant $\tau$ :

```math
\alpha = 1 - e^{-\Delta t / \tau}
```

| Time weighting | Time constant | Behavior |
|----------------|---------------|----------|
| **Fast** | 125 ms | Responsive; standard for SLM readings |
| **Slow** | 1000 ms | Smoothed; averages out fluctuations |
| **Impulse** | 35 ms attack, 1500 ms decay | Captures fast transients, holds the reading |

The **Impulse** time weighting uses an asymmetric detector per IEC 61672-1:

```math
y[n] = \begin{cases} \alpha_{\mathrm{attack}} \cdot x^2[n] + (1 - \alpha_{\mathrm{attack}}) \cdot y[n-1] & \text{if } x^2[n] > y[n-1] \\ \alpha_{\mathrm{decay}} \cdot x^2[n] + (1 - \alpha_{\mathrm{decay}}) \cdot y[n-1] & \text{otherwise} \end{cases}
```

This design rapidly captures the peak of an impulse (35 ms rise) while holding the reading for 1.5 seconds, making it specifically suited for impulsive noise like gunshots.

**Conversion to dB SPL:**

```math
L_{\mathrm{band}}[m, b] = 10 \cdot \log_{10}\!\left(\frac{y_b[m]}{p_{\mathrm{ref}}^2}\right)
```

### 7. Acoustic Metrics

**Module:** `metrics.py`

SASA computes a comprehensive set of metrics for each detected shot, following ISO/IEC standard definitions.

#### Peak Sound Pressure Level

The maximum instantaneous pressure, with optional frequency weighting:

```math
L_{\mathrm{peak}}(W) = 20 \cdot \log_{10}\!\left(\frac{\max\!\big(|p_W(t)|\big)}{p_{\mathrm{ref}}}\right)
```

where $W \in \\{Z, A, C\\}$ denotes the frequency weighting applied before peak detection.

| Metric | Description |
|--------|-------------|
| **Lpeak(Z)** | Peak physical pressure — maximum instantaneous level, unweighted |
| **Lpeak(A)** | Peak after A-weighting — perceptually weighted peak |
| **Lpeak(C)** | Peak after C-weighting — used for peak limit compliance |

#### Sound Exposure Level (SEL)

SEL integrates the total squared pressure over the duration of the event and normalizes to a 1-second reference duration. This allows comparison of events with different durations: a 10 ms gunshot and a 2-second vehicle pass can be compared on equal footing.

```math
L_{WE} = 10 \cdot \log_{10}\!\left(\frac{\int_0^T p_W^2(t)\,dt}{p_{\mathrm{ref}}^2 \cdot T_{\mathrm{ref}}}\right) \qquad T_{\mathrm{ref}} = 1\;\mathrm{s}
```

In discrete form:

```math
L_{WE} = 10 \cdot \log_{10}\!\left(\frac{\sum_{n=0}^{N-1} p_W^2[n] \cdot \Delta t}{p_{\mathrm{ref}}^2}\right) \qquad \Delta t = \frac{1}{f_s}
```

| Metric | Weighting |
|--------|-----------|
| **LAE** | A-weighted SEL |
| **LZE** | Z-weighted (unweighted) SEL |
| **LCE** | C-weighted SEL |

#### Maximum Time-Weighted Levels

The peak value of the exponentially time-weighted level curve. Fast (125 ms) captures the subjectively perceived maximum; Slow (1 s) smooths out fluctuations.

```math
L_{WF\max} = \max_t\!\left[10 \cdot \log_{10}\!\left(\frac{y_{\mathrm{fast}}(t)}{p_{\mathrm{ref}}^2}\right)\right]
```

| Metric | Time constant | Description |
|--------|--------------|-------------|
| **LAFmax** | 125 ms | Maximum A-weighted, Fast — standard impulsive noise metric |
| **LASmax** | 1000 ms | Maximum A-weighted, Slow — smoothed maximum |
| **LAImax** | 35 ms / 1500 ms | Maximum A-weighted, Impulse — captures transient peaks |
| **LZFmax** | 125 ms | Maximum Z-weighted, Fast |
| **LZSmax** | 1000 ms | Maximum Z-weighted, Slow |
| **LZImax** | 35 ms / 1500 ms | Maximum Z-weighted, Impulse |

#### Gunshot-Specific Metrics

These metrics characterize the impulsive nature of the acoustic event:

**Rise Time (10–90%)**

Measures how quickly the pressure impulse develops, from 10% to 90% of peak absolute pressure. Typical gunshot rise times: 1–50 µs for muzzle blast.

```math
t_{\mathrm{rise}} = t_{90\%} - t_{10\%}
```

where $|p(t_{10\\%})| = 0.1 \cdot |p|_{\max}$ and $|p(t_{90\\%})| = 0.9 \cdot |p|_{\max}$ , searching backwards from the peak.

**B-Duration**

The total time the signal envelope remains within 20 dB of the peak level. This measures the effective duration of the impulse including oscillatory decay. Typical gunshot B-durations: 2–20 ms.

```math
T_B = \frac{1}{f_s} \sum_{n} \mathbf{1}\!\left[|p[n]| \geq 0.1 \cdot |p|_{\max}\right]
```

(The factor 0.1 in linear amplitude corresponds to -20 dB.)

**Crest Factor**

The ratio of peak to RMS, expressed in dB. Quantifies the "peakiness" of the signal. A pure sine wave has a crest factor of 3.01 dB; gunshots typically have 15–30 dB.

```math
C = 20 \cdot \log_{10}\!\left(\frac{|p|_{\max}}{p_{\mathrm{rms}}}\right) \qquad p_{\mathrm{rms}} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} p^2[n]}
```

**Spectral Centroid**

The frequency "center of mass" of the power spectrum. Higher values indicate more high-frequency energy. Useful for distinguishing weapon types and suppressor effects.

```math
f_c = \frac{\sum_k f_k \cdot |X_k|^2}{\sum_k |X_k|^2}
```

where $X_k$ is the Hann-windowed FFT of the shot segment.

**Kurtosis (Excess)**

Measures how "peaked" the amplitude distribution is relative to a Gaussian. A Gaussian has excess kurtosis of 0; gunshots typically have kurtosis >> 10, indicating extreme impulsiveness. Used in MIL-STD-1474E for impulsive noise assessment.

```math
\kappa = \frac{\mu_4}{\mu_2^2} - 3 = \frac{\frac{1}{N}\sum(p[n] - \bar{p})^4}{\left(\frac{1}{N}\sum(p[n] - \bar{p})^2\right)^2} - 3
```

#### Aggregate Statistics

When multiple shots are detected, SASA computes energy-averaged aggregate metrics. The energy average is the correct way to average decibel values (rather than arithmetic mean of dB, which underestimates the true level):

```math
\bar{L} = 10 \cdot \log_{10}\!\left(\frac{1}{N}\sum_{i=1}^{N} 10^{L_i/10}\right)
```

### 8. Visualization

**Module:** `plots.py`

SASA generates two types of visualizations:

#### Static Plots (Matplotlib)

Publication-quality PNG/PDF/SVG figures with a consistent dark tactical theme:

- **Waveform** — pressure in Pascals vs. time, with shot markers and a secondary dB SPL axis
- **Z-weighted spectrogram** — time-frequency heatmap in dB SPL (magma colormap)
- **A/C-weighted spectrogram** — same, with perceptual weighting applied
- **1/3-octave heatmap** — band level vs. time (inferno colormap)
- **Level curves** — LAF, LAS, LZF, LZS time series
- **Band exposure bar chart** — per-band SEL in dB
- **Multi-panel shot summary** — 6-panel figure combining all of the above for each shot

#### Interactive Plots (Plotly)

HTML files with pan, zoom, and hover tooltips:

- **Waveform** — full-resolution around shot events, downsampled elsewhere for file size
- **Spectrograms** — zoomable time-frequency display

---

## Metrics Reference

### Quick Reference Table

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Lpeak(Z)** | `20 log₁₀(|p|_max / p_ref)` | Maximum instantaneous pressure (unweighted) |
| **Lpeak(A)** | `20 log₁₀(|p_A|_max / p_ref)` | Maximum instantaneous pressure (A-weighted) |
| **LAE** | `10 log₁₀(∫ p_A² dt / p_ref²)` | Total A-weighted energy in 1 second |
| **LAFmax** | `max[L_AF(t)]` | Peak of 125 ms A-weighted average |
| **LASmax** | `max[L_AS(t)]` | Peak of 1 s A-weighted average |
| **LAImax** | `max[L_AI(t)]` | Peak of Impulse A-weighted detector |
| **Rise time** | `t_90% − t_10%` | 10–90% pressure rise time (µs) |
| **B-duration** | `Σ 𝟙[|p| ≥ 0.1·|p|_max] / f_s` | Time within 20 dB of peak (ms) |
| **Crest factor** | `20 log₁₀(|p|_max / p_rms)` | Peak-to-RMS ratio (dB) |
| **Spectral centroid** | `Σ f_k·|X_k|² / Σ |X_k|²` | Frequency center of mass (Hz) |
| **Kurtosis** | `μ₄ / μ₂² − 3` | Impulsiveness (0 = Gaussian) |

### Understanding the Metrics

- **Lpeak** indicates the absolute maximum pressure — important for hearing damage risk assessment and compliance with peak exposure limits (e.g., MIL-STD-1474E: 140 dB peak).
- **LAE** (SEL) captures total auditory exposure — useful for comparing single events of different durations and computing cumulative dose.
- **LAFmax** is the standard metric for impulsive noise assessment and correlates well with perceived loudness.
- **LAImax** uses the IEC Impulse detector specifically designed for transient sounds — it captures the peak more accurately than Fast for very short impulses.
- **Rise time** differentiates weapon types: supersonic crack (~1 µs) vs. muzzle blast (~10-50 µs) vs. suppressed (~100+ µs).
- **Kurtosis** quantifies impulsiveness independent of level — useful for classifying noise environments per MIL-STD-1474E.

---

## Command Line Options

### Calibration

| Option | Default | Description |
|--------|---------|-------------|
| `--Pa-per-FS` | 143.96 | Direct calibration factor (Pa per full-scale) |
| `--sensitivity-mV` | — | Microphone sensitivity in mV/Pa |
| `--V-per-FS` | — | Recorder full-scale voltage |
| `--cal-desc` | — | Calibration description string |

### Shot Detection

| Option | Default | Description |
|--------|---------|-------------|
| `--threshold-dB` | 120 | Detection threshold in dB SPL |
| `--refractory-ms` | 200 | Minimum time between shots (ms) |
| `--pre-ms` | 50 | Pre-shot window (ms) |
| `--post-ms` | 200 | Post-shot window (ms) |

### Analysis

| Option | Default | Description |
|--------|---------|-------------|
| `--nperseg` | 2048 | STFT window size (samples) |
| `--no-bands` | false | Skip 1/3-octave band analysis |
| `--no-per-shot` | false | Skip per-shot summary plots |

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `./analysis` | Output base directory |
| `--config` | — | Load config from JSON file |
| `--formats` | `png` | Plot formats: `png`, `pdf`, `svg` |

### Config File

Save and reuse analysis settings:

```json
{
  "Pa_per_FS": 50.0,
  "detection_threshold_dB": 120.0,
  "refractory_ms": 300.0,
  "pre_shot_ms": 100.0,
  "post_shot_ms": 800.0,
  "nperseg": 4096,
  "compute_bands": true,
  "plot_formats": ["png", "pdf"]
}
```

---

## Output Files

After analysis, outputs are saved to a timestamped directory:

```
analysis/
└── recording_20260322_143052/
    ├── waveform_full.png              # Pressure waveform with shot markers
    ├── waveform_full.html             # Interactive zoomable waveform (Plotly)
    ├── spectrogram_z_full.png         # Z-weighted spectrogram (dB SPL)
    ├── spectrogram_z_full.html        # Interactive Z-weighted spectrogram
    ├── spectrogram_c_full.png         # C-weighted spectrogram (dB SPL)
    ├── spectrogram_c_full.html        # Interactive C-weighted spectrogram
    ├── bands_full.png                 # 1/3-octave band time-frequency heatmap
    ├── levels_full.png                # LAF/LAS/LZF/LZS time curves
    ├── shots/
    │   ├── shot_01_summary.png        # Multi-panel summary for shot 1
    │   ├── shot_02_summary.png        # Multi-panel summary for shot 2
    │   └── ...
    ├── metrics_summary.csv            # Per-shot metrics table (spreadsheet-ready)
    ├── analysis_metadata.json         # Complete analysis results (machine-readable)
    └── config.json                    # Configuration used for this analysis
```

---

## Module Reference

| Module | Purpose | Key Functions / Classes |
|--------|---------|------------------------|
| `main.py` | Application entry point, CLI, and orchestration | `AnalysisConfig`, `analyze_file()` |
| `calibration.py` | Digital → Pascal conversion, dB SPL math | `Calibration`, `amplitude_to_dB_SPL()`, `power_to_dB_SPL()` |
| `weighting.py` | A/C/Z frequency weighting IIR filters | `apply_a_weight()`, `apply_c_weight()`, `AWeightFilter` |
| `shot_detect.py` | Gunshot event detection with refractory period | `detect_shots()`, `ShotEvent` |
| `bands.py` | 1/3-octave band analysis (ISO 266) | `ThirdOctaveAnalyzer`, `compute_band_exposure()` |
| `STFT.py` | STFT computation with calibrated dB SPL | `compute_stft_dB_SPL()`, `analyze_stft()`, `STFTResult` |
| `metrics.py` | Per-shot and aggregate acoustic metrics | `compute_shot_metrics()`, `ShotMetrics`, `AggregateMetrics` |
| `plots.py` | Publication-quality visualization | `plot_waveform_pa()`, `plot_spectrogram_dB()`, `create_shot_summary_figure()` |
| `WavLoader.py` | WAV file loading with chunked support | `load_wav()`, `load_wav_chunk()`, `WavData` |
| `FileSelector.py` | Native OS file picker (macOS + Windows) | `choose_media_file()` |
| `ExtractAudio.py` | Audio extraction from video files | `extract_audio()` |
| `SignalGenerator.py` | Synthetic test tone generation | `synthesize_tone()` |

### Standalone Module Usage

```python
from calibration import Calibration, amplitude_to_dB_SPL
from weighting import apply_a_weight
from shot_detect import detect_shots
from metrics import compute_shot_metrics
from bands import ThirdOctaveAnalyzer
from STFT import analyze_stft

import soundfile as sf

# Load and calibrate
data, sr = sf.read('recording.wav', dtype='float32')
cal = Calibration(Pa_per_FS=50.0)
pressure_Pa = cal.to_pascals(data)

# Detect shots
shots = detect_shots(pressure_Pa, sr, threshold_dB=110)

# Analyze each shot
for shot in shots:
    segment = pressure_Pa[shot.window_start:shot.window_end]
    metrics = compute_shot_metrics(segment, sr)
    print(f"Shot {shot.shot_number}: LAE={metrics.LAE:.1f} dB, "
          f"LAFmax={metrics.LAFmax:.1f} dB, "
          f"Rise={metrics.rise_time_us:.0f} µs")
```

---

## Building from Source

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core runtime |
| pip | latest | Package management |
| Node.js | 18+ | Web UI server (optional) |
| npm | latest | UI dependency management (optional) |
| PyInstaller | 6.0+ | Standalone app building |

### macOS

```bash
# Install dependencies + PyInstaller
pip install -r requirements.txt pyinstaller

# Build the app
./build_macos.sh

# Output: dist/SASA.app
```

### Windows

```batch
REM Install dependencies + PyInstaller
pip install -r requirements.txt pyinstaller

REM Build the exe
build_windows.bat

REM Output: dist\SASA.exe
```

### GitHub Actions (CI/CD)

The repository includes a GitHub Actions workflow (`.github/workflows/build.yml`) that automatically builds macOS and Windows versions when a version tag is pushed:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Builds are automatically attached to the GitHub Release.

---

## Standards Compliance

SASA follows or references these international standards:

| Standard | Description | SASA Usage |
|----------|-------------|------------|
| **IEC 61672-1:2013** | Sound level meters — specifications | A/C/Z frequency weighting, Fast/Slow/Impulse time weighting |
| **ISO 266:1997** | Preferred frequencies for acoustics | 1/3-octave band center frequencies |
| **IEC 61260-1:2014** | Octave-band and fractional-octave-band filters | Bandpass filter design (4th-order Butterworth) |
| **ANSI S1.4-1983** | Sound level meter specification | Reference weighting curves |
| **ISO 1996-1:2016** | Environmental noise assessment | SEL computation, Leq methodology |
| **MIL-STD-1474E** | Noise limits for military materiel | Kurtosis-based impulsive noise criteria |

### Recording Recommendations

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| Sample rate | ≥ 96 kHz | Captures energy up to 48 kHz; gunshots have significant content above 20 kHz |
| Bit depth | 24-bit or 32-bit float | Provides sufficient dynamic range (~144 dB) for gunshot peak + ambient |
| Headroom | Peaks below 0 dBFS | Clipping invalidates all peak metrics |
| Calibration | Required | Without it, dB values are relative, not absolute SPL |

### Known Limitations

1. **Loudness model** — placeholder only; full ISO 532-1/532-2 implementation requires the `mosqito` library
2. **Near-field effects** — assumes far-field acoustic measurements
3. **Doppler** — not compensated for moving sources
4. **Very short events** — events < 1 ms may not be fully captured by STFT with default window size
5. **Single-channel** — multichannel recordings are mixed to mono

---

## License

Copyright © 2024–2026 Ridgeback Defense. All rights reserved.
