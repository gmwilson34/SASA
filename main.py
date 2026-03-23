#!/usr/bin/env python3
"""
SASA - Shot Acoustic Spectral Analysis

Main application for analyzing gunshot WAV recordings with calibrated acoustic measurements.
Produces Z-weighted and C-weighted analysis with:
  - Shot detection with configurable thresholds
  - Per-shot metrics (Peak SPL, LAE, LAFmax, band exposure)
  - Publication-quality plots (waveforms, spectrograms, 1/3-octave heatmaps)
  - Aggregate statistics (CSV summary, JSON metadata)

Usage:
    python main.py                          # Interactive file selection
    python main.py path/to/recording.wav    # Direct file input
    python main.py recording.wav --Pa-per-FS 50.0  # With calibration
    python main.py recording.wav --config config.json  # From config file

Audio is always used at the file's native sample rate (e.g. 192 kHz); no resampling.
Downsampling applies only to full-recording overview plots (for display/file size).

Dependencies:
    pip install numpy scipy soundfile matplotlib plotly

Author: Ridgeback Defense
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# Import custom modules
from FileSelector import choose_media_file, VIDEO_EXTS, AUDIO_EXTS
try:
    from ExtractAudio import ensure_moviepy_uses_packaged_ffmpeg, extract_audio
    _VIDEO_SUPPORT = True
except ImportError:
    _VIDEO_SUPPORT = False
from WavLoader import load_wav, load_wav_chunk, get_wav_info
from calibration import Calibration, amplitude_to_dB_SPL
from shot_detect import detect_shots, ShotEvent
from metrics import compute_shot_metrics, compute_aggregate_metrics, ShotMetrics, AggregateMetrics
from bands import ThirdOctaveAnalyzer
from STFT import analyze_stft, STFTResult
from plots import (
    setup_plot_style,
    plot_waveform_pa,
    plot_spectrogram_dB,
    plot_third_octave_heatmap,
    create_shot_summary_figure,
    save_figure,
    save_interactive_waveform_html,
    save_interactive_spectrogram_html,
)

# Check once at startup which Python we're using and if Plotly is available
_PLOTLY_AVAILABLE = False
_PLOTLY_ERROR = None
try:
    import plotly.graph_objects  # noqa: F401
    _PLOTLY_AVAILABLE = True
except ImportError as _e:
    _PLOTLY_ERROR = _e


@dataclass
class AnalysisConfig:
    """Configuration for acoustic analysis."""
    # Calibration
    # Default: derived from calibrated 114 dB SPL tone (Audio/260212_0010-1.wav)
    # Measured digital RMS vs reference → Pa_per_FS ≈ 143.96
    Pa_per_FS: float = 143.96
    sensitivity_mV_per_Pa: Optional[float] = None
    V_per_FS: Optional[float] = None
    calibration_description: str = ""

    # Shot detection
    detection_threshold_dB: float = 120.0
    refractory_ms: float = 200.0
    pre_shot_ms: float = 50.0
    post_shot_ms: float = 200.0

    # STFT parameters
    nperseg: int = 2048
    noverlap: int = 1536

    # Load format: "float32" (default) or "float64" to preserve full 32-bit recorder precision
    load_dtype: str = "float32"

    # Analysis options
    compute_bands: bool = True
    compute_time_series: bool = True

    # Output options
    save_per_shot_plots: bool = True
    save_aggregate_plots: bool = True
    plot_formats: Optional[List[str]] = None

    def __post_init__(self):
        if self.plot_formats is None:
            self.plot_formats = ['png']

    @classmethod
    def from_json(cls, path: Path) -> "AnalysisConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    def get_calibration(self) -> Calibration:
        """Create Calibration object from config."""
        if self.sensitivity_mV_per_Pa is not None and self.V_per_FS is not None:
            return Calibration.from_sensitivity(
                self.sensitivity_mV_per_Pa,
                self.V_per_FS,
                self.calibration_description,
            )
        return Calibration(
            Pa_per_FS=self.Pa_per_FS,
            description=self.calibration_description or f"Direct: {self.Pa_per_FS} Pa/FS",
        )


@dataclass
class AnalysisResult:
    """Results from complete analysis."""
    input_file: Path
    output_dir: Path
    calibration: Calibration
    sample_rate: int
    duration_s: float
    n_shots: int
    shots: List[ShotEvent]
    shot_metrics: List[ShotMetrics]
    aggregate: AggregateMetrics
    config: AnalysisConfig
    timestamp: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'input_file': str(self.input_file),
            'output_dir': str(self.output_dir),
            'calibration': {
                'Pa_per_FS': self.calibration.Pa_per_FS,
                'description': self.calibration.description,
                'is_calibrated': self.calibration.is_calibrated(),
            },
            'sample_rate': self.sample_rate,
            'duration_s': round(self.duration_s, 3),
            'n_shots': self.n_shots,
            'aggregate': self.aggregate.to_dict(),
            'timestamp': self.timestamp,
        }


# Chunked processing limits (reduce RAM for very long files)
MAX_DURATION_FULL_LOAD_S = 600.0   # If longer, use chunked path (10 min)
CHUNK_DURATION_S = 120.0          # Process in 2-minute chunks
# Display-only downsampling: analysis always uses full sample rate and full data.
# These only thin the data sent to full-recording overview plots (waveform/spectrogram HTML/PNG).
MAX_WAVEFORM_POINTS = 400_000     # Downsample waveform *plot* above this (analysis uses all samples)
SPECTROGRAM_DOWNSAMPLE = 40       # Keep every Nth STFT frame for full-file *plot* (STFT uses all samples)
# Full-res around shots: margin (seconds) each side of shot window for HTML waveform
WAVEFORM_HTML_SHOT_MARGIN_S = 0.05


def _waveform_full_res_around_shots(
    sr: int,
    pressure_Pa: np.ndarray,
    shots: List[ShotEvent],
    margin_s: float = WAVEFORM_HTML_SHOT_MARGIN_S,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (time_s, pressure_Pa) for HTML waveform: full resolution inside each shot
    window (plus margin), downsampled elsewhere. Keeps file size reasonable while
    giving full detail around detected shots.
    """
    n = len(pressure_Pa)
    if n == 0:
        return np.array([]), np.array([])
    step = max(1, n // MAX_WAVEFORM_POINTS)
    margin_samples = int(margin_s * sr)
    time_full = np.arange(n, dtype=np.float64) / sr

    # Build (start_sample, end_sample) for each "full res" region (shot window + margin)
    full_regions: List[tuple[int, int]] = []
    for s in shots:
        a = max(0, s.window_start - margin_samples)
        b = min(n, s.window_end + margin_samples)
        if b > a:
            full_regions.append((a, b))
    # Merge overlapping regions
    full_regions.sort(key=lambda r: r[0])
    merged: List[tuple[int, int]] = []
    for a, b in full_regions:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    out_t: List[np.ndarray] = []
    out_p: List[np.ndarray] = []
    last_end = 0
    for a, b in merged:
        # Downsampled segment [last_end, a)
        if a > last_end:
            i0 = (last_end + step - 1) // step * step
            if i0 < a:
                idx = np.arange(i0, min(a, n), step, dtype=np.intp)
                if len(idx) > 0:
                    out_t.append(time_full[idx])
                    out_p.append(pressure_Pa[idx])
        # Full-res segment [a, b]
        idx_full = np.arange(a, min(b, n), dtype=np.intp)
        if len(idx_full) > 0:
            out_t.append(time_full[idx_full])
            out_p.append(pressure_Pa[idx_full])
        last_end = b
    # Downsampled tail [last_end, n)
    if last_end < n:
        i0 = (last_end + step - 1) // step * step
        if i0 < n:
            idx = np.arange(i0, n, step, dtype=np.intp)
            if len(idx) > 0:
                out_t.append(time_full[idx])
                out_p.append(pressure_Pa[idx])

    if not out_t:
        # No segments (e.g. no shots): return downsampled whole
        idx = np.arange(0, n, step, dtype=np.intp)
        return time_full[idx], pressure_Pa[idx]
    return np.concatenate(out_t), np.concatenate(out_p)


def _waveform_chunked_full_res_around_shots(
    time_down: np.ndarray,
    pressure_down: np.ndarray,
    step: int,
    total_frames: int,
    sr: int,
    shots: List[ShotEvent],
    get_region: Any,  # (start_sample, end_sample) -> (time_s, pressure_Pa)
    margin_s: float = WAVEFORM_HTML_SHOT_MARGIN_S,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (time_s, pressure_Pa) for chunked HTML waveform: full resolution in each
    shot window (from get_region), downsampled elsewhere from time_down/pressure_down.
    """
    n = total_frames
    margin_samples = int(margin_s * sr)
    def down_idx(samp: int) -> int:
        return min(len(time_down) - 1, max(0, samp // step))

    full_regions: List[tuple[int, int]] = []
    for s in shots:
        a = max(0, s.window_start - margin_samples)
        b = min(n, s.window_end + margin_samples)
        if b > a:
            full_regions.append((a, b))
    full_regions.sort(key=lambda r: r[0])
    merged: List[tuple[int, int]] = []
    for a, b in full_regions:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    out_t: List[np.ndarray] = []
    out_p: List[np.ndarray] = []
    last_end = 0
    for a, b in merged:
        if a > last_end:
            i0, i1 = down_idx(last_end), down_idx(a)
            if i1 > i0:
                out_t.append(time_down[i0:i1])
                out_p.append(pressure_down[i0:i1])
        try:
            t_full, p_full = get_region(a, b)
            if len(t_full) > 0:
                out_t.append(t_full)
                out_p.append(p_full)
        except Exception:
            pass
        last_end = b
    if last_end < n:
        i0 = down_idx(last_end)
        if i0 < len(time_down):
            out_t.append(time_down[i0:])
            out_p.append(pressure_down[i0:])
    if not out_t:
        return time_down, pressure_down
    return np.concatenate(out_t), np.concatenate(out_p)


def create_output_directory(base_dir: Path, input_file: Path) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{input_file.stem}_{timestamp}"
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_csv_summary(
    output_path: Path,
    shot_metrics: List[ShotMetrics],
) -> None:
    """Save per-shot metrics to CSV file."""
    if not shot_metrics:
        return

    fieldnames = [
        'shot_number', 'duration_ms',
        'Lpeak_Z', 'Lpeak_A', 'Lpeak_C',
        'LAE', 'LZE', 'LCE',
        'LAFmax', 'LASmax', 'LZFmax', 'LZSmax',
        'LAImax', 'LZImax',
        'rise_time_us', 'b_duration_ms', 'crest_factor_dB',
        'spectral_centroid_Hz', 'kurtosis',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for m in shot_metrics:
            writer.writerow({
                'shot_number': m.shot_number,
                'duration_ms': round(m.duration_s * 1000, 1),
                'Lpeak_Z': round(m.Lpeak_Z, 1),
                'Lpeak_A': round(m.Lpeak_A, 1),
                'Lpeak_C': round(m.Lpeak_C, 1),
                'LAE': round(m.LAE, 1),
                'LZE': round(m.LZE, 1),
                'LCE': round(m.LCE, 1),
                'LAFmax': round(m.LAFmax, 1),
                'LASmax': round(m.LASmax, 1),
                'LZFmax': round(m.LZFmax, 1),
                'LZSmax': round(m.LZSmax, 1),
                'LAImax': round(m.LAImax, 1),
                'LZImax': round(m.LZImax, 1),
                'rise_time_us': round(m.rise_time_us, 1),
                'b_duration_ms': round(m.b_duration_ms, 2),
                'crest_factor_dB': round(m.crest_factor_dB, 1),
                'spectral_centroid_Hz': round(m.spectral_centroid_Hz, 0),
                'kurtosis': round(m.kurtosis, 1),
            })


def save_json_metadata(
    output_path: Path,
    result: AnalysisResult,
) -> None:
    """Save complete analysis metadata to JSON file."""
    data = result.to_dict()
    data['shots'] = [
        {
            'shot_number': s.shot_number,
            'time_s': round(s.time_s, 4),
            'peak_Pa': round(s.peak_Pa, 2),
            'peak_dB_SPL': round(s.peak_dB_SPL, 1),
        }
        for s in result.shots
    ]
    data['per_shot_metrics'] = [m.to_dict() for m in result.shot_metrics]

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def _analyze_file_chunked(
    wav_path: Path,
    config: AnalysisConfig,
    output_dir: Path,
    timestamp: str,
) -> AnalysisResult:
    """
    Run analysis on a long file by loading and processing in chunks to limit RAM.
    """
    import matplotlib.pyplot as plt

    total_frames, sr, duration_s, _ = get_wav_info(wav_path)
    cal = config.get_calibration()
    chunk_frames = int(CHUNK_DURATION_S * sr)
    refractory_samples = int(config.refractory_ms * sr / 1000.0)

    print(f"  Using chunked mode (max {CHUNK_DURATION_S:.0f} s per chunk) to limit RAM")

    # ---- Shot detection: one chunk at a time ----
    all_shots: List[ShotEvent] = []
    start_frame = 0
    while start_frame < total_frames:
        n_frames = min(chunk_frames, total_frames - start_frame)
        chunk_samples, _ = load_wav_chunk(wav_path, start_frame, n_frames, dtype=config.load_dtype, mono=True)
        if len(chunk_samples) == 0:
            break
        pressure_chunk = cal.to_pascals(chunk_samples)
        chunk_start_s = start_frame / sr
        chunk_shots = detect_shots(
            pressure_chunk,
            sr,
            threshold_dB=config.detection_threshold_dB,
            refractory_ms=config.refractory_ms,
            pre_ms=config.pre_shot_ms,
            post_ms=config.post_shot_ms,
        )
        for s in chunk_shots:
            all_shots.append(ShotEvent(
                index=s.index + start_frame,
                time_s=s.time_s + chunk_start_s,
                peak_Pa=s.peak_Pa,
                peak_dB_SPL=s.peak_dB_SPL,
                window_start=s.window_start + start_frame,
                window_end=s.window_end + start_frame,
                shot_number=0,
            ))
        start_frame += n_frames
        del pressure_chunk, chunk_samples

    # Sort by time and merge duplicates within refractory period
    all_shots.sort(key=lambda s: s.time_s)
    merged: List[ShotEvent] = []
    for s in all_shots:
        if merged and (s.time_s - merged[-1].time_s) * sr < refractory_samples:
            continue
        merged.append(ShotEvent(
            index=s.index,
            time_s=s.time_s,
            peak_Pa=s.peak_Pa,
            peak_dB_SPL=s.peak_dB_SPL,
            window_start=s.window_start,
            window_end=s.window_end,
            shot_number=len(merged) + 1,
        ))
    shots = merged
    print(f"  Detection threshold: {config.detection_threshold_dB} dB SPL")
    print(f"  Detected shots: {len(shots)}")
    for shot in shots:
        print(f"    Shot {shot.shot_number}: t={shot.time_s:.3f}s, peak={shot.peak_dB_SPL:.1f} dB SPL")

    # ---- Per-shot metrics: load each shot window only ----
    shot_metrics = []
    for shot in shots:
        n_frames = shot.window_end - shot.window_start
        shot_samples, _ = load_wav_chunk(wav_path, shot.window_start, n_frames, dtype=config.load_dtype, mono=True)
        shot_pressure = cal.to_pascals(shot_samples)
        metrics = compute_shot_metrics(
            shot_pressure,
            sr,
            compute_bands=config.compute_bands,
            compute_time_series=config.compute_time_series,
            shot_number=shot.shot_number,
        )
        shot_metrics.append(metrics)
        print(f"  Shot {shot.shot_number}: LAE={metrics.LAE:.1f} dB, LAFmax={metrics.LAFmax:.1f} dB")
        del shot_samples, shot_pressure

    aggregate = compute_aggregate_metrics(shot_metrics)
    if len(shots) > 1:
        print(f"\n  Aggregate ({len(shots)} shots): LAE mean={aggregate.LAE_mean:.1f} dB, LAFmax mean={aggregate.LAFmax_mean:.1f} dB")

    # ---- Full waveform: chunked read, downsampled for plot ----
    print("\n[5/6] Generating plots (chunked)...")
    if not _PLOTLY_AVAILABLE:
        print("  (Plotly not available — using PNG for full waveform/spectrograms.)")
        print(f"  Python in use: {sys.executable}")
        if _PLOTLY_ERROR is not None:
            print(f"  Import error: {_PLOTLY_ERROR}")
        print("  To fix, run this (installs for the Python that runs this script):")
        print(f"    {sys.executable} -m pip install plotly")
        print("  Note: If you used 'pip install plotly' and it still fails, your venv may have")
        print("  multiple Python versions; use the command above so plotly installs for the right one.")
    step = max(1, total_frames // MAX_WAVEFORM_POINTS)
    time_parts: List[np.ndarray] = []
    pressure_parts: List[np.ndarray] = []
    start_frame = 0
    while start_frame < total_frames:
        n_frames = min(chunk_frames, total_frames - start_frame)
        chunk_samples, _ = load_wav_chunk(wav_path, start_frame, n_frames, dtype=config.load_dtype, mono=True)
        if len(chunk_samples) == 0:
            break
        pressure_chunk = cal.to_pascals(chunk_samples)
        t_chunk = (start_frame + np.arange(len(pressure_chunk))) / sr
        time_parts.append(t_chunk[::step])
        pressure_parts.append(pressure_chunk[::step])
        start_frame += n_frames
        del pressure_chunk, chunk_samples
    time_full = np.concatenate(time_parts) if time_parts else np.array([])
    pressure_down = np.concatenate(pressure_parts) if pressure_parts else np.array([])
    del time_parts, pressure_parts
    gc.collect()

    if len(time_full) > 0:
        _wpath = output_dir / "waveform_full.html"
        if _PLOTLY_AVAILABLE:
            def get_region(start_samp: int, end_samp: int):
                chunk_s, _ = load_wav_chunk(wav_path, start_samp, end_samp - start_samp, dtype=config.load_dtype, mono=True)
                p = cal.to_pascals(chunk_s)
                t = (start_samp + np.arange(len(p))) / sr
                return t.astype(np.float64), p.astype(np.float64)
            time_plot, pressure_plot = _waveform_chunked_full_res_around_shots(
                time_full, pressure_down, step, total_frames, sr, shots, get_region,
            )
            if save_interactive_waveform_html(
                _wpath,
                time_plot, pressure_plot,
                shots=shots,
                title=f"Pressure Waveform: {wav_path.name}",
            ):
                print(f"  ✓ Full waveform (interactive HTML, full res around shots): {_wpath.name}")
            del time_plot, pressure_plot
        else:
            fig_wave, _ = plot_waveform_pa(
                time_full, pressure_down,
                shots=shots,
                title=f"Pressure Waveform: {wav_path.name} (downsampled for display)",
            )
            save_figure(fig_wave, output_dir / "waveform_full", formats=config.plot_formats)
            plt.close(fig_wave)
            print("  ✓ Full waveform (PNG)")
    del time_full, pressure_down
    gc.collect()

    # ---- Full spectrogram: chunked STFT; full res for HTML, downsampled for PNG ----
    config.nperseg - config.noverlap
    freq_axis: np.ndarray = np.array([])
    time_z_list: List[np.ndarray] = []
    mag_z_list: List[np.ndarray] = []
    start_frame = 0
    while start_frame < total_frames:
        n_frames = min(chunk_frames, total_frames - start_frame)
        chunk_samples, _ = load_wav_chunk(wav_path, start_frame, n_frames, dtype=config.load_dtype, mono=True)
        if len(chunk_samples) == 0:
            break
        pressure_chunk = cal.to_pascals(chunk_samples)
        stft_z = analyze_stft(pressure_chunk, sr, nperseg=config.nperseg, noverlap=config.noverlap, weighting='Z')
        if len(freq_axis) == 0:
            freq_axis = stft_z.frequencies_Hz
        chunk_start_s = start_frame / sr
        if _PLOTLY_AVAILABLE:
            time_z_list.append(stft_z.time_s + chunk_start_s)
            mag_z_list.append(stft_z.magnitude_dB)
        else:
            take = slice(None, None, SPECTROGRAM_DOWNSAMPLE)
            time_z_list.append(stft_z.time_s[take] + chunk_start_s)
            mag_z_list.append(stft_z.magnitude_dB[:, take])
        start_frame += n_frames
        del pressure_chunk, chunk_samples, stft_z
        gc.collect()
    if time_z_list and mag_z_list:
        time_z = np.concatenate(time_z_list)
        mag_z = np.concatenate(mag_z_list, axis=1)
        stft_z_full = STFTResult(
            time_s=time_z,
            frequencies_Hz=freq_axis,
            magnitude_dB=mag_z,
            weighting='Z',
            sample_rate=sr,
            nperseg=config.nperseg,
            noverlap=config.noverlap,
            window='hann',
        )
        _zpath = output_dir / "spectrogram_z_full.html"
        if _PLOTLY_AVAILABLE and save_interactive_spectrogram_html(
            _zpath,
            stft_z_full, shots=shots,
            title=f"Z-Weighted Spectrogram: {wav_path.name}",
        ):
            print(f"  ✓ Z-weighted spectrogram (interactive HTML): {_zpath.name}")
        else:
            fig_stft_z, _ = plot_spectrogram_dB(stft_z_full, shots=shots,
                                                title=f"Z-Weighted Spectrogram: {wav_path.name}")
            save_figure(fig_stft_z, output_dir / "spectrogram_z_full", formats=config.plot_formats)
            plt.close(fig_stft_z)
            print("  ✓ Z-weighted spectrogram (PNG)")
        del time_z_list, mag_z_list, time_z, mag_z, stft_z_full
    gc.collect()

    time_c_list = []
    mag_c_list = []
    start_frame = 0
    while start_frame < total_frames:
        n_frames = min(chunk_frames, total_frames - start_frame)
        chunk_samples, _ = load_wav_chunk(wav_path, start_frame, n_frames, dtype=config.load_dtype, mono=True)
        if len(chunk_samples) == 0:
            break
        pressure_chunk = cal.to_pascals(chunk_samples)
        stft_c = analyze_stft(pressure_chunk, sr, nperseg=config.nperseg, noverlap=config.noverlap, weighting='C')
        chunk_start_s = start_frame / sr
        if _PLOTLY_AVAILABLE:
            time_c_list.append(stft_c.time_s + chunk_start_s)
            mag_c_list.append(stft_c.magnitude_dB)
        else:
            take = slice(None, None, SPECTROGRAM_DOWNSAMPLE)
            time_c_list.append(stft_c.time_s[take] + chunk_start_s)
            mag_c_list.append(stft_c.magnitude_dB[:, take])
        start_frame += n_frames
        del pressure_chunk, chunk_samples, stft_c
        gc.collect()
    if time_c_list and mag_c_list:
        time_c = np.concatenate(time_c_list)
        mag_c = np.concatenate(mag_c_list, axis=1)
        stft_c_full = STFTResult(
            time_s=time_c,
            frequencies_Hz=freq_axis,
            magnitude_dB=mag_c,
            weighting='C',
            sample_rate=sr,
            nperseg=config.nperseg,
            noverlap=config.noverlap,
            window='hann',
        )
        _cpath = output_dir / "spectrogram_c_full.html"
        if _PLOTLY_AVAILABLE and save_interactive_spectrogram_html(
            _cpath,
            stft_c_full, shots=shots,
            title=f"C-Weighted Spectrogram: {wav_path.name}",
        ):
            print(f"  ✓ C-weighted spectrogram (interactive HTML): {_cpath.name}")
        else:
            fig_stft_c, _ = plot_spectrogram_dB(stft_c_full, shots=shots,
                                                title=f"C-Weighted Spectrogram: {wav_path.name}")
            save_figure(fig_stft_c, output_dir / "spectrogram_c_full", formats=config.plot_formats)
            plt.close(fig_stft_c)
            print("  ✓ C-weighted spectrogram (PNG)")
        del time_c_list, mag_c_list, time_c, mag_c, stft_c_full
    gc.collect()

    # ---- 1/3-octave bands: chunked ----
    if config.compute_bands:
        analyzer = ThirdOctaveAnalyzer(sample_rate=sr)
        band_times_list: List[np.ndarray] = []
        band_levels_list: List[np.ndarray] = []
        start_frame = 0
        while start_frame < total_frames:
            n_frames = min(chunk_frames, total_frames - start_frame)
            chunk_samples, _ = load_wav_chunk(wav_path, start_frame, n_frames, dtype=config.load_dtype, mono=True)
            if len(chunk_samples) == 0:
                break
            pressure_chunk = cal.to_pascals(chunk_samples)
            band_res = analyzer.analyze(pressure_chunk, time_weighting='fast', hop_ms=10.0)
            chunk_start_s = start_frame / sr
            band_times_list.append(band_res['time_s'] + chunk_start_s)
            band_levels_list.append(band_res['band_levels_dB'])
            start_frame += n_frames
            del pressure_chunk, chunk_samples
            gc.collect()
        if band_times_list and band_levels_list:
            band_time_s = np.concatenate(band_times_list)
            band_levels_dB = np.concatenate(band_levels_list, axis=1)
            # Downsample for plot if huge
            if band_time_s.size > MAX_WAVEFORM_POINTS:
                step_b = max(1, band_time_s.size // MAX_WAVEFORM_POINTS)
                band_time_s = band_time_s[::step_b]
                band_levels_dB = band_levels_dB[:, ::step_b]
            fig_bands, _ = plot_third_octave_heatmap(
                band_time_s,
                analyzer.center_frequencies,
                band_levels_dB,
                shots=shots,
                title=f"1/3-Octave Band Levels: {wav_path.name}",
            )
            save_figure(fig_bands, output_dir / "bands_full", formats=config.plot_formats)
            plt.close(fig_bands)
            print("  ✓ 1/3-octave bands")
        del band_times_list, band_levels_list
    gc.collect()

    # ---- Per-shot summary figures: load one shot at a time ----
    if config.save_per_shot_plots and shots:
        shot_dir = output_dir / "shots"
        shot_dir.mkdir(exist_ok=True)
        for shot, metrics in zip(shots, shot_metrics):
            n_frames = shot.window_end - shot.window_start
            shot_samples, _ = load_wav_chunk(wav_path, shot.window_start, n_frames, dtype=config.load_dtype, mono=True)
            shot_pressure = cal.to_pascals(shot_samples)
            shot_time = np.arange(len(shot_pressure)) / sr
            stft_z = analyze_stft(shot_pressure, sr, nperseg=config.nperseg, noverlap=config.noverlap, weighting='Z')
            stft_c = analyze_stft(shot_pressure, sr, nperseg=config.nperseg, noverlap=config.noverlap, weighting='C')
            fig_shot = create_shot_summary_figure(
                shot_time, shot_pressure,
                stft_z, stft_c, metrics,
                title=f"Shot {shot.shot_number} Analysis",
            )
            save_figure(fig_shot, shot_dir / f"shot_{shot.shot_number:02d}_summary", formats=config.plot_formats)
            plt.close(fig_shot)
            del shot_samples, shot_pressure, stft_z, stft_c
        print(f"  ✓ Per-shot summaries ({len(shots)} shots)")
    gc.collect()

    # ---- Save data files ----
    print("\n[6/6] Saving data files...")
    csv_path = output_dir / "metrics_summary.csv"
    save_csv_summary(csv_path, shot_metrics)
    print(f"  ✓ CSV: {csv_path.name}")

    result = AnalysisResult(
        input_file=wav_path,
        output_dir=output_dir,
        calibration=cal,
        sample_rate=sr,
        duration_s=duration_s,
        n_shots=len(shots),
        shots=shots,
        shot_metrics=shot_metrics,
        aggregate=aggregate,
        config=config,
        timestamp=timestamp,
    )
    json_path = output_dir / "analysis_metadata.json"
    save_json_metadata(json_path, result)
    print(f"  ✓ JSON: {json_path.name}")
    config_path = output_dir / "config.json"
    config.to_json(config_path)
    print(f"  ✓ Config: {config_path.name}")
    print(f"\n{'='*60}\nAnalysis complete!\nOutput directory: {output_dir}\n{'='*60}\n")
    return result


def analyze_file(
    wav_path: Path,
    config: AnalysisConfig,
    output_base: Optional[Path] = None,
) -> AnalysisResult:
    """
    Run complete analysis pipeline on a WAV file.

    Args:
        wav_path: Path to input WAV file.
        config: Analysis configuration.
        output_base: Base directory for outputs (default: same as input).

    Returns:
        AnalysisResult with all computed data.
    """
    setup_plot_style()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    if output_base is None:
        output_base = wav_path.parent / "analysis"
    output_dir = create_output_directory(output_base, wav_path)

    print(f"\n{'='*60}")
    print("SASA - Shot Acoustic Spectral Analysis")
    print(f"{'='*60}")
    print(f"Input: {wav_path}")
    print(f"Output: {output_dir}")

    # Check file length; use chunked path for very long files to limit RAM
    try:
        _frames, _sr, duration_s_pre, _ = get_wav_info(wav_path)
        if duration_s_pre > MAX_DURATION_FULL_LOAD_S:
            print("\n[1/6] Long file detected — using chunked analysis to limit RAM...")
            return _analyze_file_chunked(wav_path, config, output_dir, timestamp)
    except Exception:
        pass  # Fall back to full load (e.g. unsupported format)

    # Load audio (full file)
    # Note: ST2012 is stereo pair, but we average to mono for analysis
    print("\n[1/6] Loading audio...")
    wav_data = load_wav(wav_path, dtype=config.load_dtype, mono=True)
    sr = wav_data.sample_rate
    samples = wav_data.samples
    duration_s = len(samples) / sr

    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {duration_s:.2f} s")
    print(f"  Samples: {len(samples):,}")

    # Calibrate to Pascals
    print("\n[2/6] Applying calibration...")
    cal = config.get_calibration()
    pressure_Pa = cal.to_pascals(samples)

    peak_Pa = np.max(np.abs(pressure_Pa))
    peak_dB = amplitude_to_dB_SPL(peak_Pa)
    print(f"  Calibration: {cal.description}")
    print(f"  Pa per FS: {cal.Pa_per_FS:.2f}")
    print(f"  Peak level: {peak_dB:.1f} dB SPL")

    # Detect shots
    print("\n[3/6] Detecting shots...")
    shots = detect_shots(
        pressure_Pa,
        sr,
        threshold_dB=config.detection_threshold_dB,
        refractory_ms=config.refractory_ms,
        pre_ms=config.pre_shot_ms,
        post_ms=config.post_shot_ms,
    )

    print(f"  Detection threshold: {config.detection_threshold_dB} dB SPL")
    print(f"  Detected shots: {len(shots)}")

    for shot in shots:
        print(f"    Shot {shot.shot_number}: t={shot.time_s:.3f}s, "
              f"peak={shot.peak_dB_SPL:.1f} dB SPL")

    # Compute per-shot metrics
    print("\n[4/6] Computing per-shot metrics...")
    shot_metrics = []

    for shot in shots:
        # Extract shot window
        shot_pressure = pressure_Pa[shot.window_start:shot.window_end]

        metrics = compute_shot_metrics(
            shot_pressure,
            sr,
            compute_bands=config.compute_bands,
            compute_time_series=config.compute_time_series,
            shot_number=shot.shot_number,
        )
        shot_metrics.append(metrics)

        print(f"  Shot {shot.shot_number}: LAE={metrics.LAE:.1f} dB, "
              f"LAFmax={metrics.LAFmax:.1f} dB, Lpeak(Z)={metrics.Lpeak_Z:.1f} dB")

    # Aggregate metrics
    aggregate = compute_aggregate_metrics(shot_metrics)

    if len(shots) > 1:
        print(f"\n  Aggregate ({len(shots)} shots):")
        print(f"    LAE mean: {aggregate.LAE_mean:.1f} ± {aggregate.LAE_std:.1f} dB")
        print(f"    LAFmax mean: {aggregate.LAFmax_mean:.1f} ± {aggregate.LAFmax_std:.1f} dB")
        print(f"    Peak SPL (Z) max: {aggregate.Lpeak_Z_max:.1f} dB")

    # Generate plots
    print("\n[5/6] Generating plots...")
    import matplotlib.pyplot as plt
    if not _PLOTLY_AVAILABLE:
        print("  (Plotly not available — using PNG for full waveform/spectrograms.)")
        print(f"  Python in use: {sys.executable}")
        if _PLOTLY_ERROR is not None:
            print(f"  Import error: {_PLOTLY_ERROR}")
        print("  To fix, run this (installs for the Python that runs this script):")
        print(f"    {sys.executable} -m pip install plotly")
        print("  Note: If you used 'pip install plotly' and it still fails, your venv may have")
        print("  multiple Python versions; use the command above so plotly installs for the right one.")

    # Full recording waveform: for HTML use full resolution around shots, else downsampled for PNG
    n_samples = len(pressure_Pa)
    _wave_path = output_dir / "waveform_full.html"
    if _PLOTLY_AVAILABLE:
        time_plot, pressure_plot = _waveform_full_res_around_shots(sr, pressure_Pa, shots)
        if save_interactive_waveform_html(
            _wave_path,
            time_plot, pressure_plot,
            shots=shots,
            title=f"Pressure Waveform: {wav_path.name}",
        ):
            print(f"  ✓ Full waveform (interactive HTML, full res around shots): {_wave_path.name}")
        del time_plot, pressure_plot
    else:
        if n_samples > MAX_WAVEFORM_POINTS:
            step = n_samples // MAX_WAVEFORM_POINTS
            time_full = (np.arange(0, n_samples, step) / sr).astype(np.float64)
            pressure_plot = pressure_Pa[::step].copy()
        else:
            time_full = np.arange(n_samples, dtype=np.float64) / sr
            pressure_plot = pressure_Pa
        fig_wave, _ = plot_waveform_pa(
            time_full, pressure_plot,
            shots=shots,
            title=f"Pressure Waveform: {wav_path.name}",
        )
        save_figure(fig_wave, output_dir / "waveform_full", formats=config.plot_formats)
        plt.close(fig_wave)
        print("  ✓ Full waveform (PNG)")
        del time_full, pressure_plot
    gc.collect()

    # Full recording spectrograms (compute one at a time to limit peak RAM)
    stft_z_full = analyze_stft(pressure_Pa, sr, nperseg=config.nperseg,
                               noverlap=config.noverlap, weighting='Z')
    _z_path = output_dir / "spectrogram_z_full.html"
    if _PLOTLY_AVAILABLE and save_interactive_spectrogram_html(
        _z_path,
        stft_z_full, shots=shots,
        title=f"Z-Weighted Spectrogram: {wav_path.name}",
    ):
        print(f"  ✓ Z-weighted spectrogram (interactive HTML): {_z_path.name}")
    else:
        fig_stft_z, _ = plot_spectrogram_dB(stft_z_full, shots=shots,
                                            title=f"Z-Weighted Spectrogram: {wav_path.name}")
        save_figure(fig_stft_z, output_dir / "spectrogram_z_full", formats=config.plot_formats)
        plt.close(fig_stft_z)
        print("  ✓ Z-weighted spectrogram (PNG)")
    del stft_z_full
    gc.collect()

    stft_c_full = analyze_stft(pressure_Pa, sr, nperseg=config.nperseg,
                               noverlap=config.noverlap, weighting='C')
    _c_path = output_dir / "spectrogram_c_full.html"
    if _PLOTLY_AVAILABLE and save_interactive_spectrogram_html(
        _c_path,
        stft_c_full, shots=shots,
        title=f"C-Weighted Spectrogram: {wav_path.name}",
    ):
        print(f"  ✓ C-weighted spectrogram (interactive HTML): {_c_path.name}")
    else:
        fig_stft_c, _ = plot_spectrogram_dB(stft_c_full, shots=shots,
                                            title=f"C-Weighted Spectrogram: {wav_path.name}")
        save_figure(fig_stft_c, output_dir / "spectrogram_c_full", formats=config.plot_formats)
        plt.close(fig_stft_c)
        print("  ✓ C-weighted spectrogram (PNG)")
    del stft_c_full
    gc.collect()

    # 1/3-octave band analysis (full recording)
    if config.compute_bands:
        analyzer = ThirdOctaveAnalyzer(sample_rate=sr)
        band_results = analyzer.analyze(pressure_Pa, time_weighting='fast', hop_ms=10.0)
        fig_bands, _ = plot_third_octave_heatmap(
            band_results['time_s'],
            band_results['center_frequencies'],
            band_results['band_levels_dB'],
            shots=shots,
            title=f"1/3-Octave Band Levels: {wav_path.name}",
        )
        save_figure(fig_bands, output_dir / "bands_full", formats=config.plot_formats)
        plt.close(fig_bands)
        print("  ✓ 1/3-octave bands")
        del band_results
        gc.collect()

    # Per-shot summary figures
    if config.save_per_shot_plots:
        shot_dir = output_dir / "shots"
        shot_dir.mkdir(exist_ok=True)

        for shot, metrics in zip(shots, shot_metrics):
            shot_pressure = pressure_Pa[shot.window_start:shot.window_end]
            shot_time = np.arange(len(shot_pressure)) / sr

            stft_z = analyze_stft(shot_pressure, sr, nperseg=config.nperseg,
                                  noverlap=config.noverlap, weighting='Z')
            stft_c = analyze_stft(shot_pressure, sr, nperseg=config.nperseg,
                                  noverlap=config.noverlap, weighting='C')

            fig_shot = create_shot_summary_figure(
                shot_time, shot_pressure,
                stft_z, stft_c, metrics,
                title=f"Shot {shot.shot_number} Analysis",
            )
            save_figure(fig_shot, shot_dir / f"shot_{shot.shot_number:02d}_summary",
                       formats=config.plot_formats)
            plt.close(fig_shot)
            del stft_z, stft_c
        print(f"  ✓ Per-shot summaries ({len(shots)} shots)")

    # Save data files
    print("\n[6/6] Saving data files...")

    # CSV summary
    csv_path = output_dir / "metrics_summary.csv"
    save_csv_summary(csv_path, shot_metrics)
    print(f"  ✓ CSV: {csv_path.name}")

    # Create result object
    result = AnalysisResult(
        input_file=wav_path,
        output_dir=output_dir,
        calibration=cal,
        sample_rate=sr,
        duration_s=duration_s,
        n_shots=len(shots),
        shots=shots,
        shot_metrics=shot_metrics,
        aggregate=aggregate,
        config=config,
        timestamp=timestamp,
    )

    # JSON metadata
    json_path = output_dir / "analysis_metadata.json"
    save_json_metadata(json_path, result)
    print(f"  ✓ JSON: {json_path.name}")

    # Save config
    config_path = output_dir / "config.json"
    config.to_json(config_path)
    print(f"  ✓ Config: {config_path.name}")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    return result


def main() -> int:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="SASA - Shot Acoustic Spectral Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Interactive file selection
  python main.py recording.wav             # Analyze with defaults
  python main.py recording.wav --Pa-per-FS 50.0
  python main.py recording.wav --sensitivity-mV 10.0 --V-per-FS 1.0
  python main.py recording.wav --threshold-dB 120
  python main.py recording.wav --config my_config.json
        """,
    )

    # Input file
    parser.add_argument("input", type=Path, nargs='?', default=None,
                        help="Input WAV file (or select interactively)")

    # Calibration options
    cal_group = parser.add_argument_group("Calibration")
    cal_group.add_argument("--Pa-per-FS", type=float, default=143.96,
                          help="Direct calibration: Pascals per full-scale (default: 143.96 from 114 dB tone)")
    cal_group.add_argument("--sensitivity-mV", type=float, default=None,
                          help="Microphone sensitivity in mV/Pa")
    cal_group.add_argument("--V-per-FS", type=float, default=None,
                          help="Recorder full-scale voltage")
    cal_group.add_argument("--cal-desc", type=str, default="",
                          help="Calibration description")

    # Detection options
    det_group = parser.add_argument_group("Shot Detection")
    det_group.add_argument("--threshold-dB", type=float, default=120.0,
                          help="Detection threshold in dB SPL (default: 120)")
    det_group.add_argument("--refractory-ms", type=float, default=200.0,
                          help="Minimum time between shots in ms (default: 200)")
    det_group.add_argument("--pre-ms", type=float, default=50.0,
                          help="Pre-shot window in ms (default: 50)")
    det_group.add_argument("--post-ms", type=float, default=200.0,
                          help="Post-shot window in ms (default: 200)")

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis")
    analysis_group.add_argument("--dtype", type=str, default="float32",
                               choices=["float32", "float64"],
                               help="Sample dtype when loading WAV: float32 (default) or float64 for full 32-bit precision")
    analysis_group.add_argument("--nperseg", type=int, default=2048,
                               help="STFT window size (default: 2048)")
    analysis_group.add_argument("--no-bands", action="store_true",
                               help="Skip 1/3-octave band analysis")
    analysis_group.add_argument("--no-per-shot", action="store_true",
                               help="Skip per-shot summary plots")

    # Output options
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output", "-o", type=Path, default=None,
                          help="Output base directory")
    out_group.add_argument("--config", type=Path, default=None,
                          help="Load config from JSON file")
    out_group.add_argument("--formats", type=str, default="png",
                          help="Plot formats, comma-separated (default: png)")

    args = parser.parse_args()

    # Load or create config
    if args.config is not None:
        config = AnalysisConfig.from_json(args.config)
    else:
        config = AnalysisConfig(
            Pa_per_FS=args.Pa_per_FS,
            sensitivity_mV_per_Pa=args.sensitivity_mV,
            V_per_FS=args.V_per_FS,
            calibration_description=args.cal_desc,
            detection_threshold_dB=args.threshold_dB,
            refractory_ms=args.refractory_ms,
            pre_shot_ms=args.pre_ms,
            post_shot_ms=args.post_ms,
            load_dtype=args.dtype,
            nperseg=args.nperseg,
            compute_bands=not args.no_bands,
            save_per_shot_plots=not args.no_per_shot,
            plot_formats=args.formats.split(','),
        )

    # Get input file
    if args.input is not None:
        wav_path = args.input.resolve()
        if not wav_path.exists():
            print(f"Error: File not found: {wav_path}", file=sys.stderr)
            return 1
    else:
        print("Select an audio file to analyze...")
        selected = choose_media_file()
        if selected is None:
            print("No file selected. Exiting.")
            return 1
        wav_path = selected

        # Handle video files
        if wav_path.suffix.lower() in VIDEO_EXTS:
            if not _VIDEO_SUPPORT:
                print("Error: Video support requires moviepy and imageio-ffmpeg.", file=sys.stderr)
                print("  Install with: pip install moviepy imageio-ffmpeg", file=sys.stderr)
                return 1
            print("\n[Video detected] Extracting audio...")
            ensure_moviepy_uses_packaged_ffmpeg()
            audio_dir = Path(__file__).parent / "Audio"
            audio_dir.mkdir(exist_ok=True)
            audio_path = audio_dir / (wav_path.stem + ".wav")

            if not audio_path.exists():
                extract_audio(wav_path, audio_path, bitrate=None)
            wav_path = audio_path

    # Verify it's an audio file
    if wav_path.suffix.lower() not in AUDIO_EXTS:
        print(f"Warning: {wav_path.suffix} may not be a supported audio format.")

    # Run analysis
    try:
        result = analyze_file(wav_path, config, args.output)

        # Print final summary
        if result.n_shots > 0:
            print("Quick Summary:")
            print(f"  Total shots: {result.n_shots}")
            print(f"  Peak SPL (Z): {result.aggregate.Lpeak_Z_max:.1f} dB")
            print(f"  Mean LAE: {result.aggregate.LAE_mean:.1f} dB")
            print(f"  Mean LAFmax: {result.aggregate.LAFmax_mean:.1f} dB")
        else:
            print("No shots detected. Try lowering --threshold-dB.")

        return 0

    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
