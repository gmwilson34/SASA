#!/usr/bin/env python3
"""
plots.py - Publication-Grade Acoustic Visualization

Generates high-quality plots for gunshot acoustic analysis:
  - Waveform in Pascals with shot markers
  - Z-weighted spectrogram (dB SPL)
  - A-weighted spectrogram (dB SPL)
  - 1/3-octave time-frequency heatmap (dB SPL)
  - LAfast/LAslow time curves
  - Combined multi-panel figures

Dark tactical theme matching the SASA web UI.

Usage:
    from plots import (
        plot_waveform_pa,
        plot_spectrogram_dB,
        plot_third_octave_heatmap,
        plot_level_curves,
        create_shot_summary_figure,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for worker threads
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from calibration import amplitude_to_dB_SPL
from shot_detect import ShotEvent
from metrics import ShotMetrics
from STFT import STFTResult


# ═══════════════════════════════════════════════════════════
#  Theme — matches styles.css dark tactical palette
# ═══════════════════════════════════════════════════════════

# Backgrounds
_BG_ROOT = '#08080c'
_BG_SURFACE = '#0e0e14'
_BG_CARD = '#12121a'
_BG_ELEVATED = '#1a1a24'

# Borders & grid
_BORDER = '#1e1e2a'
_GRID = '#1e1e2a'

# Text
_TEXT_PRIMARY = '#e8e8ef'
_TEXT_SECONDARY = '#8888a0'
_TEXT_MUTED = '#555568'

# Accent — electric blue
_ACCENT = '#3b82f6'

# Metric accent colors (from CSS --metric-*)
_METRIC_PEAK = '#f97316'
_METRIC_RMS = '#3b82f6'
_METRIC_SEL = '#a855f7'
_METRIC_SHOTS = '#22c55e'

# Status
_SUCCESS = '#22c55e'
_WARNING = '#f59e0b'
_DANGER = '#ef4444'
_INFO = '#06b6d4'

# ── Analysis line colors ──
COLOR_Z_WEIGHT = _ACCENT          # Blue
COLOR_A_WEIGHT = _METRIC_PEAK     # Orange
COLOR_C_WEIGHT = _SUCCESS         # Green
COLOR_FAST = _DANGER              # Red
COLOR_SLOW = _METRIC_SEL          # Purple
COLOR_SHOT_MARKER = _DANGER       # Red

# ── Figure defaults ──
DEFAULT_DPI = 200
PUBLICATION_DPI = 300

# ── Colormaps ──
CMAP_SPECTROGRAM = 'magma'
CMAP_OCTAVE = 'inferno'

# ── dB ranges ──
DB_RANGE_DEFAULT = (20, 160)
DB_RANGE_AMBIENT = (-10, 100)

# ── Plotly dark layout (reused across all interactive charts) ──
_PLOTLY_LAYOUT = dict(
    paper_bgcolor=_BG_ROOT,
    plot_bgcolor=_BG_CARD,
    font=dict(
        color=_TEXT_PRIMARY,
        family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, system-ui, sans-serif',
        size=12,
    ),
    title_font=dict(size=14, color=_TEXT_PRIMARY),
    xaxis=dict(
        gridcolor=_GRID,
        zerolinecolor=_BORDER,
        linecolor=_BORDER,
        tickfont=dict(color=_TEXT_MUTED, size=10),
        title_font=dict(color=_TEXT_SECONDARY, size=11),
    ),
    yaxis=dict(
        gridcolor=_GRID,
        zerolinecolor=_BORDER,
        linecolor=_BORDER,
        tickfont=dict(color=_TEXT_MUTED, size=10),
        title_font=dict(color=_TEXT_SECONDARY, size=11),
    ),
    margin=dict(l=60, r=20, t=50, b=50),
    hoverlabel=dict(
        bgcolor=_BG_ELEVATED,
        bordercolor=_BORDER,
        font_color=_TEXT_PRIMARY,
        font_size=11,
    ),
)


def setup_plot_style() -> None:
    """Configure matplotlib to match the SASA dark tactical UI."""
    plt.rcParams.update({
        # Figure
        'figure.facecolor': _BG_ROOT,
        'figure.edgecolor': _BORDER,
        'figure.dpi': DEFAULT_DPI,

        # Axes
        'axes.facecolor': _BG_CARD,
        'axes.edgecolor': _BORDER,
        'axes.labelcolor': _TEXT_SECONDARY,
        'axes.titlecolor': _TEXT_PRIMARY,

        # Text
        'text.color': _TEXT_PRIMARY,

        # Ticks
        'xtick.color': _TEXT_MUTED,
        'ytick.color': _TEXT_MUTED,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Grid
        'grid.color': _GRID,
        'grid.alpha': 0.4,
        'grid.linewidth': 0.4,

        # Legend
        'legend.facecolor': _BG_ELEVATED,
        'legend.edgecolor': _BORDER,
        'legend.labelcolor': _TEXT_SECONDARY,
        'legend.framealpha': 0.9,

        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': [
            'SF Pro Display', 'Helvetica Neue',
            'Segoe UI', 'Arial', 'sans-serif',
        ],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 600,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.titleweight': 700,

        # Lines
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,
    })


def _style_colorbar(cbar: Any, label: str = 'dB SPL') -> None:
    """Apply dark-theme styling to a colorbar."""
    cbar.set_label(label, color=_TEXT_SECONDARY, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_MUTED)
    cbar.outline.set_edgecolor(_BORDER)
    cbar.outline.set_linewidth(0.5)
    for t in cbar.ax.get_yticklabels():
        t.set_color(_TEXT_MUTED)
        t.set_fontsize(8)


# ═══════════════════════════════════════════════════════════
#  Static plot functions (Matplotlib)
# ═══════════════════════════════════════════════════════════

def plot_waveform_pa(
    time_s: np.ndarray,
    pressure_Pa: np.ndarray,
    *,
    shots: Optional[List[ShotEvent]] = None,
    title: str = "Pressure Waveform",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 4),
    show_dB_secondary: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot pressure waveform in Pascals with optional shot markers.

    Args:
        time_s: Time axis in seconds.
        pressure_Pa: Pressure waveform in Pascals.
        shots: Optional list of detected shot events.
        title: Plot title.
        ax: Existing axes to plot on.
        figsize: Figure size if creating new figure.
        show_dB_secondary: Show dB SPL scale on right axis.

    Returns:
        (figure, axes) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[assignment]

    ax.plot(time_s, pressure_Pa, color=_ACCENT, linewidth=0.5, alpha=0.85)

    if shots:
        for shot in shots:
            is_first = shot.shot_number == 1
            ax.axvline(shot.time_s, color=COLOR_SHOT_MARKER, linestyle='--',
                       alpha=0.6, linewidth=1.2,
                       label='Shot' if is_first else '')
            peak_time_idx = np.argmin(np.abs(time_s - shot.time_s))
            if 0 <= peak_time_idx < len(pressure_Pa):
                ax.plot(shot.time_s, pressure_Pa[peak_time_idx],
                        'o', color=COLOR_SHOT_MARKER, markersize=4,
                        markeredgecolor='none')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (Pa)')
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    if show_dB_secondary:
        ax2 = ax.twinx()
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        ax2.spines['right'].set_edgecolor(_BORDER)
        p_max = max(abs(np.min(pressure_Pa)), abs(np.max(pressure_Pa)))
        if p_max > 0:
            dB_max = float(amplitude_to_dB_SPL(p_max))
            dB_min = max(0.0, dB_max - 80)
            ax2.set_ylim(dB_min, dB_max + 5)
            ax2.set_ylabel('Level (dB SPL)', color=_TEXT_MUTED)
            dB_ticks = np.arange(int(dB_min / 10) * 10,
                                 int(dB_max / 10) * 10 + 20, 20)
            ax2.set_yticks(dB_ticks)
            ax2.tick_params(axis='y', colors=_TEXT_MUTED)

    if shots and len(shots) <= 5:
        ax.legend(loc='upper right')

    plt.tight_layout()
    return fig, ax  # type: ignore[return-value]


def plot_spectrogram_dB(
    result: STFTResult,
    *,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 5),
    db_range: Optional[Tuple[float, float]] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    cmap: str = CMAP_SPECTROGRAM,
    shots: Optional[List[ShotEvent]] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot STFT spectrogram in dB SPL.

    Args:
        result: STFTResult object with spectrogram data.
        title: Plot title. Auto-generated if None.
        ax: Existing axes to plot on.
        figsize: Figure size if creating new figure.
        db_range: (min_dB, max_dB) for colorbar. Auto if None.
        freq_range: (min_Hz, max_Hz) for y-axis. Auto if None.
        cmap: Colormap name.
        shots: Optional shot markers.

    Returns:
        (figure, axes) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[assignment]

    if db_range is None:
        vmax = result.get_max_level()
        vmin = max(0, vmax - 80)
    else:
        vmin, vmax = db_range

    if freq_range is None:
        freq_max = min(20000, result.frequencies_Hz[-1])
        freq_range = (0, freq_max)

    pcm = ax.pcolormesh(
        result.time_s,
        result.frequencies_Hz,
        result.magnitude_dB,
        shading='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    if shots:
        for shot in shots:
            ax.axvline(shot.time_s, color=_TEXT_PRIMARY, linestyle='--',
                       alpha=0.45, linewidth=1.0)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(freq_range)

    if title is None:
        weight_label = {
            'Z': 'Z-Weighted (Flat)',
            'A': 'A-Weighted (Perceptual)',
            'C': 'C-Weighted',
        }.get(result.weighting, result.weighting)
        title = f'Spectrogram \u2014 {weight_label}'
    ax.set_title(title)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    _style_colorbar(cbar, 'Level (dB SPL)')

    plt.tight_layout()
    return fig, ax  # type: ignore[return-value]


def plot_third_octave_heatmap(
    time_s: np.ndarray,
    center_frequencies: np.ndarray,
    band_levels_dB: np.ndarray,
    *,
    title: str = "1/3-Octave Band Levels",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 6),
    db_range: Optional[Tuple[float, float]] = None,
    cmap: str = CMAP_OCTAVE,
    shots: Optional[List[ShotEvent]] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot 1/3-octave band time-frequency heatmap.

    Args:
        time_s: Time axis in seconds.
        center_frequencies: Band center frequencies in Hz.
        band_levels_dB: Band levels in dB SPL, shape (n_bands, n_frames).
        title: Plot title.
        ax: Existing axes to plot on.
        figsize: Figure size.
        db_range: (min_dB, max_dB) for colorbar.
        cmap: Colormap.
        shots: Optional shot markers.

    Returns:
        (figure, axes) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[assignment]

    if db_range is None:
        vmax = np.max(band_levels_dB)
        vmin = max(0, vmax - 60)
    else:
        vmin, vmax = db_range

    n_bands = len(center_frequencies)
    band_indices = np.arange(n_bands + 1) - 0.5

    pcm = ax.pcolormesh(
        time_s,
        band_indices[:-1],
        band_levels_dB,
        shading='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    if shots:
        for shot in shots:
            ax.axvline(shot.time_s, color=_TEXT_PRIMARY, linestyle='--',
                       alpha=0.45, linewidth=1.0)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)

    n_labels = min(15, n_bands)
    tick_indices = np.linspace(0, n_bands - 1, n_labels, dtype=int)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels([f'{center_frequencies[i]:.0f}' for i in tick_indices])

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    _style_colorbar(cbar, 'Level (dB SPL)')

    plt.tight_layout()
    return fig, ax  # type: ignore[return-value]


def plot_level_curves(
    time_s: np.ndarray,
    LAF: np.ndarray,
    LAS: np.ndarray,
    *,
    LZF: Optional[np.ndarray] = None,
    LZS: Optional[np.ndarray] = None,
    title: str = "Time-Weighted Sound Levels",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 4),
    shots: Optional[List[ShotEvent]] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot LAfast, LAslow, and optionally LZ curves.

    Args:
        time_s: Time axis in seconds.
        LAF: A-weighted Fast level (dB SPL).
        LAS: A-weighted Slow level (dB SPL).
        LZF: Optional Z-weighted Fast level.
        LZS: Optional Z-weighted Slow level.
        title: Plot title.
        ax: Existing axes.
        figsize: Figure size.
        shots: Optional shot markers.

    Returns:
        (figure, axes) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[assignment]

    ax.plot(time_s, LAF, color=COLOR_FAST, linewidth=1.0,
            label='LAF (A-wt, Fast)')
    ax.plot(time_s, LAS, color=COLOR_SLOW, linewidth=1.0, linestyle='--',
            label='LAS (A-wt, Slow)')

    if LZF is not None:
        ax.plot(time_s, LZF, color=COLOR_Z_WEIGHT, linewidth=0.8, alpha=0.5,
                label='LZF (Z-wt, Fast)')
    if LZS is not None:
        ax.plot(time_s, LZS, color=COLOR_Z_WEIGHT, linewidth=0.8, alpha=0.35,
                linestyle=':', label='LZS (Z-wt, Slow)')

    if shots:
        for shot in shots:
            ax.axvline(shot.time_s, color=COLOR_SHOT_MARKER, linestyle='--',
                       alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Level (dB SPL)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig, ax  # type: ignore[return-value]


def plot_band_exposure(
    center_frequencies: np.ndarray,
    band_exposure_dB: np.ndarray,
    *,
    title: str = "1/3-Octave Band Exposure (SEL)",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Tuple[Figure, Axes]:
    """
    Plot bar chart of per-band sound exposure levels.

    Args:
        center_frequencies: Band center frequencies in Hz.
        band_exposure_dB: Sound exposure level per band in dB.
        title: Plot title.
        ax: Existing axes.
        figsize: Figure size.

    Returns:
        (figure, axes) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[assignment]

    x = np.arange(len(center_frequencies))
    ax.bar(x, band_exposure_dB, color=_ACCENT, alpha=0.85,
           edgecolor=_BG_ELEVATED, linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{f:.0f}' for f in center_frequencies],
                       rotation=45, ha='right')
    ax.set_xlabel('Center Frequency (Hz)')
    ax.set_ylabel('Sound Exposure Level (dB)')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    return fig, ax  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════
#  Multi-panel shot summary
# ═══════════════════════════════════════════════════════════

def create_shot_summary_figure(
    time_s: np.ndarray,
    pressure_Pa: np.ndarray,
    stft_z: STFTResult,
    stft_c: STFTResult,
    metrics: ShotMetrics,
    shot: Optional[ShotEvent] = None,
    *,
    title: str = "Shot Analysis Summary",
    figsize: Tuple[float, float] = (14, 12),
) -> Figure:
    """
    Create comprehensive multi-panel summary figure for a single shot.

    Panels:
      1. Waveform in Pa
      2. LAF/LAS time curves
      3. Z-weighted spectrogram
      4. C-weighted spectrogram
      5. Band exposure bar chart
      6. Metrics summary text

    Args:
        time_s: Time axis.
        pressure_Pa: Pressure waveform.
        stft_z: Z-weighted STFT result.
        stft_c: C-weighted STFT result.
        metrics: Computed shot metrics.
        shot: Optional shot event info.
        title: Overall figure title.
        figsize: Figure size.

    Returns:
        Figure object.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1],
                          hspace=0.35, wspace=0.3)

    # ── Panel 1: Waveform (top left) ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_s, pressure_Pa, color=_ACCENT, linewidth=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pressure (Pa)')
    ax1.set_title('Waveform')
    ax1.grid(True, alpha=0.2)
    if shot:
        ax1.axvline(shot.time_s - time_s[0], color=COLOR_SHOT_MARKER,
                    linestyle='--', alpha=0.6, linewidth=1.0)

    # ── Panel 2: Level curves (top right) ──
    ax2 = fig.add_subplot(gs[0, 1])
    if len(metrics.time_s) > 0:
        ax2.plot(metrics.time_s, metrics.LAF, color=COLOR_FAST,
                 linewidth=1.0, label='LAF')
        ax2.plot(metrics.time_s, metrics.LAS, color=COLOR_SLOW,
                 linewidth=1.0, linestyle='--', label='LAS')
        ax2.legend(loc='upper right')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Level (dB SPL)')
    ax2.set_title(f'Time-Weighted Levels (LAFmax = {metrics.LAFmax:.1f} dB)')
    ax2.grid(True, alpha=0.2)

    # ── Panel 3: Z-weighted spectrogram (middle left) ──
    ax3 = fig.add_subplot(gs[1, 0])
    vmax_z = stft_z.get_max_level()
    vmin_z = max(0, vmax_z - 80)
    pcm3 = ax3.pcolormesh(stft_z.time_s, stft_z.frequencies_Hz,
                           stft_z.magnitude_dB,
                           shading='auto', cmap=CMAP_SPECTROGRAM,
                           vmin=vmin_z, vmax=vmax_z, rasterized=True)
    ax3.set_ylim(0, 20000)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('Z-Weighted Spectrogram')
    cb3 = fig.colorbar(pcm3, ax=ax3)
    _style_colorbar(cb3)

    # ── Panel 4: C-weighted spectrogram (middle right) ──
    ax4 = fig.add_subplot(gs[1, 1])
    vmax_c = stft_c.get_max_level()
    vmin_c = max(0, vmax_c - 80)
    pcm4 = ax4.pcolormesh(stft_c.time_s, stft_c.frequencies_Hz,
                           stft_c.magnitude_dB,
                           shading='auto', cmap=CMAP_SPECTROGRAM,
                           vmin=vmin_c, vmax=vmax_c, rasterized=True)
    ax4.set_ylim(0, 20000)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('C-Weighted Spectrogram')
    cb4 = fig.colorbar(pcm4, ax=ax4)
    _style_colorbar(cb4)

    # ── Panel 5: Band exposure (bottom left) ──
    ax5 = fig.add_subplot(gs[2, 0])
    if len(metrics.band_frequencies) > 0:
        x = np.arange(len(metrics.band_frequencies))
        ax5.bar(x, metrics.band_exposure_dB, color=_ACCENT,
                alpha=0.85, edgecolor=_BG_ELEVATED, linewidth=0.5)
        step = max(1, len(x) // 10)
        ax5.set_xticks(x[::step])
        ax5.set_xticklabels(
            [f'{f:.0f}' for f in metrics.band_frequencies[::step]],
            rotation=45, ha='right')
    ax5.set_xlabel('Center Frequency (Hz)')
    ax5.set_ylabel('SEL (dB)')
    ax5.set_title('1/3-Octave Band Exposure')
    ax5.grid(True, axis='y', alpha=0.2)

    # ── Panel 6: Metrics summary (bottom right) ──
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    lines = [
        ('PEAK LEVELS', None),
        ('Lpeak(Z)', f'{metrics.Lpeak_Z:.1f} dB SPL'),
        ('Lpeak(A)', f'{metrics.Lpeak_A:.1f} dB SPL'),
        ('Lpeak(C)', f'{metrics.Lpeak_C:.1f} dB SPL'),
        ('', ''),
        ('EXPOSURE (SEL)', None),
        ('LAE', f'{metrics.LAE:.1f} dB'),
        ('LZE', f'{metrics.LZE:.1f} dB'),
        ('', ''),
        ('MAX TIME-WEIGHTED', None),
        ('LAFmax', f'{metrics.LAFmax:.1f} dB SPL'),
        ('LASmax', f'{metrics.LASmax:.1f} dB SPL'),
        ('', ''),
        ('DURATION', f'{metrics.duration_s * 1000:.1f} ms'),
    ]

    y_pos = 0.95
    for label, value in lines:
        if value is None:
            # Section header
            ax6.text(0.08, y_pos, label, transform=ax6.transAxes,
                     fontsize=8, fontweight=700, color=_TEXT_MUTED,
                     fontfamily='sans-serif')
        elif value == '':
            pass  # spacer
        else:
            ax6.text(0.08, y_pos, label, transform=ax6.transAxes,
                     fontsize=9.5, color=_TEXT_SECONDARY, fontfamily='monospace')
            ax6.text(0.55, y_pos, value, transform=ax6.transAxes,
                     fontsize=9.5, color=_TEXT_PRIMARY, fontweight=600,
                     fontfamily='monospace')
        y_pos -= 0.065

    ax6.set_title('Metrics Summary')

    fig.suptitle(title, fontsize=14, fontweight='bold', color=_TEXT_PRIMARY)

    return fig


# ═══════════════════════════════════════════════════════════
#  Interactive HTML charts (Plotly)
# ═══════════════════════════════════════════════════════════

def save_interactive_waveform_html(
    output_path: Path,
    time_s: np.ndarray,
    pressure_Pa: np.ndarray,
    shots: Optional[List[ShotEvent]] = None,
    title: str = "Pressure Waveform",
) -> bool:
    """
    Save an interactive zoomable/pannable waveform chart as HTML (Plotly).
    Returns True if saved, False if Plotly is not installed or an error occurs.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    try:
        output_path = Path(output_path)
        if output_path.suffix.lower() != '.html':
            output_path = output_path.with_suffix('.html')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        x = np.asarray(time_s, dtype=np.float64).ravel()
        y = np.asarray(pressure_Pa, dtype=np.float64).ravel()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color=_ACCENT, width=1),
            name='Pressure (Pa)',
        ))

        if shots:
            for s in shots:
                fig.add_vline(
                    x=float(s.time_s), line_dash="dash",
                    line_color=COLOR_SHOT_MARKER, opacity=0.6,
                )

        fig.update_layout(
            {**_PLOTLY_LAYOUT,
             'title': str(title),
             'xaxis_title': "Time (s)",
             'yaxis_title': "Pressure (Pa)",
             'hovermode': "x unified",
             'height': 450},
        )
        fig.write_html(str(output_path), config={"scrollZoom": True})
        print(f"  -> {output_path.resolve()}")
        return True
    except Exception as e:
        print(f"  [Plotly waveform error] {e}")
        return False


def save_interactive_spectrogram_html(
    output_path: Path,
    result: STFTResult,
    shots: Optional[List[ShotEvent]] = None,
    title: Optional[str] = None,
) -> bool:
    """
    Save an interactive zoomable/pannable spectrogram as HTML (Plotly).
    Returns True if saved, False if Plotly is not installed or an error occurs.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    try:
        output_path = Path(output_path)
        if output_path.suffix.lower() != '.html':
            output_path = output_path.with_suffix('.html')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if title is None:
            w = result.weighting
            title = f"Spectrogram \u2014 {w}-Weighted"

        vmax = float(result.get_max_level())
        vmin = float(max(0, vmax - 80))

        x = np.asarray(result.time_s, dtype=np.float64)
        y = np.asarray(result.frequencies_Hz, dtype=np.float64)
        z = np.asarray(result.magnitude_dB, dtype=np.float64)

        fig = go.Figure(go.Heatmap(
            x=x, y=y, z=z,
            colorscale='Magma',
            zmin=vmin, zmax=vmax,
            colorbar=dict(
                title="dB SPL",
                title_font=dict(color=_TEXT_SECONDARY, size=11),
                tickfont=dict(color=_TEXT_MUTED, size=10),
                outlinecolor=_BORDER,
                outlinewidth=0.5,
            ),
        ))

        if shots:
            for s in shots:
                fig.add_vline(
                    x=float(s.time_s), line_dash="dash",
                    line_color=_TEXT_PRIMARY, opacity=0.4,
                )

        y_max = float(min(20000, result.frequencies_Hz[-1]))
        fig.update_layout(
            {**_PLOTLY_LAYOUT,
             'title': str(title),
             'xaxis_title': "Time (s)",
             'yaxis_title': "Frequency (Hz)",
             'height': 500,
             'yaxis': dict(
                 gridcolor=_GRID,
                 zerolinecolor=_BORDER,
                 linecolor=_BORDER,
                 tickfont=dict(color=_TEXT_MUTED, size=10),
                 title_font=dict(color=_TEXT_SECONDARY, size=11),
                 range=[0, y_max],
             )},
        )
        fig.write_html(str(output_path), config={"scrollZoom": True})
        print(f"  -> {output_path.resolve()}")
        return True
    except Exception as e:
        print(f"  [Plotly spectrogram error] {e}")
        return False


# ═══════════════════════════════════════════════════════════
#  File I/O
# ═══════════════════════════════════════════════════════════

def save_figure(
    fig: Figure,
    output_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    formats: Optional[List[str]] = None,
) -> List[Path]:
    """
    Save figure to file(s).

    Args:
        fig: Figure to save.
        output_path: Output path (extension will be replaced for each format).
        dpi: Resolution in dots per inch.
        formats: List of output formats ('png', 'pdf', 'svg'). Defaults to ['png'].

    Returns:
        List of saved file paths.
    """
    if formats is None:
        formats = ['png']

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        saved_paths.append(path)

    return saved_paths


# ── CLI for testing ──

def main() -> int:
    """Test plotting functions."""
    import argparse

    parser = argparse.ArgumentParser(description="Test acoustic plotting")
    parser.add_argument("--test", action="store_true",
                        help="Run test with synthetic data")
    args = parser.parse_args()

    if args.test:
        setup_plot_style()

        sr = 96000
        duration = 0.5
        t = np.arange(int(sr * duration)) / sr

        decay = 0.02
        signal = np.exp(-t / decay) * np.sin(2 * np.pi * 1000 * t)
        signal[0] = 1.0
        pressure_Pa = signal * 200.0

        fig1, ax1 = plot_waveform_pa(t, pressure_Pa,
                                     title="Test Impulse Waveform")
        fig1.savefig('test_waveform.png', dpi=150,
                     facecolor=fig1.get_facecolor())
        print("Saved test_waveform.png")

        from metrics import compute_time_weighted_levels
        from weighting import apply_a_weight

        x_a = apply_a_weight(pressure_Pa, sr)
        time_lev, LAF, LAS = compute_time_weighted_levels(x_a, sr)

        fig2, ax2 = plot_level_curves(time_lev, LAF, LAS,
                                      title="Test Level Curves")
        fig2.savefig('test_levels.png', dpi=150,
                     facecolor=fig2.get_facecolor())
        print("Saved test_levels.png")

        plt.close('all')
        print("\nTest plots generated successfully!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
