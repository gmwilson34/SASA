#!/usr/bin/env python3
"""
plots.py - Customer-deliverable acoustic figures

Every figure this module produces is a MEASUREMENT DOCUMENT: it is printed,
emailed to a customer, and attached to a test report. That drives four rules:

  1. LIGHT is the default theme. Saved figures go on paper. A dark theme is
     still available for on-screen use via ``setup_plot_style(theme="dark")``.
  2. Every saved figure carries a provenance footer - source file, analysis
     timestamp, calibration, sample rate, software version, and whether the
     levels are dB SPL or dB re FS. A figure without that footer is an
     unattributable claim.
  3. No function hard-codes "dB SPL". Every one takes ``level_unit`` and
     labels its axes with it. An uncalibrated analysis labelled "dB SPL" is a
     false measurement claim.
  4. Colour scales are explicit and shareable, so two recordings - or the two
     panels of one figure - can be compared by eye.

Palette
-------
The colours below are transcribed from ``ui/renderer/tokens.css``, which is the
SINGLE SOURCE OF TRUTH for the SASA "Ridgeback Instrument" design system. Every
pair in that file has been checked numerically against WCAG 2.1. If a colour
changes there it must be changed here, and nowhere else in this file - do not
introduce a hex value that does not appear in tokens.css.

Public API (all pre-existing names and signatures still work):
    setup_plot_style
    plot_waveform_pa
    plot_spectrogram_dB
    plot_third_octave_heatmap
    plot_level_curves
    plot_band_exposure
    create_shot_summary_figure
    save_figure
    save_interactive_waveform_html
    save_interactive_spectrogram_html

New in this revision:
    FigureProvenance, set_default_provenance, set_default_level_unit
    resolve_db_range, shared_db_range
    plot_insertion_loss        - the headline suppressor deliverable
    plot_shot_overlay          - spot the outlier shot at a glance
    plot_measurement_quality   - is this measurement admissible at all
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - safe for worker threads
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from calibration import P_REF
from shot_detect import ShotEvent
from metrics import ShotMetrics
from STFT import STFTResult

try:  # provenance.py owns the version string; never duplicate it
    from provenance import __version__ as _SASA_VERSION
except Exception:  # noqa: BLE001 - a plotting module must not fail to import
    _SASA_VERSION = "unknown"


# ═══════════════════════════════════════════════════════════════════════════
#  Palette - transcribed from ui/renderer/tokens.css (source of truth)
# ═══════════════════════════════════════════════════════════════════════════
#
#  tokens.css                  key here
#  --------------------------  --------------------------
#  --bg-canvas                 bg_canvas
#  --bg-surface / --plot-bg    bg_surface / plot_bg
#  --border, --border-subtle   border, border_subtle
#  --text, --text-2, --text-3  text, text_2, text_3
#  --accent, --accent-wash     accent, accent_wash
#  --ok/--warn/--danger/--info + their -wash / -border variants
#  --series-1 ... --series-6   series[0..5]
#  --shot-marker               shot_marker
#
#  Nothing below may be edited independently of tokens.css.

_LIGHT: Dict[str, Any] = {
    'name': 'light',
    'bg_canvas': '#F4F6F9',
    'bg_surface': '#FFFFFF',
    'bg_sunken': '#EAEEF3',
    'bg_inset': '#F7F9FB',
    'plot_bg': '#FFFFFF',

    'border': '#C8D1DE',
    'border_subtle': '#E1E7EF',
    'border_input': '#7C8798',

    'text': '#0F141A',
    'text_2': '#48566A',
    'text_3': '#556274',
    'text_on_accent': '#FFFFFF',

    'accent': '#14508C',
    'accent_wash': '#E8F0F9',
    'accent_border': '#A9C4E2',

    'ok': '#1B6E3C',      'ok_wash': '#E4F3EA',     'ok_border': '#A5D6BA',
    'warn': '#8A5A00',    'warn_wash': '#FBF0DC',   'warn_border': '#E4C489',
    'danger': '#A32020',  'danger_wash': '#FBE9E9', 'danger_border': '#E8AFAF',
    'info': '#0B5F73',    'info_wash': '#E2F1F5',   'info_border': '#9CCBD7',

    'series': ['#14508C', '#B45309', '#6D28D9', '#0F766E', '#9D174D', '#4D7C0F'],
    'shot_marker': '#A32020',
}

_DARK: Dict[str, Any] = {
    'name': 'dark',
    'bg_canvas': '#0E1318',
    'bg_surface': '#161C24',
    'bg_sunken': '#0A0E13',
    'bg_inset': '#131920',
    'plot_bg': '#0F1419',

    'border': '#2F3A47',
    'border_subtle': '#222B36',
    'border_input': '#5A6878',

    'text': '#E7ECF3',
    'text_2': '#AAB6C6',
    'text_3': '#8B99AB',
    'text_on_accent': '#08111C',

    'accent': '#5AA6F5',
    'accent_wash': '#15263A',
    'accent_border': '#2C4B6E',

    'ok': '#4ADE80',      'ok_wash': '#10241A',     'ok_border': '#1F4D33',
    'warn': '#FBBF24',    'warn_wash': '#241C08',   'warn_border': '#55400F',
    'danger': '#F87171',  'danger_wash': '#2A1414', 'danger_border': '#5C2626',
    'info': '#38BDF8',    'info_wash': '#0C2029',   'info_border': '#1C4557',

    'series': ['#60A5FA', '#FB923C', '#A78BFA', '#2DD4BF', '#F472B6', '#A3E635'],
    'shot_marker': '#F87171',
}

_THEMES = {'light': _LIGHT, 'dark': _DARK}
_THEME: Dict[str, Any] = _LIGHT          # current theme (light = the print default)


def _c(key: str) -> Any:
    """Look up a colour in the active theme. Never inline a hex value instead."""
    return _THEME[key]


def _series(i: int) -> str:
    """Categorical series colour, deuteranopia-safe ordering (see tokens.css)."""
    return _THEME['series'][i % len(_THEME['series'])]


def current_theme() -> str:
    """Name of the active plot theme ('light' or 'dark')."""
    return str(_THEME['name'])


# ── Semantic series assignments (kept as module names for backwards compat;
#    refreshed by setup_plot_style() so they follow the active theme) ──
COLOR_Z_WEIGHT = _LIGHT['series'][0]
COLOR_A_WEIGHT = _LIGHT['series'][1]
COLOR_C_WEIGHT = _LIGHT['series'][2]
COLOR_BAND = _LIGHT['series'][3]
COLOR_REFERENCE = _LIGHT['series'][4]   # unsuppressed reference
COLOR_TEST = _LIGHT['series'][5]        # suppressed test article
COLOR_FAST = _LIGHT['series'][1]
COLOR_SLOW = _LIGHT['series'][2]
COLOR_SHOT_MARKER = _LIGHT['shot_marker']


def _refresh_module_colors() -> None:
    g = globals()
    g['COLOR_Z_WEIGHT'] = _series(0)
    g['COLOR_A_WEIGHT'] = _series(1)
    g['COLOR_C_WEIGHT'] = _series(2)
    g['COLOR_BAND'] = _series(3)
    g['COLOR_REFERENCE'] = _series(4)
    g['COLOR_TEST'] = _series(5)
    g['COLOR_FAST'] = _series(1)
    g['COLOR_SLOW'] = _series(2)
    g['COLOR_SHOT_MARKER'] = _c('shot_marker')


# ═══════════════════════════════════════════════════════════════════════════
#  Figure geometry and type - one system, sized for a report page
# ═══════════════════════════════════════════════════════════════════════════
#
#  The old figures were 12 in wide with 9-10 pt type and were then saved with
#  bbox_inches='tight'. Dropped into a Letter/A4 report they are scaled to
#  ~6.5 in, so the type lands at ~5 pt and every figure ends up a slightly
#  different width. Here the figure is authored at its FINAL PRINTED SIZE
#  (7.5 in = US Letter minus 0.5 in margins, also fits A4), so 9 pt type prints
#  as 9 pt and every figure in the report shares one width.

FIGURE_WIDTH_IN = 7.5                    # authored = printed width
FOOTER_HEIGHT_IN = 0.60                  # reserved band for the provenance footer

SIZE_STRIP = (FIGURE_WIDTH_IN, 2.9)      # waveform, level curves
SIZE_PANEL = (FIGURE_WIDTH_IN, 3.9)      # spectrogram
SIZE_TALL = (FIGURE_WIDTH_IN, 4.6)       # 1/3-octave heatmap, band bars
SIZE_PAGE = (FIGURE_WIDTH_IN, 9.6)       # full-page multi-panel summary

# Backwards-compatible aliases for the old default figsizes.
FIGSIZE_WAVEFORM = SIZE_STRIP
FIGSIZE_SPECTROGRAM = SIZE_PANEL
FIGSIZE_HEATMAP = SIZE_TALL

FONT_2XS = 6.5    # footer, colourbar ticks
FONT_XS = 7.5     # tick labels, annotations
FONT_SM = 8.5     # legends, secondary labels
FONT_BASE = 9.0   # axis labels
FONT_MD = 10.0    # axes titles
FONT_LG = 12.0    # figure title

DEFAULT_DPI = 200
PUBLICATION_DPI = 300

_SANS = ['Helvetica Neue', 'Inter', 'Segoe UI', 'Arial', 'DejaVu Sans']
_MONO = ['SF Mono', 'JetBrains Mono', 'Menlo', 'Consolas', 'DejaVu Sans Mono']


# ═══════════════════════════════════════════════════════════════════════════
#  Colormaps
# ═══════════════════════════════════════════════════════════════════════════
#
#  Previously: 'magma' for spectrograms and 'inferno' for the 1/3-octave
#  heatmap. Both are perceptually uniform, but using TWO different maps for the
#  same physical quantity means two figures in one report cannot be compared by
#  eye, and both spend their top third in a red->yellow ramp that protanopes
#  and deuteranopes compress.
#
#  Chosen: 'cividis'. It is perceptually uniform (monotonic, near-linear
#  lightness ramp), and it was designed specifically so that a person with
#  deuteranopia or protanopia sees essentially the same image as a trichromat
#  (Nunez, Anderton & Renslow, PLoS ONE 2018). Because the ordering is carried
#  by lightness alone it also survives a greyscale photocopy - which is what
#  happens to a test report. One map is used for every level surface in this
#  module so that panels are mutually comparable. Override per call with cmap=.
CMAP_SPECTROGRAM = 'cividis'
CMAP_OCTAVE = 'cividis'

# ── dB ranges ──
DEFAULT_DYNAMIC_RANGE_DB = 70.0   # span shown below a spectrogram's own maximum
BAND_DYNAMIC_RANGE_DB = 60.0      # span for the 1/3-octave heatmap
EXPOSURE_DYNAMIC_RANGE_DB = 40.0  # span for the 1/3-octave exposure bars
DB_RANGE_DEFAULT = (20.0, 160.0)  # retained for callers that imported it
DB_RANGE_AMBIENT = (-10.0, 100.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Level units
# ═══════════════════════════════════════════════════════════════════════════

UNIT_SPL = "dB SPL"
UNIT_FS = "dB re FS"
UNIT_UNDECLARED = "dB"

_DEFAULT_LEVEL_UNIT: Optional[str] = None


def set_default_level_unit(unit: Optional[str]) -> None:
    """
    Declare the level unit for every subsequent figure ("dB SPL" / "dB re FS").

    Call once from the analysis pipeline with ``calibration.level_unit`` so no
    figure has to guess. Pass None to clear.
    """
    global _DEFAULT_LEVEL_UNIT
    _DEFAULT_LEVEL_UNIT = unit


def _is_spl(unit: str) -> bool:
    return 'SPL' in unit.upper()


def _resolve_level_unit(explicit: Optional[str] = None,
                        inferred: Optional[str] = None) -> str:
    """
    Resolve the level unit, most trustworthy source first.

    explicit argument > pipeline declaration (set_default_level_unit) > unit
    inferred from the data object > the neutral "dB".

    The pipeline's declaration outranks the data object because a flag such as
    ``STFTResult.calibrated`` carries whatever default the analysis was called
    with, whereas the declaration is made from the actual Calibration in use.
    The fallback is deliberately NOT "dB SPL": claiming SPL for an analysis
    whose calibration is unknown is a false measurement claim, whereas "dB" is
    merely incomplete - and the footer says so.
    """
    for candidate in (explicit, _DEFAULT_LEVEL_UNIT, inferred):
        if candidate:
            return str(candidate)
    return UNIT_UNDECLARED


def _unit_from_calibrated_flag(obj: Any) -> Optional[str]:
    """Infer the unit from any object exposing a .calibrated boolean."""
    flag = getattr(obj, 'calibrated', None)
    if flag is None:
        return None
    return UNIT_SPL if bool(flag) else UNIT_FS


def _level_ref(unit: str) -> float:
    """Amplitude reference for the unit: 20 uPa for SPL, full scale otherwise."""
    return P_REF if _is_spl(unit) else 1.0


def _amp_to_level(amplitude: Union[float, np.ndarray], unit: str) -> Union[float, np.ndarray]:
    ref = _level_ref(unit)
    return 20.0 * np.log10(np.maximum(np.abs(amplitude), 1e-30) / ref)


def _level_to_amp(level_dB: Union[float, np.ndarray], unit: str) -> Union[float, np.ndarray]:
    return _level_ref(unit) * np.power(10.0, np.asarray(level_dB, dtype=float) / 20.0)


def _amplitude_axis_label(unit: str) -> str:
    """
    Label the waveform's own quantity honestly.

    Pascals may only be claimed when the levels are declared as SPL; with an
    undeclared calibration the axis says so rather than asserting either unit.
    """
    if _is_spl(unit):
        return 'Pressure (Pa)'
    if unit == UNIT_FS:
        return 'Amplitude (full scale)'
    return 'Amplitude (unit not declared)'


def _level_axis_label(unit: str, quantity: str = 'Level') -> str:
    if unit == UNIT_UNDECLARED:
        return f'{quantity} (dB, unit not declared)'
    return f'{quantity} ({unit})'


# ═══════════════════════════════════════════════════════════════════════════
#  Provenance - every saved figure is attributable
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FigureProvenance:
    """
    What a reader needs in order to trust - or reproduce - a figure.

    Attach it to any plot call (``provenance=``), or set it once for the whole
    run with :func:`set_default_provenance`.
    """
    source_file: str = ""
    analysis_timestamp: str = ""      # ISO-ish local time; filled in if blank
    calibration: str = ""             # calibration.description
    sample_rate_Hz: float = 0.0
    level_unit: str = ""              # "dB SPL" or "dB re FS"
    software: str = ""                # defaults to "SASA <version>"
    operator: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.analysis_timestamp:
            self.analysis_timestamp = datetime.now().astimezone().strftime(
                '%Y-%m-%d %H:%M:%S %Z').strip()
        if not self.software:
            self.software = f"SASA {_SASA_VERSION}"

    # -- rendering -------------------------------------------------------
    def unit_statement(self) -> Tuple[str, str]:
        """(text, semantic-key) describing how levels must be read."""
        unit = self.level_unit or UNIT_UNDECLARED
        if _is_spl(unit):
            return (f"Levels: {unit} (re 20 uPa, calibrated)", 'ok')
        if unit == UNIT_FS:
            return ("Levels: dB re FS - UNCALIBRATED, not sound pressure level", 'warn')
        return ("Levels: dB - unit not declared; calibration unverified", 'warn')

    def lines(self) -> Tuple[str, str]:
        """The two footer lines, in reading order."""
        sr = (f"{self.sample_rate_Hz:,.0f} Hz".replace(',', ' ')
              if self.sample_rate_Hz else "sample rate not recorded")
        src = self.source_file or "source file not recorded"
        cal = self.calibration or "calibration not recorded"
        first = f"Source: {src}   |   Sample rate: {sr}"
        second = f"Calibration: {cal}   |   Analysed: {self.analysis_timestamp}   |   {self.software}"
        if self.operator:
            second += f"   |   Operator: {self.operator}"
        if self.notes:
            second += f"   |   {self.notes}"
        return first, second


def _ellipsize(text: str, max_chars: int) -> str:
    """Trim to a whole word and mark the cut, so nothing is silently lost."""
    if max_chars <= 3 or len(text) <= max_chars:
        return text
    cut = text[:max_chars - 3]
    space = cut.rfind(' ')
    if space > max_chars * 0.6:
        cut = cut[:space]
    return cut.rstrip(' |') + '...'


_DEFAULT_PROVENANCE: Optional[FigureProvenance] = None


def set_default_provenance(provenance: Optional[FigureProvenance]) -> None:
    """
    Set the provenance stamped on every figure that does not carry its own.

    The analysis pipeline should call this once, e.g.::

        plots.set_default_provenance(plots.FigureProvenance(
            source_file=wav_path.name,
            calibration=cal.description,
            sample_rate_Hz=sr,
            level_unit=cal.level_unit,
        ))
    """
    global _DEFAULT_PROVENANCE
    _DEFAULT_PROVENANCE = provenance
    if provenance is not None and provenance.level_unit:
        set_default_level_unit(provenance.level_unit)


def _coerce_provenance(prov: Union[None, FigureProvenance, Mapping[str, Any], str]
                       ) -> Optional[FigureProvenance]:
    if prov is None:
        return None
    if isinstance(prov, FigureProvenance):
        return prov
    if isinstance(prov, str):
        return FigureProvenance(source_file=prov)
    if isinstance(prov, Mapping):
        known = {f for f in FigureProvenance.__dataclass_fields__}  # type: ignore[attr-defined]
        return FigureProvenance(**{k: v for k, v in prov.items() if k in known})
    raise TypeError(f"Unsupported provenance type: {type(prov)!r}")


def _attach_provenance(fig: Figure,
                       prov: Union[None, FigureProvenance, Mapping[str, Any], str],
                       level_unit: Optional[str] = None) -> None:
    """Record provenance on the figure so save_figure() can stamp it."""
    resolved = _coerce_provenance(prov)
    if resolved is None:
        resolved = getattr(fig, '_sasa_provenance', None)
    if resolved is None and _DEFAULT_PROVENANCE is not None:
        resolved = replace(_DEFAULT_PROVENANCE)
    if resolved is None and level_unit:
        resolved = FigureProvenance(level_unit=level_unit)
    if resolved is None:
        return
    if level_unit and not resolved.level_unit:
        resolved = replace(resolved, level_unit=level_unit)
    fig._sasa_provenance = resolved  # type: ignore[attr-defined]


def annotate_provenance(fig: Figure,
                        provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
                        *, force: bool = False) -> None:
    """
    Draw the provenance footer on ``fig``. Idempotent.

    Called automatically by :func:`save_figure`, so no figure can leave the
    application without an attribution block.
    """
    if getattr(fig, '_sasa_footer_drawn', False) and not force:
        return

    prov = _coerce_provenance(provenance)
    if prov is None:
        prov = getattr(fig, '_sasa_provenance', None)
    if prov is None:
        prov = replace(_DEFAULT_PROVENANCE) if _DEFAULT_PROVENANCE is not None \
            else FigureProvenance(level_unit=_DEFAULT_LEVEL_UNIT or "")

    fig_h = float(fig.get_figheight())
    fig_w = float(fig.get_figwidth())
    band = min(0.5, FOOTER_HEIGHT_IN / fig_h)      # fraction of figure height
    line1, line2 = prov.lines()
    unit_text, unit_key = prov.unit_statement()

    # Character budget for the monospace footer at this figure width.
    usable_pt = 0.976 * 72.0 * fig_w
    max_chars = max(20, int(usable_pt / (0.60 * FONT_2XS)))
    chip_chars = len(unit_text) + 8

    # Hairline rule separating the plot area from the attribution block.
    fig.add_artist(plt.Line2D(
        [0.012, 0.988], [band * 0.98, band * 0.98],
        transform=fig.transFigure, color=_c('border'), linewidth=0.6,
        zorder=5,
    ))

    y = band * 0.62
    dy = band * 0.32
    fig.text(0.012, y, _ellipsize(line1, max_chars - chip_chars),
             transform=fig.transFigure,
             fontsize=FONT_2XS, family='monospace', color=_c('text_3'),
             ha='left', va='center', zorder=6)
    fig.text(0.012, y - dy, _ellipsize(line2, max_chars),
             transform=fig.transFigure,
             fontsize=FONT_2XS, family='monospace', color=_c('text_3'),
             ha='left', va='center', zorder=6)

    # The unit statement is the load-bearing one: never colour alone, so it
    # carries a glyph and words as well as a semantic hue.
    glyph = 'OK' if unit_key == 'ok' else '!'
    fig.text(0.988, y, f" {glyph}  {unit_text} ",
             transform=fig.transFigure, fontsize=FONT_2XS, family='monospace',
             color=_c(unit_key), ha='right', va='center', zorder=6,
             bbox=dict(boxstyle='round,pad=0.3',
                       facecolor=_c(f'{unit_key}_wash'),
                       edgecolor=_c(f'{unit_key}_border'), linewidth=0.6))

    fig._sasa_footer_drawn = True  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════
#  Style
# ═══════════════════════════════════════════════════════════════════════════

def setup_plot_style(theme: str = 'light') -> None:
    """
    Configure matplotlib for SASA figures.

    Args:
        theme: 'light' (default - the theme saved figures and printed reports
               use) or 'dark' (on-screen work in a low-light room).

    Every colour comes from tokens.css via the palette above.
    """
    global _THEME
    key = str(theme).strip().lower()
    if key in ('print', 'paper', 'report'):
        key = 'light'
    if key in ('screen', 'night', 'tactical'):
        key = 'dark'
    _THEME = _THEMES.get(key, _LIGHT)
    _refresh_module_colors()

    plt.rcParams.update({
        # Figure
        'figure.facecolor': _c('bg_surface'),
        'figure.edgecolor': _c('border'),
        'figure.dpi': DEFAULT_DPI,
        'savefig.facecolor': _c('bg_surface'),
        'savefig.edgecolor': 'none',

        # Axes
        'axes.facecolor': _c('plot_bg'),
        'axes.edgecolor': _c('border'),
        'axes.labelcolor': _c('text_2'),
        'axes.titlecolor': _c('text'),
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Text
        'text.color': _c('text'),

        # Ticks
        'xtick.color': _c('border'),
        'ytick.color': _c('border'),
        'xtick.labelcolor': _c('text_3'),
        'ytick.labelcolor': _c('text_3'),
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,

        # Grid
        'grid.color': _c('border_subtle'),
        'grid.alpha': 1.0,
        'grid.linewidth': 0.5,

        # Legend
        'legend.facecolor': _c('bg_surface'),
        'legend.edgecolor': _c('border'),
        'legend.labelcolor': _c('text_2'),
        'legend.framealpha': 0.94,
        'legend.borderpad': 0.45,

        # Type
        'font.family': 'sans-serif',
        'font.sans-serif': _SANS,
        'font.monospace': _MONO,
        'font.size': FONT_BASE,
        'axes.titlesize': FONT_MD,
        'axes.titleweight': 'bold',
        'axes.labelsize': FONT_BASE,
        'xtick.labelsize': FONT_XS,
        'ytick.labelsize': FONT_XS,
        'legend.fontsize': FONT_SM,
        'figure.titlesize': FONT_LG,
        'figure.titleweight': 'bold',

        # Lines
        'lines.linewidth': 0.9,
        'axes.linewidth': 0.6,

        # Output
        'savefig.dpi': DEFAULT_DPI,
        'pdf.fonttype': 42,     # embed TrueType, keep text selectable in the PDF
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


# ═══════════════════════════════════════════════════════════════════════════
#  Small helpers
# ═══════════════════════════════════════════════════════════════════════════

def _mono_ticks(*axes_objs: Axes) -> None:
    """
    Render numeric tick labels in the monospace face with tabular figures, so
    digits align in columns and a changing reading does not shift the layout.
    """
    for ax in axes_objs:
        try:
            ax.xaxis.set_tick_params(labelfontfamily='monospace')
            ax.yaxis.set_tick_params(labelfontfamily='monospace')
        except (AttributeError, TypeError):  # older matplotlib
            for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                lbl.set_family('monospace')


def _thin_time_axis(ax: Axes, nbins: int = 6) -> None:
    """Cap the number of x ticks so mono labels cannot run into one another."""
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=3))


def _grid(ax: Axes, axis: str = 'both') -> None:
    ax.grid(True, axis=axis, color=_c('border_subtle'), linewidth=0.5, alpha=1.0)
    ax.set_axisbelow(True)


def _finalize(fig: Figure, owns_figure: bool, *, footer: bool = True,
              top: float = 0.99) -> None:
    """
    Lay the figure out, reserving the provenance band at the bottom.

    Constrained layout is used rather than tight_layout because it accounts for
    colorbars and suptitles - with tight_layout a colorbar's label collides
    with the next panel's y-axis label on the multi-panel page.
    """
    if not owns_figure:
        return
    fig_h = float(fig.get_figheight())
    bottom = min(0.5, FOOTER_HEIGHT_IN / fig_h) if footer else 0.02
    rect = (0.012, bottom, 0.976, max(0.1, top - bottom))
    engine = fig.get_layout_engine()
    # The engine must already be constrained (set at figure creation): adding
    # it after a colorbar exists produces a degenerate layout grid.
    if engine is not None and hasattr(engine, 'set'):
        try:
            engine.set(rect=rect, h_pad=0.03, w_pad=0.03)
            return
        except Exception:  # noqa: BLE001
            pass
    try:
        fig.tight_layout(rect=(0.012, bottom, 0.988, top))
    except Exception:  # noqa: BLE001 - layout must never break a deliverable
        fig.subplots_adjust(left=0.10, right=0.97,
                            bottom=bottom + 0.04, top=top)


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """
    Convert cell CENTRE coordinates to the N+1 EDGE coordinates pcolormesh
    needs with shading='flat'. Passing centres where edges are expected is what
    shifted every band row by half a band in the old heatmap.
    """
    c = np.asarray(centers, dtype=float).ravel()
    if c.size == 0:
        return np.array([0.0, 1.0])
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate(([first], mid, [last]))


def _fmt_freq(f: float) -> str:
    """ISO-266-style nominal band label: 50, 250, 1k, 12.5k."""
    f = float(f)
    if abs(f) >= 1000.0:
        return f'{f / 1000.0:g}k'
    return f'{f:g}'


def _style_colorbar(cbar: Any, label: str = 'Level (dB)') -> None:
    """Apply the active theme to a colorbar."""
    cbar.set_label(label, color=_c('text_2'), fontsize=FONT_XS)
    cbar.ax.yaxis.set_tick_params(color=_c('border'), labelcolor=_c('text_3'))
    try:
        cbar.ax.yaxis.set_tick_params(labelfontfamily='monospace')
    except (AttributeError, TypeError):
        pass
    cbar.outline.set_edgecolor(_c('border'))
    cbar.outline.set_linewidth(0.6)
    for t in cbar.ax.get_yticklabels():
        t.set_color(_c('text_3'))
        t.set_fontsize(FONT_2XS)


def _annotate_scale(ax: Axes, vmin: float, vmax: float, unit: str,
                    *, loc: str = 'upper right', prefix: str = 'colour scale',
                    compact: bool = False) -> None:
    """State the colour scale ON the figure, so two panels can be compared."""
    span = '' if compact else f'  ({vmax - vmin:.0f} dB span)'
    text = f'{prefix} {vmin:.0f} to {vmax:.0f} {unit}{span}'
    x, ha = (0.985, 'right') if 'right' in loc else (0.015, 'left')
    y, va = (0.965, 'top') if 'upper' in loc else (0.035, 'bottom')
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=FONT_2XS, family='monospace', color=_c('text_2'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=_c('bg_surface'),
                      edgecolor=_c('border'), linewidth=0.5, alpha=0.92),
            zorder=6)


def resolve_db_range(data: Any,
                     *,
                     db_range: Optional[Tuple[float, float]] = None,
                     dynamic_range_dB: float = DEFAULT_DYNAMIC_RANGE_DB,
                     round_to: float = 5.0) -> Tuple[float, float]:
    """
    Return an explicit (vmin, vmax) colour range.

    With ``db_range`` given it is used verbatim - that is how two figures, or
    two panels, are pinned to one scale. Otherwise the top is this data's own
    maximum rounded up to ``round_to``, and the bottom is a FIXED
    ``dynamic_range_dB`` below it, so the span is always the same size even
    though its position follows the signal.
    """
    if db_range is not None:
        lo, hi = float(db_range[0]), float(db_range[1])
        return (lo, hi) if hi > lo else (hi, lo + 1.0)

    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, float(dynamic_range_dB))
    vmax = math.ceil(float(np.max(finite)) / round_to) * round_to
    return (vmax - float(dynamic_range_dB), float(vmax))


def shared_db_range(*sources: Any,
                    dynamic_range_dB: float = DEFAULT_DYNAMIC_RANGE_DB,
                    round_to: float = 5.0) -> Tuple[float, float]:
    """
    One colour range covering several datasets, so they are comparable.

    Accepts arrays or STFTResult objects::

        rng = shared_db_range(stft_ref, stft_test)
        plot_spectrogram_dB(stft_ref,  db_range=rng)
        plot_spectrogram_dB(stft_test, db_range=rng)
    """
    peaks: List[float] = []
    for src in sources:
        if src is None:
            continue
        arr = np.asarray(getattr(src, 'magnitude_dB', src), dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            peaks.append(float(np.max(finite)))
    if not peaks:
        return (0.0, float(dynamic_range_dB))
    vmax = math.ceil(max(peaks) / round_to) * round_to
    return (vmax - float(dynamic_range_dB), float(vmax))


def _shot_markers(ax: Axes, shots: Optional[Sequence[ShotEvent]],
                  *, label: bool = True, offset_s: float = 0.0) -> None:
    if not shots:
        return
    for i, shot in enumerate(shots):
        ax.axvline(float(shot.time_s) - offset_s, color=_c('shot_marker'),
                   linestyle=(0, (4, 3)), alpha=0.8, linewidth=0.9,
                   label='Detected shot' if (label and i == 0) else None,
                   zorder=4)


# ═══════════════════════════════════════════════════════════════════════════
#  Waveform
# ═══════════════════════════════════════════════════════════════════════════

def plot_waveform_pa(
    time_s: np.ndarray,
    pressure_Pa: np.ndarray,
    *,
    shots: Optional[List[ShotEvent]] = None,
    title: str = "Pressure Waveform",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = SIZE_STRIP,
    show_dB_secondary: bool = False,
    level_unit: Optional[str] = None,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot the pressure waveform with optional shot markers.

    Args:
        time_s: Time axis in seconds.
        pressure_Pa: Pressure in Pascals (or full-scale units when uncalibrated).
        shots: Detected shot events to mark.
        title: Axes title.
        ax: Existing axes to draw on.
        figsize: Figure size, if creating one.
        show_dB_secondary: Draw the right-hand peak-level axis. This is now a
            REAL axis: its ticks sit at the pressures that correspond to round
            level values (L = 20*log10(|p|/p_ref)), so reading a tick gives the
            peak level of that amplitude. The previous implementation invented
            an 80 dB span with no relation to the plotted data; it is gone.
            Defaults to False because the waveform's own quantity is pressure.
        level_unit: "dB SPL" or "dB re FS". Determines both the amplitude axis
            label and the reference used by the secondary axis.
        provenance: Provenance for the saved figure's footer.

    Returns:
        (figure, axes)
    """
    unit = _resolve_level_unit(level_unit)
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        fig = ax.figure  # type: ignore[assignment]

    time_s = np.asarray(time_s, dtype=float).ravel()
    pressure_Pa = np.asarray(pressure_Pa, dtype=float).ravel()

    ax.axhline(0.0, color=_c('border'), linewidth=0.6, zorder=1)
    ax.plot(time_s, pressure_Pa, color=_series(0), linewidth=0.5,
            solid_joinstyle='round', zorder=3)

    if shots:
        _shot_markers(ax, shots)
        for shot in shots:
            idx = int(np.argmin(np.abs(time_s - float(shot.time_s)))) if time_s.size else 0
            if 0 <= idx < pressure_Pa.size:
                ax.plot(time_s[idx], pressure_Pa[idx], 'o',
                        color=_c('shot_marker'), markersize=3.0,
                        markeredgecolor='none', zorder=5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(_amplitude_axis_label(unit))
    ax.set_title(title)
    if time_s.size:
        ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    _grid(ax, axis='y')
    _mono_ticks(ax)

    if show_dB_secondary and pressure_Pa.size:
        p_max = float(np.max(np.abs(pressure_Pa)))
        if p_max > 0:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())            # identical linear scale
            ax2.set_facecolor('none')
            ax2.spines['right'].set_visible(True)
            ax2.spines['right'].set_color(_c('border'))
            L_top = float(_amp_to_level(p_max, unit))
            levels = np.arange(math.floor(L_top / 10.0) * 10.0,
                               math.floor(L_top / 10.0) * 10.0 - 61.0, -10.0)
            positions = np.asarray(_level_to_amp(levels, unit), dtype=float)
            keep = positions <= p_max
            levels, positions = levels[keep], positions[keep]
            # The pressure->level map is logarithmic, so equal dB steps crowd
            # together near zero. Drop any tick closer than 8% of the axis
            # range to the one above it rather than printing a smear.
            y0, y1 = ax.get_ylim()
            span = float(y1 - y0)
            kept_L: List[float] = []
            kept_p: List[float] = []
            last = None
            for L, p_pos in zip(levels, positions):
                if last is None or (last - p_pos) >= 0.08 * span:
                    kept_L.append(float(L))
                    kept_p.append(float(p_pos))
                    last = p_pos
            levels = np.asarray(kept_L)
            positions = np.asarray(kept_p)
            ax2.set_yticks(positions)
            ax2.set_yticklabels([f'{L:.0f}' for L in levels])
            ax2.set_ylabel(_level_axis_label(unit, 'Peak level of |p|'),
                           color=_c('text_3'), fontsize=FONT_SM)
            ax2.tick_params(axis='y', colors=_c('border'),
                            labelcolor=_c('text_3'), labelsize=FONT_2XS)
            _mono_ticks(ax2)
            for p in positions:
                ax.axhline(p, color=_c('border'), linewidth=0.4,
                           linestyle=(0, (1, 3)), zorder=2)

    if shots and len(shots) <= 8:
        ax.legend(loc='upper right', fontsize=FONT_SM)

    _attach_provenance(fig, provenance, unit)
    _finalize(fig, owns)
    return fig, ax  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
#  Spectrogram
# ═══════════════════════════════════════════════════════════════════════════

def plot_spectrogram_dB(
    result: STFTResult,
    *,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = SIZE_PANEL,
    db_range: Optional[Tuple[float, float]] = None,
    dynamic_range_dB: float = DEFAULT_DYNAMIC_RANGE_DB,
    freq_range: Optional[Tuple[float, float]] = None,
    cmap: str = CMAP_SPECTROGRAM,
    shots: Optional[List[ShotEvent]] = None,
    level_unit: Optional[str] = None,
    show_scale_note: bool = True,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot an STFT spectrogram on an EXPLICIT, shareable colour scale.

    Args:
        result: STFTResult to draw.
        db_range: Explicit (vmin, vmax). Pin two panels or two recordings to
            the same value to make them comparable - see :func:`shared_db_range`.
        dynamic_range_dB: When db_range is None, the fixed span shown below
            this figure's own maximum (default 70 dB). The span never changes
            size, only its position follows the signal.
        freq_range: (min_Hz, max_Hz) for the y-axis.
        cmap: Colormap; defaults to the perceptually-uniform, CVD-safe cividis.
        level_unit: Overrides the unit inferred from ``result.calibrated``.
        show_scale_note: Print the colour range on the figure.

    Returns:
        (figure, axes)
    """
    unit = _resolve_level_unit(level_unit, _unit_from_calibrated_flag(result))
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        fig = ax.figure  # type: ignore[assignment]

    vmin, vmax = resolve_db_range(result.magnitude_dB, db_range=db_range,
                                  dynamic_range_dB=dynamic_range_dB)

    if freq_range is None:
        freq_max = float(min(20000.0, float(result.frequencies_Hz[-1])
                             if len(result.frequencies_Hz) else 20000.0))
        freq_range = (0.0, freq_max)

    # Explicit edges: result.time_s / frequencies_Hz are cell CENTRES.
    x_edges = _centers_to_edges(result.time_s)
    y_edges = _centers_to_edges(result.frequencies_Hz)

    pcm = ax.pcolormesh(x_edges, y_edges, result.magnitude_dB,
                        shading='flat', cmap=cmap, vmin=vmin, vmax=vmax,
                        rasterized=True)

    if shots:
        for shot in shots:
            ax.axvline(float(shot.time_s), color=_c('bg_surface'),
                       linestyle=(0, (4, 3)), alpha=0.85, linewidth=1.4, zorder=4)
            ax.axvline(float(shot.time_s), color=_c('shot_marker'),
                       linestyle=(0, (4, 3)), alpha=0.95, linewidth=0.8, zorder=5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(freq_range)

    if title is None:
        weight_label = {
            'Z': 'Z-weighted (flat)',
            'A': 'A-weighted (perceptual)',
            'C': 'C-weighted',
        }.get(result.weighting, str(result.weighting))
        title = f'Spectrogram — {weight_label}'
    ax.set_title(title)
    _mono_ticks(ax)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.015, fraction=0.045)
    _style_colorbar(cbar, _level_axis_label(unit))
    if show_scale_note:
        _annotate_scale(ax, vmin, vmax, unit)

    _attach_provenance(fig, provenance, unit)
    _finalize(fig, owns)
    return fig, ax  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
#  1/3-octave heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_third_octave_heatmap(
    time_s: np.ndarray,
    center_frequencies: np.ndarray,
    band_levels_dB: np.ndarray,
    *,
    title: str = "1/3-Octave Band Levels",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = SIZE_TALL,
    db_range: Optional[Tuple[float, float]] = None,
    dynamic_range_dB: float = BAND_DYNAMIC_RANGE_DB,
    cmap: str = CMAP_OCTAVE,
    shots: Optional[List[ShotEvent]] = None,
    level_unit: Optional[str] = None,
    show_scale_note: bool = True,
    max_labels: int = 15,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot the 1/3-octave band time-frequency heatmap.

    BAND ALIGNMENT: row i of ``band_levels_dB`` is drawn between the edges
    i-0.5 and i+0.5, and the tick labelled ``center_frequencies[i]`` sits at
    i - the centre of that row. The previous version passed cell edges as
    coordinates while labelling centres, so every tick landed on the boundary
    between two bands and every row was mislabelled by half a band.

    Args:
        band_levels_dB: Shape (n_bands, n_frames). A transposed array is
            detected and corrected.
        db_range / dynamic_range_dB: Explicit colour range, or a fixed span
            below this data's own maximum.
    """
    unit = _resolve_level_unit(level_unit)
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        fig = ax.figure  # type: ignore[assignment]

    fc = np.asarray(center_frequencies, dtype=float).ravel()
    levels = np.asarray(band_levels_dB, dtype=float)
    n_bands = fc.size
    if levels.ndim == 2 and levels.shape[0] != n_bands and levels.shape[1] == n_bands:
        levels = levels.T

    vmin, vmax = resolve_db_range(levels, db_range=db_range,
                                  dynamic_range_dB=dynamic_range_dB)

    # y is band INDEX space: edges at i-0.5, ticks (centres) at i.
    y_edges = np.arange(n_bands + 1, dtype=float) - 0.5
    x_edges = _centers_to_edges(time_s)

    pcm = ax.pcolormesh(x_edges, y_edges, levels, shading='flat', cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True)

    if shots:
        for shot in shots:
            ax.axvline(float(shot.time_s), color=_c('bg_surface'),
                       linestyle=(0, (4, 3)), alpha=0.85, linewidth=1.4, zorder=4)
            ax.axvline(float(shot.time_s), color=_c('shot_marker'),
                       linestyle=(0, (4, 3)), alpha=0.95, linewidth=0.8, zorder=5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('1/3-octave band centre (Hz)')
    ax.set_title(title)
    ax.set_ylim(-0.5, n_bands - 0.5)

    n_labels = int(min(max(2, max_labels), max(1, n_bands)))
    tick_idx = np.unique(np.linspace(0, n_bands - 1, n_labels).round().astype(int))
    ax.set_yticks(tick_idx)                      # centre of row i is exactly i
    ax.set_yticklabels([_fmt_freq(fc[i]) for i in tick_idx])
    _mono_ticks(ax)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.015, fraction=0.045)
    _style_colorbar(cbar, _level_axis_label(unit))
    if show_scale_note:
        _annotate_scale(ax, vmin, vmax, unit)

    _attach_provenance(fig, provenance, unit)
    _finalize(fig, owns)
    return fig, ax  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
#  Time-weighted level curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_level_curves(
    time_s: np.ndarray,
    LAF: np.ndarray,
    LAS: np.ndarray,
    *,
    LZF: Optional[np.ndarray] = None,
    LZS: Optional[np.ndarray] = None,
    title: str = "Time-Weighted Sound Levels",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = SIZE_STRIP,
    shots: Optional[List[ShotEvent]] = None,
    level_unit: Optional[str] = None,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot LAfast / LAslow and, optionally, the Z-weighted pair.

    Series are separated by colour AND dash pattern, so the figure survives
    greyscale printing and colour-vision deficiency.
    """
    unit = _resolve_level_unit(level_unit)
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        fig = ax.figure  # type: ignore[assignment]

    ax.plot(time_s, LAF, color=_series(1), linewidth=1.1, label='LAF (A, fast)')
    ax.plot(time_s, LAS, color=_series(2), linewidth=1.1, linestyle=(0, (5, 2)),
            label='LAS (A, slow)')
    if LZF is not None:
        ax.plot(time_s, LZF, color=_series(0), linewidth=0.9, alpha=0.85,
                linestyle=(0, (2, 1.5)), label='LZF (Z, fast)')
    if LZS is not None:
        ax.plot(time_s, LZS, color=_series(3), linewidth=0.9, alpha=0.85,
                linestyle=(0, (1, 2)), label='LZS (Z, slow)')

    _shot_markers(ax, shots)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(_level_axis_label(unit))
    ax.set_title(title)
    ax.legend(loc='upper right', ncols=2, fontsize=FONT_SM)
    _grid(ax)
    _mono_ticks(ax)

    _attach_provenance(fig, provenance, unit)
    _finalize(fig, owns)
    return fig, ax  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
#  1/3-octave exposure bars
# ═══════════════════════════════════════════════════════════════════════════

def _exposure_baseline(values: np.ndarray, baseline_dB: Optional[float],
                       dynamic_range_dB: float, round_to: float = 5.0) -> float:
    """
    A meaningful floor for logarithmic bars.

    Bars drawn from 0 dB are arbitrary: 0 dB is not "no sound", it is the
    hearing threshold (or full scale, uncalibrated), so a 40 dB spread across
    the bands is compressed into the top 25 % of the plot. Baselining a fixed
    span below the loudest band puts the whole interesting range on the page.
    """
    if baseline_dB is not None:
        return float(baseline_dB)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    top = math.ceil(float(np.max(finite)) / round_to) * round_to
    return float(top - dynamic_range_dB)


def plot_band_exposure(
    center_frequencies: np.ndarray,
    band_exposure_dB: np.ndarray,
    *,
    title: str = "1/3-Octave Band Exposure (SEL)",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = SIZE_TALL,
    level_unit: Optional[str] = None,
    baseline_dB: Optional[float] = None,
    dynamic_range_dB: float = EXPOSURE_DYNAMIC_RANGE_DB,
    highlight_peak: bool = True,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> Tuple[Figure, Axes]:
    """
    Per-band sound exposure level as bars on a MEANINGFUL baseline.

    Args:
        baseline_dB: Explicit bar floor. Default: a fixed ``dynamic_range_dB``
            span below the loudest band, rounded to 5 dB. Bars no longer start
            at an arbitrary 0 dB.
        highlight_peak: Mark and label the dominant band.
    """
    unit = _resolve_level_unit(level_unit)
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        fig = ax.figure  # type: ignore[assignment]

    fc = np.asarray(center_frequencies, dtype=float).ravel()
    vals = np.asarray(band_exposure_dB, dtype=float).ravel()
    n = min(fc.size, vals.size)
    fc, vals = fc[:n], vals[:n]

    base = _exposure_baseline(vals, baseline_dB, dynamic_range_dB)
    heights = np.clip(vals - base, 0.0, None)
    x = np.arange(n)

    colors = [_series(3)] * n
    peak_i = int(np.argmax(vals)) if n else -1
    if highlight_peak and peak_i >= 0:
        colors[peak_i] = _c('accent')

    ax.bar(x, heights, bottom=base, color=colors, width=0.82,
           edgecolor=_c('border'), linewidth=0.4, zorder=3)
    ax.axhline(base, color=_c('border'), linewidth=0.8, zorder=4)

    if highlight_peak and peak_i >= 0:
        ax.annotate(f'{fc[peak_i]:.0f} Hz\n{vals[peak_i]:.1f} dB',
                    xy=(peak_i, vals[peak_i]), xytext=(0, 6),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=FONT_2XS, family='monospace', color=_c('accent'),
                    zorder=6)

    step = max(1, n // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([_fmt_freq(f) for f in fc[::step]], rotation=45, ha='right')
    ax.set_xlim(-0.8, n - 0.2 if n else 1)
    # Headroom for the peak-band annotation, so it never rides into the title.
    ax.set_ylim(base, (float(np.max(vals)) + 0.16 * max(1.0, dynamic_range_dB))
                if n else base + 1)
    ax.set_xlabel('1/3-octave band centre (Hz)')
    ax.set_ylabel(_level_axis_label(unit, 'Sound exposure level'))
    ax.set_title(title)
    ax.text(0.985, 0.965, f'bars referenced to {base:.0f} {unit}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT_2XS, family='monospace', color=_c('text_2'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=_c('bg_surface'),
                      edgecolor=_c('border'), linewidth=0.5, alpha=0.92))
    _grid(ax, axis='y')
    _mono_ticks(ax)

    _attach_provenance(fig, provenance, unit)
    _finalize(fig, owns)
    return fig, ax  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
#  NEW: insertion loss - the headline suppressor deliverable
# ═══════════════════════════════════════════════════════════════════════════

def _energy_sum_dB(levels_dB: np.ndarray) -> float:
    arr = np.asarray(levels_dB, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    return float(10.0 * np.log10(np.sum(np.power(10.0, arr / 10.0))))


def plot_insertion_loss(
    reference_bands: np.ndarray,
    test_bands: np.ndarray,
    frequencies: np.ndarray,
    *,
    title: str = "Insertion Loss — Suppressed vs Unsuppressed",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = SIZE_TALL,
    level_unit: Optional[str] = None,
    reference_label: str = "Unsuppressed reference",
    test_label: str = "Suppressed test",
    broadband_reduction_dB: Optional[float] = None,
    show_spectra: bool = False,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> Tuple[Figure, Axes]:
    """
    Per-band insertion loss: how much quieter the suppressed shot is.

    This is the number the customer bought. Reduction (reference minus test) is
    plotted per band against a zero line; a band where the test article is
    LOUDER than the reference is drawn in the danger colour AND hatched AND
    labelled, so it can never be mistaken for a reduction.

    Args:
        reference_bands: Per-band levels of the unsuppressed reference (dB).
        test_bands: Per-band levels of the suppressed test article (dB).
        frequencies: Band centre frequencies (Hz).
        broadband_reduction_dB: Overrides the energy-summed broadband figure
            (use this when the pipeline computed it from LAE/LZpeak instead).
        show_spectra: Also draw the two source spectra on a right-hand axis.

    Returns:
        (figure, axes)
    """
    unit = _resolve_level_unit(level_unit)
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        fig = ax.figure  # type: ignore[assignment]

    ref = np.asarray(reference_bands, dtype=float).ravel()
    tst = np.asarray(test_bands, dtype=float).ravel()
    fc = np.asarray(frequencies, dtype=float).ravel()
    n = int(min(ref.size, tst.size, fc.size))
    ref, tst, fc = ref[:n], tst[:n], fc[:n]
    il = ref - tst
    x = np.arange(n)

    gain_mask = il < 0
    colors = np.where(gain_mask, _c('danger'), _series(5)).tolist()
    ax.bar(x[~gain_mask], il[~gain_mask], color=_series(5), width=0.82,
           edgecolor=_c('border'), linewidth=0.4, zorder=3,
           label='Reduction (quieter)')
    if np.any(gain_mask):
        ax.bar(x[gain_mask], il[gain_mask], color=_c('danger'), width=0.82,
               edgecolor=_c('danger_border'), linewidth=0.5, hatch='//',
               zorder=3, label='Increase (LOUDER than reference)')
    del colors

    ax.axhline(0.0, color=_c('text_2'), linewidth=1.0, zorder=4)

    # Broadband figure, energy-summed across the bands unless supplied.
    ref_bb = _energy_sum_dB(ref)
    tst_bb = _energy_sum_dB(tst)
    bb = float(broadband_reduction_dB) if broadband_reduction_dB is not None \
        else (ref_bb - tst_bb)
    if np.isfinite(bb):
        ax.axhline(bb, color=_c('accent'), linewidth=1.0,
                   linestyle=(0, (5, 2)), zorder=5)
        ax.annotate(f'broadband {bb:+.1f} dB', xy=(0.988, bb),
                    xycoords=('axes fraction', 'data'), xytext=(0, 3),
                    textcoords='offset points', ha='right', va='bottom',
                    fontsize=FONT_2XS, family='monospace', color=_c('accent'),
                    zorder=6)

    mean_il = float(np.nanmean(il)) if n else float('nan')
    box_lines = [
        f'Broadband reduction  {bb:+7.1f} dB',
        f'Mean per-band        {mean_il:+7.1f} dB',
        f'{_ellipsize(reference_label, 20):<20s} {ref_bb:7.1f} {unit}',
        f'{_ellipsize(test_label, 20):<20s} {tst_bb:7.1f} {unit}',
    ]
    ax.text(0.985, 0.03, '\n'.join(box_lines), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=FONT_XS, family='monospace',
            color=_c('text'), zorder=7,
            bbox=dict(boxstyle='round,pad=0.45', facecolor=_c('accent_wash'),
                      edgecolor=_c('accent_border'), linewidth=0.7))

    step = max(1, n // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([_fmt_freq(f) for f in fc[::step]], rotation=45, ha='right')
    ax.set_xlim(-0.8, n - 0.2 if n else 1)
    # Keep a clear band below the zero line for the summary box, so it never
    # sits on top of the bars it is describing.
    if n:
        lo = min(0.0, float(np.min(il)))
        hi = max(0.0, float(np.max(il)))
        rng = max(1.0, hi - lo)
        ax.set_ylim(lo - 0.45 * rng, hi + 0.20 * rng)
    ax.set_xlabel('1/3-octave band centre (Hz)')
    ax.set_ylabel('Insertion loss (dB reduction)')
    ax.set_title(title)
    _grid(ax, axis='y')
    _mono_ticks(ax)

    if not show_spectra:
        ax.legend(loc='upper left', fontsize=FONT_SM)
    else:
        ax2 = ax.twinx()
        ax2.set_facecolor('none')
        ax2.plot(x, ref, color=_series(4), linewidth=1.0, marker='o',
                 markersize=2.2, label=reference_label)
        ax2.plot(x, tst, color=_series(5), linewidth=1.0, linestyle=(0, (4, 2)),
                 marker='s', markersize=2.2, label=test_label)
        ax2.set_ylabel(_level_axis_label(unit, 'Band level'),
                       color=_c('text_3'), fontsize=FONT_SM)
        ax2.tick_params(axis='y', colors=_c('border'), labelcolor=_c('text_3'),
                        labelsize=FONT_2XS)
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color(_c('border'))
        lo2 = float(np.nanmin([np.nanmin(ref), np.nanmin(tst)]))
        hi2 = float(np.nanmax([np.nanmax(ref), np.nanmax(tst)]))
        ax2.set_ylim(lo2 - 0.05 * (hi2 - lo2 + 1.0),
                     hi2 + 0.32 * (hi2 - lo2 + 1.0))   # room for the legend
        _mono_ticks(ax2)
        # One combined legend: two overlapping legends fight for the same corner.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=FONT_SM,
                  ncols=2)

    _attach_provenance(fig, provenance, unit)
    _finalize(fig, owns)
    return fig, ax  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
#  NEW: shot overlay - spot the outlier
# ═══════════════════════════════════════════════════════════════════════════

def _normalise_shot_traces(
    shots: Any,
    sample_rate: Optional[float],
    time_s: Optional[np.ndarray],
    labels: Optional[Sequence[str]],
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Accept dict / list-of-arrays / list-of-(t, p) and return (t, p, label)."""
    items: List[Tuple[Any, str]] = []
    if isinstance(shots, Mapping):
        items = [(v, str(k)) for k, v in shots.items()]
    else:
        seq = list(shots)
        for i, entry in enumerate(seq):
            name = labels[i] if labels is not None and i < len(labels) else f'Shot {i + 1}'
            items.append((entry, name))

    traces: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for entry, name in items:
        if isinstance(entry, (tuple, list)) and len(entry) == 2 \
                and np.ndim(entry[0]) == 1 and np.ndim(entry[1]) == 1:
            t = np.asarray(entry[0], dtype=float).ravel()
            p = np.asarray(entry[1], dtype=float).ravel()
        else:
            p = np.asarray(entry, dtype=float).ravel()
            if time_s is not None and len(time_s) >= p.size:
                t = np.asarray(time_s, dtype=float).ravel()[:p.size]
            elif sample_rate:
                t = np.arange(p.size, dtype=float) / float(sample_rate)
            else:
                raise ValueError(
                    "plot_shot_overlay needs sample_rate= or time_s= when the "
                    "shots are plain waveform arrays.")
        n = min(t.size, p.size)
        traces.append((t[:n], p[:n], name))
    return traces


def plot_shot_overlay(
    shots: Any,
    *,
    sample_rate: Optional[float] = None,
    time_s: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
    align: str = 'peak',
    window_ms: Tuple[float, float] = (-2.0, 10.0),
    title: str = "Shot-to-Shot Consistency",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = SIZE_STRIP,
    level_unit: Optional[str] = None,
    outlier_threshold_dB: float = 3.0,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> Tuple[Figure, Axes]:
    """
    Overlay every detected shot on a common time origin.

    A technician reads shot-to-shot consistency from this in one glance: the
    traces should sit on top of one another. Any shot whose peak differs from
    the group median by more than ``outlier_threshold_dB`` is drawn in the
    danger colour, dashed, and named in the legend - never colour alone.

    Args:
        shots: One of
            * ``{label: waveform}`` mapping,
            * list of 1-D waveform arrays (needs ``sample_rate`` or ``time_s``),
            * list of ``(time, waveform)`` pairs.
        align: 'peak' puts each shot's peak |p| at t = 0 (default);
               'start' keeps each trace's own origin; 'none' keeps absolute time.
        window_ms: Display window around the origin, in milliseconds.
        outlier_threshold_dB: Peak deviation from the median that flags a shot.

    Returns:
        (figure, axes)
    """
    unit = _resolve_level_unit(level_unit)
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        fig = ax.figure  # type: ignore[assignment]

    traces = _normalise_shot_traces(shots, sample_rate, time_s, labels)
    if not traces:
        ax.text(0.5, 0.5, 'No shots to overlay', transform=ax.transAxes,
                ha='center', va='center', fontsize=FONT_BASE, color=_c('text_3'))
        _attach_provenance(fig, provenance, unit)
        _finalize(fig, owns)
        return fig, ax  # type: ignore[return-value]

    peaks = np.array([float(np.max(np.abs(p))) if p.size else 0.0
                      for _, p, _ in traces])
    median_peak = float(np.median(peaks[peaks > 0])) if np.any(peaks > 0) else 0.0
    dev_dB = (20.0 * np.log10(np.maximum(peaks, 1e-30) / max(median_peak, 1e-30))
              if median_peak > 0 else np.zeros_like(peaks))

    ax.axhline(0.0, color=_c('border'), linewidth=0.6, zorder=1)
    ax.axvline(0.0, color=_c('border'), linewidth=0.6, zorder=1)

    n_out = 0
    for i, (t, p, name) in enumerate(traces):
        if align == 'peak' and p.size:
            t0 = float(t[int(np.argmax(np.abs(p)))])
        elif align == 'start' and t.size:
            t0 = float(t[0])
        else:
            t0 = 0.0
        t_ms = (t - t0) * 1000.0

        is_outlier = abs(float(dev_dB[i])) > outlier_threshold_dB
        if is_outlier:
            n_out += 1
            ax.plot(t_ms, p, color=_c('danger'), linewidth=1.0,
                    linestyle=(0, (4, 2)), zorder=5,
                    label=f'{name} — OUTLIER {dev_dB[i]:+.1f} dB')
        else:
            ax.plot(t_ms, p, color=_series(i % 4), linewidth=0.7, alpha=0.85,
                    zorder=3,
                    label=name if len(traces) <= 8 else None)

    ax.set_xlim(window_ms)
    ax.set_xlabel('Time relative to peak (ms)' if align == 'peak' else 'Time (ms)')
    ax.set_ylabel(_amplitude_axis_label(unit))
    spread = float(np.max(dev_dB) - np.min(dev_dB)) if len(traces) > 1 else 0.0
    ax.set_title(f'{title} — {len(traces)} shots, peak spread {spread:.1f} dB')
    _grid(ax, axis='y')
    _mono_ticks(ax)

    verdict_key = 'danger' if n_out else 'ok'
    verdict = (f'{n_out} outlier shot(s) > {outlier_threshold_dB:.0f} dB from median'
               if n_out else
               f'all shots within {outlier_threshold_dB:.0f} dB of the median peak')
    ax.text(0.985, 0.03, f' {"!" if n_out else "OK"}  {verdict} ',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FONT_2XS, family='monospace', color=_c(verdict_key),
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor=_c(f'{verdict_key}_wash'),
                      edgecolor=_c(f'{verdict_key}_border'), linewidth=0.6),
            zorder=7)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.margins(y=0.30)   # headroom so the legend never sits on a trace
        ax.legend(loc='upper right', fontsize=FONT_SM, ncols=2)

    _attach_provenance(fig, provenance, unit)
    _finalize(fig, owns)
    return fig, ax  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
#  NEW: measurement quality panel
# ═══════════════════════════════════════════════════════════════════════════

_STATUS_ORDER = {'ok': 0, 'warn': 1, 'danger': 2, 'info': -1}
_STATUS_WORD = {'ok': 'PASS', 'warn': 'CAUTION', 'danger': 'FAIL',
                'info': 'NOT MEASURED'}
_STATUS_GLYPH = {'ok': '✓', 'warn': '!', 'danger': '✕', 'info': '–'}


def _grade(value: Optional[float], pass_at: float, caution_at: float,
           *, higher_is_better: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return 'info'
    if higher_is_better:
        if value >= pass_at:
            return 'ok'
        return 'warn' if value >= caution_at else 'danger'
    if value <= pass_at:
        return 'ok'
    return 'warn' if value <= caution_at else 'danger'


def _q(quality: Any, key: str, default: Any = None) -> Any:
    if isinstance(quality, Mapping):
        return quality.get(key, default)
    return getattr(quality, key, default)


def plot_measurement_quality(
    quality_dict: Any,
    *,
    title: str = "Measurement Quality",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (FIGURE_WIDTH_IN, 3.4),
    level_unit: Optional[str] = None,
    thresholds: Optional[Mapping[str, Tuple[float, float]]] = None,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> Tuple[Figure, Axes]:
    """
    Compact admissibility panel: is this recording fit to measure from?

    Accepts a ``SignalQuality`` object, its ``.to_dict()``, or any mapping with
    the same keys. Each row is marked PASS / CAUTION / FAIL with a glyph and a
    word as well as a colour.

    Rows: headroom, clipping, SNR, DC offset, sample-rate adequacy.

    Default thresholds (override with ``thresholds={'snr_dB': (pass, caution)}``):
        headroom_dB   pass >= 6,    caution >= 3      (higher is better)
        snr_dB        pass >= 30,   caution >= 15     (higher is better)
        dc_offset_dB  pass <= -40,  caution <= -30    (lower is better)
        clipping      pass = no clipped samples at all
        sample rate   pass = adequate flag set and >= 96 kHz for blast rise time
    """
    unit = _resolve_level_unit(level_unit)
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        fig = ax.figure  # type: ignore[assignment]
    ax.axis('off')

    th = dict(headroom_dB=(6.0, 3.0), snr_dB=(30.0, 15.0),
              dc_offset_dB=(-40.0, -30.0))
    if thresholds:
        th.update({k: (float(v[0]), float(v[1])) for k, v in thresholds.items()})

    headroom = _q(quality_dict, 'headroom_dB')
    snr = _q(quality_dict, 'snr_dB')
    dc_dB = _q(quality_dict, 'dc_offset_dB')
    dc_fs = _q(quality_dict, 'dc_offset_FS')
    clipped_n = _q(quality_dict, 'clipped_samples')
    clipped_runs = _q(quality_dict, 'clipped_runs', 0)
    clip_ratio = _q(quality_dict, 'clipping_ratio')
    sr = _q(quality_dict, 'sample_rate')
    sr_ok = _q(quality_dict, 'sample_rate_adequate')
    nyq = _q(quality_dict, 'nyquist_Hz')
    peak_dB = _q(quality_dict, 'peak_level_dB')

    rows: List[Tuple[str, str, str, str]] = []  # (metric, value, status, note)

    rows.append((
        'Headroom',
        f'{headroom:+.1f} dB' if headroom is not None else '--',
        _grade(headroom, *th['headroom_dB'], higher_is_better=True),
        f'peak {peak_dB:.1f} {unit}' if peak_dB is not None else 'below full scale',
    ))

    if clipped_n is None:
        clip_status, clip_val, clip_note = 'info', '--', 'not assessed'
    elif int(clipped_n) == 0:
        clip_status, clip_val, clip_note = 'ok', '0 samples', 'no samples at full scale'
    else:
        ratio = float(clip_ratio) if clip_ratio is not None else float('nan')
        clip_status = 'danger'
        clip_val = f'{int(clipped_n)} samples'
        clip_note = (f'{int(clipped_runs or 0)} run(s), {ratio * 100:.4f}% of record'
                     if np.isfinite(ratio) else f'{int(clipped_runs or 0)} run(s)')
    rows.append(('Clipping', clip_val, clip_status, clip_note))

    rows.append((
        'Signal-to-noise',
        f'{snr:.1f} dB' if snr is not None else '--',
        _grade(snr, *th['snr_dB'], higher_is_better=True),
        'peak above estimated noise floor',
    ))

    rows.append((
        'DC offset',
        f'{dc_dB:.1f} dB' if dc_dB is not None else '--',
        _grade(dc_dB, *th['dc_offset_dB'], higher_is_better=False),
        (f'{dc_fs:+.2e} FS, re signal RMS' if dc_fs is not None else 're signal RMS'),
    ))

    if sr is None:
        sr_status, sr_val, sr_note = 'info', '--', 'not reported'
    else:
        sr_f = float(sr)
        if sr_ok is False:
            sr_status = 'danger'
        elif sr_f < 96000.0:
            sr_status = 'warn'
        else:
            sr_status = 'ok'
        sr_val = f'{sr_f:,.0f} Hz'.replace(',', ' ')
        sr_note = (f'Nyquist {float(nyq):,.0f} Hz'.replace(',', ' ')
                   if nyq else 'blast rise time needs >= 96 kHz')
    rows.append(('Sample rate', sr_val, sr_status, sr_note))

    worst = max(rows, key=lambda r: _STATUS_ORDER.get(r[2], 0))[2]
    worst = worst if worst in _STATUS_ORDER else 'info'
    verdict_word = {'ok': 'ADMISSIBLE',
                    'warn': 'ADMISSIBLE WITH CAUTION',
                    'danger': 'NOT ADMISSIBLE',
                    'info': 'INCOMPLETE'}[worst]

    # ── Verdict banner ──
    ax.text(0.0, 0.985, title, transform=ax.transAxes, ha='left', va='top',
            fontsize=FONT_MD, fontweight='bold', color=_c('text'))
    ax.text(1.0, 0.985,
            f'  {_STATUS_GLYPH[worst]}  {verdict_word}  ',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT_SM, family='monospace', fontweight='bold',
            color=_c(worst),
            bbox=dict(boxstyle='round,pad=0.45', facecolor=_c(f'{worst}_wash'),
                      edgecolor=_c(f'{worst}_border'), linewidth=0.8))

    ax.plot([0.0, 1.0], [0.86, 0.86], transform=ax.transAxes,
            color=_c('border'), linewidth=0.7)

    y = 0.75
    dy = 0.155
    for metric, value, status, note in rows:
        ax.text(0.0, y, metric, transform=ax.transAxes, ha='left', va='center',
                fontsize=FONT_BASE, color=_c('text_2'))
        ax.text(0.42, y, value, transform=ax.transAxes, ha='right', va='center',
                fontsize=FONT_BASE, family='monospace', fontweight='bold',
                color=_c('text'))
        ax.text(0.47, y, f' {_STATUS_GLYPH[status]} {_STATUS_WORD[status]} ',
                transform=ax.transAxes, ha='left', va='center',
                fontsize=FONT_XS, family='monospace', color=_c(status),
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=_c(f'{status}_wash'),
                          edgecolor=_c(f'{status}_border'), linewidth=0.6))
        ax.text(1.0, y, note, transform=ax.transAxes, ha='right', va='center',
                fontsize=FONT_2XS, family='monospace', color=_c('text_3'))
        ax.plot([0.0, 1.0], [y - dy / 2, y - dy / 2], transform=ax.transAxes,
                color=_c('border_subtle'), linewidth=0.5)
        y -= dy

    errors = _q(quality_dict, 'errors', []) or []
    warns = _q(quality_dict, 'warnings', []) or []
    if errors or warns:
        msg = '; '.join(list(errors)[:2] + list(warns)[:2])
        budget = max(20, int(0.976 * 72.0 * float(fig.get_figwidth())
                             / (0.60 * FONT_2XS)))
        ax.text(0.0, max(0.0, y), _ellipsize(msg, budget), transform=ax.transAxes,
                ha='left', va='center', fontsize=FONT_2XS,
                color=_c('danger' if errors else 'warn'))

    _attach_provenance(fig, provenance, unit)
    _finalize(fig, owns, top=1.0)
    return fig, ax  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-panel shot summary
# ═══════════════════════════════════════════════════════════════════════════

def create_shot_summary_figure(
    time_s: np.ndarray,
    pressure_Pa: np.ndarray,
    stft_z: STFTResult,
    stft_c: STFTResult,
    metrics: ShotMetrics,
    shot: Optional[ShotEvent] = None,
    *,
    title: str = "Shot Analysis Summary",
    figsize: Tuple[float, float] = SIZE_PAGE,
    level_unit: Optional[str] = None,
    db_range: Optional[Tuple[float, float]] = None,
    dynamic_range_dB: float = DEFAULT_DYNAMIC_RANGE_DB,
    freq_range: Tuple[float, float] = (0.0, 20000.0),
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> Figure:
    """
    Full-page, six-panel summary of one shot.

    Panels: waveform, time-weighted levels, Z-weighted spectrogram,
    C-weighted spectrogram, 1/3-octave exposure, metrics table.

    The two spectrograms SHARE one colour scale (computed across both unless
    ``db_range`` is given), so the weighting difference is readable as a
    difference rather than being normalised away by per-panel auto-scaling.
    """
    unit = _resolve_level_unit(level_unit, _unit_from_calibrated_flag(stft_z))

    fig = plt.figure(figsize=figsize, layout='constrained')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.35, 1.15])

    # ── 1. Waveform ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axhline(0.0, color=_c('border'), linewidth=0.6, zorder=1)
    ax1.plot(time_s, pressure_Pa, color=_series(0), linewidth=0.5, zorder=3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(_amplitude_axis_label(unit))
    ax1.set_title('Waveform')
    if len(time_s):
        ax1.set_xlim(float(time_s[0]), float(time_s[-1]))
    if shot is not None:
        ax1.axvline(float(shot.time_s) - float(time_s[0] if len(time_s) else 0.0),
                    color=_c('shot_marker'), linestyle=(0, (4, 3)),
                    linewidth=0.9, zorder=4)
    _grid(ax1, axis='y')

    # ── 2. Time-weighted levels ──
    ax2 = fig.add_subplot(gs[0, 1])
    if len(metrics.time_s) > 0:
        ax2.plot(metrics.time_s, metrics.LAF, color=_series(1), linewidth=1.1,
                 label='LAF')
        ax2.plot(metrics.time_s, metrics.LAS, color=_series(2), linewidth=1.1,
                 linestyle=(0, (5, 2)), label='LAS')
        ax2.legend(loc='upper right', fontsize=FONT_SM)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(_level_axis_label(unit))
    ax2.set_title(f'Time-weighted levels  (LAFmax {metrics.LAFmax:.1f} dB)')
    _grid(ax2)

    # ── 3 & 4. Spectrograms on ONE shared colour scale ──
    vmin, vmax = (resolve_db_range(np.array([]), db_range=db_range)
                  if db_range is not None
                  else shared_db_range(stft_z, stft_c,
                                       dynamic_range_dB=dynamic_range_dB))

    ax3 = fig.add_subplot(gs[1, 0])
    pcm3 = ax3.pcolormesh(_centers_to_edges(stft_z.time_s),
                          _centers_to_edges(stft_z.frequencies_Hz),
                          stft_z.magnitude_dB, shading='flat',
                          cmap=CMAP_SPECTROGRAM, vmin=vmin, vmax=vmax,
                          rasterized=True)
    ax3.set_ylim(freq_range)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('Z-weighted spectrogram')
    _style_colorbar(fig.colorbar(pcm3, ax=ax3, pad=0.015, fraction=0.045),
                    _level_axis_label(unit))

    ax4 = fig.add_subplot(gs[1, 1])
    pcm4 = ax4.pcolormesh(_centers_to_edges(stft_c.time_s),
                          _centers_to_edges(stft_c.frequencies_Hz),
                          stft_c.magnitude_dB, shading='flat',
                          cmap=CMAP_SPECTROGRAM, vmin=vmin, vmax=vmax,
                          rasterized=True)
    ax4.set_ylim(freq_range)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('C-weighted spectrogram')
    _style_colorbar(fig.colorbar(pcm4, ax=ax4, pad=0.015, fraction=0.045),
                    _level_axis_label(unit))

    # Both panels state the scale they share, so neither can be read alone and
    # mistaken for an independently auto-scaled image.
    for a in (ax3, ax4):
        _annotate_scale(a, vmin, vmax, unit, prefix='shared scale', compact=True)
        _thin_time_axis(a, nbins=5)

    # ── 5. Band exposure, baselined ──
    ax5 = fig.add_subplot(gs[2, 0])
    if len(metrics.band_frequencies) > 0:
        vals = np.asarray(metrics.band_exposure_dB, dtype=float).ravel()
        fc = np.asarray(metrics.band_frequencies, dtype=float).ravel()
        n = min(vals.size, fc.size)
        vals, fc = vals[:n], fc[:n]
        base = _exposure_baseline(vals, None, EXPOSURE_DYNAMIC_RANGE_DB)
        x = np.arange(n)
        colors = [_series(3)] * n
        pk = int(np.argmax(vals))
        colors[pk] = _c('accent')
        ax5.bar(x, np.clip(vals - base, 0.0, None), bottom=base, color=colors,
                width=0.82, edgecolor=_c('border'), linewidth=0.35, zorder=3)
        ax5.axhline(base, color=_c('border'), linewidth=0.8, zorder=4)
        step = max(1, n // 10)
        ax5.set_xticks(x[::step])
        ax5.set_xticklabels([_fmt_freq(f) for f in fc[::step]],
                            rotation=45, ha='right')
        ax5.set_ylim(base, float(np.max(vals)) + 3.0)
        ax5.text(0.98, 0.95, f'from {base:.0f} {unit}', transform=ax5.transAxes,
                 ha='right', va='top', fontsize=FONT_2XS, family='monospace',
                 color=_c('text_3'))
    ax5.set_xlabel('1/3-octave band centre (Hz)')
    ax5.set_ylabel(_level_axis_label(unit, 'SEL'))
    ax5.set_title('1/3-octave band exposure')
    _grid(ax5, axis='y')

    # ── 6. Metrics table ──
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    ax6.set_title('Metrics summary')

    lines: List[Tuple[str, Optional[str]]] = [
        ('PEAK LEVELS', None),
        ('Lpeak(Z)', f'{metrics.Lpeak_Z:8.1f} {unit}'),
        ('Lpeak(A)', f'{metrics.Lpeak_A:8.1f} {unit}'),
        ('Lpeak(C)', f'{metrics.Lpeak_C:8.1f} {unit}'),
        ('EXPOSURE (SEL)', None),
        ('LAE', f'{metrics.LAE:8.1f} dB'),
        ('LZE', f'{metrics.LZE:8.1f} dB'),
        ('MAX TIME-WEIGHTED', None),
        ('LAFmax', f'{metrics.LAFmax:8.1f} {unit}'),
        ('LASmax', f'{metrics.LASmax:8.1f} {unit}'),
        ('WAVEFORM', None),
        ('Rise time', f'{metrics.rise_time_us:8.1f} us'
                      + ('' if metrics.rise_time_resolved else '  (unresolved)')),
        ('A-duration', f'{metrics.a_duration_ms:8.2f} ms'),
        ('Window', f'{metrics.duration_s * 1000:8.1f} ms'),
    ]

    y_pos = 0.96
    for label, value in lines:
        if value is None:
            y_pos -= 0.02
            ax6.text(0.02, y_pos, label, transform=ax6.transAxes,
                     fontsize=FONT_2XS, fontweight='bold', color=_c('text_3'))
        else:
            ax6.text(0.02, y_pos, label, transform=ax6.transAxes,
                     fontsize=FONT_XS, family='monospace', color=_c('text_2'))
            ax6.text(0.98, y_pos, value, transform=ax6.transAxes, ha='right',
                     fontsize=FONT_XS, family='monospace', fontweight='bold',
                     color=_c('text'))
        y_pos -= 0.068

    if metrics.clipped or not metrics.valid:
        ax6.text(0.02, max(0.0, y_pos), ' !  CLIPPED - level is a lower bound ',
                 transform=ax6.transAxes, fontsize=FONT_2XS, family='monospace',
                 color=_c('danger'),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=_c('danger_wash'),
                           edgecolor=_c('danger_border'), linewidth=0.6))

    for a in (ax1, ax2, ax3, ax4, ax5):
        _mono_ticks(a)
    _thin_time_axis(ax1, nbins=5)
    _thin_time_axis(ax2, nbins=5)

    fig.suptitle(title, fontsize=FONT_LG, fontweight='bold', color=_c('text'))

    _attach_provenance(fig, provenance, unit)
    _finalize(fig, True, top=0.985)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Interactive HTML charts (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

def _plotly_layout() -> Dict[str, Any]:
    """Plotly layout built from the active theme (tokens.css palette)."""
    return dict(
        paper_bgcolor=_c('bg_surface'),
        plot_bgcolor=_c('plot_bg'),
        font=dict(color=_c('text'),
                  family='-apple-system, BlinkMacSystemFont, Inter, Segoe UI, system-ui, sans-serif',
                  size=12),
        title_font=dict(size=15, color=_c('text')),
        xaxis=dict(gridcolor=_c('border_subtle'), zerolinecolor=_c('border'),
                   linecolor=_c('border'),
                   tickfont=dict(color=_c('text_3'), size=11),
                   title_font=dict(color=_c('text_2'), size=12)),
        yaxis=dict(gridcolor=_c('border_subtle'), zerolinecolor=_c('border'),
                   linecolor=_c('border'),
                   tickfont=dict(color=_c('text_3'), size=11),
                   title_font=dict(color=_c('text_2'), size=12)),
        margin=dict(l=64, r=20, t=54, b=76),
        hoverlabel=dict(bgcolor=_c('bg_surface'), bordercolor=_c('border'),
                        font_color=_c('text'), font_size=11),
    )


def _plotly_footer(prov: Optional[FigureProvenance]) -> List[Dict[str, Any]]:
    """The same provenance block, as Plotly annotations."""
    prov = prov or _DEFAULT_PROVENANCE or FigureProvenance(
        level_unit=_DEFAULT_LEVEL_UNIT or "")
    line1, line2 = prov.lines()
    unit_text, unit_key = prov.unit_statement()
    common = dict(xref='paper', yref='paper', showarrow=False, xanchor='left',
                  font=dict(size=9, color=_c('text_3'), family='monospace'))
    return [
        dict(x=0, y=-0.16, text=line1, **common),
        dict(x=0, y=-0.215, text=line2, **common),
        dict(x=1, y=-0.16, text=unit_text, xref='paper', yref='paper',
             showarrow=False, xanchor='right',
             font=dict(size=9, color=_c(unit_key), family='monospace')),
    ]


def save_interactive_waveform_html(
    output_path: Path,
    time_s: np.ndarray,
    pressure_Pa: np.ndarray,
    shots: Optional[List[ShotEvent]] = None,
    title: str = "Pressure Waveform",
    *,
    level_unit: Optional[str] = None,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> bool:
    """
    Save an interactive zoomable waveform as HTML (Plotly).
    Returns True if saved, False if Plotly is missing or an error occurs.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    try:
        unit = _resolve_level_unit(level_unit)
        output_path = Path(output_path)
        if output_path.suffix.lower() != '.html':
            output_path = output_path.with_suffix('.html')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        x = np.asarray(time_s, dtype=np.float64).ravel()
        y = np.asarray(pressure_Pa, dtype=np.float64).ravel()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                 line=dict(color=_series(0), width=1),
                                 name=_amplitude_axis_label(unit)))
        if shots:
            for s in shots:
                fig.add_vline(x=float(s.time_s), line_dash="dash",
                              line_color=_c('shot_marker'), opacity=0.7)

        fig.update_layout({**_plotly_layout(),
                           'title': str(title),
                           'xaxis_title': "Time (s)",
                           'yaxis_title': _amplitude_axis_label(unit),
                           'hovermode': "x unified",
                           'height': 480,
                           'annotations': _plotly_footer(
                               _coerce_provenance(provenance))})
        fig.write_html(str(output_path), config={"scrollZoom": True})
        print(f"  -> {output_path.resolve()}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [Plotly waveform error] {e}")
        return False


def save_interactive_spectrogram_html(
    output_path: Path,
    result: STFTResult,
    shots: Optional[List[ShotEvent]] = None,
    title: Optional[str] = None,
    *,
    db_range: Optional[Tuple[float, float]] = None,
    dynamic_range_dB: float = DEFAULT_DYNAMIC_RANGE_DB,
    level_unit: Optional[str] = None,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
) -> bool:
    """
    Save an interactive zoomable spectrogram as HTML (Plotly).
    Uses the same explicit dB range and CVD-safe colormap as the static figure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    try:
        unit = _resolve_level_unit(level_unit, _unit_from_calibrated_flag(result))
        output_path = Path(output_path)
        if output_path.suffix.lower() != '.html':
            output_path = output_path.with_suffix('.html')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if title is None:
            title = f"Spectrogram — {result.weighting}-weighted"

        vmin, vmax = resolve_db_range(result.magnitude_dB, db_range=db_range,
                                      dynamic_range_dB=dynamic_range_dB)

        fig = go.Figure(go.Heatmap(
            x=np.asarray(result.time_s, dtype=np.float64),
            y=np.asarray(result.frequencies_Hz, dtype=np.float64),
            z=np.asarray(result.magnitude_dB, dtype=np.float64),
            colorscale='Cividis',   # matches CMAP_SPECTROGRAM
            zmin=vmin, zmax=vmax,
            colorbar=dict(title=_level_axis_label(unit),
                          title_font=dict(color=_c('text_2'), size=12),
                          tickfont=dict(color=_c('text_3'), size=10),
                          outlinecolor=_c('border'), outlinewidth=0.5),
        ))
        if shots:
            for s in shots:
                fig.add_vline(x=float(s.time_s), line_dash="dash",
                              line_color=_c('shot_marker'), opacity=0.7)

        y_max = float(min(20000.0, float(result.frequencies_Hz[-1])
                          if len(result.frequencies_Hz) else 20000.0))
        layout = _plotly_layout()
        layout['yaxis'] = {**layout['yaxis'], 'range': [0, y_max]}
        fig.update_layout({**layout,
                           'title': f'{title}   ({vmin:.0f} to {vmax:.0f} {unit})',
                           'xaxis_title': "Time (s)",
                           'yaxis_title': "Frequency (Hz)",
                           'height': 540,
                           'annotations': _plotly_footer(
                               _coerce_provenance(provenance))})
        fig.write_html(str(output_path), config={"scrollZoom": True})
        print(f"  -> {output_path.resolve()}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [Plotly spectrogram error] {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  File I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_figure(
    fig: Figure,
    output_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    formats: Optional[List[str]] = None,
    provenance: Union[None, FigureProvenance, Mapping[str, Any], str] = None,
    bbox_inches: Optional[str] = None,
) -> List[Path]:
    """
    Save a figure, stamping the provenance footer first.

    ``bbox_inches`` defaults to None (NOT 'tight'): the figure was authored at
    its final printed size, and cropping to the ink would make every figure in
    the report a slightly different width. Pass 'tight' deliberately if needed.

    Returns the list of written paths.
    """
    if formats is None:
        formats = ['png']

    annotate_provenance(fig, provenance)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for fmt in formats:
        path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches,
                    facecolor=fig.get_facecolor(), edgecolor='none')
        saved_paths.append(path)

    return saved_paths


# ═══════════════════════════════════════════════════════════════════════════
#  Smoke test CLI
# ═══════════════════════════════════════════════════════════════════════════

def _synthetic_shot(sr: int = 96000, duration: float = 0.05,
                    peak_Pa: float = 200.0, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration)) / sr
    env = np.exp(-t / 0.004)
    sig = env * np.sin(2 * np.pi * 700 * t) + 0.35 * env * np.sin(2 * np.pi * 2300 * t)
    sig[:8] += np.linspace(1.0, 0.2, 8)
    sig += 0.002 * rng.standard_normal(t.size)
    sig = sig / np.max(np.abs(sig)) * peak_Pa
    return t, sig


def main() -> int:
    """Generate every figure with synthetic data (smoke test)."""
    import argparse

    parser = argparse.ArgumentParser(description="SASA figure smoke test")
    parser.add_argument("--test", action="store_true", help="Run the smoke test")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: a temp dir)")
    parser.add_argument("--theme", default="light", choices=["light", "dark"])
    args = parser.parse_args()

    if not args.test:
        parser.print_help()
        return 0

    import tempfile
    outdir = Path(args.outdir) if args.outdir else Path(tempfile.mkdtemp(prefix="sasa_plots_"))
    outdir.mkdir(parents=True, exist_ok=True)

    setup_plot_style(theme=args.theme)
    set_default_provenance(FigureProvenance(
        source_file="synthetic_smoke_test.wav",
        calibration="Synthetic - 1 kHz 94 dB pistonphone, 0.8912 Pa/FS",
        sample_rate_Hz=96000,
        level_unit=UNIT_SPL,
        operator="smoke test",
    ))

    from STFT import analyze_stft
    from metrics import compute_shot_metrics
    from calibration import Calibration, assess_signal_quality

    sr = 96000
    t, p = _synthetic_shot(sr)
    written: List[Path] = []

    def record(fig: Figure, name: str) -> None:
        paths = save_figure(fig, outdir / name)
        written.extend(paths)
        plt.close(fig)

    # 1. waveform (with the real secondary level axis exercised)
    fig, _ = plot_waveform_pa(t, p, title="Synthetic impulse", level_unit=UNIT_SPL,
                              show_dB_secondary=True)
    record(fig, "01_waveform")

    # 2. spectrogram
    stft_z = analyze_stft(p, sr, nperseg=1024, noverlap=768, weighting='Z')
    stft_c = analyze_stft(p, sr, nperseg=1024, noverlap=768, weighting='C')
    fig, _ = plot_spectrogram_dB(stft_z)
    record(fig, "02_spectrogram_z")
    rng = shared_db_range(stft_z, stft_c)
    fig, _ = plot_spectrogram_dB(stft_c, db_range=rng)
    record(fig, "03_spectrogram_c_shared_scale")

    # 3. 1/3-octave heatmap + band alignment verification
    from bands import ThirdOctaveAnalyzer
    analyzer = ThirdOctaveAnalyzer(sample_rate=sr)
    fc = analyzer.center_frequencies
    n_bands = len(fc)
    n_frames = 60
    target = n_bands // 2
    synth = np.full((n_bands, n_frames), 40.0)
    synth[target, :] = 120.0                       # exactly one band is hot
    tb = np.arange(n_frames) * 0.01
    fig, ax = plot_third_octave_heatmap(tb, fc, synth,
                                        title="Band alignment check")
    fig.canvas.draw()
    mesh = list(ax.collections)[0]
    coords = mesh._coordinates                      # (n_bands+1, n_frames+1, 2)
    lo = float(coords[target, 0, 1])
    hi = float(coords[target + 1, 0, 1])
    centre = 0.5 * (lo + hi)
    ok_align = abs(centre - target) < 1e-9
    print(f"  band-alignment: only band index {target} "
          f"(fc = {fc[target]:.1f} Hz) is hot; its cell spans "
          f"y = [{lo:g}, {hi:g}], centre {centre:g} -> "
          f"{'OK' if ok_align else 'FAIL'}")
    ticks = [int(round(v)) for v in ax.get_yticks()]
    labels = [lbl.get_text() for lbl in ax.get_yticklabels()]
    bad = [(i, lab) for i, lab in zip(ticks, labels)
           if 0 <= i < n_bands and lab != _fmt_freq(fc[i])]
    print(f"  band-alignment: {len(ticks)} y ticks checked against "
          f"fc[index]; mismatches = {len(bad)} -> {'OK' if not bad else bad}")
    # A tick either side of the hot row must bracket it, never sit on it by
    # half a band (the old bug put the label on the cell boundary).
    below = max([i for i in ticks if i < target], default=None)
    above = min([i for i in ticks if i > target], default=None)
    print(f"  band-alignment: neighbouring ticks {below} "
          f"({_fmt_freq(fc[below]) if below is not None else '-'} Hz) and "
          f"{above} ({_fmt_freq(fc[above]) if above is not None else '-'} Hz) "
          f"bracket the hot row -> "
          f"{'OK' if (below is None or below < target) and (above is None or above > target) else 'FAIL'}")
    record(fig, "04_third_octave_heatmap")

    # 4. level curves
    metrics = compute_shot_metrics(p, sr, compute_bands=True,
                                   compute_time_series=True, shot_number=1)
    fig, _ = plot_level_curves(metrics.time_s, metrics.LAF, metrics.LAS,
                               LZF=metrics.LZF, LZS=metrics.LZS,
                               level_unit=UNIT_SPL)
    record(fig, "05_level_curves")

    # 5. band exposure
    fig, _ = plot_band_exposure(metrics.band_frequencies, metrics.band_exposure_dB,
                                level_unit=UNIT_SPL)
    record(fig, "06_band_exposure")

    # 6. shot summary page
    fig = create_shot_summary_figure(t, p, stft_z, stft_c, metrics,
                                     title="Shot 1 Analysis", level_unit=UNIT_SPL)
    record(fig, "07_shot_summary")

    # 7. insertion loss
    ref_bands = metrics.band_exposure_dB
    loss = 18.0 + 14.0 * np.exp(-((np.arange(len(ref_bands)) - len(ref_bands) * 0.7) ** 2)
                                / (2 * 6.0 ** 2))
    loss[:3] -= 20.0                                 # a low band that gets louder
    fig, _ = plot_insertion_loss(ref_bands, ref_bands - loss,
                                 metrics.band_frequencies, level_unit=UNIT_SPL,
                                 show_spectra=True)
    record(fig, "08_insertion_loss")

    # 8. shot overlay, with a deliberate outlier
    shot_set = {}
    for i in range(5):
        pk = 200.0 if i != 3 else 380.0
        _, pi = _synthetic_shot(sr, duration=0.03, peak_Pa=pk, seed=i)
        shot_set[f'Shot {i + 1}'] = pi
    fig, _ = plot_shot_overlay(shot_set, sample_rate=sr, level_unit=UNIT_SPL)
    record(fig, "09_shot_overlay")

    # 9. measurement quality
    cal = Calibration.preset(200.0, "smoke-test", "synthetic 1 kHz reference")
    quality = assess_signal_quality(p / (np.max(np.abs(p)) * 1.2), sr, cal)
    fig, _ = plot_measurement_quality(quality.to_dict(), level_unit=UNIT_SPL)
    record(fig, "10_measurement_quality")

    # 10. uncalibrated labelling must NOT say SPL
    set_default_provenance(FigureProvenance(
        source_file="synthetic_uncalibrated.wav",
        calibration="UNCALIBRATED - relative units (dB re FS)",
        sample_rate_Hz=sr, level_unit=UNIT_FS))
    fig, _ = plot_waveform_pa(t, p / np.max(np.abs(p)),
                              title="Uncalibrated waveform", level_unit=UNIT_FS)
    record(fig, "11_uncalibrated_waveform")

    # 11. dark theme still works (back on the calibrated provenance)
    set_default_provenance(FigureProvenance(
        source_file="synthetic_smoke_test.wav",
        calibration="Synthetic - 1 kHz 94 dB pistonphone, 0.8912 Pa/FS",
        sample_rate_Hz=sr, level_unit=UNIT_SPL))
    setup_plot_style(theme='dark')
    fig, _ = plot_spectrogram_dB(stft_z, title="Dark theme spectrogram")
    record(fig, "12_dark_spectrogram")
    setup_plot_style(theme='light')

    print(f"\nWrote {len(written)} files to {outdir}")
    for path in written:
        print(f"  {path.stat().st_size:>9,d} B  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
