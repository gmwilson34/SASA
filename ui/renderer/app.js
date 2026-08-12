/* ============================================================================
   SASA — Shot Acoustic Spectral Analysis
   Renderer application logic.

   Ridgeback Defense.

   ORGANISATION
     0.  Constants and the metric registry
     1.  Small DOM / number / string utilities
     2.  Persistence
     3.  State  (one object; render() derives the DOM from it)
     4.  Theme
     5.  Announcements, toasts, dialogs
     6.  Navigation, tabs, keyboard
     7.  Analyze — recording, calibration, metadata, settings
     8.  Validation and the run configuration
     9.  Transport — WebSocket with capped backoff, run lifecycle
    10.  Results — validity, metrics, per-shot review, aggregate recompute
    11.  Charts (canvas, theme-aware, devicePixelRatio-correct)
    12.  Compare — insertion loss
    13.  History
    14.  Settings and calibration profiles
    15.  Boot

   SECURITY NOTE
     This file builds NO HTML strings. There is not one innerHTML, outerHTML,
     insertAdjacentHTML, document.write or Function/eval call in it: every node
     comes from a <template> clone or document.createElement, and every piece of
     text is assigned with textContent. That removes the injection surface the
     previous esc() helper was trying (and failing) to cover, so no escaper is
     needed. If you ever add one, escape & < > " ' and use it for attributes too.

   RENDER MODEL
     setState()/commit() marks the app dirty and schedules render() on the next
     animation frame. render() writes DERIVED regions only. It never writes back
     into a control the operator types into — those controls are the source of
     truth for their slice of state and are hydrated only on load and on an
     explicit reset. That is what keeps a re-render from eating a keystroke.
   ============================================================================ */

'use strict';

(() => {

/* ===========================================================================
   0. CONSTANTS AND THE METRIC REGISTRY
   =========================================================================== */

const STORAGE = {
  theme:    'sasa.theme',          // shared with the FOUC guard in index.html
  settings: 'sasa.settings',
  metadata: 'sasa.metadata',
  profiles: 'sasa.profiles',
  recent:   'sasa.recentFiles',
  prefs:    'sasa.prefs',
  rejects:  'sasa.rejects',
  sidebar:  'sasa.sidebar',
};

const MAX_LOG_LINES  = 2000;
const MAX_RECENT     = 12;
const PA_REF         = 20e-6;      // 20 uPa
const EIGHT_HOURS_S  = 28800;

/* Reconnection: capped exponential backoff with jitter. The old code retried
   every 2 s for ever, which hammered a server that was not coming back. */
const BACKOFF = { base: 400, factor: 1.8, max: 20000, jitter: 0.25, maxAttempts: 12 };

/* Level metrics are energy-averaged; everything else is arithmetic-averaged.
   `stat` is the key used in aggregate.statistics, `key` the key used in
   per_shot_metrics — the backend does not spell them the same way. */
const METRICS = [
  { key: 'Lpeak_Z_dB', stat: 'Lpeak_Z', label: ['L', 'peak', ' Z'],  plain: 'Peak level, Z-weighted',        unit: 'level', digits: 1, level: true,  weight: null, headline: 1, compare: true },
  { key: 'Lpeak_A_dB', stat: 'Lpeak_A', label: ['L', 'peak', ' A'],  plain: 'Peak level, A-weighted',        unit: 'level', digits: 1, level: true,  weight: 'A',  headline: 2, compare: true },
  { key: 'Lpeak_C_dB', stat: 'Lpeak_C', label: ['L', 'peak', ' C'],  plain: 'Peak level, C-weighted',        unit: 'level', digits: 1, level: true,  weight: 'C',  headline: 0, compare: true },
  { key: 'LAE_dB',     stat: 'LAE',     label: ['L', 'AE'],          plain: 'Sound exposure, A-weighted',    unit: 'level', digits: 1, level: true,  weight: 'A',  headline: 3, compare: true },
  { key: 'LZE_dB',     stat: 'LZE',     label: ['L', 'ZE'],          plain: 'Sound exposure, unweighted',    unit: 'level', digits: 1, level: true,  weight: null, headline: 0, compare: true },
  { key: 'LCE_dB',     stat: 'LCE',     label: ['L', 'CE'],          plain: 'Sound exposure, C-weighted',    unit: 'level', digits: 1, level: true,  weight: 'C',  headline: 0, compare: true },
  { key: 'LAFmax_dB',  stat: 'LAFmax',  label: ['L', 'AFmax'],       plain: 'Max level, A fast',             unit: 'level', digits: 1, level: true,  weight: 'A',  headline: 0, compare: false },
  { key: 'LASmax_dB',  stat: 'LASmax',  label: ['L', 'ASmax'],       plain: 'Max level, A slow',             unit: 'level', digits: 1, level: true,  weight: 'A',  headline: 0, compare: false },
  { key: 'LAImax_dB',  stat: 'LAImax',  label: ['L', 'AImax'],       plain: 'Max level, A impulse',          unit: 'level', digits: 1, level: true,  weight: 'A',  headline: 4, compare: true },
  { key: 'LZImax_dB',  stat: 'LZImax',  label: ['L', 'ZImax'],       plain: 'Max level, Z impulse',          unit: 'level', digits: 1, level: true,  weight: null, headline: 0, compare: false },
  { key: 'a_duration_ms',        stat: 'a_duration_ms',        label: ['A-duration'],  plain: 'A-duration',        unit: 'ms',   digits: 2, level: false, weight: null, headline: 5, compare: false },
  { key: 'b_duration_ms',        stat: 'b_duration_ms',        label: ['B-duration'],  plain: 'B-duration',        unit: 'ms',   digits: 2, level: false, weight: null, headline: 0, compare: false },
  { key: 'rise_time_us',         stat: 'rise_time_us',         label: ['Rise time'],   plain: 'Rise time',         unit: 'us',   digits: 1, level: false, weight: null, headline: 6, compare: false },
  { key: 'specific_impulse_Pa_s',stat: 'specific_impulse_Pa_s',label: ['Impulse'],     plain: 'Specific impulse',  unit: 'Pa·s', digits: 4, level: false, weight: null, headline: 0, compare: false },
  { key: 'crest_factor_dB',      stat: 'crest_factor_dB',      label: ['Crest factor'],plain: 'Crest factor',      unit: 'dB',   digits: 1, level: false, weight: null, headline: 0, compare: false },
  { key: 'spectral_centroid_Hz', stat: 'spectral_centroid_Hz', label: ['Centroid'],    plain: 'Spectral centroid', unit: 'Hz',   digits: 0, level: false, weight: null, headline: 0, compare: false },
  { key: 'kurtosis',             stat: 'kurtosis',             label: ['Kurtosis'],    plain: 'Kurtosis',          unit: '',     digits: 1, level: false, weight: null, headline: 0, compare: false },
];

const METRIC_BY_KEY = new Map(METRICS.map(m => [m.key, m]));

/* Required test-metadata fields. `field` is the wrapper id used for the error
   state; `input` is the control. Environment fields are required because they
   fix the reference impedance and the speed of sound: a record without them
   cannot be recomputed later. */
const REQUIRED_METADATA = [
  { input: 'md-operator',      field: 'field-md-operator',      name: 'Operator' },
  { input: 'md-date',          field: 'field-md-date',          name: 'Test date' },
  { input: 'md-configuration', field: 'field-md-configuration', name: 'Configuration' },
  { input: 'md-weapon',        field: 'field-md-weapon',        name: 'Weapon' },
  { input: 'md-ammunition',    field: 'field-md-ammunition',    name: 'Ammunition' },
  { input: 'md-mic-model',     field: 'field-md-mic-model',     name: 'Microphone' },
  { input: 'md-distance',      field: 'field-md-distance',      name: 'Distance' },
  { input: 'md-angle',         field: 'field-md-angle',         name: 'Angle' },
  { input: 'md-height',        field: 'field-md-height',        name: 'Height' },
  { input: 'md-temp',          field: 'field-md-temp',          name: 'Temperature' },
  { input: 'md-humidity',      field: 'field-md-humidity',      name: 'Relative humidity' },
  { input: 'md-pressure',      field: 'field-md-pressure',      name: 'Static pressure' },
];

/* id -> { name (config key sent to the bridge), min, max, integer, required } */
const NUMERIC_INPUTS = {
  'cal-tone-freq':  { min: 31.5,  max: 10000,  label: 'Calibrator frequency' },
  'cal-sensitivity':{ min: 0.001, max: 1e6,    label: 'Microphone sensitivity', exclusiveMin: true },
  'cal-gain':       { min: -20,   max: 80,     label: 'Preamplifier gain' },
  'cal-fsv':        { min: 0.01,  max: 1e6,    label: 'ADC full scale', exclusiveMin: true },
  'threshold-value':{ min: 1,     max: 90,     label: 'Threshold' },
  'refractory-ms':  { min: 10,    max: 5000,   label: 'Refractory period' },
  'min-snr':        { min: 0,     max: 80,     label: 'Minimum shot SNR' },
  'pre-ms':         { min: 1,     max: 500,    label: 'Pre-trigger window' },
  'post-ms':        { min: 50,    max: 5000,   label: 'Post-trigger window' },
  'stft-overlap':   { min: 0,     max: 95,     label: 'STFT overlap' },
  'band-low':       { min: 10,    max: 1000,   label: 'Lowest band centre' },
  'band-high':      { min: 1000,  max: 40000,  label: 'Highest band centre' },
  'hazard-rounds':  { min: 1,     max: 100000, label: 'Rounds per working day', integer: true },
  'md-barrel':      { min: 1,     max: 60,     label: 'Barrel length' },
  'md-distance':    { min: 0.1,   max: 100,    label: 'Distance' },
  'md-angle':       { min: 0,     max: 360,    label: 'Angle' },
  'md-height':      { min: 0,     max: 10,     label: 'Height' },
  'md-temp':        { min: -40,   max: 60,     label: 'Temperature' },
  'md-humidity':    { min: 0,     max: 100,    label: 'Relative humidity' },
  'md-pressure':    { min: 50,    max: 110,    label: 'Static pressure' },
  'md-wind':        { min: 0,     max: 40,     label: 'Wind speed' },
};

/* Analysis settings persisted between sessions, with their factory values.
   Threshold is 20 dB below peak; post-trigger is 500 ms because B-duration
   truncates at 200 ms on most centrefire shots. */
const SETTING_DEFAULTS = {
  'threshold-mode':  'relative',
  'threshold-value': '20',
  'refractory-ms':   '200',
  'min-snr':         '20',
  'pre-ms':          '20',
  'post-ms':         '500',
  'stft-nperseg':    '2048',
  'stft-overlap':    '75',
  'stft-window':     'hann',
  'band-low':        '25',
  'band-high':       '20000',
  'plot-format':     'png',
  'opt-weight-a':    true,
  'opt-weight-c':    true,
  'opt-bands':       true,
  'opt-per-shot-plots': true,
  'hazard-criterion':'85',
  'hazard-rounds':   '100',
};

const METADATA_IDS = [
  'md-operator', 'md-date', 'md-location', 'md-configuration', 'md-weapon',
  'md-barrel', 'md-ammunition', 'md-suppressor', 'md-mic-model', 'md-mic-serial',
  'md-distance', 'md-angle', 'md-height', 'md-temp', 'md-humidity',
  'md-pressure', 'md-wind', 'md-notes',
];

/* DOM id -> bridge metadata key. The bridge validates and rejects unknown
   keys, so this map is the contract, not a convenience. */
const METADATA_TO_CONFIG = {
  'md-operator':      { key: 'operator',       type: 'string' },
  'md-date':          { key: 'date',           type: 'string' },
  'md-location':      { key: 'location',       type: 'string' },
  'md-configuration': { key: 'configuration',  type: 'string' },
  'md-weapon':        { key: 'weapon',         type: 'string' },
  'md-barrel':        { key: 'barrelLengthIn', type: 'number' },
  'md-ammunition':    { key: 'ammunition',     type: 'string' },
  'md-suppressor':    { key: 'suppressor',     type: 'string' },
  'md-mic-model':     { key: 'micModel',       type: 'string' },
  'md-mic-serial':    { key: 'micSerial',      type: 'string' },
  'md-distance':      { key: 'micDistanceM',   type: 'number' },
  'md-angle':         { key: 'micAngleDeg',    type: 'number' },
  'md-height':        { key: 'micHeightM',     type: 'number' },
  'md-temp':          { key: 'temperatureC',   type: 'number' },
  'md-humidity':      { key: 'humidityPct',    type: 'number' },
  'md-pressure':      { key: 'pressureKPa',    type: 'number' },
  'md-wind':          { key: 'windMps',        type: 'number' },
  'md-notes':         { key: 'notes',          type: 'string' },
};

/* Reverse map for server-side "invalid-config" field errors, so a rejection
   lands on the control that caused it instead of in a generic toast. */
const CONFIG_TO_FIELD = {
  filePath:            { field: null,               input: 'file-path-input' },
  calibratorTone:      { field: null,               input: 'cal-tone-input' },
  calibratorLevelDb:   { field: 'field-cal-tone-level',  input: 'cal-tone-level' },
  calibratorFreqHz:    { field: null,               input: 'cal-tone-freq' },
  sensitivityMv:       { field: 'field-cal-sensitivity', input: 'cal-sensitivity' },
  preampGainDb:        { field: 'field-cal-gain',   input: 'cal-gain' },
  adcFullScaleV:       { field: 'field-cal-fsv',    input: 'cal-fsv' },
  paPerFS:             { field: null,               input: 'cal-profile-select' },
  thresholdDb:         { field: 'field-threshold',  input: 'threshold-value' },
  thresholdRelativeDb: { field: 'field-threshold',  input: 'threshold-value' },
  refractoryMs:        { field: null,               input: 'refractory-ms' },
  preMs:               { field: null,               input: 'pre-ms' },
  postMs:              { field: null,               input: 'post-ms' },
  overlapFraction:     { field: null,               input: 'stft-overlap' },
  nperseg:             { field: null,               input: 'stft-nperseg' },
  formats:             { field: null,               input: 'plot-format' },
  outputDir:           { field: null,               input: 'setting-output-dir' },
  calDesc:             { field: null,               input: 'cal-description' },
};

const VIEW_IDS = {
  analyze:  'view-analyze',
  results:  'view-results',
  compare:  'view-compare',
  history:  'view-history',
  settings: 'view-settings',
};

// Order must match the tablist in index.html: arrow-key navigation walks this
// array, so a tab missing here is unreachable by keyboard AND never revealed.
const TABS = ['overview', 'spectrogram', 'bands', 'shots', 'string', 'table', 'hazard'];

/* The guided flow. Order is the route; `owns` is the set of concerns whose
   blockers are attributed to that stop, so a missing microphone distance is
   reported on Test record and not as an anonymous item at the end. */
const STEPS = [
  { id: 'recording',   name: 'Recording',   panel: 'step-recording',   ready: 'Recording loaded' },
  { id: 'calibration', name: 'Calibration', panel: 'step-calibration', ready: 'Method chosen' },
  { id: 'metadata',    name: 'Test record', panel: 'step-metadata',    ready: 'Record complete' },
  { id: 'settings',    name: 'Detection',   panel: 'step-settings',    ready: 'Settings valid' },
  { id: 'run',         name: 'Run',         panel: 'step-run',         ready: 'Ready to run' },
];

const STEP_IDS = STEPS.map(s => s.id);

/* Which stop owns each numeric control. Anything not listed belongs to
   Detection, which is where the analysis parameters live. */
const INPUT_STEP = {
  'cal-tone-freq': 'calibration', 'cal-sensitivity': 'calibration',
  'cal-gain': 'calibration', 'cal-fsv': 'calibration', 'cal-tone-level': 'calibration',
  'md-barrel': 'metadata', 'md-distance': 'metadata', 'md-angle': 'metadata',
  'md-height': 'metadata', 'md-temp': 'metadata', 'md-humidity': 'metadata',
  'md-pressure': 'metadata', 'md-wind': 'metadata',
};

/* On a Mac the palette is Cmd-K, everywhere else Ctrl-K. Detected once, so the
   keycap the operator is shown is the key they actually have. */
const IS_APPLE = typeof navigator !== 'undefined'
  && /Mac|iPhone|iPad|iPod/i.test(navigator.platform || navigator.userAgent || '');
const MOD_LABEL = IS_APPLE ? '⌘' : 'Ctrl';


/* ===========================================================================
   1. UTILITIES
   =========================================================================== */

const $  = (id) => document.getElementById(id);
const qs = (sel, root = document) => root.querySelector(sel);
const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Fill point inside a cloned template. */
const slot = (root, name) => root.querySelector(`[data-slot="${name}"]`);

/** Clone a <template> and return its first element. */
function fromTemplate(id) {
  const tpl = $(id);
  if (!tpl) { console.error(`template #${id} is missing`); return null; }
  return tpl.content.cloneNode(true);
}

function setText(node, text) {
  if (!node) return;
  const value = (text === null || text === undefined || text === '') ? '—' : String(text);
  if (node.textContent !== value) node.textContent = value;
}

/** Text without the em-dash fallback, for places where blank means blank. */
function setRaw(node, text) {
  if (!node) return;
  const value = text === null || text === undefined ? '' : String(text);
  if (node.textContent !== value) node.textContent = value;
}

/** Build a label like L{AE} as real <sub> elements — never an HTML string. */
function labelNodes(parts) {
  const frag = document.createDocumentFragment();
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      const sub = document.createElement('sub');
      sub.textContent = part;
      frag.appendChild(sub);
    } else {
      frag.appendChild(document.createTextNode(part));
    }
  });
  return frag;
}

function setLabel(node, parts) {
  if (!node) return;
  node.textContent = '';
  node.appendChild(labelNodes(parts));
}

function show(node, visible) {
  if (!node) return;
  if (visible) node.removeAttribute('hidden');
  else node.setAttribute('hidden', '');
}

function setAttr(node, name, value) {
  if (!node) return;
  if (value === null || value === undefined || value === false) node.removeAttribute(name);
  else node.setAttribute(name, value === true ? '' : String(value));
}

function setDisabled(node, disabled) {
  if (!node) return;
  node.disabled = Boolean(disabled);
}

/** Swap the sprite an <svg><use> points at. */
function setIcon(svgOrHost, iconId) {
  if (!svgOrHost) return;
  const svg = svgOrHost.tagName && svgOrHost.tagName.toLowerCase() === 'svg'
    ? svgOrHost : svgOrHost.querySelector('svg');
  const use = svg && svg.querySelector('use');
  if (use) use.setAttribute('href', `#${iconId}`);
}

const TONE_ICON = { ok: 'i-check-circle', warn: 'i-alert', danger: 'i-error', info: 'i-info', accent: 'i-info' };

/** Never signal state by colour alone: tone always drags an icon with it. */
function setTone(node, tone, iconId) {
  if (!node) return;
  node.setAttribute('data-tone', tone);
  setIcon(node, iconId || TONE_ICON[tone] || 'i-info');
}

/* ---- numbers ------------------------------------------------------------ */

/**
 * Read a numeric control, distinguishing EMPTY from ZERO.
 *
 * THE ZERO BUG: the previous renderer used `Number(el.value) || DEFAULT`, so a
 * deliberate 0 for threshold, refractory, pre- or post-trigger was silently
 * replaced by a built-in. Nothing here is coerced: empty is empty, 0 is 0, and
 * anything unparseable is an error the operator is shown.
 *
 * @returns {{empty:boolean, ok:boolean, value:number|null, error:string|null}}
 */
function readNumber(id, spec = {}) {
  const el = $(id);
  const rule = { ...(NUMERIC_INPUTS[id] || {}), ...spec };
  const label = rule.label || id;
  if (!el) return { empty: true, ok: false, value: null, error: `${label} control is missing.` };

  const raw = String(el.value).trim();
  if (raw === '') {
    return rule.required
      ? { empty: true, ok: false, value: null, error: `${label} is required.` }
      : { empty: true, ok: true, value: null, error: null };
  }

  const value = Number(raw);
  if (!Number.isFinite(value)) {
    return { empty: false, ok: false, value: null, error: `${label} must be a number.` };
  }
  if (rule.integer && !Number.isInteger(value)) {
    return { empty: false, ok: false, value: null, error: `${label} must be a whole number.` };
  }
  if (rule.min !== undefined) {
    const below = rule.exclusiveMin ? value <= rule.min : value < rule.min;
    if (below) {
      return {
        empty: false, ok: false, value: null,
        error: `${label} must be ${rule.exclusiveMin ? 'greater than' : 'at least'} ${rule.min}.`,
      };
    }
  }
  if (rule.max !== undefined && value > rule.max) {
    return { empty: false, ok: false, value: null, error: `${label} must be at most ${rule.max}.` };
  }
  return { empty: false, ok: true, value, error: null };
}

function isNum(v) { return typeof v === 'number' && Number.isFinite(v); }

/** Format a measured number. Everything numeric lands in a .num element, which
    styles.css sets in the mono face with tabular figures. */
function fmt(value, digits = 1) {
  if (!isNum(value)) return '—';
  if (digits === 4) {              // specific impulse: significant figures
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 0.01 || abs >= 10000)) return value.toExponential(2);
    return String(Number(value.toPrecision(4)));
  }
  return value.toFixed(digits);
}

function fmtSigned(value, digits = 1) {
  if (!isNum(value)) return '—';
  const s = value.toFixed(digits);
  return value > 0 ? `+${s}` : s;
}

function fmtInt(value) { return isNum(value) ? String(Math.round(value)) : '—'; }

function fmtBytes(n) {
  if (!isNum(n)) return '—';
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} kB`;
  return `${n} B`;
}

function fmtDuration(s) {
  if (!isNum(s)) return '—';
  if (s < 60) return `${s.toFixed(2)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${(s - m * 60).toFixed(0)} s`;
}

function fmtHz(hz) {
  if (!isNum(hz)) return '—';
  if (hz >= 1000) return `${(hz / 1000).toFixed(hz % 1000 === 0 ? 0 : 2)} kHz`;
  return `${hz.toFixed(0)} Hz`;
}

function fmtTimestamp(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function basename(p) {
  if (typeof p !== 'string') return '—';
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1] || p;
}

/* ---- statistics --------------------------------------------------------- */

/** Energy (ISO) average of a set of levels: 10*log10(mean(10^(L/10))). */
function energyAverageDb(values) {
  const finite = values.filter(isNum);
  if (finite.length === 0) return NaN;
  const mean = finite.reduce((acc, v) => acc + 10 ** (v / 10), 0) / finite.length;
  return 10 * Math.log10(mean);
}

/** Sample standard deviation (ddof = 1) — the shots are a sample, not a census. */
function sampleStd(values) {
  const finite = values.filter(isNum);
  const n = finite.length;
  if (n < 2) return 0;
  const mean = finite.reduce((a, v) => a + v, 0) / n;
  const ss = finite.reduce((a, v) => a + (v - mean) ** 2, 0);
  return Math.sqrt(ss / (n - 1));
}

function median(values) {
  const finite = values.filter(isNum).slice().sort((a, b) => a - b);
  const n = finite.length;
  if (n === 0) return NaN;
  return n % 2 ? finite[(n - 1) / 2] : (finite[n / 2 - 1] + finite[n / 2]) / 2;
}

/** Distribution summary matching metrics.py::_summarize exactly. */
function summarise(values, isLevel) {
  const finite = values.filter(isNum);
  const n = finite.length;
  if (n === 0) {
    return { n: 0, mean: NaN, std: NaN, min: NaN, max: NaN, median: NaN, ci95_half_width: NaN };
  }
  const std = sampleStd(finite);
  return {
    n,
    mean: isLevel ? energyAverageDb(finite) : finite.reduce((a, v) => a + v, 0) / n,
    std,
    min: Math.min(...finite),
    max: Math.max(...finite),
    median: median(finite),
    ci95_half_width: n > 1 ? 1.96 * std / Math.sqrt(n) : 0,
  };
}

/* ---- misc --------------------------------------------------------------- */

function uid(prefix = 'r') {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return prefix + Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

function isTypingTarget(node) {
  if (!node) return false;
  const tag = node.tagName ? node.tagName.toLowerCase() : '';
  return tag === 'input' || tag === 'textarea' || tag === 'select' || node.isContentEditable === true;
}

async function copyText(text) {
  if (!text) return false;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall through to the legacy path */ }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.setAttribute('aria-hidden', 'true');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch { return false; }
}

function downloadText(filename, text, mime = 'text/csv;charset=utf-8') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

/** RFC 4180 quoting. Nothing here reaches the DOM, so this is not an escaper. */
function csvCell(value) {
  const s = value === null || value === undefined ? '' : String(value);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function csvRows(rows) {
  return rows.map(r => r.map(csvCell).join(',')).join('\r\n') + '\r\n';
}


/* ===========================================================================
   2. PERSISTENCE
   =========================================================================== */

const store = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      if (raw === null) return fallback;
      return JSON.parse(raw);
    } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; }
    catch { return false; }
  },
  setRaw(key, value) {
    try { localStorage.setItem(key, value); return true; } catch { return false; }
  },
  getRaw(key, fallback) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : v; }
    catch { return fallback; }
  },
  remove(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
};


/* ===========================================================================
   3. STATE
   =========================================================================== */

const state = {
  view: 'analyze',
  theme: 'system',

  /** Chrome that is not a measurement: which stop the flow is on, whether the
      sidebar is collapsed, which floating surface is open. */
  ui: {
    step: 'recording',
    stepDir: 'fwd',          // fwd | back — which way the panel should enter
    visited: new Set(['recording']),
    sidebar: 'full',         // full | rail
    menu: null,              // id of the open menu, or null
    paletteIndex: 0,
    paletteQuery: '',
  },

  /** Transport. status is the real socket state, never a guess. */
  ws: {
    socket: null,
    status: 'idle',        // idle | connecting | open | closing | closed | error
    attempts: 0,
    timer: null,
    lastError: null,
    serverVersion: null,
    protocolVersion: null,
  },

  /** The recording under analysis. */
  input: { name: null, path: null, size: null, source: null, uploading: false, progress: 0, error: null },

  /** Calibration is an explicit choice. There is deliberately no default. */
  calibration: {
    method: '',            // '' | tone | chain | profile | none
    tone: { name: null, path: null, size: null, uploading: false, error: null },
    derivedPaPerFS: null,  // filled from a completed tone run
    derivedResidual: null,
    profileId: '',
  },

  /** Field-level errors keyed by control id. */
  fieldErrors: new Map(),

  /** Run lifecycle. */
  run: {
    status: 'idle',        // idle | starting | running | cancelling | complete | error | cancelled | lost
    requestId: null,
    percent: 0,
    stage: '',
    determinate: false,
    startedAt: null,
    finishedAt: null,
    outputDir: null,
    error: null,           // { code, message, stderr, exitCode }
    logCount: 0,
  },

  /** Loaded measurement. */
  results: {
    status: 'empty',       // empty | loading | loaded | error
    dir: null,
    payload: null,         // { metadata, images, shotImages, csv, outputDir }
    levelUnit: 'dB',
    rejected: new Set(),   // shot numbers the technician has excluded
    shotIndex: 0,
    tab: 'overview',
    filter: '',
    showRejected: true,
    error: null,
    aggregate: null,       // client-side recompute over the included shots
  },

  compare: {
    refDir: '', testDir: '',
    ref: null, test: null,
    status: 'empty',       // empty | loading | loaded | error | refused
    error: null,
    blockers: [],
    rows: [],
    bands: null,
  },

  history: { status: 'idle', items: [], filter: '', config: 'all', error: null },

  profiles: [],
  recent: [],
  prefs: { outputDir: '', openOnComplete: true },

  app: { version: null, engine: null, commit: null },
};

let dirty = false;
let frame = null;
let frameTimer = null;

/**
 * Mark the app dirty; render() runs once however many times we ask within a
 * frame.
 *
 * requestAnimationFrame alone is NOT enough: a browser stops servicing it in a
 * background tab, and an operator watching a twenty-minute analysis will switch
 * tabs. Without the timer fallback the interface silently freezes on whatever
 * it last showed — which for a run in progress means it goes on claiming the
 * analysis is running long after it has finished or failed. The two are raced
 * and whichever fires first renders.
 */
function commit() {
  dirty = true;
  if (frame !== null || frameTimer !== null) return;

  const run = () => {
    if (frame !== null) { cancelAnimationFrame(frame); frame = null; }
    if (frameTimer !== null) { clearTimeout(frameTimer); frameTimer = null; }
    if (!dirty) return;
    dirty = false;
    try { render(); } catch (err) { console.error('render failed', err); }
  };

  frame = requestAnimationFrame(run);
  frameTimer = setTimeout(run, 200);
}

/** The single place the DOM is derived from state. */
function render() {
  renderNav();
  renderTheme();
  renderInput();
  renderCalibration();
  renderMetadataCompleteness();
  renderSettingsDerived();
  renderBlockers();
  renderFlow();
  renderRun();
  renderResults();
  renderCompare();
  renderHistory();
  renderProfiles();
  renderPrefs();
}

/* ===========================================================================
   4. THEME
   The FOUC guard in index.html has already applied the stored theme before
   first paint. Here we only keep the two controls, storage and the charts in
   step. "system" means REMOVE data-theme so tokens.css falls through to
   prefers-color-scheme.
   =========================================================================== */

const darkQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

function applyTheme(choice) {
  const value = (choice === 'light' || choice === 'dark') ? choice : 'system';
  state.theme = value;
  if (value === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', value);
  store.setRaw(STORAGE.theme, value);
  redrawCharts();
  commit();
}

function renderTheme() {
  const radio = $(`theme-${state.theme}`);
  if (radio && !radio.checked) radio.checked = true;
  const select = $('theme-select');
  if (select && select.value !== state.theme) select.value = state.theme;
}

function cycleTheme() {
  const order = ['light', 'dark', 'system'];
  const next = order[(order.indexOf(state.theme) + 1) % order.length];
  applyTheme(next);
  announce(`Theme: ${next}`);
}

function wireTheme() {
  qsa('input[name="theme"]').forEach(input => {
    input.addEventListener('change', () => { if (input.checked) applyTheme(input.value); });
  });
  const select = $('theme-select');
  if (select) select.addEventListener('change', () => applyTheme(select.value));

  const cycle = $('btn-theme-cycle');
  if (cycle) cycle.addEventListener('click', cycleTheme);

  if (darkQuery) {
    const onSystemChange = () => { if (state.theme === 'system') redrawCharts(); };
    if (darkQuery.addEventListener) darkQuery.addEventListener('change', onSystemChange);
    else if (darkQuery.addListener) darkQuery.addListener(onSystemChange);
  }
}


/* ===========================================================================
   5. ANNOUNCEMENTS, TOASTS, DIALOGS
   =========================================================================== */

let announceTimer = null;

/** Polite announcement for something that has no visible home of its own. */
function announce(message) {
  const node = $('live-announcer');
  if (!node) return;
  // Re-assigning identical text does not re-announce; blank it first.
  node.textContent = '';
  clearTimeout(announceTimer);
  announceTimer = setTimeout(() => { node.textContent = message; }, 40);
}

const TOAST_TTL = { info: 5000, ok: 4000, warn: 9000, danger: 0 /* sticky */ };

function toast({ title, text = '', tone = 'info', icon = null, ttl = null }) {
  const region = $('toast-region');
  const frag = fromTemplate('tpl-toast');
  if (!region || !frag) return null;

  const node = frag.firstElementChild;
  setTone(node, tone, icon);
  setIcon(slot(node, 'icon'), icon || TONE_ICON[tone] || 'i-info');
  setRaw(slot(node, 'title'), title);
  setRaw(slot(node, 'text'), text);
  show(slot(node, 'text'), Boolean(text));

  const dismiss = () => {
    node.removeEventListener('click', onClick);
    node.remove();
  };
  const onClick = (ev) => {
    if (ev.target.closest('[data-action="dismiss"]')) dismiss();
  };
  node.addEventListener('click', onClick);

  region.appendChild(node);

  const life = ttl === null ? (TOAST_TTL[tone] ?? 5000) : ttl;
  if (life > 0) setTimeout(dismiss, life);
  return node;
}

/* ---- dialogs ------------------------------------------------------------ */

/* <dialog>.showModal() traps focus and restores it natively in every browser
   this ships to, but we track the opener explicitly so focus restoration is
   guaranteed even when the dialog is closed programmatically, and we add a
   Tab guard for the case where the dialog has been re-parented. */
const dialogState = new WeakMap();

function openDialog(dialog, { opener = null, focus = null } = {}) {
  if (!dialog) return;
  dialogState.set(dialog, { opener: opener || document.activeElement });
  if (typeof dialog.showModal === 'function') {
    if (!dialog.open) dialog.showModal();
  } else {
    dialog.setAttribute('open', '');
  }
  const target = focus || qs('button, [href], input, select, textarea', dialog);
  if (target) target.focus();
}

function closeDialog(dialog) {
  if (!dialog) return;
  const record = dialogState.get(dialog);
  if (typeof dialog.close === 'function' && dialog.open) dialog.close();
  else dialog.removeAttribute('open');
  const opener = record && record.opener;
  const usable = opener && opener !== document.body && document.contains(opener)
    && typeof opener.focus === 'function';
  // A dialog raised by an event rather than a click has no opener to go back
  // to; dropping focus on <body> would strand a keyboard user at the top of the
  // document, so the main region takes it instead.
  if (usable) opener.focus();
  else if ($('main')) $('main').focus({ preventScroll: true });
  dialogState.delete(dialog);
}

function trapTab(ev) {
  if (ev.key !== 'Tab') return;
  const dialog = ev.currentTarget;
  const focusables = qsa(
    'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    dialog,
  ).filter(el => el.offsetParent !== null || el === document.activeElement);
  if (focusables.length === 0) return;
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  if (ev.shiftKey && document.activeElement === first) { ev.preventDefault(); last.focus(); }
  else if (!ev.shiftKey && document.activeElement === last) { ev.preventDefault(); first.focus(); }
}

/**
 * Confirmation dialog. Resolves true on accept, false on cancel or dismiss.
 * `mode: 'info'` hides the accept button and turns Cancel into Close, which is
 * how the keyboard-shortcut sheet and the About panel are shown.
 */
function confirmDialog({ title, build, acceptLabel = 'Confirm', acceptTone = 'danger', mode = 'confirm' }) {
  return new Promise(resolve => {
    const dialog = $('modal-confirm');
    const body = $('modal-confirm-body');
    const accept = $('btn-modal-confirm-accept');
    const cancel = $('btn-modal-confirm-cancel');
    const close = $('btn-modal-confirm-close');
    if (!dialog || !body) { resolve(false); return; }

    setRaw($('modal-confirm-title'), title);
    body.textContent = '';
    if (typeof build === 'function') build(body);

    show(accept, mode === 'confirm');
    if (accept) {
      setRaw(accept, acceptLabel);
      accept.classList.toggle('btn-danger', acceptTone === 'danger');
      accept.classList.toggle('btn-primary', acceptTone !== 'danger');
    }
    setRaw(cancel, mode === 'confirm' ? 'Cancel' : 'Close');

    const finish = (value) => {
      accept && accept.removeEventListener('click', onAccept);
      cancel && cancel.removeEventListener('click', onCancel);
      close && close.removeEventListener('click', onCancel);
      dialog.removeEventListener('cancel', onNativeCancel);
      dialog.removeEventListener('keydown', trapTab);
      closeDialog(dialog);
      resolve(value);
    };
    const onAccept = () => finish(true);
    const onCancel = () => finish(false);
    const onNativeCancel = (ev) => { ev.preventDefault(); finish(false); };

    accept && accept.addEventListener('click', onAccept);
    cancel && cancel.addEventListener('click', onCancel);
    close && close.addEventListener('click', onCancel);
    dialog.addEventListener('cancel', onNativeCancel);
    dialog.addEventListener('keydown', trapTab);

    openDialog(dialog, { focus: mode === 'confirm' ? accept : cancel });
  });
}

/** Simple paragraph appender used by the dialog builders. */
function addParagraph(parent, text, className = 'prose') {
  const p = document.createElement('p');
  p.className = className;
  p.textContent = text;
  parent.appendChild(p);
  return p;
}

function addKeyList(parent, pairs) {
  const dl = document.createElement('dl');
  dl.className = 'kv-list';
  for (const [key, value] of pairs) {
    const dt = document.createElement('dt');
    dt.textContent = key;
    const dd = document.createElement('dd');
    dd.textContent = value;
    dl.append(dt, dd);
  }
  parent.appendChild(dl);
  return dl;
}

function wireDialogs() {
  // The two static dialogs close from their own header and footer buttons.
  const pairs = [
    ['modal-profiles', ['btn-modal-profiles-close', 'btn-modal-profiles-done']],
    ['modal-validity', ['btn-modal-validity-close', 'btn-modal-validity-done']],
  ];
  for (const [dialogId, buttonIds] of pairs) {
    const dialog = $(dialogId);
    if (!dialog) continue;
    dialog.addEventListener('keydown', trapTab);
    dialog.addEventListener('cancel', (ev) => { ev.preventDefault(); closeDialog(dialog); });
    for (const id of buttonIds) {
      const btn = $(id);
      if (btn) btn.addEventListener('click', () => closeDialog(dialog));
    }
  }
}


/* ===========================================================================
   6. NAVIGATION, TABS, KEYBOARD
   =========================================================================== */

function setView(view, { focus = true } = {}) {
  if (!VIEW_IDS[view]) return;
  if (state.view === view) return;
  state.view = view;

  if (view === 'history' && state.history.status === 'idle') loadHistory();
  if (view === 'compare') ensureCompareOptions();

  commit();
  if (focus) {
    const main = $('main');
    if (main) main.focus({ preventScroll: false });
    window.scrollTo({ top: 0, behavior: 'auto' });
  }
  const title = qs(`#${VIEW_IDS[view]} .view-title`);
  announce(title ? `${title.textContent} view` : `${view} view`);
}

function renderNav() {
  for (const [view, id] of Object.entries(VIEW_IDS)) {
    show($(id), view === state.view);
  }
  qsa('.nav-item[data-view]').forEach(item => {
    const active = item.dataset.view === state.view;
    if (active) item.setAttribute('aria-current', 'page');
    else item.removeAttribute('aria-current');
  });

  // Results is unreachable until there is something to show.
  const resultsNav = qs('.nav-item[data-view="results"]');
  setDisabled(resultsNav, state.results.status === 'empty');

  const shotBadge = $('nav-badge-shots');
  const shots = shotList().length;
  show(shotBadge, shots > 0);
  setRaw(shotBadge, String(shots));

  const historyBadge = $('nav-badge-history');
  show(historyBadge, state.history.items.length > 0);
  setRaw(historyBadge, String(state.history.items.length));
}

function wireNavigation() {
  // One delegated listener covers the sidebar, the brand button and every
  // button inside an empty state.
  document.addEventListener('click', (ev) => {
    const trigger = ev.target.closest('[data-view]');
    if (!trigger) return;
    if (trigger.classList.contains('view')) return;    // the <section> itself
    if (trigger.disabled) return;
    ev.preventDefault();
    setView(trigger.dataset.view);
  });
}

/* ---- result tabs -------------------------------------------------------- */

function setTab(tab, { focus = false } = {}) {
  if (!TABS.includes(tab)) return;
  state.results.tab = tab;
  for (const name of TABS) {
    const button = $(`tab-${name}`);
    const panel = $(`panel-${name}`);
    const active = name === tab;
    if (button) {
      button.setAttribute('aria-selected', active ? 'true' : 'false');
      button.tabIndex = active ? 0 : -1;
      if (active && focus) button.focus();
    }
    show(panel, active);
  }
  if (tab === 'shots') drawShotBandChart();
  commit();
}

function wireTabs() {
  const list = $('results-tablist');
  if (!list) return;

  list.addEventListener('click', (ev) => {
    const tab = ev.target.closest('.tab');
    if (!tab || !tab.id.startsWith('tab-')) return;
    setTab(tab.id.slice(4));
  });

  list.addEventListener('keydown', (ev) => {
    const index = TABS.indexOf(state.results.tab);
    let next = null;
    if (ev.key === 'ArrowRight') next = (index + 1) % TABS.length;
    else if (ev.key === 'ArrowLeft') next = (index - 1 + TABS.length) % TABS.length;
    else if (ev.key === 'Home') next = 0;
    else if (ev.key === 'End') next = TABS.length - 1;
    if (next === null) return;
    ev.preventDefault();
    setTab(TABS[next], { focus: true });
  });
}

/* ---- keyboard shortcuts ------------------------------------------------- */

const SHORTCUTS = [
  [`${MOD_LABEL}K`, 'Command palette — every action by name'],
  ['1 … 5', 'Analyze, Results, Compare, History, Settings'],
  ['← and →', 'Previous / next step, while the rail has focus'],
  ['R', 'Run the analysis'],
  ['Escape', 'Cancel a running analysis, or close a dialog'],
  ['[ and ]', 'Previous / next shot'],
  ['X', 'Reject or restore the selected shot'],
  ['T', 'Cycle theme: light, dark, system'],
  ['B', 'Collapse or expand the sidebar'],
  ['/', 'Focus the filter in the current view'],
  ['?', 'This help'],
];

function showShortcuts() {
  confirmDialog({
    title: 'Keyboard shortcuts',
    mode: 'info',
    build: (body) => {
      addParagraph(body, 'Shortcuts are inactive while the focus is in a text field.');
      addKeyList(body, SHORTCUTS);
    },
  });
}

function showAbout() {
  confirmDialog({
    title: 'About SASA',
    mode: 'info',
    build: (body) => {
      addParagraph(body,
        'SASA reports acoustic measurements of firearm discharge. Every analysis runs '
        + 'locally; nothing is uploaded. The result is only as defensible as its '
        + 'calibration and its test record.');
      addKeyList(body, [
        ['Version', state.app.version || '—'],
        ['Engine', state.app.engine || '—'],
        ['Connection', wsStatusLabel()],
      ]);
    },
  });
}

function wireKeyboard() {
  document.addEventListener('keydown', (ev) => {
    // The one modified combination we do claim. Cmd/Ctrl-K is not reserved by
    // any browser and is the near-universal binding for a command palette, so
    // an operator arriving from any other professional tool already knows it.
    // It works from inside a text field, which is the whole point.
    if ((ev.metaKey || ev.ctrlKey) && !ev.altKey && (ev.key === 'k' || ev.key === 'K')) {
      ev.preventDefault();
      const palette = $('palette');
      if (palette && palette.open) closePalette();
      else openPalette();
      return;
    }

    // Never hijack any other browser-reserved combination.
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;

    if (ev.key === 'Escape') {
      // Whatever is on top goes first. Without this, dismissing the palette or
      // a dialog during a run ALSO cancelled the run — one keystroke, two
      // actions, and the destructive one invisible behind the other.
      if (state.ui.menu) { ev.preventDefault(); closeMenu({ restoreFocus: true }); return; }
      const palette = $('palette');
      if (palette && palette.open) return;
      if (qsa('dialog[open]').length > 0) return;
      if (state.run.status === 'running' || state.run.status === 'starting') {
        ev.preventDefault();
        cancelRun();
      }
      return;
    }

    if (isTypingTarget(ev.target)) return;
    // A modal is modal. R, X, [ and ] were still driving the page behind it.
    if (qsa('dialog[open]').length > 0) return;

    switch (ev.key) {
      case '1': setView('analyze'); break;
      case '2': if (state.results.status !== 'empty') setView('results'); break;
      case '3': setView('compare'); break;
      case '4': setView('history'); break;
      case '5': setView('settings'); break;
      case 'r': case 'R':
        if (!$('btn-run') || $('btn-run').disabled) return;
        ev.preventDefault();
        startRun();
        break;
      case '[': stepShot(-1); break;
      case ']': stepShot(1); break;
      case 'x': case 'X':
        if (state.view === 'results') { ev.preventDefault(); toggleReject(currentShotNumber()); }
        break;
      case 't': case 'T': ev.preventDefault(); cycleTheme(); break;
      case 'b': case 'B': ev.preventDefault(); toggleSidebar(); break;
      case '?': ev.preventDefault(); showShortcuts(); break;
      case '/': {
        const target = state.view === 'history' ? $('history-filter')
          : state.view === 'results' ? $('metrics-filter') : null;
        if (target) { ev.preventDefault(); target.focus(); target.select(); }
        break;
      }
      default: break;
    }
  });
}

/* ===========================================================================
   7. ANALYZE
   =========================================================================== */

/* ---- field-level errors ------------------------------------------------- */

/**
 * Every invalid control gets data-invalid="true" on its .field wrapper AND a
 * visible .field-error. Most fields ship one in the markup; the rest get one
 * built here (createElement only) so no control can ever go red without saying
 * why.
 */
function fieldErrorNode(input) {
  const field = input.closest('.field');
  if (!field) return null;
  let node = field.querySelector('.field-error');
  if (node) return node;

  node = document.createElement('p');
  node.className = 'field-error';
  node.id = `${input.id}-error-auto`;
  node.setAttribute('hidden', '');

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#i-error');
  svg.appendChild(use);

  const span = document.createElement('span');
  node.append(svg, span);
  field.appendChild(node);

  const described = (input.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
  if (!described.includes(node.id)) {
    described.push(node.id);
    input.setAttribute('aria-describedby', described.join(' '));
  }
  return node;
}

function setFieldError(inputId, message) {
  const input = $(inputId);
  if (!input) return;
  const field = input.closest('.field');
  const node = fieldErrorNode(input);

  if (message) {
    state.fieldErrors.set(inputId, message);
    if (field) field.setAttribute('data-invalid', 'true');
    input.setAttribute('aria-invalid', 'true');
    if (node) {
      const span = node.querySelector('span');
      setRaw(span || node, message);
      show(node, true);
    } else {
      // No .field wrapper to hang a message on. Marking the control without
      // saying why is exactly the failure this function exists to prevent, so
      // the reason goes somewhere the operator will see it.
      toast({ title: 'Rejected', text: message, tone: 'danger' });
    }
  } else {
    state.fieldErrors.delete(inputId);
    if (field) field.removeAttribute('data-invalid');
    input.removeAttribute('aria-invalid');
    if (node) show(node, false);
  }
}

function clearFieldErrors() {
  for (const id of Array.from(state.fieldErrors.keys())) setFieldError(id, null);
}

/** Validate one numeric control and surface the result on the control itself. */
function validateNumericInput(id) {
  const result = readNumber(id);
  setFieldError(id, result.ok ? null : result.error);
  return result;
}

/* ---- the recording ------------------------------------------------------ */

const AUDIO_EXTENSIONS = ['.wav', '.wave', '.flac', '.aif', '.aiff', '.aifc', '.w64',
  '.rf64', '.bwf', '.caf', '.ogg', '.mp3', '.m4a'];
const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.mkv', '.avi', '.mts', '.mxf'];

function extensionOf(name) {
  const match = /\.[A-Za-z0-9]+$/.exec(String(name || ''));
  return match ? match[0].toLowerCase() : '';
}

function describeFormat(name) {
  const ext = extensionOf(name);
  if (!ext) return { text: 'Unknown container', tone: 'warn' };
  if (VIDEO_EXTENSIONS.includes(ext)) {
    return { text: `${ext.slice(1).toUpperCase()} container — audio track will be extracted`, tone: 'warn' };
  }
  if (AUDIO_EXTENSIONS.includes(ext)) {
    const lossy = ext === '.mp3' || ext === '.ogg' || ext === '.m4a';
    return {
      text: lossy
        ? `${ext.slice(1).toUpperCase()} — lossy coding will bias impulse metrics`
        : `${ext.slice(1).toUpperCase()}`,
      tone: lossy ? 'warn' : 'info',
    };
  }
  return { text: `${ext.slice(1).toUpperCase()} — unsupported`, tone: 'danger' };
}

function renderInput() {
  const chip = $('file-chip');
  const has = Boolean(state.input.path) || state.input.uploading;
  show(chip, has);
  show($('file-dropzone'), !has);

  setText($('file-chip-name'), state.input.name);
  let meta;
  if (state.input.uploading) {
    meta = `Uploading… ${Math.round(state.input.progress)}%`;
  } else if (state.input.source === 'path') {
    meta = state.input.path;
  } else {
    meta = `${fmtBytes(state.input.size)} · uploaded to the local analysis area`;
  }
  setText($('file-chip-meta'), meta);

  const pill = $('input-format-pill');
  if (state.input.name) {
    const format = describeFormat(state.input.name);
    show(pill, true);
    setTone(pill, format.tone);
    setRaw($('input-format-text'), format.text);
  } else {
    show(pill, false);
  }

  const errorRow = $('file-path-error');
  show(errorRow, Boolean(state.input.error));
  setRaw($('file-path-error-text'), state.input.error || '');
  const pathInput = $('file-path-input');
  if (pathInput) {
    const field = pathInput.closest('.field');
    if (field) setAttr(field, 'data-invalid', state.input.error ? 'true' : null);
  }
}

function clearInput() {
  state.input = { name: null, path: null, size: null, source: null, uploading: false, progress: 0, error: null };
  const fileInput = $('file-input');
  if (fileInput) fileInput.value = '';
  commit();
}

/** Upload with XHR so large recordings report progress instead of freezing. */
function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('file', file, file.name);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);
    xhr.responseType = 'json';

    xhr.upload.addEventListener('progress', (ev) => {
      if (!ev.lengthComputable) return;
      state.input.progress = (ev.loaded / ev.total) * 100;
      commit();
    });
    xhr.addEventListener('load', () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300 && body && body.path) resolve(body);
      else reject(new Error((body && (body.message || body.error)) || `Upload failed (HTTP ${xhr.status}).`));
    });
    xhr.addEventListener('error', () => reject(new Error('The upload could not reach the local service.')));
    xhr.addEventListener('abort', () => reject(new Error('The upload was cancelled.')));
    xhr.send(form);
  });
}

async function acceptRecording(file) {
  state.input = {
    name: file.name, path: null, size: file.size, source: 'upload',
    uploading: true, progress: 0, error: null,
  };
  commit();
  try {
    const result = await uploadFile(file);
    state.input.path = result.path;
    state.input.name = result.name || file.name;
    state.input.size = result.size ?? file.size;
    state.input.uploading = false;
    rememberRecent(result.path, state.input.name);
    announce(`Recording ready: ${state.input.name}`);
  } catch (err) {
    state.input = { name: null, path: null, size: null, source: null, uploading: false, progress: 0, error: err.message };
    toast({ title: 'Upload failed', text: err.message, tone: 'danger' });
  }
  commit();
}

async function acceptCalibratorTone(file) {
  state.calibration.tone = { name: file.name, path: null, size: file.size, uploading: true, error: null };
  commit();
  try {
    const result = await uploadFile(file);
    state.calibration.tone = {
      name: result.name || file.name, path: result.path,
      size: result.size ?? file.size, uploading: false, error: null,
    };
  } catch (err) {
    state.calibration.tone = { name: null, path: null, size: null, uploading: false, error: null };
    toast({ title: 'Calibrator upload failed', text: err.message, tone: 'danger' });
  }
  commit();
}

/** Drop zones: keyboard operation comes free (the file input covers the zone). */
function wireDropzone(zoneId, inputId, onFile) {
  const zone = $(zoneId);
  const input = $(inputId);
  if (!zone || !input) return;

  let depth = 0;
  const setDrag = (on) => zone.classList.toggle('is-dragover', on);

  zone.addEventListener('dragenter', (ev) => { ev.preventDefault(); depth += 1; setDrag(true); });
  zone.addEventListener('dragover', (ev) => { ev.preventDefault(); ev.dataTransfer.dropEffect = 'copy'; });
  zone.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (depth === 0) setDrag(false); });
  zone.addEventListener('drop', (ev) => {
    ev.preventDefault();
    depth = 0; setDrag(false);
    const file = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
    if (file) onFile(file);
  });
  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    if (file) onFile(file);
  });
}

function rememberRecent(path, name) {
  if (!path) return;
  const list = state.recent.filter(item => item.path !== path);
  list.unshift({ path, name: name || basename(path), at: Date.now() });
  state.recent = list.slice(0, MAX_RECENT);
  store.set(STORAGE.recent, state.recent);
  renderRecent();
}

/** The recent-files list lives in a <datalist> attached to the path field —
    the markup has no dedicated container, and a datalist adds no new styling. */
function renderRecent() {
  const input = $('file-path-input');
  if (!input) return;
  let list = $('file-path-recent');
  if (!list) {
    list = document.createElement('datalist');
    list.id = 'file-path-recent';
    input.parentNode.appendChild(list);
    input.setAttribute('list', list.id);
  }
  list.textContent = '';
  for (const item of state.recent) {
    const option = document.createElement('option');
    option.value = item.path;
    option.label = item.name;
    list.appendChild(option);
  }
}

function useTypedPath() {
  const input = $('file-path-input');
  if (!input) return;
  const value = input.value.trim();
  if (value === '') {
    state.input.error = 'Enter a path, or drop a file above.';
    commit();
    return;
  }
  const ext = extensionOf(value);
  if (!AUDIO_EXTENSIONS.includes(ext) && !VIDEO_EXTENSIONS.includes(ext)) {
    state.input.error = 'That file type is not one SASA can read.';
    commit();
    return;
  }
  // The server confines the path to its permitted roots and reports back if it
  // will not accept it; we do not pretend to have verified it here.
  state.input = {
    name: basename(value), path: value, size: null, source: 'path',
    uploading: false, progress: 0, error: null,
  };
  rememberRecent(value, basename(value));
  commit();
  announce(`Path set: ${basename(value)}`);
}

/* ---- calibration -------------------------------------------------------- */

function chainPaPerFS() {
  const sens = readNumber('cal-sensitivity');
  const gain = readNumber('cal-gain');
  const fsv = readNumber('cal-fsv');
  if (!sens.ok || !gain.ok || !fsv.ok) return null;
  if (sens.empty || gain.empty || fsv.empty) return null;
  // Pa_per_FS = V_fullscale / (S_V_per_Pa * 10^(gain/20))   [calibration.py]
  const sensitivityVPerPa = (sens.value / 1000) * 10 ** (gain.value / 20);
  if (!(sensitivityVPerPa > 0)) return null;
  return fsv.value / sensitivityVPerPa;
}

function fullScaleDb(paPerFS) {
  return isNum(paPerFS) && paPerFS > 0 ? 20 * Math.log10(paPerFS / PA_REF) : NaN;
}

function activeProfile() {
  return state.profiles.find(p => p.id === state.calibration.profileId) || null;
}

/** The Pa/FS this run will use, or null when the choice is incomplete. */
function resolvedPaPerFS() {
  switch (state.calibration.method) {
    case 'chain': return chainPaPerFS();
    case 'profile': {
      const profile = activeProfile();
      return profile && isNum(profile.paPerFS) ? profile.paPerFS : null;
    }
    case 'tone': return state.calibration.derivedPaPerFS;   // the backend derives it
    default: return null;
  }
}

function levelUnitForMethod() {
  if (state.calibration.method === '') return null;
  return state.calibration.method === 'none' ? 'dB re FS' : 'dB SPL';
}

function renderCalibration() {
  const method = state.calibration.method;

  for (const name of ['tone', 'chain', 'profile', 'none']) {
    show($(`cal-panel-${name}`), method === name);
  }

  // Status pill — never colour alone.
  const pill = $('cal-status-pill');
  const text = $('cal-status-text');
  if (method === '') {
    setTone(pill, 'warn'); setRaw(text, 'Not set');
  } else if (method === 'none') {
    const ack = $('cal-uncal-ack');
    if (ack && ack.checked) { setTone(pill, 'warn'); setRaw(text, 'Uncalibrated — acknowledged'); }
    else { setTone(pill, 'warn'); setRaw(text, 'Uncalibrated — not acknowledged'); }
  } else if (method === 'tone') {
    const ready = Boolean(state.calibration.tone.path) && $('cal-tone-level') && $('cal-tone-level').value !== '';
    setTone(pill, ready ? 'ok' : 'warn');
    setRaw(text, ready ? 'Calibrator tone' : 'Calibrator tone — incomplete');
  } else {
    const pa = resolvedPaPerFS();
    setTone(pill, isNum(pa) ? 'ok' : 'warn');
    setRaw(text, isNum(pa)
      ? (method === 'chain' ? 'Recording chain' : 'Saved profile')
      : (method === 'chain' ? 'Recording chain — incomplete' : 'No profile chosen'));
  }

  // Chain readout
  const chainPa = chainPaPerFS();
  const chainResult = $('cal-chain-result');
  if (chainResult) chainResult.value = isNum(chainPa) ? fmt(chainPa, 4) : '—';
  setText($('cal-chain-fsdb'), isNum(chainPa) ? fmt(fullScaleDb(chainPa), 1) : null);

  // Tone readout — Pa/FS is derived by the backend from the tone recording, so
  // it stays blank until a run using this tone has completed.
  const toneResult = $('cal-tone-result');
  if (toneResult) {
    toneResult.value = isNum(state.calibration.derivedPaPerFS)
      ? fmt(state.calibration.derivedPaPerFS, 4) : '—';
  }
  setText($('cal-tone-residual'), isNum(state.calibration.derivedResidual)
    ? fmt(state.calibration.derivedResidual, 2) : null);

  const toneChip = $('cal-tone-chip');
  const hasTone = Boolean(state.calibration.tone.name);
  show(toneChip, hasTone);
  show($('cal-tone-dropzone'), !hasTone);
  setText($('cal-tone-name'), state.calibration.tone.name);
  setText($('cal-tone-meta'), state.calibration.tone.error
    ? state.calibration.tone.error
    : (state.calibration.tone.uploading ? 'Uploading…' : fmtBytes(state.calibration.tone.size)));
  if (state.calibration.tone.error && method === 'tone') {
    setTone(pill, 'danger', 'i-error');
    setRaw(text, 'Calibrator tone rejected');
  }

  // Profile readout
  const profile = activeProfile();
  setText($('cal-profile-pafs'), profile ? `${fmt(profile.paPerFS, 4)} Pa/FS` : null);
  setText($('cal-profile-method'), profile ? profile.method : null);
  setText($('cal-profile-date'), profile ? profile.verified : null);

  // Footer
  setText($('cal-level-unit'), levelUnitForMethod());
  const canSave = isNum(resolvedPaPerFS()) && (method === 'chain' || method === 'tone');
  setDisabled($('btn-cal-save-profile'), !canSave);
}

function wireCalibration() {
  qsa('input[name="cal-method"]').forEach(radio => {
    radio.addEventListener('change', () => {
      if (!radio.checked) return;
      state.calibration.method = radio.value;
      // Moving away from a method clears the errors it raised.
      ['cal-tone-level', 'cal-sensitivity', 'cal-gain', 'cal-fsv'].forEach(id => setFieldError(id, null));
      commit();
    });
  });

  ['cal-sensitivity', 'cal-gain', 'cal-fsv', 'cal-tone-freq'].forEach(id => {
    const input = $(id);
    if (!input) return;
    input.addEventListener('input', () => { validateNumericInput(id); commit(); });
  });

  const level = $('cal-tone-level');
  if (level) level.addEventListener('change', () => {
    setFieldError('cal-tone-level', null);
    commit();
  });

  const ack = $('cal-uncal-ack');
  if (ack) ack.addEventListener('change', commit);

  const profileSelect = $('cal-profile-select');
  if (profileSelect) profileSelect.addEventListener('change', () => {
    state.calibration.profileId = profileSelect.value;
    commit();
  });

  const clearTone = $('btn-cal-tone-clear');
  if (clearTone) clearTone.addEventListener('click', () => {
    state.calibration.tone = { name: null, path: null, size: null, uploading: false, error: null };
    const input = $('cal-tone-input');
    if (input) input.value = '';
    commit();
  });

  const save = $('btn-cal-save-profile');
  if (save) save.addEventListener('click', saveCurrentAsProfile);

  const manage = $('btn-manage-profiles');
  if (manage) manage.addEventListener('click', () => openProfiles(manage));
}

/* ---- test metadata ------------------------------------------------------ */

function metadataValues() {
  const out = {};
  for (const id of METADATA_IDS) {
    const el = $(id);
    out[id] = el ? el.value : '';
  }
  return out;
}

function missingRequiredMetadata() {
  const missing = [];
  for (const entry of REQUIRED_METADATA) {
    const el = $(entry.input);
    if (!el || String(el.value).trim() === '') missing.push(entry);
  }
  const configuration = $('md-configuration');
  const suppressor = $('md-suppressor');
  if (configuration && configuration.value === 'suppressed'
      && suppressor && suppressor.value.trim() === '') {
    missing.push({ input: 'md-suppressor', field: 'field-md-suppressor', name: 'Suppressor' });
  }
  return missing;
}

/**
 * Per-group completion. Derived from the DOM rather than from a second list,
 * so a field moved between groups cannot leave the counter lying.
 */
function renderFieldsetCounts(missingIds) {
  const form = $('metadata-form');
  if (!form) return;

  for (const fieldset of qsa('fieldset', form)) {
    const legend = qs('.fieldset-legend', fieldset);
    if (!legend) continue;

    const required = qsa('.field-req', fieldset)
      .map(mark => mark.closest('.field'))
      .filter(Boolean);
    if (required.length === 0) {
      const stale = qs('.fieldset-count', legend);
      if (stale) stale.remove();
      continue;
    }

    const outstanding = required.filter(field => {
      const input = qs('input, select, textarea', field);
      return input && missingIds.has(input.id);
    }).length;

    let count = qs('.fieldset-count', legend);
    if (!count) {
      count = document.createElement('span');
      count.className = 'fieldset-count';
      legend.appendChild(count);
    }
    const done = required.length - outstanding;
    count.textContent = `${done}/${required.length}`;
    count.dataset.state = outstanding === 0 ? 'complete' : 'outstanding';
    count.setAttribute('aria-label',
      `${done} of ${required.length} required fields complete in this group`);
  }
}

function renderMetadataCompleteness() {
  const missing = missingRequiredMetadata();
  const missingIds = new Set(missing.map(entry => entry.input));

  // A required marker escalates only once the operator has been past the field
  // and left it empty. Before that it is information, not a fault.
  for (const entry of REQUIRED_METADATA) {
    const field = $(entry.field);
    if (!field) continue;
    const outstanding = missingIds.has(entry.input) && field.dataset.touched === 'true';
    field.dataset.outstanding = outstanding ? 'true' : 'false';
  }
  renderFieldsetCounts(missingIds);

  setRaw($('metadata-missing-count'), String(missing.length));
  const wrapper = $('metadata-completeness');
  if (wrapper) {
    wrapper.textContent = '';
    const count = document.createElement('span');
    count.className = 'num';
    count.id = 'metadata-missing-count';
    count.textContent = String(missing.length);
    wrapper.append(count, document.createTextNode(
      missing.length === 1 ? ' required field still empty' : ' required fields still empty'));
  }

  // The suppressor requirement escalates with the configuration.
  const configuration = $('md-configuration');
  const req = $('md-suppressor-req');
  if (req && configuration) {
    const suppressed = configuration.value === 'suppressed';
    setRaw(req, suppressed ? 'Required' : 'Required when suppressed');
    req.classList.toggle('field-req', suppressed);
    req.classList.toggle('field-opt', !suppressed);
  }
}

function wireMetadata() {
  const form = $('metadata-form');
  if (!form) return;

  form.addEventListener('input', (ev) => {
    const id = ev.target.id;
    if (id && NUMERIC_INPUTS[id]) validateNumericInput(id);
    if (id && state.fieldErrors.has(id) && String(ev.target.value).trim() !== '' && !NUMERIC_INPUTS[id]) {
      setFieldError(id, null);
    }
    persistMetadata();
    commit();
  });
  form.addEventListener('change', () => { persistMetadata(); commit(); });
  form.addEventListener('submit', (ev) => ev.preventDefault());

  // "Touched" is what separates a field the operator has not reached yet from
  // one they looked at and left empty. Only the second deserves a warning.
  // Captured, because blur does not bubble.
  form.addEventListener('blur', (ev) => {
    const field = ev.target.closest ? ev.target.closest('.field') : null;
    if (!field) return;
    if (field.dataset.touched === 'true') return;
    field.dataset.touched = 'true';
    commit();
  }, true);
}

/** Mark every test-record field as seen — used when the flow leaves the step. */
function touchMetadataFields() {
  let changed = false;
  for (const entry of REQUIRED_METADATA) {
    const field = $(entry.field);
    if (!field || field.dataset.touched === 'true') continue;
    field.dataset.touched = 'true';
    changed = true;
  }
  if (changed) commit();
}

const persistMetadata = debounce(() => store.set(STORAGE.metadata, metadataValues()), 400);
const persistSettings = debounce(() => store.set(STORAGE.settings, settingsValues()), 400);

function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

/* ---- detection and analysis settings ------------------------------------ */

function settingsValues() {
  const out = {};
  for (const key of Object.keys(SETTING_DEFAULTS)) {
    if (key === 'threshold-mode') {
      const relative = $('threshold-mode-relative');
      out[key] = relative && relative.checked ? 'relative' : 'absolute';
      continue;
    }
    const el = $(key);
    if (!el) continue;
    out[key] = el.type === 'checkbox' ? el.checked : el.value;
  }
  return out;
}

function applySettings(values) {
  for (const [key, value] of Object.entries(values || {})) {
    if (key === 'threshold-mode') {
      const relative = $('threshold-mode-relative');
      const absolute = $('threshold-mode-absolute');
      if (relative && absolute) {
        relative.checked = value !== 'absolute';
        absolute.checked = value === 'absolute';
      }
      continue;
    }
    const el = $(key);
    if (!el) continue;
    if (el.type === 'checkbox') el.checked = Boolean(value);
    else el.value = String(value);
  }
}

function thresholdMode() {
  const absolute = $('threshold-mode-absolute');
  return absolute && absolute.checked ? 'absolute' : 'relative';
}

function renderSettingsDerived() {
  // Threshold unit follows the mode.
  const mode = thresholdMode();
  setRaw($('threshold-unit'), mode === 'absolute' ? 'dB SPL' : 'dB below peak');
  setRaw($('threshold-hint'), mode === 'absolute'
    ? 'An absolute level. Only meaningful on a calibrated recording.'
    : 'Range 1–90 dB below peak. Typical 15–25.');

  // An absolute threshold on an uncalibrated recording is a category error.
  const absolute = $('threshold-mode-absolute');
  if (absolute) {
    const uncalibrated = state.calibration.method === 'none';
    setDisabled(absolute, uncalibrated);
    if (uncalibrated && absolute.checked) {
      const relative = $('threshold-mode-relative');
      if (relative) relative.checked = true;
    }
  }

  // STFT resolution readout: sample rate is unknown until a recording is read,
  // so it is quoted per the source when we have one and left as a ratio if not.
  const nperseg = Number($('stft-nperseg') ? $('stft-nperseg').value : NaN);
  const sampleRate = state.results.payload
    && state.results.payload.metadata
    && state.results.payload.metadata.source
    && state.results.payload.metadata.source.sample_rate;
  const resolution = $('stft-resolution');
  if (isNum(nperseg) && nperseg > 0 && isNum(sampleRate)) setText(resolution, fmt(sampleRate / nperseg, 2));
  else if (isNum(nperseg) && nperseg > 0) setText(resolution, `fs ÷ ${nperseg}`);
  else setText(resolution, null);
}

function wireSettings() {
  const ids = Object.keys(SETTING_DEFAULTS).filter(k => k !== 'threshold-mode');
  for (const id of ids) {
    const el = $(id);
    if (!el) continue;
    const handler = () => {
      if (NUMERIC_INPUTS[id]) validateNumericInput(id);
      persistSettings();
      commit();
    };
    el.addEventListener('input', handler);
    el.addEventListener('change', handler);
  }
  qsa('input[name="threshold-mode"]').forEach(radio => {
    radio.addEventListener('change', () => { persistSettings(); commit(); });
  });

  const restore = $('btn-load-defaults');
  if (restore) restore.addEventListener('click', async () => {
    const yes = await confirmDialog({
      title: 'Restore defaults',
      acceptLabel: 'Restore',
      build: (body) => addParagraph(body,
        'Detection and analysis settings return to their factory values. The recording, '
        + 'the calibration choice and the test record are left alone.'),
    });
    if (!yes) return;
    applySettings(SETTING_DEFAULTS);
    clearFieldErrors();
    store.remove(STORAGE.settings);
    commit();
    toast({ title: 'Defaults restored', tone: 'ok' });
  });
}


/* ===========================================================================
   8. VALIDATION AND THE RUN CONFIGURATION
   =========================================================================== */

/**
 * Everything standing between the operator and a run, in the order it should
 * be fixed, each item attributed to the stop that owns it.
 *
 * The attribution is the point. A blocker with nowhere to go is a dead end:
 * the operator reads "Complete 4 required test-record fields" and then has to
 * hunt. Tagged with its step, the same blocker lights up a disc on the rail,
 * names itself on the bar, and the Continue button walks straight to it.
 *
 * @returns {{step: string, message: string}[]}
 */
function detailedBlockers() {
  const out = [];
  const add = (step, message) => out.push({ step, message });

  if (state.run.status === 'starting' || state.run.status === 'running' || state.run.status === 'cancelling') {
    add('run', 'An analysis is already running.');
    return out;
  }
  if (state.ws.status !== 'open') {
    add('run', state.ws.status === 'connecting'
      ? 'Connecting to the local analysis service…'
      : 'Not connected to the local analysis service.');
  }
  if (state.input.uploading) add('recording', 'The recording is still uploading.');
  else if (!state.input.path) add('recording', 'Select a recording.');
  else if (state.input.error) add('recording', `Recording: ${state.input.error}`);

  const method = state.calibration.method;
  if (method === '') {
    add('calibration', 'Choose a calibration method — there is no default.');
  } else if (method === 'tone') {
    if (state.calibration.tone.error) add('calibration', `Calibrator tone: ${state.calibration.tone.error}`);
    if (!state.calibration.tone.path) add('calibration', 'Add the calibrator tone recording.');
    const level = $('cal-tone-level');
    if (!level || level.value === '') add('calibration', 'State the calibrator level printed on its body.');
  } else if (method === 'chain') {
    const sens = readNumber('cal-sensitivity');
    const gain = readNumber('cal-gain');
    const fsv = readNumber('cal-fsv');
    if (sens.empty) add('calibration', 'Enter the microphone sensitivity.');
    if (gain.empty) add('calibration', 'Enter the preamplifier gain (0 if unity).');
    if (fsv.empty) add('calibration', 'Enter the ADC full-scale voltage.');
    if (!sens.ok && !sens.empty) add('calibration', sens.error);
    if (!gain.ok && !gain.empty) add('calibration', gain.error);
    if (!fsv.ok && !fsv.empty) add('calibration', fsv.error);
  } else if (method === 'profile') {
    if (!activeProfile()) add('calibration', 'Choose a saved calibration profile.');
  } else if (method === 'none') {
    const ack = $('cal-uncal-ack');
    if (!ack || !ack.checked) add('calibration', 'Acknowledge that the results will be relative levels.');
  }

  const missing = missingRequiredMetadata();
  if (missing.length > 0) {
    add('metadata', missing.length === 1
      ? `Complete the test record: ${missing[0].name}.`
      : `Complete ${missing.length} required test-record fields: ${missing.slice(0, 3).map(m => m.name).join(', ')}${missing.length > 3 ? '…' : ''}.`);
  }

  // Any numeric control that is present but wrong stops the run outright.
  for (const id of Object.keys(NUMERIC_INPUTS)) {
    const el = $(id);
    if (!el) continue;
    const result = readNumber(id);
    if (!result.ok && !result.empty) add(INPUT_STEP[id] || 'settings', result.error);
  }
  // Required detection numbers must not be blank either — a blank here used to
  // be silently replaced by a built-in default.
  for (const id of ['threshold-value', 'refractory-ms', 'pre-ms', 'post-ms', 'stft-overlap']) {
    const result = readNumber(id);
    if (result.empty) add(INPUT_STEP[id] || 'settings', `${NUMERIC_INPUTS[id].label} must have a value.`);
  }

  return out;
}

/** The same list, flattened, for the run banner and the disabled Run button. */
function runBlockers() {
  return detailedBlockers().map(b => b.message);
}

/** Blockers owned by one stop. */
function blockersForStep(step) {
  return detailedBlockers().filter(b => b.step === step).map(b => b.message);
}

function renderBlockers() {
  const blockers = runBlockers();
  const banner = $('run-blockers');
  const list = $('run-blockers-list');
  const runButton = $('btn-run');

  if (list) {
    list.textContent = '';
    for (const message of blockers) {
      const frag = fromTemplate('tpl-blocker');
      if (!frag) break;
      const item = frag.firstElementChild;
      item.dataset.blocker = message.slice(0, 40);
      item.textContent = message;
      list.appendChild(item);
    }
  }
  const busy = state.run.status === 'running' || state.run.status === 'starting' || state.run.status === 'cancelling';

  // "Not ready to run: an analysis is already running" is a true sentence and a
  // useless one — the progress card directly below it is already saying so, in
  // more detail, with a stage and a percentage. While a run is in flight the
  // banner has nothing to add and only competes with it.
  show(banner, blockers.length > 0 && !busy);
  setDisabled(runButton, blockers.length > 0);

  show($('btn-cancel-run'), busy);
  setDisabled($('btn-cancel-run'), state.run.status === 'cancelling');

  setRaw($('run-hint'), busy
    ? 'Cancelling stops the engine; whatever it had already written is partial and is not a result.'
    : 'Analysis runs locally; nothing leaves this machine.');
}

/**
 * Build the run configuration.
 *
 * Only fields the bridge knows are sent — it REJECTS unknown keys rather than
 * ignoring them, so an extra field here would fail the whole run. Presence is
 * tested explicitly so 0 and false survive, and no blank field is ever
 * replaced by a built-in constant.
 *
 * @returns {{ok: boolean, config: object|null, problems: string[]}}
 */
function buildRunConfig() {
  const problems = [];
  const config = {};

  if (!state.input.path) problems.push('No recording selected.');
  else config.filePath = state.input.path;

  // ---- detection ----
  const threshold = readNumber('threshold-value');
  if (threshold.ok && !threshold.empty) {
    if (thresholdMode() === 'absolute') config.thresholdDb = threshold.value;
    else config.thresholdRelativeDb = threshold.value;
  } else problems.push(threshold.error || 'Threshold must have a value.');

  for (const [id, key] of [['refractory-ms', 'refractoryMs'], ['pre-ms', 'preMs'], ['post-ms', 'postMs']]) {
    const value = readNumber(id);
    if (value.ok && !value.empty) config[key] = value.value;
    else problems.push(value.error || `${NUMERIC_INPUTS[id].label} must have a value.`);
  }

  // ---- spectral ----
  const nperseg = Number($('stft-nperseg') ? $('stft-nperseg').value : NaN);
  if (Number.isInteger(nperseg)) config.nperseg = nperseg;
  const overlap = readNumber('stft-overlap');
  if (overlap.ok && !overlap.empty) config.overlapFraction = overlap.value / 100;
  else problems.push(overlap.error || 'STFT overlap must have a value.');

  // ---- calibration: explicit, never defaulted ----
  switch (state.calibration.method) {
    case 'tone': {
      if (!state.calibration.tone.path) problems.push('No calibrator tone recording.');
      else config.calibratorTone = state.calibration.tone.path;
      const level = $('cal-tone-level');
      const levelValue = level ? Number(level.value) : NaN;
      if (Number.isFinite(levelValue)) config.calibratorLevelDb = levelValue;
      else problems.push('No calibrator level chosen.');
      const freq = readNumber('cal-tone-freq');
      if (freq.ok && !freq.empty) config.calibratorFreqHz = freq.value;
      break;
    }
    case 'chain': {
      const sens = readNumber('cal-sensitivity');
      const gain = readNumber('cal-gain');
      const fsv = readNumber('cal-fsv');
      if (sens.ok && !sens.empty) config.sensitivityMv = sens.value;
      else problems.push(sens.error || 'Microphone sensitivity is required.');
      if (gain.ok && !gain.empty) config.preampGainDb = gain.value;   // 0 is a value
      else problems.push(gain.error || 'Preamplifier gain is required.');
      if (fsv.ok && !fsv.empty) config.adcFullScaleV = fsv.value;
      else problems.push(fsv.error || 'ADC full scale is required.');
      break;
    }
    case 'profile': {
      const profile = activeProfile();
      if (profile && isNum(profile.paPerFS)) config.paPerFS = profile.paPerFS;
      else problems.push('No calibration profile chosen.');
      break;
    }
    case 'none':
      config.uncalibrated = true;
      break;
    default:
      problems.push('No calibration method chosen.');
  }

  const description = $('cal-description');
  const descriptionText = description ? description.value.trim() : '';
  if (descriptionText !== '') config.calDesc = descriptionText.slice(0, 200);
  else if (state.calibration.method === 'profile' && activeProfile()) {
    config.calDesc = `Profile: ${activeProfile().name}`.slice(0, 200);
  }

  // ---- output ----
  const formats = $('plot-format');
  if (formats && formats.value) config.formats = formats.value;
  const bands = $('opt-bands');
  if (bands && !bands.checked) config.noBands = true;
  const perShot = $('opt-per-shot-plots');
  if (perShot && !perShot.checked) config.noPerShot = true;

  const outputDir = $('setting-output-dir');
  if (outputDir && outputDir.value.trim() !== '') config.outputDir = outputDir.value.trim();

  // ---- test metadata ----
  const metadata = {};
  for (const [id, spec] of Object.entries(METADATA_TO_CONFIG)) {
    const el = $(id);
    if (!el) continue;
    const raw = String(el.value).trim();
    if (raw === '') continue;
    if (spec.type === 'number') {
      const parsed = readNumber(id);
      if (parsed.ok && !parsed.empty) metadata[spec.key] = parsed.value;
      else problems.push(parsed.error || `${id} is not a number.`);
    } else {
      metadata[spec.key] = raw;
    }
  }
  if (Object.keys(metadata).length > 0) config.metadata = metadata;

  return { ok: problems.length === 0, config: problems.length === 0 ? config : null, problems };
}

/* ===========================================================================
   9. TRANSPORT AND THE RUN LIFECYCLE

   The old UI could sit on "Running…" for ever: it handled only onmessage, and
   a dropped socket or a non-zero exit left the interface lying about what the
   backend was doing. Every socket state is handled here, reconnection is
   capped exponential backoff, and there is no path that leaves the UI claiming
   an analysis is in progress when it is not.
   =========================================================================== */

function wsStatusLabel() {
  switch (state.ws.status) {
    case 'open': return 'Connected';
    case 'connecting': return 'Connecting…';
    case 'closing': return 'Closing…';
    case 'error': return 'Connection error';
    case 'closed': return state.ws.attempts >= BACKOFF.maxAttempts ? 'Disconnected' : 'Reconnecting…';
    default: return 'Not connected';
  }
}

function backoffDelay(attempt) {
  const raw = Math.min(BACKOFF.max, BACKOFF.base * BACKOFF.factor ** attempt);
  const jitter = raw * BACKOFF.jitter * (Math.random() * 2 - 1);
  return Math.max(BACKOFF.base, Math.round(raw + jitter));
}

function connect({ manual = false } = {}) {
  const ws = state.ws;
  if (ws.socket && (ws.socket.readyState === WebSocket.OPEN || ws.socket.readyState === WebSocket.CONNECTING)) return;
  clearTimeout(ws.timer);
  ws.timer = null;
  if (manual) ws.attempts = 0;

  ws.status = 'connecting';
  commit();

  let socket;
  try {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${location.host}/ws`);
  } catch (err) {
    ws.status = 'error';
    ws.lastError = err.message;
    scheduleReconnect();
    commit();
    return;
  }
  ws.socket = socket;

  socket.addEventListener('open', () => {
    ws.status = 'open';
    ws.attempts = 0;
    ws.lastError = null;
    commit();
  });

  socket.addEventListener('message', (ev) => {
    let message;
    try { message = JSON.parse(ev.data); } catch {
      console.warn('unparseable message from the analysis service');
      return;
    }
    if (message && typeof message === 'object') handleServerMessage(message);
  });

  socket.addEventListener('error', () => {
    // The error event carries no detail by design; close follows it.
    ws.status = 'error';
    ws.lastError = 'The connection to the local analysis service failed.';
    commit();
  });

  socket.addEventListener('close', (ev) => {
    ws.socket = null;
    ws.status = 'closed';
    commit();

    // Closing the socket cancels the backend process server-side, so a drop
    // during a run has really ended that run. Say so rather than spinning.
    if (state.run.status === 'running' || state.run.status === 'starting' || state.run.status === 'cancelling') {
      state.run.status = 'lost';
      state.run.error = {
        code: 'connection-lost',
        message: `The connection to the local analysis service closed (${ev.code}${ev.reason ? `: ${ev.reason}` : ''}). `
          + 'The analysis was stopped — the service ends a run when its client disconnects.',
        stderr: null,
        exitCode: null,
      };
      state.run.finishedAt = Date.now();
      appendLog('Connection lost. The analysis was stopped.', 'error');
      commit();
      offerReattach();
    }
    scheduleReconnect();
  });
}

function scheduleReconnect() {
  const ws = state.ws;
  if (ws.timer !== null) return;
  if (ws.attempts >= BACKOFF.maxAttempts) {
    toast({
      title: 'Not connected',
      text: 'SASA has stopped trying to reach the local analysis service. Use Reconnect once it is running.',
      tone: 'danger',
    });
    commit();
    return;
  }
  const delay = backoffDelay(ws.attempts);
  ws.attempts += 1;
  ws.timer = setTimeout(() => { ws.timer = null; connect(); }, delay);
}

function sendMessage(payload) {
  const socket = state.ws.socket;
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  try { socket.send(JSON.stringify(payload)); return true; }
  catch (err) {
    console.error('send failed', err);
    return false;
  }
}

async function offerReattach() {
  const again = await confirmDialog({
    title: 'The analysis stopped',
    acceptLabel: 'Run again',
    acceptTone: 'primary',
    build: (body) => {
      addParagraph(body, state.run.error ? state.run.error.message : 'The connection dropped.');
      addParagraph(body,
        'Any output this run had already written is partial and must not be read as a result. '
        + 'Reconnect and start it again, or dismiss this and keep the settings for later.');
    },
  });
  if (!again) return;
  connect({ manual: true });
  // Give the socket a moment to come up before retrying the run.
  setTimeout(() => { if (state.ws.status === 'open') startRun(); }, 600);
}

/* ---- log ---------------------------------------------------------------- */

/* The log is an append-only sink, not derived state: rebuilding thousands of
   lines every frame would be pointless, so it is written here and trimmed. */
function appendLog(line, level = 'info') {
  const log = $('run-log');
  if (!log) return;
  const frag = fromTemplate('tpl-log-line');
  if (!frag) return;
  const node = frag.firstElementChild;
  node.dataset.level = level;
  node.textContent = line;

  const pinned = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
  log.appendChild(node);
  state.run.logCount += 1;

  while (log.childElementCount > MAX_LOG_LINES) log.removeChild(log.firstElementChild);
  if (pinned) log.scrollTop = log.scrollHeight;
}

function clearLog() {
  const log = $('run-log');
  if (log) log.textContent = '';
  state.run.logCount = 0;
}

function classifyLogLine(line, stream) {
  const text = String(line);
  if (/\b(error|failed|traceback|exception)\b/i.test(text)) return 'error';
  if (/\bwarn(ing)?\b/i.test(text)) return 'warn';
  if (stream === 'stderr') return 'warn';
  if (/\b(complete|written|ok)\b/i.test(text)) return 'ok';
  return 'info';
}

/* ---- run lifecycle ------------------------------------------------------ */

function startRun() {
  // Re-entry guard. #btn-run is disabled while a run is in flight, but that is
  // applied on the next animation frame; a second activation inside the same
  // frame (or a programmatic call) must not throw away the requestId of a run
  // that is still going, because every message about it would then be dropped
  // as "not mine" and the interface would sit on "Starting…" for ever.
  if (state.run.status === 'starting' || state.run.status === 'running' || state.run.status === 'cancelling') {
    announce('An analysis is already running.');
    return;
  }
  const blockers = runBlockers();
  if (blockers.length > 0) {
    announce(`Cannot run: ${blockers[0]}`);
    return;
  }
  const built = buildRunConfig();
  if (!built.ok) {
    toast({ title: 'Settings incomplete', text: built.problems[0], tone: 'danger' });
    return;
  }

  // The progress bar and the log live on the last step. A run started from the
  // palette or the R key can come from anywhere, and an operator who cannot
  // see the run has no way to tell it apart from a run that never started.
  setStep('run', { focus: false });

  const requestId = uid('run-');
  state.run = {
    status: 'starting',
    requestId,
    percent: 0,
    stage: 'Starting…',
    determinate: false,
    startedAt: Date.now(),
    finishedAt: null,
    outputDir: null,
    error: null,
    logCount: 0,
  };
  clearLog();
  clearFieldErrors();
  commit();

  const sent = sendMessage({ type: 'run-analysis', requestId, config: built.config });
  if (!sent) {
    state.run.status = 'error';
    state.run.error = { code: 'not-connected', message: 'The analysis could not be sent: the local service is not connected.', stderr: null, exitCode: null };
    commit();
    return;
  }
  appendLog(`Requested analysis of ${state.input.name}`, 'info');
  announce('Analysis started');
}

function cancelRun() {
  if (state.run.status !== 'running' && state.run.status !== 'starting') return;
  state.run.status = 'cancelling';
  state.run.stage = 'Cancelling…';
  commit();
  const sent = sendMessage({ type: 'cancel', requestId: state.run.requestId });
  if (!sent) {
    state.run.status = 'error';
    state.run.error = { code: 'not-connected', message: 'Cancellation could not be sent: the local service is not connected.', stderr: null, exitCode: null };
    commit();
  }
  announce('Cancelling the analysis');
}

function handleServerMessage(message) {
  const run = state.run;
  // A message for a run we are not tracking is noise from a previous socket.
  const mine = !message.requestId || message.requestId === run.requestId;

  switch (message.type) {
    case 'ready':
      state.ws.serverVersion = message.version || null;
      state.ws.protocolVersion = message.protocolVersion || null;
      if (message.version) {
        state.app.version = message.version;
        setRaw($('app-version'), message.version);
        setText($('about-version'), message.version);
      }
      commit();
      break;

    case 'started':
      if (!mine) break;
      run.status = 'running';
      run.stage = 'Reading the recording…';
      run.determinate = false;
      appendLog(`Analysis started at ${fmtTimestamp(message.startedAt)}`, 'ok');
      commit();
      break;

    case 'progress': {
      if (!mine) break;
      run.status = run.status === 'cancelling' ? 'cancelling' : 'running';
      if (isNum(message.percent)) {
        run.percent = clamp(message.percent, 0, 100);
        run.determinate = true;
      }
      if (message.message) run.stage = String(message.message);
      commit();
      break;
    }

    case 'log': {
      if (!mine) break;
      if (isNum(message.dropped) && message.dropped > 0) {
        appendLog(`[${message.dropped} log line(s) omitted to keep the interface responsive]`, 'warn');
      }
      if (message.line !== undefined && message.line !== null) {
        appendLog(String(message.line), classifyLogLine(message.line, message.stream));
      }
      break;
    }

    case 'cancelling':
      if (!mine) break;
      run.status = 'cancelling';
      run.stage = 'Cancelling…';
      commit();
      break;

    case 'cancelled':
      if (!mine) break;
      run.status = 'cancelled';
      run.finishedAt = Date.now();
      run.stage = 'Cancelled';
      appendLog('The analysis was cancelled. Anything the engine had already written is partial and must not be read as a result.', 'warn');
      commit();
      toast({ title: 'Analysis cancelled', tone: 'warn' });
      announce('Analysis cancelled');
      break;

    case 'complete': {
      if (!mine) break;
      // A "complete" without a readable output directory is a FAILURE. The old
      // UI showed it as a clean 100%.
      if (!message.outputDir) {
        run.status = 'error';
        run.finishedAt = Date.now();
        run.error = {
          code: 'output-missing',
          message: 'The analysis reported completion but named no readable results directory. Nothing can be shown.',
          stderr: null,
          exitCode: isNum(message.exitCode) ? message.exitCode : null,
        };
        appendLog('Completed without a results directory — treated as a failure.', 'error');
        commit();
        toast({ title: 'Analysis produced no results', text: run.error.message, tone: 'danger' });
        break;
      }
      run.status = 'complete';
      run.percent = 100;
      run.determinate = true;
      run.stage = 'Complete';
      run.finishedAt = Date.now();
      run.outputDir = message.outputDir;
      appendLog(`Complete in ${isNum(message.elapsedMs) ? (message.elapsedMs / 1000).toFixed(1) : '—'} s`, 'ok');
      commit();
      announce('Analysis complete');
      toast({ title: 'Analysis complete', text: basename(message.outputDir), tone: 'ok' });
      loadHistory();
      loadResults(message.outputDir, { navigate: state.prefs.openOnComplete });
      break;
    }

    case 'error': {
      // An error carrying no requestId is a protocol-level complaint about a
      // message, not a verdict on the running analysis. Reporting it as a run
      // failure would tell the operator a healthy run had died while the engine
      // carried on writing files behind the interface.
      if (!message.requestId) {
        appendLog(`Service error (${message.code || 'error'}): ${message.message || 'unspecified'}`, 'error');
        toast({ title: 'The local service rejected a message', text: message.message || '', tone: 'warn' });
        break;
      }
      if (!mine) break;
      const code = message.code || 'error';
      if (code === 'invalid-config') {
        applyServerFieldErrors(message.fields);
        run.status = 'error';
        run.finishedAt = Date.now();
        run.error = {
          code,
          message: 'The analysis settings were rejected by the local service. The offending fields are marked below.',
          stderr: null, exitCode: null,
        };
        appendLog('Settings rejected before the analysis started.', 'error');
        commit();
        toast({ title: 'Settings rejected', text: 'Fields marked in red need attention.', tone: 'danger' });
        break;
      }
      if (code === 'busy' || code === 'not-running') {
        toast({ title: 'Not possible right now', text: message.message || code, tone: 'warn' });
        if (code === 'not-running' && run.status === 'cancelling') {
          run.status = 'idle';
          commit();
        }
        // "busy" means this request was never accepted, so the run state that
        // startRun() optimistically created describes nothing. Leaving it at
        // "starting" would freeze the interface — the button stays disabled and
        // no message will ever arrive under this requestId.
        if (code === 'busy' && (run.status === 'starting' || run.status === 'running')) {
          run.status = 'error';
          run.finishedAt = Date.now();
          run.error = {
            code: 'busy',
            message: message.message
              || 'An analysis is already running on this connection. Wait for it to finish, or reconnect to stop it.',
            stderr: null,
            exitCode: null,
          };
          appendLog('The request was refused: an analysis is already running on this connection.', 'error');
          commit();
        }
        break;
      }
      run.status = 'error';
      run.finishedAt = Date.now();
      run.error = {
        code,
        message: message.message || 'The analysis failed.',
        stderr: message.stderr || null,
        exitCode: isNum(message.exitCode) ? message.exitCode : null,
        signal: message.signal || null,
        outputDir: message.outputDir || null,
      };
      appendLog(`ERROR (${code}): ${run.error.message}`, 'error');
      if (run.error.stderr) {
        appendLog('— stderr tail —', 'error');
        String(run.error.stderr).split(/\r?\n/).slice(-40).forEach(line => {
          if (line.trim() !== '') appendLog(line, 'error');
        });
      }
      commit();
      toast({ title: 'Analysis failed', text: run.error.message, tone: 'danger' });
      announce('Analysis failed');

      // "no shots" and "inadmissible" are verdicts: a directory still exists
      // and the operator needs to see why the run is unusable.
      if (message.outputDir) loadResults(message.outputDir, { navigate: false });
      break;
    }

    case 'pong':
      break;

    default:
      console.warn('unrecognised message type from the analysis service:', message.type);
  }
}

/** Land a server-side config rejection on the control that caused it. */
function applyServerFieldErrors(fields) {
  if (!Array.isArray(fields)) return;
  let unmapped = 0;
  for (const entry of fields) {
    if (!entry || typeof entry.field !== 'string') continue;

    /* The recording and the calibrator tone are not plain .field controls:
       they have their own purpose-built error lines, driven from state. Routing
       them through setFieldError would turn the control red and then let the
       next render blank the message — colour with no explanation. */
    if (entry.field === 'filePath') {
      state.input.error = entry.message || 'The local service will not read that recording.';
      appendLog(`Rejected: filePath — ${state.input.error}`, 'error');
      commit();
      continue;
    }
    if (entry.field === 'calibratorTone') {
      state.calibration.tone.error = entry.message || 'The local service will not read that calibrator recording.';
      appendLog(`Rejected: calibratorTone — ${state.calibration.tone.error}`, 'error');
      commit();
      continue;
    }

    let target = CONFIG_TO_FIELD[entry.field];
    if (!target && entry.field.startsWith('metadata.')) {
      const key = entry.field.slice('metadata.'.length);
      const id = Object.keys(METADATA_TO_CONFIG).find(k => METADATA_TO_CONFIG[k].key === key);
      if (id) target = { input: id };
    }
    if (target && target.input && $(target.input)) setFieldError(target.input, entry.message || 'Rejected by the analysis service.');
    else unmapped += 1;
    appendLog(`Rejected: ${entry.field} — ${entry.message || 'invalid'}`, 'error');
  }
  if (unmapped > 0) {
    toast({
      title: 'Some settings were rejected',
      text: `${unmapped} rejected field(s) have no control on this page; see the analysis log.`,
      tone: 'warn',
    });
  }
}

const RUN_PILL = {
  idle:       { tone: 'info',   icon: 'i-info',         text: 'Idle' },
  starting:   { tone: 'info',   icon: 'i-info',         text: 'Starting' },
  running:    { tone: 'info',   icon: 'i-info',         text: 'Running' },
  cancelling: { tone: 'warn',   icon: 'i-alert',        text: 'Cancelling' },
  cancelled:  { tone: 'warn',   icon: 'i-ban',          text: 'Cancelled' },
  complete:   { tone: 'ok',     icon: 'i-check-circle', text: 'Complete' },
  error:      { tone: 'danger', icon: 'i-error',        text: 'Failed' },
  lost:       { tone: 'danger', icon: 'i-error',        text: 'Connection lost' },
};

function renderRun() {
  const run = state.run;
  const card = $('run-progress-card');
  show(card, run.status !== 'idle');

  const pill = $('run-state-pill');
  const spec = RUN_PILL[run.status] || RUN_PILL.idle;
  setTone(pill, spec.tone, spec.icon);
  setRaw($('run-state-text'), spec.text);

  const progress = $('run-progress');
  const fill = $('run-progress-fill');
  if (progress) {
    if (run.status === 'complete') progress.setAttribute('data-state', 'done');
    else if (run.status === 'error' || run.status === 'lost' || run.status === 'cancelled') progress.setAttribute('data-state', 'error');
    else if (!run.determinate) progress.setAttribute('data-state', 'indeterminate');
    else progress.removeAttribute('data-state');
  }
  const percent = clamp(run.percent, 0, 100);
  if (fill) {
    fill.style.width = `${percent}%`;
    if (run.determinate) {
      fill.setAttribute('aria-valuenow', String(Math.round(percent)));
      fill.setAttribute('aria-valuetext', `${Math.round(percent)}% — ${run.stage || 'working'}`);
    } else {
      fill.removeAttribute('aria-valuenow');
      fill.setAttribute('aria-valuetext', run.stage || 'Working');
    }
  }
  setRaw($('run-progress-pct'), `${Math.round(percent)}%`);

  let stage = run.stage || '';
  if (run.status === 'error' && run.error) stage = run.error.message;
  if (run.status === 'lost' && run.error) stage = run.error.message;
  setRaw($('run-progress-stage'), stage || 'Working…');
}

function wireRun() {
  const runButton = $('btn-run');
  if (runButton) runButton.addEventListener('click', startRun);
  const cancel = $('btn-cancel-run');
  if (cancel) cancel.addEventListener('click', cancelRun);
}

/* ===========================================================================
   10. RESULTS
   =========================================================================== */

/* ---- accessors ---------------------------------------------------------- */

function meta() {
  return (state.results.payload && state.results.payload.metadata) || null;
}
function metaBlock(name) {
  const m = meta();
  return (m && m[name] && typeof m[name] === 'object') ? m[name] : {};
}
function shotList() {
  const m = meta();
  return Array.isArray(m && m.per_shot_metrics) ? m.per_shot_metrics : [];
}
function detectedShots() {
  const m = meta();
  return Array.isArray(m && m.shots) ? m.shots : [];
}
function shotEvent(shotNumber) {
  return detectedShots().find(s => s.shot_number === shotNumber) || null;
}

/** The level unit for THIS measurement. Never hard-code "dB SPL". */
function levelUnit() {
  return state.results.levelUnit || 'dB';
}

function unitFor(metric) {
  if (metric.unit === 'level') return levelUnit();
  if (metric.unit === 'us') return 'µs';
  return metric.unit;
}

/** A shot is excluded if the backend called it invalid or the operator did. */
function isRejected(shot) {
  return state.results.rejected.has(shot.shot_number) || shot.valid === false;
}
function includedShots() {
  return shotList().filter(shot => !isRejected(shot));
}

function artifactUrl(dir, file, sub) {
  const params = new URLSearchParams({ dir, file });
  if (sub) params.set('sub', sub);
  return `/api/image?${params.toString()}`;
}

/** Prefer a raster rendering; the HTML ones are for opening full size. */
function pickImage(keys, extension = 'png') {
  const payload = state.results.payload;
  if (!payload || !payload.images) return null;
  for (const key of keys) {
    const entry = payload.images[key];
    if (entry && entry[extension]) return entry[extension];
  }
  return null;
}

function setFigure(imgId, file, sub, altText) {
  const img = $(imgId);
  if (!img) return false;
  if (!file || !state.results.dir) {
    img.removeAttribute('src');
    show(img, false);
    return false;
  }
  const url = artifactUrl(state.results.dir, file, sub);
  if (img.getAttribute('src') !== url) img.setAttribute('src', url);
  if (altText) img.setAttribute('alt', altText);
  show(img, true);
  return true;
}

/* ---- loading ------------------------------------------------------------ */

async function loadResults(dir, { navigate = true } = {}) {
  if (!dir) return;
  resetChartState();
  state.results.status = 'loading';
  state.results.error = null;
  state.results.dir = dir;
  commit();
  if (navigate) setView('results');

  let response;
  try {
    response = await fetch(`/api/results?dir=${encodeURIComponent(dir)}`, { headers: { Accept: 'application/json' } });
  } catch (err) {
    // Never swallow this and navigate to a stale Results view.
    failResults(`The results could not be fetched: ${err.message}`);
    return;
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.message) detail = body.message;
    } catch { /* keep the status line */ }
    failResults(detail);
    return;
  }

  let payload;
  try { payload = await response.json(); } catch (err) {
    failResults(`The results file could not be read: ${err.message}`);
    return;
  }
  if (!payload || !payload.metadata) {
    failResults('That directory does not contain a completed analysis.');
    return;
  }

  state.results.payload = payload;
  state.results.dir = payload.outputDir || dir;
  state.results.levelUnit = (payload.metadata.calibration && payload.metadata.calibration.level_unit) || 'dB';
  state.results.status = 'loaded';
  state.results.shotIndex = 0;
  state.results.error = null;

  // Restore the operator's own rejections for this directory.
  const saved = store.get(STORAGE.rejects, {}) || {};
  const list = Array.isArray(saved[state.results.dir]) ? saved[state.results.dir] : [];
  state.results.rejected = new Set(list);

  // A tone calibration only yields its Pa/FS once the backend has derived it,
  // and ONLY the run the operator just started may write it back into the
  // Analyze form. Opening an archived analysis from History, or loading one
  // into Compare, used to overwrite the live calibration with that record's
  // factor — so the next run would be scaled by a number that came from a
  // different microphone on a different day, and the form would show it as
  // this session's calibration.
  const calibration = payload.metadata.calibration || {};
  const isOwnRun = Boolean(state.run.outputDir)
    && (state.results.dir === state.run.outputDir);
  if (isOwnRun && calibration.method
      && String(calibration.method).toLowerCase().includes('tone')) {
    state.calibration.derivedPaPerFS = isNum(calibration.Pa_per_FS) ? calibration.Pa_per_FS : null;
    state.calibration.derivedResidual = isNum(calibration.residual_dB) ? calibration.residual_dB : null;
  }

  setText($('about-commit'), (payload.metadata.software && payload.metadata.software.git_commit) || null);
  recomputeAggregate();
  commit();
  if (navigate) setView('results');
}

function failResults(message) {
  state.results.status = 'error';
  state.results.error = message;
  state.results.payload = null;
  commit();
  toast({ title: 'Results could not be loaded', text: message, tone: 'danger' });
  announce(`Results could not be loaded: ${message}`);
}

/* ---- client-side aggregate --------------------------------------------- */

/**
 * Recompute the displayed aggregate over the INCLUDED shots, so rejecting a
 * shot shows its effect immediately instead of waiting for a re-run.
 *
 * Mirrors metrics.py exactly: levels are energy-averaged with
 * 10*log10(mean(10^(L/10))), other quantities arithmetically, dispersion is
 * the sample standard deviation (ddof = 1) of the decibel values, and the 95 %
 * half-width is 1.96*sigma/sqrt(n).
 */
function recomputeAggregate() {
  const shots = includedShots();
  const statistics = {};
  for (const metric of METRICS) {
    const values = shots.map(shot => shot[metric.key]).filter(isNum);
    statistics[metric.stat] = { ...summarise(values, metric.level), name: metric.stat, unit: metric.unit };
  }

  // Mean band exposure, energy-averaged per band across the included shots.
  let bandFrequencies = [];
  let bandMean = [];
  const withBands = shots.filter(s => Array.isArray(s.band_exposure_dB) && s.band_exposure_dB.length > 0);
  if (withBands.length > 0) {
    const width = withBands[0].band_exposure_dB.length;
    if (withBands.every(s => s.band_exposure_dB.length === width)) {
      bandFrequencies = withBands[0].band_frequencies_Hz || [];
      bandMean = [];
      for (let i = 0; i < width; i += 1) {
        bandMean.push(energyAverageDb(withBands.map(s => s.band_exposure_dB[i])));
      }
    }
  }

  state.results.aggregate = {
    n_shots: shotList().length,
    n_valid: shots.length,
    statistics,
    band_frequencies_Hz: bandFrequencies,
    band_exposure_mean_dB: bandMean,
    band_exposure_std_dB: bandStats(withBands, 'std'),
    band_exposure_min_dB: bandStats(withBands, 'min'),
    band_exposure_max_dB: bandStats(withBands, 'max'),
    hazard: recomputeHazard(shots),
  };
}

function bandStats(shots, which) {
  if (shots.length === 0) return [];
  const width = shots[0].band_exposure_dB.length;
  if (!shots.every(s => s.band_exposure_dB.length === width)) return [];
  const out = [];
  for (let i = 0; i < width; i += 1) {
    const column = shots.map(s => s.band_exposure_dB[i]).filter(isNum);
    if (column.length === 0) { out.push(NaN); continue; }
    if (which === 'std') out.push(sampleStd(column));
    else if (which === 'min') out.push(Math.min(...column));
    else out.push(Math.max(...column));
  }
  return out;
}

/**
 * Hearing hazard, recomputed for the rounds-per-day and criterion the operator
 * has set, over the included shots only. Mirrors metrics.py::compute_hazard.
 */
function recomputeHazard(shots) {
  const lae = shots.map(s => s.LAE_dB).filter(isNum);
  const criterionInput = $('hazard-criterion');
  const criterion = criterionInput ? Number(criterionInput.value) : 85;
  const roundsRead = readNumber('hazard-rounds');
  const rounds = (roundsRead.ok && !roundsRead.empty) ? roundsRead.value : lae.length;

  if (lae.length === 0 || !isNum(criterion) || !isNum(rounds) || rounds <= 0) {
    return {
      n_rounds: rounds || 0, LAE_mean_dB: NaN, LAeq8h_dB: NaN,
      criterion_dB: criterion, dose_percent: NaN, allowable_rounds: NaN,
      exceeds_limit: false, n_shots_used: lae.length,
      method: 'Energy-based LAeq8h, 3 dB exchange rate (MIL-STD-1474E / NIOSH)',
    };
  }
  const mean = energyAverageDb(lae);
  const laeq8h = mean + 10 * Math.log10(rounds) - 10 * Math.log10(EIGHT_HOURS_S);
  const dose = 100 * 10 ** ((laeq8h - criterion) / 10);
  const allowable = 10 ** ((criterion - mean + 10 * Math.log10(EIGHT_HOURS_S)) / 10);
  return {
    n_rounds: rounds,
    LAE_mean_dB: mean,
    LAeq8h_dB: laeq8h,
    criterion_dB: criterion,
    dose_percent: dose,
    allowable_rounds: allowable,
    exceeds_limit: laeq8h > criterion,
    n_shots_used: lae.length,
    method: 'Energy-based LAeq8h, 3 dB exchange rate (MIL-STD-1474E / NIOSH)',
  };
}

/** Statistics for one metric: the client recompute, or the record's own. */
function statsFor(metric) {
  const live = state.results.aggregate && state.results.aggregate.statistics[metric.stat];
  if (live && live.n > 0) return live;
  const recorded = metaBlock('aggregate').statistics || {};
  return recorded[metric.stat] || recorded[metric.key] || null;
}

/* ---- validity ----------------------------------------------------------- */

/**
 * The validity block is rendered FIRST and derived from quality{}, detection{}
 * and calibration{}. An inadmissible measurement has to be unmistakable, and
 * must never be dressed up as a clean result.
 */
function validityChecks() {
  const quality = metaBlock('quality');
  const detection = metaBlock('detection');
  const calibration = metaBlock('calibration');
  const source = metaBlock('source');
  const shots = shotList();
  const checks = [];

  const push = (id, name, value, stateName) => checks.push({ id, name, value, state: stateName });

  // Calibration
  if (calibration.calibrated === true) {
    push('calibration', 'Calibration', `${calibration.method || 'calibrated'} · ${fmt(calibration.Pa_per_FS, 4)} Pa/FS`, 'pass');
  } else {
    push('calibration', 'Calibration', `Uncalibrated — levels are ${calibration.level_unit || 'dB re FS'}`, 'warn');
  }

  // Clipping
  if (quality.is_clipped) {
    push('clipping', 'Clipping', `${fmtInt(quality.clipped_samples)} samples in ${fmtInt(quality.clipped_runs)} run(s)`, 'fail');
  } else {
    push('clipping', 'Clipping', 'None detected', 'pass');
  }

  // Headroom
  if (isNum(quality.headroom_dB)) {
    const headroom = quality.headroom_dB;
    push('headroom', 'Headroom', `${fmt(headroom, 1)} dB below full scale`,
      headroom < 1 ? 'fail' : headroom < 6 ? 'warn' : 'pass');
  }

  // Signal-to-noise
  if (isNum(quality.snr_dB)) {
    push('snr', 'Signal-to-noise', `${fmt(quality.snr_dB, 1)} dB`,
      quality.snr_dB < 20 ? 'fail' : quality.snr_dB < 40 ? 'warn' : 'pass');
  }

  // Sample rate
  if (isNum(source.sample_rate)) {
    push('samplerate', 'Sample rate', `${fmt(source.sample_rate / 1000, 1)} kHz`,
      quality.sample_rate_adequate === false ? 'fail' : 'pass');
  }

  // Detection
  const detected = isNum(detection.n_detected) ? detection.n_detected : shots.length;
  push('detection', 'Shots detected',
    `${fmtInt(detected)} of ${fmtInt(detection.n_candidates)} candidates at ${fmt(detection.threshold_dB, 1)} dB (${detection.threshold_mode || '—'})`,
    detected === 0 ? 'fail' : 'pass');

  if (isNum(detection.n_suppressed_by_refractory) && detection.n_suppressed_by_refractory > 0) {
    push('refractory', 'Refractory suppressions',
      `${fmtInt(detection.n_suppressed_by_refractory)} candidate(s) discarded as too close together`, 'warn');
  }

  // Truncated windows
  const truncated = shots.filter(s => s.window_truncated).length;
  if (shots.length > 0) {
    push('truncation', 'Analysis windows',
      truncated === 0 ? 'None truncated' : `${truncated} of ${shots.length} truncated — B-duration is a lower bound`,
      truncated === 0 ? 'pass' : 'warn');
  }

  // Per-shot clipping
  const clippedShots = shots.filter(s => s.clipped).length;
  if (clippedShots > 0) {
    push('shot-clipping', 'Clipped shots', `${clippedShots} of ${shots.length} saturated`, 'fail');
  }

  // DC offset
  if (isNum(quality.dc_offset_FS) && Math.abs(quality.dc_offset_FS) > 0.001) {
    push('dc', 'DC offset', `${fmt(quality.dc_offset_FS * 100, 3)} % of full scale`,
      Math.abs(quality.dc_offset_FS) > 0.01 ? 'warn' : 'pass');
  }

  // Wind / low-frequency energy
  if (isNum(quality.lf_energy_fraction)) {
    push('lf', 'Low-frequency energy', `${fmt(quality.lf_energy_fraction * 100, 1)} % below 20 Hz`,
      quality.lf_energy_fraction > 0.5 ? 'warn' : 'pass');
  }

  // Noise floor, informational
  if (isNum(quality.noise_floor_dB)) {
    push('noise', 'Noise floor', `${fmt(quality.noise_floor_dB, 1)} ${levelUnit()}`, 'pass');
  }

  return checks;
}

/**
 * Name a failing or cautioning check the way a technician needs to read it.
 *
 * The check's name alone is not a reason: "headroom." tells an operator nothing
 * about what is wrong or how close to the edge they are. The value carries the
 * measurement that tripped it, so both are stated.
 */
function describeCheck(check) {
  // The NAME is prose and lowercases safely inside a sentence. The VALUE is not:
  // it carries units, and "db" is not a unit — "dB" is. Casing in a measured
  // quantity is part of the quantity, so the value is passed through untouched.
  const name = String(check.name || '').toLowerCase();
  const value = String(check.value || '').trim();
  if (!value) return name;
  return `${name} (${value})`;
}

/**
 * Reasons are joined into one sentence with semicolons, so each has to arrive
 * without its own terminator. Engine warnings are written as full sentences
 * and check descriptions are not, which is how "...the muzzle blast.." got
 * onto the verdict banner.
 */
function joinReasons(reasons, limit = 4) {
  const cleaned = reasons
    .map(reason => String(reason).trim().replace(/[.;]+$/, ''))
    .filter(Boolean);
  if (cleaned.length === 0) return '';
  const shown = cleaned.slice(0, limit).join('; ');
  const rest = cleaned.length - limit;
  return `${shown}${rest > 0 ? `; and ${rest} more` : ''}.`;
}

function renderValidity() {
  const quality = metaBlock('quality');
  const calibration = metaBlock('calibration');
  const recorded = metaBlock('validity');
  const checks = validityChecks();

  const errors = Array.isArray(quality.errors) ? quality.errors : [];
  const warnings = [
    ...(Array.isArray(quality.warnings) ? quality.warnings : []),
    ...(Array.isArray(metaBlock('detection').warnings) ? metaBlock('detection').warnings : []),
    ...(Array.isArray(meta() && meta().warnings) ? meta().warnings : []),
  ];

  const failed = checks.filter(c => c.state === 'fail');
  const warned = checks.filter(c => c.state === 'warn');
  const noShots = shotList().length === 0;
  const inadmissible = failed.length > 0 || errors.length > 0 || noShots
    || quality.is_valid === false || recorded.measurement_valid === false;

  const banner = $('validity-banner');
  const pill = $('validity-pill');
  let tone;
  let verdict;
  let headline;
  let text;

  if (inadmissible) {
    tone = 'danger';
    verdict = 'Not admissible';
    headline = 'This measurement must not be reported as a result';
    const reasons = [
      ...(noShots ? ['no shots were detected'] : []),
      ...failed.map(describeCheck),
      ...errors,
    ];
    text = `${joinReasons(reasons)} Fix the recording or the setup and measure again.`;
  } else if (warned.length > 0 || warnings.length > 0) {
    tone = 'warn';
    verdict = 'Admissible with cautions';
    headline = 'Usable, but qualify the result';
    text = joinReasons([...warned.map(describeCheck), ...warnings]);
  } else {
    tone = 'ok';
    verdict = 'Admissible';
    headline = 'Measurement passes every automatic check';
    text = `${shotList().length} shot(s) analysed; every quality check passed.`;
  }

  if (calibration.calibrated === false) {
    text += ` Levels are ${calibration.level_unit || 'dB re FS'} — they are not sound pressure levels and no hazard figure is defensible.`;
  }

  setTone(banner, tone);
  setIcon($('validity-icon'), TONE_ICON[tone]);
  setTone(pill, tone);
  setRaw($('validity-verdict'), verdict);
  setRaw($('validity-headline'), headline);
  setRaw($('validity-text'), text);
  announceValidity(verdict);

  // Individual checks
  const grid = $('validity-checks');
  if (grid) {
    grid.textContent = '';
    const icons = { pass: 'i-check-circle', warn: 'i-alert', fail: 'i-error', unknown: 'i-info' };
    for (const check of checks) {
      const frag = fromTemplate('tpl-check-item');
      if (!frag) break;
      const node = frag.firstElementChild;
      node.dataset.check = check.id;
      node.setAttribute('data-state', check.state);
      setIcon(slot(node, 'icon'), icons[check.state] || 'i-info');
      setRaw(slot(node, 'name'), check.name);
      setRaw(slot(node, 'value'), check.value);
      grid.appendChild(node);
    }
  }

  const hasDetail = errors.length > 0 || warnings.length > 0;
  show($('validity-actions'), hasDetail);
  fillList($('validity-errors-list'), errors, 'No errors were recorded.');
  fillList($('validity-warnings-list'), warnings, 'No warnings were recorded.');
}

let lastAnnouncedVerdict = null;
function announceValidity(verdict) {
  if (verdict === lastAnnouncedVerdict) return;
  lastAnnouncedVerdict = verdict;
  announce(`Measurement validity: ${verdict}`);
}

function fillList(list, items, emptyText) {
  if (!list) return;
  list.textContent = '';
  const values = Array.isArray(items) && items.length > 0 ? items : [emptyText];
  for (const value of values) {
    const li = document.createElement('li');
    li.className = 'banner-text';
    li.textContent = String(value);
    list.appendChild(li);
  }
}

/* ---- metric tiles ------------------------------------------------------- */

function weightingAllowed(metric) {
  if (metric.weight === 'A') { const el = $('opt-weight-a'); return !el || el.checked; }
  if (metric.weight === 'C') { const el = $('opt-weight-c'); return !el || el.checked; }
  return true;
}

/**
 * A tile shows the central tendency AND its dispersion: mean, the 95 %
 * half-width, sigma, the range and n. A mean on its own is not a measurement.
 */
function metricTile(metric, stats, { tone = null } = {}) {
  const frag = fromTemplate('tpl-metric-tile');
  if (!frag) return null;
  const node = frag.firstElementChild;
  node.dataset.metric = metric.key;
  if (tone) node.setAttribute('data-tone', tone);

  const label = slot(node, 'label');
  setLabel(label, metric.label);
  label.setAttribute('title', metric.plain);

  const hasValue = stats && isNum(stats.mean);
  setRaw(slot(node, 'value'), hasValue ? fmt(stats.mean, metric.digits) : '—');
  if (!hasValue) node.setAttribute('data-state', 'empty');
  setRaw(slot(node, 'unit'), unitFor(metric));

  let caption;
  if (!stats || !isNum(stats.mean)) {
    caption = 'Not available for the included shots';
  } else if (stats.n <= 1) {
    caption = `n = ${stats.n} · no dispersion from a single shot`;
  } else {
    caption = `95% CI ±${fmt(stats.ci95_half_width, metric.digits)}`
      + ` · σ ${fmt(stats.std, metric.digits)}`
      + ` · ${fmt(stats.min, metric.digits)}–${fmt(stats.max, metric.digits)}`
      + ` · n ${stats.n}`;
  }
  setRaw(slot(node, 'caption'), caption);
  return node;
}

function renderHeadlineMetrics() {
  const grid = $('results-metrics');
  if (!grid) return;
  grid.textContent = '';

  const headline = METRICS
    .filter(m => m.headline > 0 && weightingAllowed(m))
    .sort((a, b) => a.headline - b.headline);

  for (const metric of headline) {
    const stats = statsFor(metric);
    const node = metricTile(metric, stats);
    if (node) grid.appendChild(node);
  }
  if (grid.childElementCount === 0) {
    addParagraph(grid, 'No metrics are available: every shot has been excluded.', 'prose');
  }
}

/* ---- overview ----------------------------------------------------------- */

function renderOverview() {
  const source = metaBlock('source');
  const calibration = metaBlock('calibration');
  const analysis = metaBlock('analysis');
  const software = metaBlock('software');
  const detection = metaBlock('detection');

  setText($('src-file'), basename(analysis.input_file || source.path));
  setText($('src-samplerate'), isNum(source.sample_rate) ? `${fmt(source.sample_rate / 1000, 1)} kHz` : null);
  setText($('src-channels'), isNum(source.channels) ? String(source.channels) : null);
  setText($('src-subtype'), source.subtype);
  setText($('src-duration'), fmtDuration(source.duration_s));

  const calibrated = calibration.calibrated === true;
  setTone($('cal-result-pill'), calibrated ? 'ok' : 'warn');
  setRaw($('cal-result-text'), calibrated ? 'Calibrated' : 'Uncalibrated');
  setText($('cal-out-unit'), calibration.level_unit);
  setText($('cal-out-pafs'), isNum(calibration.Pa_per_FS) ? `${fmt(calibration.Pa_per_FS, 4)} Pa/FS` : null);
  setText($('cal-out-method'), calibration.method);
  setText($('cal-out-fullscale'), isNum(calibration.full_scale_dB)
    ? `${fmt(calibration.full_scale_dB, 1)} ${calibration.level_unit || 'dB'}` : null);
  setText($('cal-out-residual'), isNum(calibration.residual_dB) ? `${fmt(calibration.residual_dB, 2)} dB` : 'Not applicable');

  setText($('prov-software'), software.name ? `${software.name} ${software.version || ''}`.trim() : null);
  setText($('prov-commit'), software.git_commit);
  setText($('prov-timestamp'), fmtTimestamp(analysis.timestamp));
  setText($('prov-sha256'), analysis.input_sha256);
  setText($('prov-elapsed'), fmtDuration(analysis.elapsed_s));
  setText($('prov-schema'), meta() && meta().schema_version);

  // Waveform
  const waveform = pickImage(['waveform_full', 'waveform']);
  const drawn = setFigure('img-waveform', waveform, null,
    'Waveform of the recording with each detected shot marked.');
  setText($('caption-waveform'), drawn
    ? `${fmtInt(detection.n_detected)} shot(s) detected at ${fmt(detection.threshold_dB, 1)} dB `
      + `(${detection.threshold_mode || 'mode unknown'}); peak ${fmt(detection.peak_level_dB, 1)} ${levelUnit()}, `
      + `noise floor ${fmt(detection.noise_floor_dB, 1)} ${levelUnit()}.`
    : 'No waveform figure was produced for this analysis.');

  const openFullSize = $('btn-overview-open');
  const interactive = pickImage(['waveform_full'], 'html');
  setDisabled(openFullSize, !waveform && !interactive);

  // Test record
  renderTestMetadata();
}

const TEST_METADATA_LABELS = [
  ['operator', 'Operator'], ['date', 'Test date'], ['location', 'Location'],
  ['configuration', 'Configuration'], ['weapon', 'Weapon'],
  ['barrel_length_in', 'Barrel length (in)'], ['ammunition', 'Ammunition'],
  ['suppressor', 'Suppressor'], ['mic_model', 'Microphone'], ['mic_serial', 'Serial'],
  ['mic_distance_m', 'Distance (m)'], ['mic_angle_deg', 'Angle (deg)'],
  ['mic_height_m', 'Height (m)'], ['temperature_C', 'Temperature (°C)'],
  ['humidity_pct', 'Humidity (%)'], ['pressure_kPa', 'Pressure (kPa)'],
  ['wind_mps', 'Wind (m/s)'], ['notes', 'Notes'],
];

function renderTestMetadata() {
  const list = $('test-metadata-list');
  if (!list) return;
  list.textContent = '';
  const record = metaBlock('test_metadata');
  let written = 0;
  for (const [key, label] of TEST_METADATA_LABELS) {
    const value = record[key];
    if (value === undefined || value === null || value === '') continue;
    const frag = fromTemplate('tpl-kv-row');
    if (!frag) break;
    setRaw(slot(frag, 'key'), label);
    setRaw(slot(frag, 'value'), String(value));
    list.appendChild(frag);
    written += 1;
  }
  if (written === 0) {
    const frag = fromTemplate('tpl-kv-row');
    if (frag) {
      setRaw(slot(frag, 'key'), 'Test record');
      setRaw(slot(frag, 'value'), 'Empty — this measurement is not attributable');
      list.appendChild(frag);
    }
  }
}

/* ---- spectrogram -------------------------------------------------------- */

function renderSpectrogramPanel() {
  const weightingSelect = $('spectrogram-weighting');
  const shotSelect = $('spectrogram-shot');
  const weighting = weightingSelect ? weightingSelect.value : 'Z';

  // The backend renders Z- and C-weighted spectrograms only. Rather than show
  // an option that silently falls back, the unavailable one is disabled.
  if (weightingSelect) {
    for (const option of Array.from(weightingSelect.options)) {
      const available = option.value === 'Z'
        ? Boolean(pickImage(['spectrogram_z_full', 'spectrogram_z']))
        : option.value === 'C'
          ? Boolean(pickImage(['spectrogram_c_full', 'spectrogram_c']))
          : false;
      option.disabled = !available;
      if (option.value === 'A') option.textContent = 'A-weighted — not produced by the engine';
    }
    if (weightingSelect.selectedOptions[0] && weightingSelect.selectedOptions[0].disabled) {
      weightingSelect.value = pickImage(['spectrogram_z_full']) ? 'Z' : 'C';
    }
  }

  // Shot list
  if (shotSelect) {
    const wanted = shotList().map(s => String(s.shot_number));
    const present = Array.from(shotSelect.options).slice(1).map(o => o.value);
    if (wanted.join(',') !== present.join(',')) {
      const current = shotSelect.value;
      while (shotSelect.options.length > 1) shotSelect.remove(1);
      for (const shot of shotList()) {
        const option = document.createElement('option');
        option.value = String(shot.shot_number);
        option.textContent = `Shot ${shot.shot_number}`;
        shotSelect.appendChild(option);
      }
      shotSelect.value = wanted.includes(current) ? current : 'all';
    }
  }

  const selectedShot = shotSelect ? shotSelect.value : 'all';
  if (selectedShot !== 'all') {
    const file = shotImageFor(Number(selectedShot));
    const drawn = setFigure('img-spectrogram', file, 'shots',
      `Summary plot for shot ${selectedShot}, including its spectrograms.`);
    setText($('caption-spectrogram'), drawn
      ? `Shot ${selectedShot} summary. The engine renders per-shot spectrograms inside the shot summary figure, `
        + `not as a separate weighted image; the weighting control above applies to the whole-recording view.`
      : `No summary figure was produced for shot ${selectedShot}.`);
    return;
  }

  const key = (weightingSelect ? weightingSelect.value : 'Z') === 'C'
    ? ['spectrogram_c_full', 'spectrogram_c']
    : ['spectrogram_z_full', 'spectrogram_z'];
  const file = pickImage(key);
  const drawn = setFigure('img-spectrogram', file, null,
    `${weighting}-weighted spectrogram of the recording.`);
  const source = metaBlock('source');
  const settings = metaBlock('settings');
  setText($('caption-spectrogram'), drawn
    ? `${weighting}-weighted, ${fmtInt(settings.nperseg)}-point window`
      + `${isNum(source.sample_rate) && isNum(settings.nperseg) ? ` (${fmt(source.sample_rate / settings.nperseg, 1)} Hz per bin)` : ''}`
      + `, levels in ${levelUnit()}.`
    : 'No spectrogram was produced for this analysis.');

  // Last, so it can take the panel over when the matrix is available. Called
  // earlier, everything above would overwrite the canvas's caption and put the
  // engine's picture back on top of it.
  drawSpectrogramChart();
}

function shotImageFor(shotNumber) {
  const payload = state.results.payload;
  if (!payload || !Array.isArray(payload.shotImages)) return null;
  const padded = String(shotNumber).padStart(2, '0');
  return payload.shotImages.find(name => name.includes(`shot_${padded}_`))
    || payload.shotImages.find(name => name.includes(`shot_${shotNumber}_`))
    || null;
}

/* ---- bands -------------------------------------------------------------- */

function bandRange() {
  const low = readNumber('band-low');
  const high = readNumber('band-high');
  return {
    low: low.ok && !low.empty ? low.value : 0,
    high: high.ok && !high.empty ? high.value : Infinity,
  };
}

function renderBandsPanel() {
  const drawn = setFigure('img-bands', pickImage(['bands_full', 'bands']), null,
    'Mean one-third-octave band exposure level against frequency.');
  const aggregate = state.results.aggregate;
  const frequencies = (aggregate && aggregate.band_frequencies_Hz) || [];
  setText($('caption-bands'), drawn
    ? `Energy-averaged over the ${aggregate ? aggregate.n_valid : 0} included shot(s); levels in ${levelUnit()}.`
    : (frequencies.length > 0
      ? 'No band figure was produced, but the band data below is from the record.'
      : 'Band analysis was not performed for this recording.'));

  const tbody = $('bands-tbody');
  if (!tbody) return;
  tbody.textContent = '';
  if (!aggregate || frequencies.length === 0) return;

  const range = bandRange();
  for (let i = 0; i < frequencies.length; i += 1) {
    const frequency = frequencies[i];
    if (frequency < range.low || frequency > range.high) continue;
    const frag = fromTemplate('tpl-bands-row');
    if (!frag) break;
    const row = frag.firstElementChild;
    row.dataset.band = String(frequency);
    setRaw(slot(row, 'frequency'), fmtHz(frequency));
    setRaw(slot(row, 'mean'), fmt(aggregate.band_exposure_mean_dB[i], 1));
    setRaw(slot(row, 'std'), fmt(aggregate.band_exposure_std_dB[i], 1));
    setRaw(slot(row, 'min'), fmt(aggregate.band_exposure_min_dB[i], 1));
    setRaw(slot(row, 'max'), fmt(aggregate.band_exposure_max_dB[i], 1));
    tbody.appendChild(row);
  }

  drawBandsChart();
}

/* ---- per-shot review ---------------------------------------------------- */

function currentShot() {
  const shots = shotList();
  if (shots.length === 0) return null;
  const index = clamp(state.results.shotIndex, 0, shots.length - 1);
  return shots[index] || null;
}

function currentShotNumber() {
  const shot = currentShot();
  return shot ? shot.shot_number : null;
}

function selectShot(index, { focusStrip = false } = {}) {
  const shots = shotList();
  if (shots.length === 0) return;
  state.results.shotIndex = clamp(index, 0, shots.length - 1);
  commit();
  const shot = currentShot();
  if (shot) announce(`Shot ${shot.shot_number} of ${shots.length}`);
  if (focusStrip) {
    // shot_number comes out of a metadata file: escape it rather than letting
    // an unexpected value throw a selector syntax error out of the renderer.
    const strip = $('shot-strip');
    const wanted = String(shot.shot_number);
    const chip = strip
      ? qsa('.shot-chip[data-shot]', strip).find(node => node.dataset.shot === wanted)
      : null;
    if (chip) chip.focus();
  }
}

function stepShot(delta) {
  if (state.view !== 'results') return;
  if (shotList().length === 0) return;
  if (state.results.tab !== 'shots') setTab('shots');
  selectShot(state.results.shotIndex + delta);
}

function toggleReject(shotNumber) {
  if (shotNumber === null || shotNumber === undefined) return;
  const shot = shotList().find(s => s.shot_number === shotNumber);
  if (!shot) return;
  if (shot.valid === false && !state.results.rejected.has(shotNumber)) {
    toast({
      title: `Shot ${shotNumber} is already excluded`,
      text: 'The engine marked it invalid; it never entered the aggregate.',
      tone: 'info',
    });
    return;
  }
  if (state.results.rejected.has(shotNumber)) state.results.rejected.delete(shotNumber);
  else state.results.rejected.add(shotNumber);

  persistRejections();
  recomputeAggregate();
  commit();
  drawShotBandChart();
  const excluded = state.results.rejected.has(shotNumber);
  announce(`Shot ${shotNumber} ${excluded ? 'rejected' : 'restored'}. `
    + `${includedShots().length} of ${shotList().length} shots now included.`);
}

function persistRejections() {
  const saved = store.get(STORAGE.rejects, {}) || {};
  if (state.results.rejected.size === 0) delete saved[state.results.dir];
  else saved[state.results.dir] = Array.from(state.results.rejected);
  store.set(STORAGE.rejects, saved);
}

function shotFlag(shot) {
  const minSnr = readNumber('min-snr');
  const threshold = (minSnr.ok && !minSnr.empty) ? minSnr.value : null;
  if (state.results.rejected.has(shot.shot_number)) {
    return { state: 'rejected', label: 'Rejected', icon: 'i-ban', tone: 'info', flag: null };
  }
  if (shot.valid === false || shot.clipped) {
    return {
      state: 'invalid',
      label: shot.clipped ? 'Clipped' : 'Invalid',
      icon: 'i-error', tone: 'danger', flag: 'danger',
    };
  }
  if (shot.window_truncated || (threshold !== null && isNum(shot.snr_dB) && shot.snr_dB < threshold)) {
    return {
      state: 'warn',
      label: shot.window_truncated ? 'Truncated' : 'Low SNR',
      icon: 'i-alert', tone: 'warn', flag: 'warn',
    };
  }
  return { state: 'valid', label: 'Valid', icon: 'i-check-circle', tone: 'ok', flag: null };
}

function renderShotsPanel() {
  const shots = shotList();
  const shot = currentShot();

  setRaw($('tab-shots-count'), String(shots.length));
  setRaw($('shot-position'), shots.length === 0 ? '— / —'
    : `${state.results.shotIndex + 1} / ${shots.length}`);
  setDisabled($('btn-shot-prev'), state.results.shotIndex <= 0);
  setDisabled($('btn-shot-next'), state.results.shotIndex >= shots.length - 1);

  // Jump-to select
  const select = $('shot-select');
  if (select) {
    const wanted = shots.map(s => String(s.shot_number));
    const present = Array.from(select.options).map(o => o.value);
    if (wanted.join(',') !== present.join(',')) {
      select.textContent = '';
      for (const item of shots) {
        const option = document.createElement('option');
        option.value = String(item.shot_number);
        option.textContent = `Shot ${item.shot_number}`;
        select.appendChild(option);
      }
    }
    if (shot && select.value !== String(shot.shot_number)) select.value = String(shot.shot_number);
    setDisabled(select, shots.length === 0);
  }

  // Flag pill and the reject/restore pair
  const flag = shot ? shotFlag(shot) : { tone: 'info', label: 'No shot', icon: 'i-info', state: 'unknown' };
  setTone($('shot-flag-pill'), flag.tone, flag.icon);
  setRaw($('shot-flag-text'), flag.label);
  const rejected = shot ? state.results.rejected.has(shot.shot_number) : false;
  show($('btn-shot-reject'), Boolean(shot) && !rejected);
  show($('btn-shot-restore'), Boolean(shot) && rejected);

  // Strip
  const strip = $('shot-strip');
  if (strip) {
    strip.textContent = '';
    for (const item of shots) {
      const frag = fromTemplate('tpl-shot-chip');
      if (!frag) break;
      const chip = frag.firstElementChild;
      const itemFlag = shotFlag(item);
      chip.dataset.shot = String(item.shot_number);
      chip.dataset.rejected = String(isRejected(item));
      if (itemFlag.flag) chip.dataset.flag = itemFlag.flag;
      chip.setAttribute('aria-current', shot && item.shot_number === shot.shot_number ? 'true' : 'false');
      chip.tabIndex = shot && item.shot_number === shot.shot_number ? 0 : -1;
      setIcon(slot(chip, 'icon'), itemFlag.icon);
      setRaw(slot(chip, 'label'), String(item.shot_number));
      chip.setAttribute('aria-label', `Shot ${item.shot_number}, ${itemFlag.label}`);
      strip.appendChild(chip);
    }
  }

  // Aggregate status
  const included = includedShots().length;
  const manual = state.results.rejected.size;
  const engineInvalid = shots.filter(s => s.valid === false).length;
  const status = $('aggregate-status');
  if (status) {
    status.textContent = '';
    const count = document.createElement('span');
    count.className = 'num';
    count.id = 'aggregate-included';
    count.textContent = String(included);
    status.append(
      document.createTextNode('Aggregates include '),
      count,
      document.createTextNode(` of ${shots.length} shot(s)`
        + (engineInvalid > 0 ? `; ${engineInvalid} excluded by the engine` : '')
        + (manual > 0 ? `; ${manual} rejected here` : '')
        + '. Every figure on this page is recomputed from the included shots.'),
    );
  }

  // Per-shot metric tiles
  const grid = $('shot-metrics');
  if (grid) {
    grid.textContent = '';
    if (!shot) {
      addParagraph(grid, 'No shots were detected in this recording.', 'prose');
    } else {
      const event = shotEvent(shot.shot_number);
      const tiles = [
        ['Time', isNum(event && event.time_s) ? fmt(event.time_s, 3) : '—', 's', 'Position in the recording'],
        ...METRICS.filter(m => m.headline > 0 && weightingAllowed(m)).map(metric => [
          metric.plain, fmt(shot[metric.key], metric.digits), unitFor(metric),
          statsFor(metric) && isNum(statsFor(metric).mean)
            ? `Aggregate mean ${fmt(statsFor(metric).mean, metric.digits)}` : '',
          metric,
        ]),
        ['SNR', fmt(shot.snr_dB, 1), 'dB', 'Shot against the local noise floor'],
        ['Rise time resolved', shot.rise_time_resolved === false ? 'No' : 'Yes', '',
          shot.rise_time_resolved === false ? 'Sample rate too low to resolve the rise' : 'Rise resolved at this sample rate'],
      ];
      for (const [label, value, unit, caption, metric] of tiles) {
        const frag = fromTemplate('tpl-metric-tile');
        if (!frag) break;
        const node = frag.firstElementChild;
        node.dataset.metric = metric ? metric.key : label;
        if (metric) setLabel(slot(node, 'label'), metric.label);
        else setRaw(slot(node, 'label'), label);
        slot(node, 'label').setAttribute('title', label);
        setRaw(slot(node, 'value'), value);
        setRaw(slot(node, 'unit'), unit);
        setRaw(slot(node, 'caption'), caption || '');

        // Deviation of this shot from the aggregate mean.
        if (metric) {
          const stats = statsFor(metric);
          if (stats && isNum(stats.mean) && isNum(shot[metric.key])) {
            const delta = shot[metric.key] - stats.mean;
            const deltaNode = slot(node, 'delta');
            const direction = Math.abs(delta) < 0.05 ? 'flat' : (delta > 0 ? 'up' : 'down');
            deltaNode.setAttribute('data-direction', direction);
            setIcon(slot(node, 'delta-icon'),
              direction === 'up' ? 'i-arrow-up' : direction === 'down' ? 'i-arrow-down' : 'i-minus');
            setRaw(slot(node, 'delta-value'), `${fmtSigned(delta, metric.digits)} vs mean`);
            show(deltaNode, true);
          }
        }
        grid.appendChild(node);
      }

      // Shot notes — the engine's own reasons for flagging it.
      const notes = Array.isArray(shot.notes) ? shot.notes : [];
      if (notes.length > 0) {
        const paragraph = addParagraph(grid, `Engine notes: ${notes.join('; ')}`, 'prose');
        paragraph.setAttribute('data-shot', String(shot.shot_number));
      }
    }
  }

  // Shot summary figure
  const file = shot ? shotImageFor(shot.shot_number) : null;
  const drawn = setFigure('img-shot-detail', file, 'shots',
    shot ? `Summary plot for shot ${shot.shot_number}.` : 'No shot selected.');
  setText($('caption-shot-detail'), drawn
    ? `Shot ${shot.shot_number}: pressure history, Z- and C-weighted spectrograms and the derived metrics, in ${levelUnit()}.`
    : (shot ? 'No summary figure was produced for this shot.' : 'No shot selected.'));

  drawShotBandChart();

  drawShotWaveChart();
  drawShotLevelsChart();
}

function wireShotNav() {
  const prev = $('btn-shot-prev');
  const next = $('btn-shot-next');
  if (prev) prev.addEventListener('click', () => selectShot(state.results.shotIndex - 1));
  if (next) next.addEventListener('click', () => selectShot(state.results.shotIndex + 1));

  const select = $('shot-select');
  if (select) select.addEventListener('change', () => {
    const index = shotList().findIndex(s => String(s.shot_number) === select.value);
    if (index >= 0) selectShot(index);
  });

  const reject = $('btn-shot-reject');
  const restore = $('btn-shot-restore');
  if (reject) reject.addEventListener('click', () => toggleReject(currentShotNumber()));
  if (restore) restore.addEventListener('click', () => toggleReject(currentShotNumber()));

  const strip = $('shot-strip');
  if (strip) {
    strip.addEventListener('click', (ev) => {
      const chip = ev.target.closest('.shot-chip');
      if (!chip) return;
      const index = shotList().findIndex(s => String(s.shot_number) === chip.dataset.shot);
      if (index >= 0) selectShot(index);
    });
    // Roving focus across the strip.
    strip.addEventListener('keydown', (ev) => {
      let delta = 0;
      if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') delta = 1;
      else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') delta = -1;
      else if (ev.key === 'Home') { ev.preventDefault(); selectShot(0, { focusStrip: true }); return; }
      else if (ev.key === 'End') { ev.preventDefault(); selectShot(shotList().length - 1, { focusStrip: true }); return; }
      else return;
      ev.preventDefault();
      selectShot(state.results.shotIndex + delta, { focusStrip: true });
    });
  }
}

/* ---- metrics table ------------------------------------------------------ */

const TABLE_SLOTS = [
  'time_s', 'Lpeak_Z_dB', 'Lpeak_A_dB', 'Lpeak_C_dB', 'LAE_dB', 'LZE_dB',
  'a_duration_ms', 'b_duration_ms', 'rise_time_us', 'specific_impulse_Pa_s',
  'crest_factor_dB', 'spectral_centroid_Hz', 'snr_dB',
];

function renderMetricsTable() {
  const tbody = $('metrics-tbody');
  if (!tbody) return;
  tbody.textContent = '';

  const filter = state.results.filter.trim().toLowerCase();
  const showRejected = state.results.showRejected;

  for (const shot of shotList()) {
    const rejected = isRejected(shot);
    if (!showRejected && rejected) continue;

    const flag = shotFlag(shot);
    const event = shotEvent(shot.shot_number);
    const haystack = [
      `shot ${shot.shot_number}`, flag.label,
      ...(Array.isArray(shot.notes) ? shot.notes : []),
    ].join(' ').toLowerCase();
    if (filter && !haystack.includes(filter)) continue;

    const frag = fromTemplate('tpl-metrics-row');
    if (!frag) break;
    const row = frag.firstElementChild;
    row.dataset.shot = String(shot.shot_number);
    row.dataset.rejected = String(rejected);

    setRaw(slot(row, 'shot-number'), String(shot.shot_number));
    const rowFlag = row.querySelector('.row-flag');
    if (rowFlag) rowFlag.setAttribute('data-state', flag.state);
    setIcon(slot(row, 'flag-icon'), flag.icon);
    setRaw(slot(row, 'flag-label'), flag.label);

    // One key loop: the template's slot names are the per_shot_metrics keys.
    for (const key of TABLE_SLOTS) {
      const target = slot(row, key);
      if (!target) continue;
      if (key === 'time_s') {
        setRaw(target, isNum(event && event.time_s) ? fmt(event.time_s, 3) : '—');
        continue;
      }
      const metric = METRIC_BY_KEY.get(key);
      setRaw(target, fmt(shot[key], metric ? metric.digits : 1));
    }

    const notes = Array.isArray(shot.notes) ? shot.notes : [];
    if (notes.length > 0) row.setAttribute('title', notes.join('; '));

    const toggle = row.querySelector('[data-action="toggle-reject"]');
    if (toggle) {
      toggle.setAttribute('aria-label',
        rejected ? `Restore shot ${shot.shot_number}` : `Reject shot ${shot.shot_number}`);
      setIcon(toggle, rejected ? 'i-restore' : 'i-ban');
      setDisabled(toggle, shot.valid === false && !state.results.rejected.has(shot.shot_number));
    }
    const view = row.querySelector('[data-action="view-shot"]');
    if (view) view.setAttribute('aria-label', `View shot ${shot.shot_number}`);

    tbody.appendChild(row);
  }

  if (tbody.childElementCount === 0) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 16;
    cell.textContent = shotList().length === 0
      ? 'No shots were detected in this recording.'
      : 'No shots match the current filter.';
    row.appendChild(cell);
    tbody.appendChild(row);
  }
}

function wireMetricsTable() {
  const tbody = $('metrics-tbody');
  if (tbody) {
    tbody.addEventListener('click', (ev) => {
      const button = ev.target.closest('[data-action]');
      if (!button) return;
      const row = button.closest('tr[data-shot]');
      if (!row) return;
      const shotNumber = Number(row.dataset.shot);
      if (button.dataset.action === 'toggle-reject') toggleReject(shotNumber);
      else if (button.dataset.action === 'view-shot') {
        const index = shotList().findIndex(s => s.shot_number === shotNumber);
        setTab('shots');
        if (index >= 0) selectShot(index);
      }
    });
  }

  const filter = $('metrics-filter');
  if (filter) filter.addEventListener('input', () => {
    state.results.filter = filter.value;
    commit();
  });

  const showRejected = $('metrics-show-rejected');
  if (showRejected) showRejected.addEventListener('change', () => {
    state.results.showRejected = showRejected.checked;
    commit();
  });

  const csv = $('btn-metrics-csv');
  if (csv) csv.addEventListener('click', () => downloadMetricsCsv());
}

function downloadMetricsCsv() {
  const shots = shotList();
  if (shots.length === 0) {
    toast({ title: 'Nothing to export', text: 'No shots were detected.', tone: 'warn' });
    return;
  }
  const header = ['shot_number', 'status', 'included', 'time_s', ...METRICS.map(m => m.key), 'notes'];
  const rows = [header];
  for (const shot of shots) {
    const event = shotEvent(shot.shot_number);
    rows.push([
      shot.shot_number,
      shotFlag(shot).label,
      isRejected(shot) ? 'no' : 'yes',
      isNum(event && event.time_s) ? event.time_s : '',
      ...METRICS.map(m => (isNum(shot[m.key]) ? shot[m.key] : '')),
      Array.isArray(shot.notes) ? shot.notes.join('; ') : '',
    ]);
  }
  rows.push([]);
  rows.push([`level_unit`, levelUnit()]);
  rows.push([`included_shots`, includedShots().length, 'of', shots.length]);
  downloadText(`${basename(state.results.dir)}-per-shot.csv`, csvRows(rows));
  toast({ title: 'CSV exported', text: 'Per-shot metrics for the shots shown.', tone: 'ok' });
}

/* ---- hazard ------------------------------------------------------------- */

function renderHazardPanel() {
  const hazard = (state.results.aggregate && state.results.aggregate.hazard) || null;
  const recorded = metaBlock('aggregate').hazard || null;
  const calibration = metaBlock('calibration');
  const uncalibrated = calibration.calibrated === false;

  const banner = $('hazard-banner');
  if (uncalibrated) {
    setTone(banner, 'danger', 'i-ban');
    setRaw($('hazard-headline'), 'No hazard figure can be given');
    setRaw($('hazard-text'),
      `This measurement is uncalibrated: levels are ${calibration.level_unit || 'dB re FS'}, not sound `
      + 'pressure levels. A dose computed from them would be meaningless.');
  } else if (!hazard || !isNum(hazard.LAeq8h_dB)) {
    setTone(banner, 'info', 'i-info');
    setRaw($('hazard-headline'), 'Hearing hazard assessment');
    setRaw($('hazard-text'), 'No A-weighted exposure is available for the included shots.');
  } else if (hazard.exceeds_limit) {
    setTone(banner, 'danger', 'i-error');
    setRaw($('hazard-headline'), 'Daily exposure exceeds the criterion');
    setRaw($('hazard-text'),
      `${fmtInt(hazard.n_rounds)} rounds per day give L${'Aeq,8h'} ${fmt(hazard.LAeq8h_dB, 1)} dB against a `
      + `${fmt(hazard.criterion_dB, 0)} dB criterion — ${fmt(hazard.dose_percent, 0)} % of the daily dose. `
      + `Unprotected exposure should stop at about ${fmtInt(hazard.allowable_rounds)} rounds.`);
  } else {
    setTone(banner, 'ok', 'i-check-circle');
    setRaw($('hazard-headline'), 'Daily exposure is within the criterion');
    setRaw($('hazard-text'),
      `${fmtInt(hazard.n_rounds)} rounds per day give ${fmt(hazard.LAeq8h_dB, 1)} dB against a `
      + `${fmt(hazard.criterion_dB, 0)} dB criterion — ${fmt(hazard.dose_percent, 0)} % of the daily dose. `
      + `The allowance is about ${fmtInt(hazard.allowable_rounds)} rounds.`);
  }

  const grid = $('hazard-metrics');
  if (grid) {
    grid.textContent = '';
    if (!uncalibrated && hazard) {
      const tiles = [
        ['LAeq,8h', fmt(hazard.LAeq8h_dB, 1), 'dB', `Criterion ${fmt(hazard.criterion_dB, 0)} dB`, hazard.exceeds_limit ? 'danger' : 'ok'],
        ['Daily dose', fmt(hazard.dose_percent, 0), '%', '100 % is the criterion', hazard.exceeds_limit ? 'danger' : 'ok'],
        ['Allowable rounds', fmtInt(hazard.allowable_rounds), 'per day', 'Unprotected', 'info'],
        ['Mean LAE', fmt(hazard.LAE_mean_dB, 1), levelUnit(), `Energy mean of ${hazard.n_shots_used} shot(s)`, 'info'],
      ];
      for (const [label, value, unit, caption, tone] of tiles) {
        const frag = fromTemplate('tpl-metric-tile');
        if (!frag) break;
        const node = frag.firstElementChild;
        node.dataset.metric = label;
        node.setAttribute('data-tone', tone);
        setRaw(slot(node, 'label'), label);
        setRaw(slot(node, 'value'), value);
        setRaw(slot(node, 'unit'), unit);
        setRaw(slot(node, 'caption'), caption);
        grid.appendChild(node);
      }
    }
  }

  const method = hazard ? hazard.method : (recorded && recorded.method);
  const recomputed = hazard && recorded
    && (hazard.n_rounds !== recorded.n_rounds || hazard.criterion_dB !== recorded.criterion_dB
      || hazard.n_shots_used !== recorded.n_rounds);
  setText($('hazard-method'), method
    ? `${method}${recomputed ? ' — recomputed in this view for the rounds and criterion set on the Analyze page' : ''}`
    : null);
  setText($('hazard-criterion-out'), hazard ? `${fmt(hazard.criterion_dB, 0)} dB LAeq,8h` : null);
  setText($('hazard-rounds-out'), hazard
    ? `${fmtInt(hazard.n_rounds)} per working day, from ${hazard.n_shots_used} included shot(s)` : null);
  setText($('hazard-lae-mean'), hazard && isNum(hazard.LAE_mean_dB)
    ? `${fmt(hazard.LAE_mean_dB, 1)} ${levelUnit()}` : null);

  renderAhaahPanel();
}

/**
 * The second metric MIL-STD-1474E approves, and the one customers ask for by
 * name. It is computed on every analysis and it declines to emit a number.
 *
 * This card exists so that the refusal REACHES the operator. Before it, the
 * model ran nowhere and the interface was simply silent on the ARU, which
 * reads as "not implemented" rather than as "implemented, and here is the
 * argument for why it will not give you a figure". The distinction is the
 * whole point: an ARU produced from four unspecified parameters would be an
 * authoritative-looking number that is wrong.
 */
function renderAhaahPanel() {
  const card = $('card-ahaah');
  if (!card) return;
  const block = metaBlock('ahaah');
  const present = block && Object.keys(block).length > 0;
  show(card, Boolean(present));
  if (!present) return;

  const unwarned = block.unwarned || null;
  // valid is never true in this build; branch on it anyway so the day the
  // model validates, this panel reports the number instead of the refusal.
  const available = Boolean(unwarned && unwarned.valid);

  setTone($('ahaah-pill'), available ? 'ok' : 'warn');
  setRaw($('ahaah-pill-text'), available ? 'Available' : 'No number');

  setRaw($('ahaah-headline'), block.attempted
    ? (block.headline || 'AHAAH unavailable')
    : `Not run — ${block.reason || 'no reason recorded'}`);

  const notes = $('ahaah-notes');
  if (notes) {
    notes.textContent = '';
    for (const note of (block.notes || []).slice(0, 4)) {
      const item = document.createElement('li');
      item.className = 'finding';
      const body = document.createElement('div');
      body.className = 'finding-body';
      const message = document.createElement('p');
      message.className = 'finding-message';
      message.textContent = note;
      body.appendChild(message);
      item.appendChild(body);
      notes.appendChild(item);
    }
    show(notes, (block.notes || []).length > 0);
  }

  const detail = $('ahaah-detail');
  if (!detail) return;
  detail.textContent = '';
  const rows = [
    ['Standing', block.validation_status === 'not_validated'
      ? 'Not validated against the ARL reference case'
      : String(block.validation_status || '—')],
    ['Shot submitted', block.shot_number === null || block.shot_number === undefined
      ? '—' : `Shot ${block.shot_number} — the loudest valid impulse`],
  ];
  if (unwarned) {
    if (isNum(unwarned.peak_pressure_dB)) {
      rows.push(['Peak submitted', `${fmt(unwarned.peak_pressure_dB, 1)} dB`]);
    }
    if (isNum(unwarned.category_c_count)) {
      rows.push(['Unspecified parameters',
        `${fmtInt(unwarned.category_c_count)} choices ARL's public release does not define`]);
    }
    if (unwarned.man_coe_md5) rows.push(['man.coe checksum', String(unwarned.man_coe_md5)]);
  }
  rows.push(['Metric in use', `A-weighted energy (MIL-STD-1474E), shown above`]);

  for (const [key, value] of rows) {
    const dt = document.createElement('dt');
    dt.textContent = key;
    const dd = document.createElement('dd');
    dd.textContent = value;
    detail.append(dt, dd);
  }
}

/* ---- results dispatcher ------------------------------------------------- */

/* ═══════════════════════════════════════════════════════════════════════════
   STRING ANALYSIS PANEL

   Everything the string says about itself beyond its average: whether the first
   round popped, what the average costs with and without it, whether the string
   drifted, and which shots a technician should look at before any of it is
   reported. Every verdict here is one the Python side already decided; the
   renderer states it and never re-derives it, so the UI and the record on disk
   can never disagree.
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * Put a card's leading rail into the tone of the verdict it carries.
 *
 * The rail is the one ornament in this interface, so it must always mean
 * something. A card whose verdict is neutral gets no rail at all rather than a
 * grey one, because a grey rail still reads as a status.
 */
function setCardTone(id, tone) {
  const card = $(id);
  if (!card) return;
  if (tone) card.setAttribute('data-tone', tone);
  else card.removeAttribute('data-tone');
}

/** The level unit this analysis earned: real dB SPL, or dB re FS. */
function levelUnit() {
  return metaBlock('calibration').level_unit || 'dB';
}

/** One entry in a findings list, built without innerHTML. */
function findingRow(identifier, message, severity, amount) {
  const li = document.createElement('li');
  li.className = 'finding';
  li.setAttribute('data-severity', severity || 'info');

  const id = document.createElement('span');
  id.className = 'finding-id';
  id.textContent = identifier;
  li.appendChild(id);

  const body = document.createElement('div');
  body.className = 'finding-body';

  const text = document.createElement('p');
  text.className = 'finding-message';
  text.textContent = message;
  body.appendChild(text);

  if (amount) {
    const quantity = document.createElement('span');
    quantity.className = 'finding-amount';
    quantity.textContent = amount;
    body.appendChild(quantity);
  }

  li.appendChild(body);
  return li;
}

function renderFirstRoundPop(stats) {
  const pop = stats && stats.first_round_pop;
  const unit = levelUnit();
  const pill = $('frp-pill');
  const note = $('frp-note');

  if (!pop || pop.refusal) {
    setCardTone('card-frp', 'info');
    setTone(pill, 'info', 'i-info');
    setText($('frp-pill-text'), 'Not measured');
    ['frp-first', 'frp-rest', 'frp-interval', 'frp-p', 'frp-basis']
      .forEach(id => setText($(id), null));
    if (pop && pop.refusal) {
      setText(note, `Not measured: ${pop.refusal}`);
      note.className = 'note note-info';
      show(note, true);
    } else {
      show(note, false);
    }
    return;
  }

  if (pop.established) {
    setCardTone('card-frp', 'danger');
    setTone(pill, 'danger', 'i-alert');
    setText($('frp-pill-text'), `Pop established · ${fmtSigned(pop.observed_dB, 2)} dB`);
  } else if (pop.first_shot_quieter) {
    setCardTone('card-frp', 'warn');
    setTone(pill, 'warn', 'i-alert');
    setText($('frp-pill-text'), 'First round QUIETER than the string');
  } else {
    setCardTone('card-frp', 'ok');
    setTone(pill, 'ok', 'i-check-circle');
    setText($('frp-pill-text'), 'No pop this measurement can resolve');
  }

  setText($('frp-first'), isNum(pop.first_shot_dB) ? `${fmt(pop.first_shot_dB, 1)} ${unit}` : null);
  setText($('frp-rest'), isNum(pop.subsequent_mean_dB)
    ? `${fmt(pop.subsequent_mean_dB, 1)} ${unit} over ${pop.n_subsequent} shots (σ ${fmt(pop.subsequent_sd_dB, 2)})`
    : null);
  setText($('frp-interval'), (isNum(pop.prediction_lower_dB) && isNum(pop.prediction_upper_dB))
    ? `${fmt(pop.prediction_lower_dB, 1)} to ${fmt(pop.prediction_upper_dB, 1)} ${unit}`
    : null);
  setText($('frp-p'), isNum(pop.p_value) ? pop.p_value.toFixed(4) : null);
  setText($('frp-basis'), pop.basis === 'across-strings'
    ? `Across ${pop.n_strings} strings`
    : 'One string — a single observation');

  if (pop.first_shot_quieter) {
    setText(note, 'The first round was quieter than the rest of the string explains. '
      + 'That is not first-round pop; check for a squib, a misfire or a detection error.');
    note.className = 'note note-warn';
    show(note, true);
  } else if (Array.isArray(pop.notes) && pop.notes.length) {
    setText(note, pop.notes.join(' '));
    note.className = 'note note-info';
    show(note, true);
  } else {
    show(note, false);
  }
}

function renderStringMeans(breakdown) {
  const body = $('string-means-tbody');
  if (!body) return;
  while (body.firstChild) body.removeChild(body.firstChild);

  const unit = levelUnit();
  const order = ['Lpeak_Z', 'Lpeak_A', 'LAE'];
  const labels = { Lpeak_Z: 'Peak, Z-weighted', Lpeak_A: 'Peak, A-weighted', LAE: 'Sound exposure' };

  order.forEach(key => {
    const stats = breakdown[key];
    if (!stats) return;
    const row = document.createElement('tr');

    const name = document.createElement('th');
    name.setAttribute('scope', 'row');
    name.textContent = labels[key] || key;
    row.appendChild(name);

    [
      isNum(stats.energy_mean_dB) ? `${fmt(stats.energy_mean_dB, 1)} ${unit}` : '—',
      isNum(stats.energy_mean_excluding_first_dB)
        ? `${fmt(stats.energy_mean_excluding_first_dB, 1)} ${unit}` : '—',
      isNum(stats.first_round_cost_dB) ? `${fmtSigned(stats.first_round_cost_dB, 2)} dB` : '—',
    ].forEach(value => {
      const cell = document.createElement('td');
      cell.className = 'num';
      cell.textContent = value;
      row.appendChild(cell);
    });

    body.appendChild(row);
  });
}

function renderStringDrift(stats) {
  const unit = levelUnit();
  const pill = $('drift-pill');

  if (!stats || !isNum(stats.trend_dB_per_shot)) {
    setCardTone('card-drift', 'info');
    setTone(pill, 'info', 'i-info');
    setText($('drift-pill-text'), 'Not tested');
    setText($('drift-slope'), null);
    setText($('drift-p'), null);
  } else if (stats.trend_established) {
    setCardTone('card-drift', 'warn');
    setTone(pill, 'warn', 'i-alert');
    setText($('drift-pill-text'), 'Drift established');
    setText($('drift-slope'), `${fmtSigned(stats.trend_dB_per_shot, 3)} dB per shot`);
    setText($('drift-p'), `p = ${Number(stats.trend_p_value).toFixed(3)}`);
  } else {
    setCardTone('card-drift', 'ok');
    setTone(pill, 'ok', 'i-check-circle');
    setText($('drift-pill-text'), 'No drift established');
    setText($('drift-slope'), `${fmtSigned(stats.trend_dB_per_shot, 3)} dB per shot`);
    setText($('drift-p'), `p = ${Number(stats.trend_p_value).toFixed(3)}`);
  }

  if (stats && isNum(stats.min_dB) && isNum(stats.max_dB)) {
    setText($('string-range'),
      `${fmt(stats.min_dB, 1)} to ${fmt(stats.max_dB, 1)} ${unit} (range ${fmt(stats.range_dB, 2)} dB)`);
  } else {
    setText($('string-range'), null);
  }

  const percentiles = stats && stats.percentiles_dB;
  if (percentiles && Object.keys(percentiles).length) {
    setText($('string-percentiles'), Object.keys(percentiles)
      .sort((a, b) => Number(a) - Number(b))
      .map(k => `p${k} ${fmt(percentiles[k], 1)}`)
      .join('  ·  '));
  } else {
    setText($('string-percentiles'), null);
  }
}

function renderShotReview() {
  const review = metaBlock('shot_review');
  const list = $('review-list');
  const pill = $('review-pill');
  const counter = $('tab-string-flags');
  if (!list) return;

  while (list.firstChild) list.removeChild(list.firstChild);

  setText($('review-sensitivity'), review.sensitivity || '');

  const flags = Array.isArray(review.flags) ? review.flags : [];
  const actionable = flags.filter(f => f.severity === 'exclude' || f.severity === 'review');
  const excluded = Array.isArray(review.shots_to_exclude) ? review.shots_to_exclude : [];

  if (!actionable.length) {
    setCardTone('card-review', 'ok');
    setTone(pill, 'ok', 'i-check-circle');
    setText($('review-pill-text'), 'No shot departs from the string');
    show(counter, false);
  } else {
    const tone = excluded.length ? 'danger' : 'warn';
    setCardTone('card-review', tone);
    setTone(pill, tone, 'i-alert');
    setText($('review-pill-text'), excluded.length
      ? `${excluded.length} to exclude, ${actionable.length - excluded.length} to review`
      : `${actionable.length} shot(s) to review`);
    setText(counter, String(actionable.length));
    counter.setAttribute('data-tone', tone);
    show(counter, true);
  }

  // Worst first, so the shot that cannot carry a number is read before the
  // shots that merely want a look.
  const rank = { exclude: 0, review: 1, info: 2 };
  flags.slice()
    .sort((a, b) => (rank[a.severity] ?? 3) - (rank[b.severity] ?? 3)
      || a.shot_number - b.shot_number)
    .forEach(flag => {
      const amount = isNum(flag.esd_statistic) && isNum(flag.esd_critical)
        ? `ESD ${fmt(flag.esd_statistic, 2)} against a critical value of ${fmt(flag.esd_critical, 2)}`
        : null;
      list.appendChild(findingRow(
        `Shot ${flag.shot_number}`, flag.message, flag.severity, amount,
      ));
    });
}

function renderAtmospherePanel() {
  const air = metaBlock('atmosphere');
  const test = metaBlock('test_metadata');
  const pill = $('atmosphere-pill');
  const note = $('atmosphere-note');

  const defaulted = Array.isArray(air.defaulted) ? air.defaulted : [];
  const outOfRange = Array.isArray(air.out_of_standard_range) ? air.out_of_standard_range : [];

  if (!defaulted.length) {
    setCardTone('card-atmosphere', 'ok');
    setTone(pill, 'ok', 'i-check-circle');
    setText($('atmosphere-pill-text'), 'Recorded');
  } else if (defaulted.length >= 3) {
    setCardTone('card-atmosphere', 'warn');
    setTone(pill, 'warn', 'i-alert');
    setText($('atmosphere-pill-text'), 'Assumed — no weather recorded');
  } else {
    setCardTone('card-atmosphere', 'warn');
    setTone(pill, 'warn', 'i-alert');
    setText($('atmosphere-pill-text'), `${defaulted.length} field(s) assumed`);
  }

  const assumed = key => defaulted.includes(key) ? ' (assumed)' : '';
  setText($('atm-temperature'), isNum(air.temperature_C)
    ? `${fmt(air.temperature_C, 1)} °C${assumed('temperature_C')}` : null);
  setText($('atm-humidity'), isNum(air.humidity_pct)
    ? `${fmt(air.humidity_pct, 0)} %${assumed('humidity_pct')}` : null);
  setText($('atm-pressure'), isNum(air.pressure_kPa)
    ? `${fmt(air.pressure_kPa, 2)} kPa${assumed('pressure_kPa')}` : null);
  setText($('atm-speed'), isNum(air.speed_of_sound_m_per_s)
    ? `${fmt(air.speed_of_sound_m_per_s, 1)} m/s` : null);
  setText($('atm-density'), isNum(air.density_kg_per_m3)
    ? `${fmt(air.density_kg_per_m3, 4)} kg/m³` : null);
  setText($('atm-distance'), isNum(test.mic_distance_m)
    ? `${fmt(test.mic_distance_m, 2)} m` : 'not recorded');
  setText($('atm-angle'), isNum(test.mic_angle_deg)
    ? `${fmt(test.mic_angle_deg, 0)}° from the line of fire` : 'not recorded');

  const problems = [];
  if (defaulted.length) {
    problems.push('Weather was not recorded, so ISO 9613-1 reference conditions were '
      + 'assumed. Any distance or absorption correction computed from them carries '
      + 'that assumption.');
  }
  outOfRange.forEach(text => problems.push(text));

  if (problems.length) {
    setText(note, problems.join(' '));
    note.className = 'note note-warn';
    show(note, true);
  } else {
    show(note, false);
  }

  drawAbsorptionChart();
}

/* ---------------------------------------------------------------------------
   BANDS, DISTRIBUTION, SHOT-TO-SHOT BEHAVIOUR

   Three charts that needed nothing from the engine: every value is already in
   analysis_metadata.json. They also respond to shots being excluded, which a
   figure rendered once at analysis time cannot.
   --------------------------------------------------------------------------- */

function drawBandsChart() {
  const canvas = $('bands-canvas');
  const image = $('img-bands');
  if (!canvas) return;
  chartRegistry.set('bands-canvas', drawBandsChart);

  const aggregate = state.results.aggregate || {};
  const frequencies = aggregate.band_frequencies_Hz || [];
  const mean = aggregate.band_exposure_mean_dB || [];
  if (frequencies.length < 2 || mean.length !== frequencies.length) {
    invalidateChart(canvas);
    show(canvas, false);
    show(image, true);
    return;
  }
  show(image, false);
  show(canvas, true);

  const palette = chartPalette();
  drawBandedChart(canvas, {
    frequencies,
    unit: levelUnit(),
    series: [{ label: 'Mean band exposure', color: palette.series[3], kind: 'bar', values: mean }],
  });
  setRaw($('caption-bands'),
    `Energy mean over the ${includedShots().length} included shot(s), `
    + `${frequencies.length} one-third-octave bands, levels in ${levelUnit()}. `
    + 'Recomputed here when a shot is excluded.');
}

/** Freedman–Diaconis bin width, falling back to Sturges on a degenerate IQR. */
function histogramBins(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  if (n < 2) return { edges: [], counts: [] };
  const quantile = (q) => {
    const pos = (n - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    return sorted[base] + (sorted[Math.min(base + 1, n - 1)] - sorted[base]) * rest;
  };
  const iqr = quantile(0.75) - quantile(0.25);
  const lo = sorted[0];
  const hi = sorted[n - 1];
  const span = hi - lo;
  if (!(span > 0)) return { edges: [lo, lo + 1], counts: [n] };
  // Freedman-Diaconis adapts the width to the spread rather than assuming a
  // shape; with a handful of shots the IQR can be zero, and Sturges is the
  // standard fallback.
  let width = iqr > 0 ? (2 * iqr) / Math.cbrt(n) : span / (Math.ceil(Math.log2(n)) + 1);
  if (!(width > 0)) width = span / 4;
  const count = clamp(Math.ceil(span / width), 3, 24);
  const step = span / count;
  const edges = Array.from({ length: count + 1 }, (_, i) => lo + i * step);
  const counts = new Array(count).fill(0);
  for (const v of sorted) counts[clamp(Math.floor((v - lo) / step), 0, count - 1)] += 1;
  return { edges, counts };
}

function drawDistributionChart() {
  const canvas = $('distribution-canvas');
  const image = $('img-distribution');
  const select = $('distribution-metric');
  if (!canvas) return;
  chartRegistry.set('distribution-canvas', drawDistributionChart);

  if (select && select.options.length === 0) {
    for (const metric of METRICS.filter(m => m.level)) {
      const option = document.createElement('option');
      option.value = metric.key;
      option.textContent = metric.label || metric.key;
      select.appendChild(option);
    }
  }
  const key = select && select.value ? select.value : 'Lpeak_Z_dB';
  const values = includedShots().map(s => s[key]).filter(isNum);

  if (values.length < 2) {
    invalidateChart(canvas);
    show(canvas, false);
    show(image, true);
    return;
  }
  show(image, false);
  show(canvas, true);

  const { edges, counts } = histogramBins(values);
  const palette = chartPalette();
  const centres = counts.map((_, i) => (edges[i] + edges[i + 1]) / 2);

  // Counts, and a cumulative count on the same axis — not a percentage on a
  // second one. With eight shots a percentage implies a precision the sample
  // does not have, and one axis keeps the two readable against each other.
  let running = 0;
  const cumulative = counts.map(c => (running += c));

  drawXYChart(canvas, {
    x: centres,
    xScale: 'linear',
    xFormat: (v) => fmt(v, 1),
    xTitle: `${(METRICS.find(m => m.key === key) || {}).label || key}, ${levelUnit()}`,
    unit: 'shots',
    yMin: 0,
    xNoun: 'level range',
    series: [
      { label: 'Shots in bin', color: palette.series[3], kind: 'step', values: counts },
      { label: 'Cumulative', color: palette.series[2], kind: 'line', values: cumulative },
    ],
  });

  // Computed from the same array that was binned, rather than looked up from
  // the aggregate: the two would disagree the moment a shot is excluded.
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  setRaw($('caption-distribution'),
    `${values.length} included shot(s) in ${counts.length} bins of `
    + `${fmt(edges[1] - edges[0], 2)} ${levelUnit()}, width chosen by Freedman–Diaconis. `
    + `Mean ${fmt(mean, 1)}, sigma ${fmt(sampleStd(values), 2)} ${levelUnit()}.`);
}

function drawVariabilityChart() {
  const canvas = $('variability-canvas');
  const figure = $('figure-variability');
  if (!canvas) return;
  chartRegistry.set('variability-canvas', drawVariabilityChart);

  const shots = includedShots();
  const key = ($('distribution-metric') && $('distribution-metric').value) || 'Lpeak_Z_dB';
  const values = shots.map(s => (isNum(s[key]) ? s[key] : null));
  const numbers = values.filter(isNum);
  show(figure, numbers.length >= 2);
  if (numbers.length < 2) return;

  const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
  const sd = sampleStd(numbers);
  const palette = chartPalette();
  const order = shots.map((s, i) => (isNum(s.shot_number) ? s.shot_number : i + 1));

  drawXYChart(canvas, {
    x: order,
    xScale: 'linear',
    xFormat: (v) => `#${Math.round(v)}`,
    xTitle: 'Position in the string',
    unit: levelUnit(),
    xNoun: 'string',
    series: [
      // The sigma band is drawn first so the shots sit on top of it.
      ...(isNum(sd) && sd > 0 ? [{
        label: 'Mean ± 1σ',
        color: palette.series[3],
        kind: 'band',
        alpha: 0.16,
        lower: order.map(() => mean - sd),
        upper: order.map(() => mean + sd),
      }] : []),
      { label: 'Shot level', color: palette.series[0], kind: 'line', values },
      { label: 'Shot level', color: palette.series[0], kind: 'marker', values },
    ],
  });

  const first = values[0];
  setRaw($('caption-variability'),
    `${numbers.length} included shot(s). Mean ${fmt(mean, 1)} ${levelUnit()}, `
    + `sigma ${fmt(sd, 2)} ${levelUnit()}, range ${fmt(Math.max(...numbers) - Math.min(...numbers), 2)} dB.`
    + (isNum(first) ? ` First round ${fmtSigned(first - mean, 2)} dB against the mean.` : ''));
}


/* ---------------------------------------------------------------------------
   THE SELECTED SHOT

   Two of the six panels of the engine's summary plate, drawn from data so
   they can be interrogated: the blast itself, and how its level develops.
   The other four are already here as live components — the metric tiles, the
   band chart — or are the two spectrograms, which the whole-recording
   spectrogram covers.
   --------------------------------------------------------------------------- */

const shotLevelsCache = new Map();

function loadShotLevels() {
  const dir = state.results.dir;
  const file = (metaBlock('artifacts') || {}).shot_levels;
  if (!dir || !file) return;
  const key = `${dir}::${file}`;
  if (shotLevelsCache.has(key)) { drawShotLevelsChart(); return; }
  shotLevelsCache.set(key, null);
  fetch(artifactUrl(dir, file))
    .then(r => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
    .then(payload => { shotLevelsCache.set(key, payload); drawShotLevelsChart(); })
    .catch(err => {
      console.warn('shot level curves unavailable', err);
      shotLevelsCache.set(key, false);
      drawShotLevelsChart();
    });
}

function currentShotLevels() {
  const dir = state.results.dir;
  const file = (metaBlock('artifacts') || {}).shot_levels;
  if (!dir || !file) return null;
  return shotLevelsCache.get(`${dir}::${file}`) || null;
}

/** The selected shot's own window, at the envelope's finest resolution. */
function drawShotWaveChart() {
  const canvas = $('shot-wave-canvas');
  const figure = $('figure-shot-wave');
  if (!canvas) return;
  chartRegistry.set('shot-wave-canvas', drawShotWaveChart);

  const envelope = currentEnvelope();
  const shot = currentShot();
  const window_ = envelope && shot
    ? (envelope.shots || []).find(s => s.shot_number === shot.shot_number)
    : null;

  // Hidden rather than empty when there is no envelope for this shot: a figure
  // frame around nothing is a claim that something should be there.
  show(figure, Boolean(window_));
  if (!window_) return;

  const columns = Math.min(window_.lo.length, window_.hi.length);
  if (columns < 2) { show(figure, false); return; }
  const step = (window_.t1_s - window_.t0_s) / Math.max(1, columns - 1);
  const times = Array.from({ length: columns }, (_, i) => (i * step) * 1000);

  const palette = chartPalette();
  drawXYChart(canvas, {
    x: times,
    xScale: 'linear',
    xFormat: (t) => `${t.toFixed(t > 100 ? 0 : 1)} ms`,
    xTitle: 'Time from the start of the window',
    xNoun: 'window',
    unit: envelope.unit || 'Pa',
    zeroLine: true,
    series: [{
      label: 'Pressure',
      color: palette.series[0],
      kind: 'band',
      lower: window_.lo,
      upper: window_.hi,
      alpha: 0.9,
    }],
    markers: [{
      x: (window_.peak_time_s - window_.t0_s) * 1000,
      label: 'peak',
    }],
  });

  const peak = [...window_.hi, ...window_.lo].reduce(
    (a, v) => (isNum(v) && Math.abs(v) > a ? Math.abs(v) : a), 0);
  // The unit comes from the record. An uncalibrated recording is in full-scale
  // units and saying "Pa" over it would be an absolute claim about a relative
  // measurement — the same mistake the axis label already avoids.
  const unit = envelope.unit || 'Pa';
  setRaw($('caption-shot-wave'),
    `Shot ${shot.shot_number}: ${fmt((window_.t1_s - window_.t0_s) * 1000, 0)} ms window, `
    + `peak ${fmt(peak, 2)} ${unit}, ${columns} columns at ${fmt(step * 1e6, 0)} µs each.`);
  canvas.setAttribute('aria-label', `Pressure against time for shot ${shot.shot_number}.`);
}

/** LAF / LAS / LZF against time, for the selected shot. */
function drawShotLevelsChart() {
  const canvas = $('shot-levels-canvas');
  const figure = $('figure-shot-levels');
  if (!canvas) return;
  chartRegistry.set('shot-levels-canvas', drawShotLevelsChart);

  const payload = currentShotLevels();
  const shot = currentShot();
  const entry = payload && shot
    ? (payload.shots || []).find(s => s.shot_number === shot.shot_number)
    : null;

  show(figure, Boolean(entry));
  if (!entry) return;

  const palette = chartPalette();
  const curves = entry.curves || {};
  const series = [
    ['LAF', 'A-weighted, fast', palette.series[1]],
    ['LAS', 'A-weighted, slow', palette.series[2]],
    ['LZF', 'Z-weighted, fast', palette.series[0]],
  ].filter(([key]) => Array.isArray(curves[key]) && curves[key].some(isNum))
    .map(([key, label, color]) => ({ label, color, kind: 'line', values: curves[key] }));

  if (series.length === 0) { show(figure, false); return; }

  drawXYChart(canvas, {
    x: entry.time_s.map(t => t * 1000),
    xScale: 'linear',
    xFormat: (t) => `${t.toFixed(t > 100 ? 0 : 1)} ms`,
    xTitle: 'Time from the start of the window',
    xNoun: 'window',
    unit: payload.level_unit || levelUnit(),
    series,
  });

  const peakOf = (key) => {
    const values = (curves[key] || []).filter(isNum);
    return values.length ? Math.max(...values) : null;
  };
  const laf = peakOf('LAF');
  const las = peakOf('LAS');
  setRaw($('caption-shot-levels'),
    `Shot ${shot.shot_number}, ${entry.time_s.length} points at 1 ms. `
    + (laf === null ? '' : `LAFmax ${fmt(laf, 1)} ${payload.level_unit}. `)
    + (las === null ? '' : `LASmax ${fmt(las, 1)} ${payload.level_unit}. `)
    + 'Fast and slow are the standard 125 ms and 1 s exponential averages; '
    + 'the gap between them is how impulsive the event is.');
  canvas.setAttribute('aria-label',
    `Time-weighted levels against time for shot ${shot.shot_number}.`);
}


/* ---------------------------------------------------------------------------
   SPECTROGRAM

   Read from spectrogram_{z,c}_matrix.json, which holds the same 0.1 dB values
   the engine plotted, quantised losslessly into uint16. Painting it here
   rather than shipping a picture is what lets the operator put the cursor on
   a smear at 4 kHz and be told what level it actually is.
   --------------------------------------------------------------------------- */

const spectrogramCache = new Map();

function spectrogramArtifact(weighting) {
  const artifacts = metaBlock('artifacts') || {};
  return artifacts[`spectrogram_${String(weighting).toLowerCase()}_matrix`] || null;
}

function loadSpectrogramMatrix(weighting) {
  const dir = state.results.dir;
  const file = spectrogramArtifact(weighting);
  if (!dir || !file) { drawSpectrogramChart(); return; }
  const key = `${dir}::${file}`;
  if (spectrogramCache.has(key)) { drawSpectrogramChart(); return; }
  spectrogramCache.set(key, null);

  fetch(artifactUrl(dir, file))
    .then(r => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
    .then(payload => {
      payload.decoded = decodeMatrix(payload.magnitude_dB_b64);
      spectrogramCache.set(key, payload);
      drawSpectrogramChart();
    })
    .catch(err => {
      console.warn('spectrogram matrix unavailable', err);
      spectrogramCache.set(key, false);
      drawSpectrogramChart();
    });
}

function currentSpectrogram() {
  const dir = state.results.dir;
  const select = $('spectrogram-weighting');
  const file = spectrogramArtifact(select ? select.value : 'Z');
  if (!dir || !file) return null;
  return spectrogramCache.get(`${dir}::${file}`) || null;
}

function drawSpectrogramChart() {
  const canvas = $('spectrogram-canvas');
  const image = $('img-spectrogram');
  if (!canvas) return;
  chartRegistry.set('spectrogram-canvas', drawSpectrogramChart);

  const payload = currentSpectrogram();
  const shotSelect = $('spectrogram-shot');
  const perShot = shotSelect && shotSelect.value !== 'all';

  // The per-shot spectrograms are separate figures the engine renders; only
  // the full-recording matrix is emitted as data. Rather than pretend, the
  // per-shot selection falls back to the engine's picture and says so.
  if (!payload || perShot) {
    invalidateChart(canvas);
    show(canvas, false);
    show(image, true);
    return;
  }
  show(image, false);
  show(canvas, true);

  const q = payload.quantisation || {};
  // The colour scale spans the top 70 dB. Anchoring it at the true minimum
  // would spend most of the ramp on the noise floor and leave the blasts —
  // the measurement — compressed into the last few shades.
  const vmax = Math.ceil(payload.max_dB);
  const vmin = Math.max(Math.floor(payload.min_dB), vmax - 70);

  drawHeatmapChart(canvas, {
    matrix: payload.decoded,
    frames: payload.frames,
    bins: payload.bins,
    time: payload.time_s,
    frequencies: payload.frequencies_Hz,
    offset: q.offset_dB,
    step: q.step_dB,
    missing: q.missing,
    vmin,
    vmax,
    unit: payload.calibrated ? 'dB SPL' : 'dB re FS',
    xTitle: 'Time',
    yTitle: 'Hz',
  });

  setRaw($('caption-spectrogram'),
    `${payload.weighting}-weighted, ${payload.nperseg}-point ${payload.window} window `
    + `(${fmt(payload.enbw_Hz, 0)} Hz bins), ${payload.frames} frames. `
    + `Colour scale ${vmin} to ${vmax} ${payload.calibrated ? 'dB SPL' : 'dB re FS'}. `
    + 'Point at the plot to read a level off it.');
}


/* ---------------------------------------------------------------------------
   WAVEFORM

   Drawn from waveform_envelope.json, which the engine writes in the data stage
   as a min/max envelope. The band between the two bounds is the signal: at
   this scale a single line would be a line through the middle of a blast, and
   the height of a blast is the measurement.

   The whole recording and one shot are two different pictures, and the file
   carries both. Zooming past the overview's own column width swaps in the
   shot's higher-resolution envelope rather than magnifying columns that have
   no more detail in them.
   --------------------------------------------------------------------------- */

/** dir -> the parsed envelope, so switching tabs does not refetch it. */
const waveformCache = new Map();

/**
 * Forget everything that describes the analysis being replaced.
 *
 * These are module-level, and every one of them outlived the measurement it
 * belonged to: opening a second analysis kept the first one's zoom, the first
 * one's selected shot, and — while the new envelope was still being fetched —
 * the first one's WAVEFORM, painted under the new analysis's heading. A chart
 * showing the wrong recording is the worst failure this interface has.
 */
function resetChartState() {
  waveformRange = null;
  waveformFocus = '';
  const jump = $('waveform-jump');
  if (jump) jump.value = '';
  for (const id of ['waveform-canvas', 'spectrogram-canvas', 'bands-canvas',
    'distribution-canvas', 'variability-canvas', 'absorption-canvas',
    'shot-band-canvas', 'shot-wave-canvas', 'shot-levels-canvas']) {
    const canvas = $(id);
    if (!canvas) continue;
    const record = chartState.get(canvas);
    if (!record) continue;
    record.view = null;
    record.spec = null;
    record.geom = null;
    record.cursor = null;
    record.drag = null;
    record.focusKey = null;
    if (record.readout) show(record.readout, false);
    if (record.reset) show(record.reset, false);
  }
}

function loadWaveformEnvelope() {
  const dir = state.results.dir;
  const file = (metaBlock('artifacts') || {}).waveform_envelope;
  if (!dir || !file) return;
  const key = `${dir}::${file}`;
  if (waveformCache.has(key)) { drawWaveformChart(); return; }
  waveformCache.set(key, null);                    // in flight; do not refetch

  fetch(artifactUrl(dir, file))
    .then(response => (response.ok ? response.json() : Promise.reject(new Error(response.status))))
    .then(payload => { waveformCache.set(key, payload); drawWaveformChart(); })
    .catch(err => {
      console.warn('waveform envelope unavailable', err);
      waveformCache.set(key, false);               // known absent; fall back to the PNG
      drawWaveformChart();
    });
}

function currentEnvelope() {
  const dir = state.results.dir;
  const file = (metaBlock('artifacts') || {}).waveform_envelope;
  if (!dir || !file) return null;
  return waveformCache.get(`${dir}::${file}`) || null;
}

/** Which shot the waveform is zoomed to, or '' for the whole recording. */
let waveformFocus = '';

/** The visible time window, in seconds, or null for everything. */
let waveformRange = null;

/**
 * Choose the coarsest pyramid level that still puts a column in every pixel.
 *
 * This is what makes the picture gain detail as the zoom goes in. A single
 * envelope magnifies: 2048 columns across twelve seconds is 5.9 ms per column,
 * so zooming to one blast just draws the same handful of columns wider. Here
 * the visible span is measured against each level's column width and the first
 * one fine enough is used — so a 5 ms window is drawn from the level with
 * 131072 columns, at roughly one sample per pixel, and nothing deeper is ever
 * decoded because there would be nothing left to see.
 */
function pickWaveformLevel(envelope, t0, t1, pixels) {
  const levels = Array.isArray(envelope.levels) && envelope.levels.length
    ? envelope.levels
    : [{ columns: envelope.columns, lo: envelope.lo, hi: envelope.hi }];
  const duration = envelope.duration_s || 1;
  const span = Math.max(1e-9, t1 - t0);
  const wanted = Math.max(1, pixels);
  for (const level of levels) {
    // Columns that will land inside the visible span at this level.
    if ((level.columns * span) / duration >= wanted) return level;
  }
  return levels[levels.length - 1];
}

function drawWaveformChart() {
  const canvas = $('waveform-canvas');
  if (!canvas) return;
  chartRegistry.set('waveform-canvas', drawWaveformChart);

  const envelope = currentEnvelope();
  const image = $('img-waveform');
  const jump = $('waveform-jump');

  // No envelope on this record — an analysis from an older build, or the
  // chunked path. Fall back to the engine's PNG rather than an empty box.
  if (!envelope) {
    invalidateChart(canvas);
    show(canvas, false);
    show(image, true);
    show(jump, false);
    return;
  }
  show(image, false);
  show(canvas, true);

  const shots = Array.isArray(envelope.shots) ? envelope.shots : [];
  if (jump) {
    show(jump, shots.length > 0);
    if (jump.options.length !== shots.length + 1) {
      jump.textContent = '';
      const whole = document.createElement('option');
      whole.value = '';
      whole.textContent = 'Whole recording';
      jump.appendChild(whole);
      for (const shot of shots) {
        const option = document.createElement('option');
        option.value = String(shot.shot_number);
        option.textContent = `Shot ${shot.shot_number}`;
        jump.appendChild(option);
      }
    }
    if (jump.value !== waveformFocus) jump.value = waveformFocus;
  }

  const palette = chartPalette();
  const focused = waveformFocus
    ? shots.find(s => String(s.shot_number) === waveformFocus) || null
    : null;

  const pixels = Math.max(200, Math.round(canvas.getBoundingClientRect().width) - 72);

  // A shot's own envelope is the finest thing available over its window, so it
  // wins whenever the visible range lies inside one.
  let lo;
  let hi;
  let t0;
  let t1;
  let sourceColumns;
  let sourceLabel;

  // A drag inside a focused shot must still narrow the view. Taking the
  // window from the shot whenever one was selected made the zoom a no-op
  // there — the selector chooses the STARTING window, not a fixed one.
  const fallback0 = focused ? focused.t0_s : 0;
  const fallback1 = focused ? focused.t1_s : (envelope.duration_s || 1);
  let window0 = waveformRange ? Math.max(waveformRange.t0, fallback0) : fallback0;
  let window1 = waveformRange ? Math.min(waveformRange.t1, fallback1) : fallback1;
  // A range that no longer overlaps the signal — a zoom carried into a shorter
  // recording, or a shot at the very end — inverts, and every slice below then
  // collapses to a single column and reports "No waveform data" for a
  // recording that has plenty. Fall back to the whole window instead.
  if (!(window1 > window0)) {
    waveformRange = null;
    window0 = fallback0;
    window1 = fallback1;
  }

  const inside = (focused && window0 >= focused.t0_s && window1 <= focused.t1_s)
    ? focused
    : shots.find(s => window0 >= s.t0_s && window1 <= s.t1_s) || focused || null;

  if (inside) {
    const step = (inside.t1_s - inside.t0_s) / Math.max(1, inside.columns - 1);
    const a = clamp(Math.floor((window0 - inside.t0_s) / step), 0, inside.columns - 1);
    const b = clamp(Math.ceil((window1 - inside.t0_s) / step), a + 1, inside.columns - 1);
    lo = inside.lo.slice(a, b + 1);
    hi = inside.hi.slice(a, b + 1);
    t0 = inside.t0_s + a * step;
    t1 = inside.t0_s + b * step;
    sourceColumns = inside.columns;
    sourceLabel = `shot ${inside.shot_number}`;
  } else {
    const level = pickWaveformLevel(envelope, window0, window1, pixels);
    const duration = envelope.duration_s || 1;
    const step = duration / Math.max(1, level.columns - 1);
    const a = clamp(Math.floor(window0 / step), 0, level.columns - 1);
    const b = clamp(Math.ceil(window1 / step), a + 1, level.columns - 1);
    lo = level.lo.slice(a, b + 1);
    hi = level.hi.slice(a, b + 1);
    t0 = a * step;
    t1 = b * step;
    sourceColumns = level.columns;
    sourceLabel = `level ${level.columns}`;
  }

  const columns = Math.min(lo.length, hi.length);
  if (columns < 2) { chartMessage(canvas, 'No waveform data'); return; }
  const step = (t1 - t0) / Math.max(1, columns - 1);
  const times = Array.from({ length: columns }, (_, i) => t0 + i * step);

  // The zoom lives in seconds here, not in array indices, precisely because
  // the array changes when a finer level is chosen.
  const record = chartRecord(canvas);
  record.view = null;

  const markers = focused
    ? [{ x: focused.peak_time_s, label: `#${focused.shot_number}` }]
    : shots.filter(s => s.peak_time_s >= t0 && s.peak_time_s <= t1)
      .map(s => ({ x: s.peak_time_s, label: String(s.shot_number) }));

  const brief = (t1 - t0) < 1;
  drawXYChart(canvas, {
    x: times,
    xScale: 'linear',
    xFormat: (t) => (brief ? `${(t * 1000).toFixed(1)} ms` : `${t.toFixed(2)} s`),
    xTitle: 'Time',
    xNoun: 'recording',
    unit: envelope.unit || 'Pa',
    zeroLine: true,
    series: [{
      label: 'Pressure',
      color: palette.series[0],
      kind: 'band',
      lower: lo,
      upper: hi,
      alpha: 0.9,
    }],
    markers,
    onZoom: (a, b) => {
      if (!isNum(a) || !isNum(b) || b <= a) return;
      waveformRange = { t0: a, t1: b };
      drawWaveformChart();
    },
    onReset: () => {
      waveformRange = null;
      waveformFocus = '';
      const control = $('waveform-jump');
      if (control) control.value = '';
    },
    isZoomed: () => Boolean(waveformRange) || Boolean(waveformFocus),
  });

  const peak = [...hi, ...lo].reduce(
    (a, v) => (isNum(v) && Math.abs(v) > a ? Math.abs(v) : a), 0);
  const resolution = (t1 - t0) / Math.max(1, columns - 1);
  setRaw($('caption-waveform'),
    `${brief ? `${fmt((t1 - t0) * 1000, 1)} ms` : `${fmt(t1 - t0, 2)} s`} shown, `
    + `peak ${fmt(peak, 2)} ${envelope.unit || 'Pa'}. ${columns} columns at `
    + `${resolution < 1e-3 ? `${fmt(resolution * 1e6, 0)} µs` : `${fmt(resolution * 1e3, 2)} ms`} each, `
    + `drawn from ${sourceLabel}. Each column is the extremes of its span, so the band always contains the true peak.`);
  canvas.setAttribute('aria-label', focused
    ? `Pressure against time for shot ${focused.shot_number}.`
    : 'Pressure against time, with each detected shot marked.');
}

function wireWaveform() {
  const jump = $('waveform-jump');
  if (!jump) return;
  jump.addEventListener('change', () => {
    waveformFocus = jump.value;
    waveformRange = null;
    drawWaveformChart();
    announce(waveformFocus ? `Waveform zoomed to shot ${waveformFocus}` : 'Waveform showing the whole recording');
  });
}

/**
 * Atmospheric absorption against frequency, drawn from the record.
 *
 * Log frequency, because absorption is a decade phenomenon: at 1 m the whole
 * curve is under a tenth of a decibel until the last third of the audio range
 * and then climbs by two orders of magnitude, and a linear axis shows that as
 * a flat line with a spike at the right-hand edge.
 *
 * The alpha curve and the path loss are plotted TOGETHER on purpose. Alpha
 * alone (dB per kilometre) sounds alarming at 20 kHz; over the metre that was
 * actually measured it is half a decibel, and it is the second number that
 * governs whether the measurement needed correcting.
 */
function drawAbsorptionChart() {
  const canvas = $('absorption-canvas');
  const card = $('card-absorption');
  if (!canvas || !card) return;
  chartRegistry.set('absorption-canvas', drawAbsorptionChart);

  const effect = metaBlock('atmospheric_effect');
  const frequencies = Array.isArray(effect.frequencies_Hz) ? effect.frequencies_Hz : [];
  const alpha = Array.isArray(effect.alpha_dB_per_km) ? effect.alpha_dB_per_km : [];
  const absorption = Array.isArray(effect.absorption_dB) ? effect.absorption_dB : [];

  show(card, frequencies.length > 1 && absorption.length === frequencies.length);
  if (card.hidden) return;

  const palette = chartPalette();
  const distance = isNum(effect.distance_m) ? effect.distance_m : null;

  drawXYChart(canvas, {
    x: frequencies,
    xScale: 'log',
    xFormat: (hz) => (hz >= 1000 ? `${hz / 1000}k` : String(Math.round(hz))),
    xTitle: 'Frequency, Hz',
    unit: 'dB',
    // One quantity, one axis. The coefficient (dB/km) and the path loss (dB)
    // differ by three orders of magnitude here, so drawing both against one
    // scale would flatten the one that governs the decision. The coefficient
    // is stated in the caption instead of being plotted on a borrowed axis.
    series: [{
      label: distance === null ? 'Absorption over the path' : `Absorption over ${fmt(distance, 2)} m`,
      color: palette.series[3],
      kind: 'line',
      values: absorption,
    }],
  });

  const worst = isNum(effect.worst_absorption_dB) ? effect.worst_absorption_dB : null;
  const band = isNum(effect.worst_band_Hz) ? effect.worst_band_Hz : null;
  setTone($('absorption-pill'), effect.matters ? 'warn' : 'ok');
  setRaw($('absorption-pill-text'), effect.matters
    ? 'Correction matters at this distance'
    : 'Negligible at this distance');
  const worstAlpha = alpha.length ? Math.max(...alpha.filter(isNum)) : null;
  setRaw($('absorption-caption'), worst === null || band === null
    ? 'Computed from the recorded atmosphere by ISO 9613-1.'
    : `Worst band ${fmt(worst, 2)} dB at ${fmtHz(band)} over `
      + `${distance === null ? 'the path' : `${fmt(distance, 2)} m`}`
      + `${worstAlpha === null ? '' : `, from a coefficient of ${fmt(worstAlpha, 1)} dB/km`}. `
      + 'ISO 9613-1, from the recorded atmosphere.');
  canvas.setAttribute('aria-label',
    `Atmospheric absorption against frequency; worst ${worst === null ? 'unknown' : fmt(worst, 2)} dB.`);
}

function renderStringPanel() {
  const breakdown = metaBlock('string_statistics');
  const headline = breakdown.Lpeak_Z || breakdown.Lpeak_A || breakdown.LAE || null;

  renderFirstRoundPop(headline);
  renderStringMeans(breakdown);
  renderStringDrift(headline);
  renderShotReview();
  renderAtmospherePanel();

  const distribution = pickImage(['level_distribution', 'distribution']);
  const drawn = setFigure('img-distribution', distribution, null,
    'Histogram and cumulative distribution of the per-shot levels.');
  setText($('caption-distribution'), drawn
    ? `Per-shot ${levelUnit()} levels: the histogram shows the shape, the cumulative `
      + 'curve assumes no distribution at all.'
    : 'Not generated for this analysis.');

  drawDistributionChart();
  drawVariabilityChart();
}

function renderResults() {
  const status = state.results.status;
  show($('results-empty'), status === 'empty' || status === 'error');
  show($('results-skeleton'), status === 'loading');
  show($('results-loaded'), status === 'loaded');

  const subtitle = $('results-subtitle');
  if (status === 'loading') setRaw(subtitle, 'Loading…');
  else if (status === 'error') setRaw(subtitle, state.results.error || 'The results could not be loaded');
  else if (status === 'loaded') {
    const analysis = metaBlock('analysis');
    const record = metaBlock('test_metadata');
    setRaw(subtitle, [
      basename(analysis.input_file) || basename(state.results.dir),
      record.configuration ? record.configuration : null,
      fmtTimestamp(analysis.timestamp),
    ].filter(Boolean).join(' · '));
  } else setRaw(subtitle, 'No analysis loaded');

  // The empty state doubles as the error state, with the reason stated.
  const emptyTitle = qs('#results-empty .empty-state-title');
  const emptyText = qs('#results-empty .empty-state-text');
  if (status === 'error') {
    setRaw(emptyTitle, 'Those results could not be loaded');
    setRaw(emptyText, state.results.error || 'The local service did not return a usable analysis.');
  } else {
    setRaw(emptyTitle, 'No measurement loaded');
    setRaw(emptyText, 'Run an analysis, or open a previous one from History.');
  }

  const actionable = status === 'loaded';
  ['btn-results-print', 'btn-results-csv', 'btn-results-folder', 'btn-results-copy-path',
   'btn-results-menu']
    .forEach(id => setDisabled($(id), !actionable));

  if (status !== 'loaded') return;

  renderValidity();
  renderHeadlineMetrics();
  renderOverview();
  loadWaveformEnvelope();
  loadShotLevels();
  loadSpectrogramMatrix($('spectrogram-weighting') ? $('spectrogram-weighting').value : 'Z');
  renderSpectrogramPanel();
  renderBandsPanel();
  renderShotsPanel();
  renderStringPanel();
  renderMetricsTable();
  renderHazardPanel();
}

function wireResultsActions() {
  const print = () => window.print();
  ['btn-print', 'btn-results-print', 'btn-compare-print'].forEach(id => {
    const button = $(id);
    if (button) button.addEventListener('click', print);
  });

  const help = $('btn-help');
  if (help) help.addEventListener('click', showAbout);

  const shortcuts = $('btn-shortcuts');
  if (shortcuts) shortcuts.addEventListener('click', showShortcuts);

  const copyPath = $('btn-results-copy-path');
  if (copyPath) copyPath.addEventListener('click', async () => {
    const ok = await copyText(state.results.dir);
    toast({
      title: ok ? 'Output path copied' : 'Copy failed',
      text: ok ? state.results.dir : 'The clipboard is not available in this context.',
      tone: ok ? 'ok' : 'danger',
    });
  });

  const copySha = $('btn-copy-sha');
  if (copySha) copySha.addEventListener('click', async () => {
    const sha = metaBlock('analysis').input_sha256;
    const ok = await copyText(sha);
    toast({ title: ok ? 'Input hash copied' : 'Copy failed', text: ok ? sha : '', tone: ok ? 'ok' : 'danger' });
  });

  const folder = $('btn-results-folder');
  if (folder) folder.addEventListener('click', async () => {
    // The local service exposes no "reveal in file manager" endpoint, and the
    // browser cannot open one. Hand over the path rather than pretend.
    const ok = await copyText(state.results.dir);
    toast({
      title: 'Output folder',
      text: ok
        ? `Path copied to the clipboard: ${state.results.dir}`
        : `Output directory: ${state.results.dir}`,
      tone: 'info',
      ttl: 12000,
    });
  });

  const csv = $('btn-results-csv');
  if (csv) csv.addEventListener('click', () => {
    const payload = state.results.payload;
    if (payload && payload.csv) {
      downloadText(`${basename(state.results.dir)}-metrics_summary.csv`, payload.csv);
      return;
    }
    // Fall back to the artifact the engine wrote, which the server serves as
    // an attachment; failing that, build one from what is on screen.
    const file = metaBlock('artifacts').metrics_csv;
    if (file) window.location.assign(artifactUrl(state.results.dir, file));
    else downloadMetricsCsv();
  });

  const details = $('btn-validity-details');
  if (details) details.addEventListener('click', () => openDialog($('modal-validity'), { opener: details }));

  const distributionMetric = $('distribution-metric');
  if (distributionMetric) distributionMetric.addEventListener('change', () => {
    // One control, two charts: the distribution and the drift plot are the
    // same measurement seen two ways and must never show different metrics.
    drawDistributionChart();
    drawVariabilityChart();
  });

  const weighting = $('spectrogram-weighting');
  if (weighting) weighting.addEventListener('change', () => {
    // Each weighting is its own matrix; fetched once and then cached.
    loadSpectrogramMatrix(weighting.value);
    commit();
  });
  const spectrogramShot = $('spectrogram-shot');
  if (spectrogramShot) spectrogramShot.addEventListener('change', () => commit());

  const overviewOpen = $('btn-overview-open');
  if (overviewOpen) overviewOpen.addEventListener('click', () => {
    const html = pickImage(['waveform_full'], 'html');
    const png = pickImage(['waveform_full', 'waveform']);
    const file = html || png;
    if (!file) return;
    window.open(artifactUrl(state.results.dir, file), '_blank', 'noopener');
  });
}

/* ---- the recording, wired ------------------------------------------------ */

function wireInput() {
  wireDropzone('file-dropzone', 'file-input', acceptRecording);
  wireDropzone('cal-tone-dropzone', 'cal-tone-input', acceptCalibratorTone);

  const clear = $('btn-file-clear');
  if (clear) clear.addEventListener('click', () => {
    clearInput();
    const zone = $('file-dropzone');
    const picker = zone && qs('input[type="file"]', zone);
    if (picker) picker.focus();
    announce('Recording cleared');
  });

  const setPath = $('btn-file-path-set');
  if (setPath) setPath.addEventListener('click', useTypedPath);

  const pathInput = $('file-path-input');
  if (pathInput) {
    pathInput.addEventListener('keydown', (ev) => {
      if (ev.key !== 'Enter') return;
      ev.preventDefault();
      useTypedPath();
    });
    // Typing again clears the previous rejection rather than leaving it red.
    pathInput.addEventListener('input', () => {
      if (!state.input.error) return;
      state.input.error = null;
      commit();
    });
  }

  renderRecent();
}


/* ===========================================================================
   11. CHARTS

   Two charts are drawn in the browser rather than by the engine, because both
   answer a question that only exists once the operator starts excluding shots:
   "what does THIS shot look like against the aggregate" and "what is the
   insertion loss per band". Both read every colour from the design tokens with
   getComputedStyle, are drawn at devicePixelRatio, and are re-drawn whenever
   the theme or the layout changes.
   =========================================================================== */

/** id -> draw function, so a theme or size change can repaint everything. */
const chartRegistry = new Map();

/**
 * Per-canvas interaction state.
 *
 *   spec    the last spec drawn, so the overlay can read values without the
 *           caller having to hand them over again
 *   geom    the plot geometry from that draw: enough to map a pixel to a band
 *           and a value back to a pixel
 *   view    {i0, i1} inclusive band indices currently shown — the zoom
 *   cursor  the band index under the pointer, or null
 *   drag    {from, to} in band indices while a range is being dragged
 *
 * Kept in a WeakMap keyed by the canvas so a redraw (theme change, resize)
 * preserves the zoom the operator set, and so nothing leaks if a canvas is
 * removed from the document.
 */
const chartState = new WeakMap();

function chartRecord(canvas) {
  let record = chartState.get(canvas);
  if (!record) {
    record = { spec: null, geom: null, view: null, cursor: null, drag: null, wired: false };
    chartState.set(canvas, record);
  }
  return record;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * Every colour a chart uses, resolved from tokens.css for the CURRENT theme.
 * There is deliberately no literal colour anywhere in this file: if a token
 * does not resolve, the stylesheet has not loaded and the chart says so rather
 * than inventing a value that would not follow the theme.
 */
function chartPalette() {
  return {
    series: [1, 2, 3, 4, 5, 6].map(i => cssVar(`--series-${i}`)),
    text: cssVar('--text-2'),
    muted: cssVar('--text-3'),
    axis: cssVar('--border'),
    grid: cssVar('--border-subtle'),
    surface: cssVar('--plot-bg'),
    accent: cssVar('--accent'),
    ok: cssVar('--ok'),
    warn: cssVar('--warn'),
    shotMarker: cssVar('--shot-marker'),
    mono: cssVar('--font-mono') || 'monospace',
  };
}

/**
 * Size the backing store to devicePixelRatio and scale the context, so text and
 * hairlines are crisp on a HiDPI display instead of being upsampled.
 * Returns null when the canvas has no layout yet (a hidden panel).
 */
function prepareCanvas(canvas) {
  if (!canvas) return null;
  if (!canvas.dataset.aspect) {
    const w = Number(canvas.getAttribute('width')) || 900;
    const h = Number(canvas.getAttribute('height')) || 300;
    canvas.dataset.aspect = String(h / w);
  }
  const host = canvas.parentElement || canvas;
  const width = Math.floor(host.clientWidth || 0);
  if (width <= 0) return null;
  const height = Math.round(width * Number(canvas.dataset.aspect));
  // An export re-runs the SAME drawing code at a higher pixel ratio rather
  // than scaling the finished picture up. Every line, glyph and axis label is
  // rasterised at the export resolution, so zooming into the saved file finds
  // real detail instead of interpolated pixels.
  const dpr = Number(canvas.dataset.exportScale) || window.devicePixelRatio || 1;

  canvas.style.width = '100%';
  canvas.style.height = `${height}px`;
  const backingWidth = Math.round(width * dpr);
  const backingHeight = Math.round(height * dpr);
  if (canvas.width !== backingWidth) canvas.width = backingWidth;
  if (canvas.height !== backingHeight) canvas.height = backingHeight;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

/**
 * Forget the draw this canvas is no longer showing.
 *
 * Hiding a canvas or painting a message over it does NOT remove the .chart
 * shell that owns the pointer listeners, the overlay and the floating
 * readout. Left behind, the crosshair goes on resolving pointer positions
 * through the previous draw's geometry and printing the previous shot's
 * levels over a chart that says it has no data — an authoritative number
 * belonging to a different measurement, which is the one failure this
 * application exists to prevent.
 */
function invalidateChart(canvas) {
  const record = chartState.get(canvas);
  if (!record) return;
  record.spec = null;
  record.geom = null;
  record.cursor = null;
  record.drag = null;
  if (record.readout) show(record.readout, false);
  if (record.reset) show(record.reset, false);
  if (record.overlay) {
    const ctx = record.overlay.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, record.overlay.width, record.overlay.height);
  }
}

function chartMessage(canvas, message) {
  invalidateChart(canvas);
  const surface = prepareCanvas(canvas);
  if (!surface) return;
  const { ctx, width, height } = surface;
  const palette = chartPalette();
  if (!palette.surface || !palette.muted) return;   // tokens unavailable
  ctx.fillStyle = palette.surface;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = palette.muted;
  ctx.font = `13px ${palette.mono}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message, width / 2, height / 2);
}

/** Nominal 1/1-octave centres — the only band labels worth printing. */
const OCTAVE_CENTRES = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];

function isOctaveCentre(hz) {
  return OCTAVE_CENTRES.some(c => Math.abs(hz - c) / c < 0.06);
}

/** Axis-width band label. fmtHz() is the prose one; this one has ~30px. */
function axisHz(hz) {
  if (!isNum(hz)) return '—';
  if (hz >= 1000) {
    const k = hz / 1000;
    return `${k % 1 === 0 ? k : k.toFixed(1)}k`;
  }
  return String(Math.round(hz * 10) / 10);
}

/**
 * A banded chart: categorical x (one slot per 1/3-octave band), any number of
 * bar or line series, an optional zero rule.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{frequencies:number[], series:{label:string,color:string,kind:'bar'|'line',values:number[]}[],
 *          unit:string, zeroLine?:boolean}} spec
 */
function drawBandedChart(canvas, spec) {
  const surface = prepareCanvas(canvas);
  if (!surface) return;
  const { ctx, width, height } = surface;
  const palette = chartPalette();

  if (!palette.surface || !palette.axis) {
    console.error('tokens.css did not resolve; the chart cannot be drawn in the current theme');
    return;
  }

  const allFrequencies = spec.frequencies || [];
  const allSeries = (spec.series || []).filter(s =>
    Array.isArray(s.values) && s.values.length > 0 && Boolean(s.color));
  if (allFrequencies.length === 0 || allSeries.flatMap(s => s.values).filter(isNum).length === 0) {
    chartMessage(canvas, 'No band data for the included shots');
    return;
  }

  // The zoom, clamped to whatever data is actually present now: a shot with
  // fewer bands than the one that was zoomed must not leave a view pointing
  // past the end of the array.
  const record = chartRecord(canvas);
  record.spec = { ...spec, frequencies: allFrequencies, series: allSeries };
  const last = allFrequencies.length - 1;
  let view = record.view;
  if (!view || view.i0 > last || view.i1 > last || view.i1 - view.i0 < 1) {
    view = { i0: 0, i1: last };
    record.view = null;                       // not zoomed
  }

  const frequencies = allFrequencies.slice(view.i0, view.i1 + 1);
  const series = allSeries.map(s => ({ ...s, values: s.values.slice(view.i0, view.i1 + 1) }));
  const values = series.flatMap(s => s.values).filter(isNum);
  if (values.length === 0) {
    // record.spec has already been replaced above; returning here without
    // clearing it would pair the NEW series with the PREVIOUS draw's
    // geometry. chartMessage invalidates both.
    chartMessage(canvas, 'No band data in the selected range');
    return;
  }

  ctx.fillStyle = palette.surface;
  ctx.fillRect(0, 0, width, height);

  const pad = { left: 54, right: 14, top: 14, bottom: 36 };
  const plotW = Math.max(10, width - pad.left - pad.right);
  const plotH = Math.max(10, height - pad.top - pad.bottom);

  // The y range is recomputed from the VISIBLE bands, so zooming in on a
  // narrow range actually resolves it instead of leaving it a flat line at
  // the bottom of a scale set by bands that are no longer on screen.
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (spec.zeroLine) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (hi - lo < 1) { hi += 0.5; lo -= 0.5; }
  const span = hi - lo;
  lo -= span * 0.08;
  hi += span * 0.08;

  const yOf = (v) => pad.top + plotH - ((v - lo) / (hi - lo)) * plotH;
  const slot = plotW / frequencies.length;
  const xOf = (i) => pad.left + slot * (i + 0.5);

  record.geom = {
    kind: 'band', pad, plotW, plotH, slot, lo, hi, width, height,
    i0: view.i0, i1: view.i1,
    xValues: allFrequencies, xFormat: fmtHz, xNoun: 'band range',
  };

  // ---- grid and y axis ----
  const ticks = 5;
  ctx.font = `11px ${palette.mono}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 1;
  for (let t = 0; t <= ticks; t += 1) {
    const value = lo + ((hi - lo) * t) / ticks;
    const y = Math.round(yOf(value)) + 0.5;
    ctx.strokeStyle = palette.grid;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = palette.muted;
    ctx.fillText(value.toFixed(Math.abs(hi - lo) < 5 ? 1 : 0), pad.left - 8, y);
  }

  ctx.strokeStyle = palette.axis;
  ctx.beginPath();
  ctx.moveTo(pad.left + 0.5, pad.top);
  ctx.lineTo(pad.left + 0.5, pad.top + plotH + 0.5);
  ctx.lineTo(pad.left + plotW, pad.top + plotH + 0.5);
  ctx.stroke();

  // ---- x labels: octave centres only, so they never collide ----
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle = palette.muted;
  let labelled = 0;
  frequencies.forEach((hz, i) => {
    if (!isOctaveCentre(hz)) return;
    labelled += 1;
    ctx.fillText(axisHz(hz), xOf(i), pad.top + plotH + 8);
  });
  // Zoomed in far enough and there may be no octave centre in view at all,
  // which would leave the x axis unlabelled. Fall back to the ends.
  if (labelled === 0 && frequencies.length > 0) {
    const ends = frequencies.length === 1 ? [0] : [0, frequencies.length - 1];
    for (const i of ends) ctx.fillText(axisHz(frequencies[i]), xOf(i), pad.top + plotH + 8);
  }
  ctx.fillStyle = palette.text;
  ctx.textAlign = 'left';
  ctx.fillText(`Hz — ${spec.unit || 'dB'}`, pad.left, pad.top + plotH + 22);

  // ---- zero rule ----
  if (spec.zeroLine && lo < 0 && hi > 0) {
    const y = Math.round(yOf(0)) + 0.5;
    ctx.strokeStyle = palette.axis;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // ---- series ----
  const bars = series.filter(s => s.kind === 'bar');
  const barWidth = Math.max(1, (slot * 0.72) / Math.max(1, bars.length));

  series.forEach((s) => {
    if (s.kind === 'bar') {
      const index = bars.indexOf(s);
      const offset = -((bars.length * barWidth) / 2) + index * barWidth;
      ctx.fillStyle = s.color;
      const base = spec.zeroLine ? yOf(Math.max(lo, Math.min(hi, 0))) : pad.top + plotH;
      s.values.forEach((v, i) => {
        if (!isNum(v)) return;
        const y = yOf(v);
        ctx.fillRect(xOf(i) + offset, Math.min(y, base), barWidth, Math.max(1, Math.abs(base - y)));
      });
      return;
    }
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let started = false;
    s.values.forEach((v, i) => {
      if (!isNum(v)) { started = false; return; }
      const x = xOf(i);
      const y = yOf(v);
      if (started) ctx.lineTo(x, y);
      else { ctx.moveTo(x, y); started = true; }
    });
    ctx.stroke();
    ctx.lineWidth = 1;
  });

  ensureChartInteraction(canvas);
  drawChartOverlay(canvas);
}


/* ---------------------------------------------------------------------------
   CHART INTERACTION — cursor readout and drag-to-zoom

   Two canvases stacked in one container. The lower one holds the plot and is
   repainted only when the DATA changes; the upper one holds the crosshair and
   the drag selection and is repainted on every pointer move. That split is the
   whole performance story: moving the pointer never re-runs the axis, grid,
   label and series drawing underneath, so the cursor tracks at frame rate even
   on a chart with forty bands and three series.

   The readout SNAPS to a band. It never interpolates between two of them,
   because there is no measurement between two 1/3-octave centres — a readout
   that slid smoothly between bands would be showing a number nobody measured.
   --------------------------------------------------------------------------- */

/** Build (once) the container, the overlay canvas, the readout and the reset. */
function ensureChartInteraction(canvas) {
  const record = chartRecord(canvas);
  if (record.wired) return record;

  const host = canvas.parentElement;
  if (!host) return record;

  const shell = document.createElement('div');
  shell.className = 'chart';
  host.insertBefore(shell, canvas);
  shell.appendChild(canvas);

  const overlay = document.createElement('canvas');
  overlay.className = 'chart-overlay';
  overlay.setAttribute('aria-hidden', 'true');
  shell.appendChild(overlay);

  const readout = document.createElement('div');
  readout.className = 'chart-readout';
  readout.setAttribute('aria-hidden', 'true');
  readout.hidden = true;
  shell.appendChild(readout);

  const reset = document.createElement('button');
  reset.type = 'button';
  reset.className = 'btn btn-secondary btn-sm chart-reset';
  reset.textContent = 'Reset zoom';
  reset.hidden = true;
  shell.appendChild(reset);

  const save = document.createElement('button');
  save.type = 'button';
  save.className = 'btn-icon btn-icon-sm chart-save';
  save.setAttribute('aria-label', 'Save this chart as a high-resolution image');
  save.innerHTML = '';
  const saveIcon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  saveIcon.setAttribute('class', 'icon');
  saveIcon.setAttribute('aria-hidden', 'true');
  const saveUse = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  saveUse.setAttribute('href', '#i-download');
  saveIcon.appendChild(saveUse);
  save.appendChild(saveIcon);
  shell.appendChild(save);

  const hint = document.createElement('p');
  hint.className = 'chart-hint';
  hint.textContent = 'Drag to zoom a range · double-click to reset · arrow keys to step';
  shell.appendChild(hint);

  // The container, not the canvas, takes focus: the canvas stays role="img"
  // describing the picture, and the group around it is what you operate.
  shell.tabIndex = 0;
  shell.setAttribute('role', 'group');
  shell.setAttribute('aria-label', canvas.getAttribute('aria-label') || 'Chart');

  record.shell = shell;
  record.overlay = overlay;
  record.readout = readout;
  record.reset = reset;
  record.save = save;
  record.wired = true;

  save.addEventListener('click', () => exportChart(canvas));

  wireChartPointer(canvas, record);
  reset.addEventListener('click', () => {
    record.view = null;
    // A chart that owns its own zoom has to be told to drop it; clearing the
    // index view alone would leave the waveform on its narrowed time window.
    if (record.spec && typeof record.spec.onReset === 'function') record.spec.onReset();
    redrawChart(canvas);
  });
  return record;
}

/* ===========================================================================
   CONTINUOUS-X CHARTS

   The banded chart above puts one measurement in each of N equal slots. This
   one puts them at their real position on a linear or logarithmic axis, which
   is what a waveform (time), an absorption curve (frequency, decades) and a
   drift plot (shot number) each need.

   It shares everything it can with the banded chart — the same canvas
   preparation, the same palette, the same two-layer interaction — and differs
   only in how a value becomes an x coordinate.
   =========================================================================== */

/** Nice round tick values covering [lo, hi] on a linear axis. */
function linearTicks(lo, hi, target = 6) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / target;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalised = raw / magnitude;
  const step = (normalised >= 5 ? 10 : normalised >= 2 ? 5 : normalised >= 1 ? 2 : 1) * magnitude;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return ticks;
}

/** 1-2-5 ticks per decade, which is what a log frequency axis is read in. */
function logTicks(lo, hi) {
  if (!(lo > 0) || !(hi > lo)) return [];
  const ticks = [];
  const first = Math.floor(Math.log10(lo));
  const last = Math.ceil(Math.log10(hi));
  for (let decade = first; decade <= last; decade += 1) {
    for (const mantissa of [1, 2, 5]) {
      const value = mantissa * Math.pow(10, decade);
      if (value >= lo && value <= hi) ticks.push(value);
    }
  }
  return ticks;
}

/**
 * A chart with a real x axis.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{
 *   x: number[], xScale?: 'linear'|'log', xFormat?: function, xTitle?: string,
 *   unit?: string, zeroLine?: boolean, yMin?: number, yMax?: number,
 *   series: {label:string, color:string, kind:'line'|'step'|'area'|'band'|'marker',
 *            values?: number[], lower?: number[], upper?: number[]}[],
 *   markers?: {x:number, label:string}[],
 * }} spec
 */
function drawXYChart(canvas, spec) {
  const surface = prepareCanvas(canvas);
  if (!surface) return;
  const { ctx, width, height } = surface;
  const palette = chartPalette();
  if (!palette.surface || !palette.axis) {
    console.error('tokens.css did not resolve; the chart cannot be drawn in the current theme');
    return;
  }

  const allX = (spec.x || []).filter(isNum);
  const allSeries = (spec.series || []).filter(s => Boolean(s.color)
    && (Array.isArray(s.values) || (Array.isArray(s.lower) && Array.isArray(s.upper))));
  if (allX.length < 2 || allSeries.length === 0) {
    chartMessage(canvas, 'No data to plot');
    return;
  }

  const record = chartRecord(canvas);
  record.spec = { ...spec, x: allX, series: allSeries, frequencies: allX };
  const last = allX.length - 1;
  let view = record.view;
  if (!view || view.i0 > last || view.i1 > last || view.i1 - view.i0 < 1) {
    view = { i0: 0, i1: last };
    record.view = null;
  }

  const slice = (a) => (Array.isArray(a) ? a.slice(view.i0, view.i1 + 1) : null);
  const xs = slice(allX);
  const series = allSeries.map(s => ({
    ...s, values: slice(s.values), lower: slice(s.lower), upper: slice(s.upper),
  }));

  const yPool = series.flatMap(s => [...(s.values || []), ...(s.lower || []), ...(s.upper || [])])
    .filter(isNum);
  // Same as above: spec is already the new one, geom is still the old one.
  if (yPool.length === 0) { chartMessage(canvas, 'No data in the selected range'); return; }

  ctx.fillStyle = palette.surface;
  ctx.fillRect(0, 0, width, height);

  const pad = { left: 58, right: 14, top: 14, bottom: 40 };
  const plotW = Math.max(10, width - pad.left - pad.right);
  const plotH = Math.max(10, height - pad.top - pad.bottom);

  // Reduced, not spread. Math.min(...array) passes every element as an
  // argument and throws RangeError once the array outgrows the engine's
  // argument limit — which a waveform envelope is already within one zoom
  // level of reaching.
  let lo = isNum(spec.yMin) ? spec.yMin : yPool.reduce((a, b) => (b < a ? b : a), Infinity);
  let hi = isNum(spec.yMax) ? spec.yMax : yPool.reduce((a, b) => (b > a ? b : a), -Infinity);
  if (spec.zeroLine) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
  if (!isNum(spec.yMin) && !isNum(spec.yMax)) {
    const span = hi - lo;
    lo -= span * 0.08;
    hi += span * 0.08;
  }

  const logX = spec.xScale === 'log';
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  const lx0 = logX ? Math.log10(Math.max(x0, 1e-9)) : x0;
  const lx1 = logX ? Math.log10(Math.max(x1, 1e-9)) : x1;
  const spanX = (lx1 - lx0) || 1;
  const xOf = (v) => pad.left + ((logX ? Math.log10(Math.max(v, 1e-9)) : v) - lx0) / spanX * plotW;
  const yOf = (v) => pad.top + plotH - ((v - lo) / (hi - lo)) * plotH;

  // ---- grid, y ticks ----
  ctx.font = `11px ${palette.mono}`;
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const decimals = (hi - lo) < 5 ? 1 : 0;
  for (const value of linearTicks(lo, hi, 5)) {
    const y = Math.round(yOf(value)) + 0.5;
    if (y < pad.top - 1 || y > pad.top + plotH + 1) continue;
    ctx.strokeStyle = palette.grid;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = palette.muted;
    ctx.fillText(value.toFixed(decimals), pad.left - 8, y);
  }

  // ---- x ticks ----
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const format = spec.xFormat || ((v) => String(v));
  const ticks = logX ? logTicks(x0, x1) : linearTicks(x0, x1, 7);
  let lastRight = -Infinity;
  for (const value of ticks) {
    const x = xOf(value);
    if (x < pad.left - 1 || x > pad.left + plotW + 1) continue;
    const text = format(value);
    const halfWidth = ctx.measureText(text).width / 2;
    if (x - halfWidth < lastRight + 6) continue;      // never let two labels touch
    lastRight = x + halfWidth;
    ctx.strokeStyle = palette.grid;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, pad.top);
    ctx.lineTo(Math.round(x) + 0.5, pad.top + plotH);
    ctx.stroke();
    ctx.fillStyle = palette.muted;
    ctx.fillText(text, x, pad.top + plotH + 8);
  }

  ctx.strokeStyle = palette.axis;
  ctx.beginPath();
  ctx.moveTo(pad.left + 0.5, pad.top);
  ctx.lineTo(pad.left + 0.5, pad.top + plotH + 0.5);
  ctx.lineTo(pad.left + plotW, pad.top + plotH + 0.5);
  ctx.stroke();

  // The x title belongs under the x axis and the y unit beside the y axis.
  // Joining them into one string put "Frequency, Hz - dB" under the abscissa,
  // which reads as a single compound unit and is not one.
  ctx.fillStyle = palette.text;
  ctx.textAlign = 'center';
  if (spec.xTitle) ctx.fillText(spec.xTitle, pad.left + plotW / 2, pad.top + plotH + 24);
  if (spec.unit) {
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = palette.muted;
    ctx.fillText(spec.unit, pad.left - 46, pad.top - 2);
    ctx.textBaseline = 'top';
  }

  if (spec.zeroLine && lo < 0 && hi > 0) {
    const y = Math.round(yOf(0)) + 0.5;
    ctx.strokeStyle = palette.axis;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // ---- series ----
  for (const s of series) {
    if (s.kind === 'band') {
      // A filled region between two measured bounds — an envelope, or an
      // interval. Never a smoothed ribbon around a single line.
      ctx.fillStyle = s.color;
      ctx.globalAlpha = isNum(s.alpha) ? s.alpha : 0.85;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < xs.length; i += 1) {
        if (!isNum(s.upper[i])) continue;
        const x = xOf(xs[i]);
        const y = yOf(s.upper[i]);
        if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
      }
      for (let i = xs.length - 1; i >= 0; i -= 1) {
        if (!isNum(s.lower[i])) continue;
        ctx.lineTo(xOf(xs[i]), yOf(s.lower[i]));
      }
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
      continue;
    }

    if (s.kind === 'marker') {
      ctx.fillStyle = s.color;
      for (let i = 0; i < xs.length; i += 1) {
        if (!isNum(s.values[i])) continue;
        ctx.beginPath();
        ctx.arc(xOf(xs[i]), yOf(s.values[i]), 3, 0, Math.PI * 2);
        ctx.fill();
      }
      continue;
    }

    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.kind === 'area' ? 1.5 : 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    let started = false;
    let prevY = 0;
    for (let i = 0; i < xs.length; i += 1) {
      const v = s.values[i];
      if (!isNum(v)) { started = false; continue; }
      const x = xOf(xs[i]);
      const y = yOf(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else if (s.kind === 'step') { ctx.lineTo(x, prevY); ctx.lineTo(x, y); }
      else ctx.lineTo(x, y);
      prevY = y;
    }
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  // ---- event markers (shot times, thresholds) ----
  for (const marker of spec.markers || []) {
    if (!isNum(marker.x) || marker.x < x0 || marker.x > x1) continue;
    const x = Math.round(xOf(marker.x)) + 0.5;
    ctx.strokeStyle = palette.shotMarker || palette.warn;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    if (marker.label) {
      ctx.fillStyle = palette.muted;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(marker.label, x, pad.top + 2);
    }
  }

  record.geom = {
    kind: 'xy', pad, plotW, plotH, lo, hi, width, height,
    i0: view.i0, i1: view.i1,
    xs: xs.map(xOf),
    xValues: allX,
    xFormat: format,
    xNoun: spec.xNoun || (spec.xTitle ? spec.xTitle.toLowerCase() : 'range'),
    slot: plotW / Math.max(1, xs.length),
  };

  ensureChartInteraction(canvas);
  drawChartOverlay(canvas);
}


/* ===========================================================================
   HEATMAP — the spectrogram

   The one chart whose marks are pixels rather than shapes. The matrix is
   painted once into an offscreen canvas at its own resolution and then scaled
   into place, so panning and zooming never re-run the colour lookup over a
   hundred thousand cells.

   CIVIDIS, and not by preference. plots.py sets CMAP_SPECTROGRAM = 'cividis'
   for its figures; if this used a different ramp the same measurement would
   be two different pictures depending on whether you were looking at the
   screen or the report. It is also perceptually uniform and safe for the
   commonest colour-vision deficiencies, which a rainbow ramp is not: on a
   rainbow, equal steps in level are unequal steps in apparent brightness, so
   the eye invents structure the data does not have.
   =========================================================================== */

const CIVIDIS_STOPS = [
  [0.0000, 0, 34, 78], [0.0667, 0, 46, 108], [0.1333, 30, 58, 111],
  [0.2000, 53, 69, 108], [0.2667, 71, 81, 108], [0.3333, 87, 93, 109],
  [0.4000, 102, 105, 112], [0.4667, 117, 117, 117], [0.5333, 132, 130, 121],
  [0.6000, 148, 142, 119], [0.6667, 165, 156, 116], [0.7333, 183, 169, 110],
  [0.8000, 200, 184, 102], [0.8667, 219, 199, 90], [0.9333, 238, 214, 73],
  [1.0000, 254, 232, 56],
];

let cividisLUT = null;

/** 256-entry RGB lookup, built once. */
function colourLUT() {
  if (cividisLUT) return cividisLUT;
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i += 1) {
    const t = i / 255;
    let k = 0;
    while (k < CIVIDIS_STOPS.length - 2 && CIVIDIS_STOPS[k + 1][0] < t) k += 1;
    const [t0, r0, g0, b0] = CIVIDIS_STOPS[k];
    const [t1, r1, g1, b1] = CIVIDIS_STOPS[k + 1];
    const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
    lut[i * 3] = r0 + (r1 - r0) * f;
    lut[i * 3 + 1] = g0 + (g1 - g0) * f;
    lut[i * 3 + 2] = b0 + (b1 - b0) * f;
  }
  cividisLUT = lut;
  return lut;
}

/** base64 big-endian uint16 -> Uint16Array, in the order it was written. */
function decodeMatrix(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  const out = new Uint16Array(bytes.length >> 1);
  for (let i = 0; i < out.length; i += 1) out[i] = (bytes[i * 2] << 8) | bytes[i * 2 + 1];
  return out;
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {{matrix:Uint16Array, frames:number, bins:number, time:number[],
 *          frequencies:number[], offset:number, step:number, missing:number,
 *          vmin:number, vmax:number, unit:string, xTitle:string, yTitle:string}} spec
 */
function drawHeatmapChart(canvas, spec) {
  const surface = prepareCanvas(canvas);
  if (!surface) return;
  const { ctx, width, height } = surface;
  const palette = chartPalette();
  if (!palette.surface) return;

  const { frames, bins } = spec;
  if (!frames || !bins || !spec.matrix) { chartMessage(canvas, 'No spectrogram data'); return; }

  const record = chartRecord(canvas);
  record.spec = { ...spec, frequencies: spec.time, unit: spec.unit || 'dB', series: [] };
  let view = record.view;
  if (!view || view.i1 > frames - 1 || view.i1 - view.i0 < 1) {
    view = { i0: 0, i1: frames - 1 };
    record.view = null;
  }

  ctx.fillStyle = palette.surface;
  ctx.fillRect(0, 0, width, height);

  const pad = { left: 58, right: 70, top: 14, bottom: 40 };
  const plotW = Math.max(10, width - pad.left - pad.right);
  const plotH = Math.max(10, height - pad.top - pad.bottom);

  const span = Math.max(1e-6, spec.vmax - spec.vmin);
  const lut = colourLUT();
  const cols = view.i1 - view.i0 + 1;

  // Painted at the matrix's own resolution, then scaled by the compositor.
  // Rasterising per screen pixel would redo the colour lookup on every zoom.
  const tile = document.createElement('canvas');
  tile.width = cols;
  tile.height = bins;
  const tileCtx = tile.getContext('2d');
  const image = tileCtx.createImageData(cols, bins);
  const pixels = image.data;
  for (let row = 0; row < bins; row += 1) {
    // Frequency increases upward on the axis and downward in image rows.
    const sourceRow = bins - 1 - row;
    for (let col = 0; col < cols; col += 1) {
      const raw = spec.matrix[sourceRow * frames + (view.i0 + col)];
      const target = (row * cols + col) * 4;
      if (raw === spec.missing) { pixels[target + 3] = 0; continue; }
      const dB = spec.offset + raw * spec.step;
      const index = clamp(Math.round(((dB - spec.vmin) / span) * 255), 0, 255);
      pixels[target] = lut[index * 3];
      pixels[target + 1] = lut[index * 3 + 1];
      pixels[target + 2] = lut[index * 3 + 2];
      pixels[target + 3] = 255;
    }
  }
  tileCtx.putImageData(image, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tile, pad.left, pad.top, plotW, plotH);
  ctx.imageSmoothingEnabled = true;

  // ---- axes ----
  const f0 = spec.frequencies[0];
  const f1 = spec.frequencies[spec.frequencies.length - 1];
  const yOf = (hz) => pad.top + plotH - ((hz - f0) / Math.max(1e-9, f1 - f0)) * plotH;
  ctx.font = `11px ${palette.mono}`;
  ctx.fillStyle = palette.muted;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const hz of linearTicks(f0, f1, 5)) {
    const y = yOf(hz);
    if (y < pad.top - 1 || y > pad.top + plotH + 1) continue;
    ctx.fillText(hz >= 1000 ? `${Math.round(hz / 1000)}k` : String(Math.round(hz)), pad.left - 8, y);
  }

  const t0 = spec.time[view.i0];
  const t1 = spec.time[view.i1];
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  let lastRight = -Infinity;
  for (const t of linearTicks(t0, t1, 7)) {
    const x = pad.left + ((t - t0) / Math.max(1e-9, t1 - t0)) * plotW;
    if (x < pad.left - 1 || x > pad.left + plotW + 1) continue;
    const text = `${t.toFixed(t1 - t0 < 1 ? 3 : 2)} s`;
    const half = ctx.measureText(text).width / 2;
    if (x - half < lastRight + 6) continue;
    lastRight = x + half;
    ctx.fillText(text, x, pad.top + plotH + 8);
  }
  ctx.fillStyle = palette.text;
  ctx.fillText(spec.xTitle || 'Time', pad.left + plotW / 2, pad.top + plotH + 24);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillStyle = palette.muted;
  ctx.fillText(spec.yTitle || 'Hz', pad.left - 46, pad.top - 2);

  ctx.strokeStyle = palette.axis;
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left + 0.5, pad.top + 0.5, plotW, plotH);

  // ---- colour scale ----
  const barX = pad.left + plotW + 14;
  const barW = 12;
  for (let i = 0; i < plotH; i += 1) {
    const index = clamp(Math.round((1 - i / plotH) * 255), 0, 255);
    ctx.fillStyle = `rgb(${lut[index * 3]},${lut[index * 3 + 1]},${lut[index * 3 + 2]})`;
    ctx.fillRect(barX, pad.top + i, barW, 1);
  }
  ctx.strokeRect(barX + 0.5, pad.top + 0.5, barW, plotH);
  ctx.fillStyle = palette.muted;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(spec.vmax.toFixed(0), barX + barW + 4, pad.top + 4);
  ctx.fillText(spec.vmin.toFixed(0), barX + barW + 4, pad.top + plotH - 4);

  record.geom = {
    kind: 'xy',                 // time is the interactive axis, as on a waveform
    pad, plotW, plotH, width, height,
    lo: f0, hi: f1,
    i0: view.i0, i1: view.i1,
    xs: Array.from({ length: cols }, (_, i) => pad.left + ((i + 0.5) / cols) * plotW),
    xValues: spec.time,
    xFormat: (t) => `${Number(t).toFixed(3)} s`,
    xNoun: 'recording',
    heatmap: { bins, frames, yOf, f0, f1 },
  };

  ensureChartInteraction(canvas);
  drawChartOverlay(canvas);
}


/* ---- mapping a pixel to a measurement, and back ---------------------------
   Two chart shapes share one interaction layer. A BAND chart has evenly
   spaced categorical slots; an XY chart has samples at arbitrary positions on
   a linear or logarithmic axis. Both resolve a pointer to the INDEX of a
   measured point, never to a position between two of them, so the readout can
   only ever show a number that was measured. */

/** Pixel x -> absolute index of the nearest measured point, clamped to view. */
function indexAtX(record, x) {
  const g = record.geom;
  if (!g) return null;
  if (g.kind === 'xy') {
    // The positions are monotonic, so a binary search finds the neighbours and
    // the nearer of the two wins. Linear scanning was fine at 31 bands and is
    // not at a few thousand waveform columns.
    const xs = g.xs;
    if (!xs || xs.length === 0) return null;
    let lo = 0;
    let hi = xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= x) lo = mid; else hi = mid;
    }
    const nearest = Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
    return g.i0 + nearest;
  }
  const visible = g.i1 - g.i0 + 1;
  const local = clamp(Math.floor((x - g.pad.left) / g.slot), 0, visible - 1);
  return g.i0 + local;
}

/** Absolute index -> pixel x. */
function xAt(record, index) {
  const g = record.geom;
  if (g.kind === 'xy') {
    const local = clamp(index - g.i0, 0, (g.xs || []).length - 1);
    return g.xs[local];
  }
  return g.pad.left + g.slot * (index - g.i0 + 0.5);
}

/** How the x value at an index is written in the readout and announcements. */
function xLabelAt(record, index) {
  const g = record.geom;
  const spec = record.spec;
  if (!g || !spec) return '';
  const format = g.xFormat || fmtHz;
  const values = g.xValues || spec.frequencies || [];
  return format(values[index]);
}

function wireChartPointer(canvas, record) {
  const shell = record.shell;
  let frame = null;

  const schedule = () => {
    if (frame !== null) return;
    frame = requestAnimationFrame(() => { frame = null; drawChartOverlay(canvas); });
  };

  const localX = (ev) => {
    const rect = canvas.getBoundingClientRect();
    // A heatmap needs both axes to resolve a cell; the line charts ignore y.
    record.cursorY = ev.clientY - rect.top;
    return ev.clientX - rect.left;
  };

  shell.addEventListener('pointermove', (ev) => {
    if (!record.geom) return;
    const x = localX(ev);
    const index = indexAtX(record, x);
    if (record.drag) record.drag.to = index;
    if (record.cursor === index && !record.drag) return;
    record.cursor = index;
    schedule();
  });

  shell.addEventListener('pointerleave', () => {
    if (record.drag) return;
    record.cursor = null;
    schedule();
  });

  shell.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0 || !record.geom) return;
    const index = indexAtX(record, localX(ev));
    record.drag = { from: index, to: index };
    record.cursor = index;
    shell.setPointerCapture(ev.pointerId);
    schedule();
  });

  shell.addEventListener('pointerup', (ev) => {
    if (!record.drag) return;
    const { from, to } = record.drag;
    record.drag = null;
    if (shell.hasPointerCapture(ev.pointerId)) shell.releasePointerCapture(ev.pointerId);
    // A drag of a single band is a click, not a range: zooming to one bar
    // would leave a chart with nothing to compare against.
    if (Math.abs(to - from) >= 1) {
      const i0 = Math.min(from, to);
      const i1 = Math.max(from, to);
      // A chart that can fetch more detail for a range owns the zoom itself:
      // it keeps the range in DATA units, picks a finer source, and redraws
      // from scratch. Index-based zoom cannot survive the array changing
      // underneath it.
      const spec = record.spec;
      if (spec && typeof spec.onZoom === 'function') {
        const xs = spec.x || spec.frequencies || [];
        record.view = null;
        spec.onZoom(xs[i0], xs[i1]);
      } else {
        record.view = { i0, i1 };
        redrawChart(canvas);
      }
      announceChartRange(canvas, record);
    } else {
      schedule();
    }
  });

  // A drag interrupted by the system — a phone call, a gesture the browser
  // decides is a scroll — fires pointercancel and NOT pointerup, so without
  // this the selection rectangle and the crosshair stay on screen for ever.
  shell.addEventListener('pointercancel', (ev) => {
    if (!record.drag) return;
    record.drag = null;
    if (shell.hasPointerCapture && shell.hasPointerCapture(ev.pointerId)) {
      shell.releasePointerCapture(ev.pointerId);
    }
    record.cursor = null;
    drawChartOverlay(canvas);
  });

  shell.addEventListener('dblclick', () => {
    // chartIsZoomed, not record.view: a chart that owns its own zoom (the
    // waveform keeps a time window) has no index view, so double-click was
    // dead on the one chart people zoom most.
    if (!chartIsZoomed(record)) return;
    record.view = null;
    if (record.spec && typeof record.spec.onReset === 'function') record.spec.onReset();
    redrawChart(canvas);
    announce(`Zoom reset; the whole ${record.geom.xNoun || 'range'} is shown.`);
  });

  shell.addEventListener('keydown', (ev) => {
    if (!record.geom) return;
    const g = record.geom;
    const step = ev.shiftKey ? 5 : 1;
    let handled = true;
    if (ev.key === 'ArrowRight') {
      record.cursor = clamp((record.cursor === null ? g.i0 - 1 : record.cursor) + step, g.i0, g.i1);
    } else if (ev.key === 'ArrowLeft') {
      record.cursor = clamp((record.cursor === null ? g.i1 + 1 : record.cursor) - step, g.i0, g.i1);
    } else if (ev.key === 'Home') { record.cursor = g.i0; }
    else if (ev.key === 'End') { record.cursor = g.i1; }
    else if (ev.key === 'Escape') {
      if (record.view || (record.spec && record.spec.onReset)) {
        record.view = null;
        if (record.spec && typeof record.spec.onReset === 'function') record.spec.onReset();
        redrawChart(canvas);
        announce(`Zoom reset; the whole ${g.xNoun || 'range'} is shown.`);
      }
      record.cursor = null;
    } else { handled = false; }
    if (!handled) return;
    ev.preventDefault();
    drawChartOverlay(canvas);
    announceChartCursor(canvas, record);
  });
}

/**
 * Save the chart as a high-resolution PNG.
 *
 * Not a screenshot. The chart is RE-DRAWN at EXPORT_SCALE times the screen's
 * pixel ratio, so every rule, glyph and axis label is rasterised at that
 * resolution — zooming into the saved file finds real detail rather than
 * interpolation. A 960 px chart lands at 3840 px, which prints past 300 dpi at
 * a full page width.
 *
 * What is exported is exactly what is on screen, INCLUDING the current zoom.
 * A saved figure that quietly reverted to the full range would not be the
 * thing the operator was looking at when they pressed the button.
 */
const EXPORT_TARGET_PX = 3840;

function exportChart(canvas) {
  const draw = chartRegistry.get(canvas.id);
  if (!draw) return;
  const record = chartRecord(canvas);

  // The crosshair lives on the overlay and is not part of the figure.
  const cursor = record.cursor;
  record.cursor = null;

  // A fixed TARGET WIDTH, not a multiple of the screen's pixel ratio. Scaling
  // by the ratio would hand a Retina user an 8x render — four times the pixels
  // of a standard display for the same figure, and close to the browser's
  // canvas limits on a wide chart. 3840 px is past 300 dpi across a printed
  // page and is the same file whatever machine saved it.
  const cssWidth = canvas.getBoundingClientRect().width || 960;
  canvas.dataset.exportScale = String(clamp(EXPORT_TARGET_PX / cssWidth, 2, 8));
  try {
    draw();
  } catch (err) {
    console.error('export redraw failed', err);
  }

  const finish = () => {
    delete canvas.dataset.exportScale;
    record.cursor = cursor;
    try { draw(); } catch { /* the on-screen redraw is best effort */ }
  };

  const name = `${(canvas.id || 'chart').replace(/-canvas$/, '')}.png`;
  if (typeof canvas.toBlob !== 'function') { finish(); return; }
  canvas.toBlob((blob) => {
    if (blob) {
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      toast({
        title: 'Chart saved',
        text: `${name} at ${Math.round(canvas.width)} × ${Math.round(canvas.height)} pixels.`,
        tone: 'ok',
      });
    }
    finish();
  }, 'image/png');
}

/** Whether this chart is showing less than all of its data. */
function chartIsZoomed(record) {
  if (!record) return false;
  if (record.view) return true;
  const spec = record.spec;
  return Boolean(spec && typeof spec.isZoomed === 'function' && spec.isZoomed());
}

/** Redraw the plot layer through whatever function owns this canvas. */
function redrawChart(canvas) {
  const draw = chartRegistry.get(canvas.id);
  if (draw) { draw(); return; }
  redrawCharts();
}

function chartValuesAt(record, index) {
  const spec = record.spec;
  if (!spec) return [];

  const grid = record.geom && record.geom.heatmap;
  if (grid) {
    // One cell, addressed by both axes. Reporting the column alone would give
    // a level with no frequency attached to it, which is not a measurement.
    const { bins, frames, f0, f1 } = grid;
    const g = record.geom;
    const y = clamp(record.cursorY === undefined ? g.pad.top : record.cursorY,
      g.pad.top, g.pad.top + g.plotH);
    const fraction = 1 - (y - g.pad.top) / g.plotH;
    const row = clamp(Math.round(fraction * (bins - 1)), 0, bins - 1);
    const raw = spec.matrix[row * frames + index];
    const hz = f0 + (f1 - f0) * (row / Math.max(1, bins - 1));
    if (raw === spec.missing) {
      return [{ label: fmtHz(hz), color: cssVar('--text-3'), text: 'no value' }];
    }
    return [{
      label: fmtHz(hz),
      color: cssVar('--accent'),
      value: spec.offset + raw * spec.step,
    }];
  }

  const out = [];
  for (const s of spec.series) {
    // A band has no single value at a point — it has two, and reporting one of
    // them (or their average, which was measured by nobody) would misstate it.
    if (s.kind === 'band') {
      const lower = (s.lower || [])[index];
      const upper = (s.upper || [])[index];
      if (isNum(lower) && isNum(upper)) {
        out.push({ label: s.label, color: s.color, value: upper, pair: [lower, upper],
          unit: s.unit, signed: s.signed });
      }
      continue;
    }
    const value = (s.values || [])[index];
    // unit and signed are per-series: one chart can legitimately carry a
    // DIFFERENCE and the two ABSOLUTE levels it was computed from, and those
    // are not the same quantity. Reading "+126.5 dB" off a reference level
    // states a change where there is a measurement.
    if (isNum(value)) {
      out.push({ label: s.label, color: s.color, value, unit: s.unit, signed: s.signed });
    }
  }
  return out;
}

function announceChartRange(canvas, record) {
  const spec = record.spec;
  if (!spec) return;
  if (!record.view) { announce('Zoomed in.'); return; }
  announce(`Zoomed to ${xLabelAt(record, record.view.i0)} through `
    + `${xLabelAt(record, record.view.i1)}.`);
}

function announceChartCursor(canvas, record) {
  const spec = record.spec;
  if (!spec || record.cursor === null) return;
  const digits = readoutDigits(record.geom);
  const parts = chartValuesAt(record, record.cursor).map((entry) => {
    const unit = entry.unit || spec.unit || 'dB';
    const signed = entry.signed === undefined ? Boolean(spec.zeroLine) : Boolean(entry.signed);
    if (entry.pair) {
      return `${entry.label} ${fmt(entry.pair[0], digits)} to ${fmt(entry.pair[1], digits)} ${unit}`;
    }
    return `${entry.label} ${signed ? fmtSigned(entry.value, digits) : fmt(entry.value, digits)} ${unit}`;
  });
  announce(`${xLabelAt(record, record.cursor)}: ${parts.join(', ') || 'no value'}`);
}

/**
 * The cheap layer. Clears and redraws only the crosshair, the highlighted
 * slot, the value dots and the drag selection.
 */
function drawChartOverlay(canvas) {
  const record = chartState.get(canvas);
  if (!record || !record.overlay || !record.geom || !record.spec) return;
  const g = record.geom;
  const overlay = record.overlay;
  const dpr = window.devicePixelRatio || 1;

  overlay.style.width = `${g.width}px`;
  overlay.style.height = `${g.height}px`;
  const backingWidth = Math.round(g.width * dpr);
  const backingHeight = Math.round(g.height * dpr);
  if (overlay.width !== backingWidth) overlay.width = backingWidth;
  if (overlay.height !== backingHeight) overlay.height = backingHeight;

  const ctx = overlay.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, g.width, g.height);

  const palette = chartPalette();
  // Asked of the chart, not inferred from record.view. A chart that owns its
  // own zoom (the waveform keeps a time window, not an index range) has
  // record.view === null while zoomed, so this used to hide the reset button
  // on the very charts that most needed it — and hide it again on every
  // pointer move for the ones that had it.
  show(record.reset, chartIsZoomed(record));

  // ---- drag selection ----
  if (record.drag && Math.abs(record.drag.to - record.drag.from) >= 1) {
    const half = g.kind === 'xy' ? 0 : g.slot / 2;
    const a = xAt(record, Math.min(record.drag.from, record.drag.to)) - half;
    const b = xAt(record, Math.max(record.drag.from, record.drag.to)) + half;
    ctx.fillStyle = palette.accent;
    ctx.globalAlpha = 0.14;
    ctx.fillRect(a, g.pad.top, b - a, g.plotH);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = palette.accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(a) + 0.5, g.pad.top);
    ctx.lineTo(Math.round(a) + 0.5, g.pad.top + g.plotH);
    ctx.moveTo(Math.round(b) + 0.5, g.pad.top);
    ctx.lineTo(Math.round(b) + 0.5, g.pad.top + g.plotH);
    ctx.stroke();
  }

  const index = record.cursor;
  if (index === null || index < g.i0 || index > g.i1) {
    show(record.readout, false);
    return;
  }

  // ---- crosshair ----
  const x = Math.round(xAt(record, index)) + 0.5;
  ctx.strokeStyle = palette.accent;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(x, g.pad.top);
  ctx.lineTo(x, g.pad.top + g.plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // ---- a dot on each series at this band ----
  const entries = chartValuesAt(record, index);
  const yOf = (v) => g.pad.top + g.plotH - ((v - g.lo) / (g.hi - g.lo)) * g.plotH;
  for (const entry of entries) {
    // On a heatmap the y axis is FREQUENCY and the value is a LEVEL, so the
    // value cannot be placed on it — it was landing on the floor of the plot
    // every time. The mark belongs at the cell the pointer is over.
    const y = g.heatmap
      ? clamp(record.cursorY === undefined ? g.pad.top : record.cursorY,
        g.pad.top, g.pad.top + g.plotH)
      : yOf(entry.value);
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = entry.color;
    ctx.fill();
    ctx.strokeStyle = palette.surface;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  renderChartReadout(record, index, entries, x);
}

/** Decimal places that resolve the visible range into about 500 steps. */
function readoutDigits(geom) {
  const span = geom && isNum(geom.hi) && isNum(geom.lo) ? Math.abs(geom.hi - geom.lo) : 0;
  if (!(span > 0)) return 1;
  if (span < 0.5) return 4;
  if (span < 5) return 3;
  if (span < 50) return 2;
  return 1;
}

function renderChartReadout(record, index, entries, x) {
  const readout = record.readout;
  const spec = record.spec;
  if (!readout || !spec) return;

  readout.textContent = '';
  const head = document.createElement('p');
  head.className = 'chart-readout-head';
  head.textContent = xLabelAt(record, index);
  readout.appendChild(head);

  for (const entry of entries) {
    const row = document.createElement('p');
    row.className = 'chart-readout-row';

    const swatch = document.createElement('span');
    swatch.className = 'chart-readout-swatch';
    swatch.style.background = entry.color;
    row.appendChild(swatch);

    const label = document.createElement('span');
    label.className = 'chart-readout-label';
    label.textContent = entry.label;
    row.appendChild(label);

    const value = document.createElement('span');
    value.className = 'chart-readout-value';
    if (entry.text) {
      value.textContent = entry.text;
      row.appendChild(value);
      readout.appendChild(row);
      continue;
    }
    // A difference carries its sign; an absolute level does not. zeroLine is
    // set exactly on the charts whose values are differences.
    // Precision follows the range actually on screen. One decimal is right
    // for a 40 dB band chart and useless on an absorption curve whose whole
    // span is half a decibel — every reading there would print as "0.0".
    const digits = readoutDigits(record.geom);
    const unit = entry.unit || spec.unit || 'dB';
    const signed = entry.signed === undefined ? Boolean(spec.zeroLine) : Boolean(entry.signed);
    value.textContent = entry.pair
      ? `${fmt(entry.pair[0], digits)} to ${fmt(entry.pair[1], digits)} ${unit}`
      : `${signed ? fmtSigned(entry.value, digits) : fmt(entry.value, digits)} ${unit}`;
    row.appendChild(value);

    readout.appendChild(row);
  }

  show(readout, entries.length > 0);
  if (entries.length === 0) return;

  // Flip to whichever side of the crosshair has room, so the readout never
  // covers the part of the trace the operator is pointing at.
  const g = record.geom;
  const box = readout.getBoundingClientRect();
  const wantLeft = x + 12 + box.width > g.width - g.pad.right;
  readout.style.left = wantLeft
    ? `${Math.max(g.pad.left, x - 12 - box.width)}px`
    : `${x + 12}px`;
  readout.style.top = `${g.pad.top + 4}px`;
}

/** Band exposure for the selected shot against the aggregate of the rest. */
function drawShotBandChart() {
  const canvas = $('shot-band-canvas');
  if (!canvas) return;
  chartRegistry.set('shot-band-canvas', drawShotBandChart);

  const shot = currentShot();
  const aggregate = state.results.aggregate;
  if (!shot || !Array.isArray(shot.band_exposure_dB) || shot.band_exposure_dB.length === 0) {
    chartMessage(canvas, shot ? 'No band exposure was computed for this shot' : 'No shot selected');
    return;
  }
  const palette = chartPalette();
  const frequencies = shot.band_frequencies_Hz
    || (aggregate && aggregate.band_frequencies_Hz)
    || [];
  const mean = aggregate && Array.isArray(aggregate.band_exposure_mean_dB)
    && aggregate.band_exposure_mean_dB.length === shot.band_exposure_dB.length
    ? aggregate.band_exposure_mean_dB : [];

  drawBandedChart(canvas, {
    frequencies,
    unit: levelUnit(),
    series: [
      { label: 'This shot', color: palette.series[3], kind: 'bar', values: shot.band_exposure_dB },
      { label: 'Aggregate mean', color: palette.series[2], kind: 'line', values: mean },
    ],
  });
  canvas.setAttribute('aria-label',
    `Band exposure for shot ${shot.shot_number} against the aggregate mean, in ${levelUnit()}.`);
}

function redrawCharts() {
  for (const draw of chartRegistry.values()) {
    try { draw(); } catch (err) { console.error('chart redraw failed', err); }
  }
}

function wireCharts() {
  // Layout changes (window resize, sidebar collapse) need a repaint too.
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(redrawCharts, 120);
  });
  window.addEventListener('beforeprint', redrawCharts);
}


/* ===========================================================================
   12. COMPARE — INSERTION LOSS

   Insertion loss is reference (unsuppressed) minus test (suppressed), so a
   POSITIVE figure means the suppressed configuration is quieter. The two 95 %
   half-widths are combined in quadrature. The comparison is REFUSED outright
   when the two records cannot legitimately be differenced.
   =========================================================================== */

const COMPARE_ROLES = { reference: 'ref', test: 'test' };

function compareSide(role) {
  return state.compare[role === 'reference' ? 'ref' : 'test'];
}

function savedRejections(dir) {
  const saved = store.get(STORAGE.rejects, {}) || {};
  return new Set(Array.isArray(saved[dir]) ? saved[dir] : []);
}

/**
 * The same aggregate arithmetic as the Results view, applied to any record:
 * levels energy-averaged, everything else arithmetic, sample sigma (ddof = 1),
 * 95 % half-width 1.96*sigma/sqrt(n). Engine-invalid and operator-rejected
 * shots are excluded on both sides identically.
 */
function aggregateFrom(metadata, dir) {
  const all = Array.isArray(metadata && metadata.per_shot_metrics) ? metadata.per_shot_metrics : [];
  const rejected = savedRejections(dir);
  const shots = all.filter(s => s.valid !== false && !rejected.has(s.shot_number));

  const statistics = {};
  for (const metric of METRICS) {
    statistics[metric.stat] = summarise(shots.map(s => s[metric.key]).filter(isNum), metric.level);
  }

  let frequencies = [];
  let bands = [];
  const withBands = shots.filter(s => Array.isArray(s.band_exposure_dB) && s.band_exposure_dB.length > 0);
  if (withBands.length > 0) {
    const width = withBands[0].band_exposure_dB.length;
    if (withBands.every(s => s.band_exposure_dB.length === width)) {
      frequencies = withBands[0].band_frequencies_Hz || [];
      for (let i = 0; i < width; i += 1) {
        bands.push(energyAverageDb(withBands.map(s => s.band_exposure_dB[i])));
      }
    }
  }
  return {
    n_total: all.length,
    n_included: shots.length,
    n_rejected: all.length - shots.length,
    statistics,
    band_frequencies_Hz: frequencies,
    band_exposure_mean_dB: bands,
  };
}

/** A one-line verdict for a whole record, used by History and by Compare. */
function quickVerdict(metadata) {
  const quality = (metadata && metadata.quality) || {};
  const calibration = (metadata && metadata.calibration) || {};
  const shots = Array.isArray(metadata && metadata.per_shot_metrics) ? metadata.per_shot_metrics : [];
  const errors = Array.isArray(quality.errors) ? quality.errors : [];
  const warnings = Array.isArray(quality.warnings) ? quality.warnings : [];

  if (quality.is_valid === false || errors.length > 0 || shots.length === 0 || quality.is_clipped === true) {
    return { tone: 'danger', icon: 'i-error', label: 'Not admissible' };
  }
  if (calibration.calibrated === false) {
    return { tone: 'warn', icon: 'i-alert', label: 'Uncalibrated' };
  }
  if (warnings.length > 0) {
    return { tone: 'warn', icon: 'i-alert', label: 'Cautions' };
  }
  return { tone: 'ok', icon: 'i-check-circle', label: 'Admissible' };
}

/**
 * Reasons the two records cannot, or should not, be differenced.
 * `fatal` refuses the comparison outright; the rest qualify it.
 */
function compareObjections(ref, test) {
  const objections = [];
  const add = (fatal, text) => objections.push({ fatal, text });

  const refCal = (ref.metadata.calibration) || {};
  const testCal = (test.metadata.calibration) || {};
  const refRecord = (ref.metadata.test_metadata) || {};
  const testRecord = (test.metadata.test_metadata) || {};
  const refSource = (ref.metadata.source) || {};
  const testSource = (test.metadata.source) || {};

  // ---- calibration ----
  if (refCal.calibrated !== testCal.calibrated) {
    add(true, 'One recording is calibrated and the other is not. Their level scales are '
      + 'unrelated, so the difference between them is not an insertion loss.');
  } else if ((refCal.level_unit || '') !== (testCal.level_unit || '')) {
    add(true, `The two records report different level units (${refCal.level_unit || '—'} against `
      + `${testCal.level_unit || '—'}), so they cannot be differenced.`);
  } else if (refCal.calibrated === false) {
    add(false, 'Both recordings are uncalibrated. Insertion loss is only valid here if the '
      + 'microphone, gain and converter were identical and untouched between the two runs — '
      + 'the unknown scale factor cancels only then.');
  } else if ((refCal.method || '') !== (testCal.method || '')) {
    add(false, `The two calibrations were established by different methods (${refCal.method || '—'} `
      + `against ${testCal.method || '—'}). Check that both are traceable to the same reference.`);
  } else if (isNum(refCal.Pa_per_FS) && isNum(testCal.Pa_per_FS)
             && Math.abs(20 * Math.log10(refCal.Pa_per_FS / testCal.Pa_per_FS)) > 0.5) {
    add(false, `The scale factors differ by ${fmt(Math.abs(20 * Math.log10(refCal.Pa_per_FS / testCal.Pa_per_FS)), 2)} dB. `
      + 'That is legitimate if the chain was re-calibrated, but it means the two runs were not identical.');
  }

  // ---- bands ----
  const refBands = ref.aggregate.band_frequencies_Hz;
  const testBands = test.aggregate.band_frequencies_Hz;
  if (refBands.length > 0 && testBands.length > 0) {
    const same = refBands.length === testBands.length
      && refBands.every((hz, i) => Math.abs(hz - testBands[i]) / Math.max(1, hz) < 0.01);
    if (!same) {
      add(true, `The two analyses used different band sets (${refBands.length} against `
        + `${testBands.length} bands). Per-band insertion loss cannot be computed.`);
    }
  }

  // ---- configuration ----
  if (refRecord.configuration && refRecord.configuration !== 'unsuppressed') {
    add(true, 'The reference is not recorded as an unsuppressed measurement.');
  } else if (!refRecord.configuration) {
    add(false, 'The reference record does not state its configuration.');
  }
  if (testRecord.configuration && testRecord.configuration !== 'suppressed') {
    add(true, 'The test is not recorded as a suppressed measurement.');
  } else if (!testRecord.configuration) {
    add(false, 'The test record does not state its configuration.');
  }

  // ---- geometry and setup ----
  const geometry = [
    ['mic_distance_m', 'microphone distance', 'm', 0.01],
    ['mic_angle_deg', 'microphone angle', '°', 0.5],
    ['mic_height_m', 'microphone height', 'm', 0.01],
  ];
  for (const [key, name, unit, tolerance] of geometry) {
    const a = refRecord[key];
    const b = testRecord[key];
    if (isNum(a) && isNum(b) && Math.abs(a - b) > tolerance) {
      add(false, `The ${name} differs between the two runs (${a} ${unit} against ${b} ${unit}). `
        + 'Insertion loss assumes an unchanged geometry.');
    } else if (!isNum(a) || !isNum(b)) {
      add(false, `The ${name} is missing from at least one record, so the geometry cannot be shown to match.`);
    }
  }
  if ((refRecord.mic_model || '') !== (testRecord.mic_model || '')) {
    add(false, 'The two runs name different microphones.');
  }
  if ((refRecord.weapon || '') !== (testRecord.weapon || '')) {
    add(false, 'The two runs name different weapons.');
  }
  if ((refRecord.ammunition || '') !== (testRecord.ammunition || '')) {
    add(false, 'The two runs name different ammunition. Insertion loss is only attributable to the '
      + 'suppressor when everything else is held constant.');
  }
  if (isNum(refSource.sample_rate) && isNum(testSource.sample_rate)
      && refSource.sample_rate !== testSource.sample_rate) {
    add(false, `The recordings were sampled at different rates (${fmtInt(refSource.sample_rate)} against `
      + `${fmtInt(testSource.sample_rate)} Hz), which biases rise time and peak level.`);
  }

  // ---- statistical strength and admissibility ----
  for (const [side, data] of [['reference', ref], ['test', test]]) {
    const verdict = quickVerdict(data.metadata);
    if (verdict.tone === 'danger') {
      add(true, `The ${side} measurement is not admissible on its own (${verdict.label}); it cannot `
        + 'be used as one half of a comparison.');
    }
    if (data.aggregate.n_included === 0) {
      add(true, `The ${side} has no included shots.`);
    } else if (data.aggregate.n_included < 3) {
      add(false, `The ${side} contributes only ${data.aggregate.n_included} shot(s); the confidence `
        + 'interval on that mean is very wide.');
    }
  }

  return objections;
}

/** Insertion-loss rows: reference mean minus test mean, CIs in quadrature. */
function insertionLossRows(ref, test) {
  const rows = [];
  for (const metric of METRICS) {
    if (!metric.compare) continue;
    const a = ref.aggregate.statistics[metric.stat];
    const b = test.aggregate.statistics[metric.stat];
    if (!a || !b || !isNum(a.mean) || !isNum(b.mean)) continue;
    const il = a.mean - b.mean;
    const ci = Math.sqrt((isNum(a.ci95_half_width) ? a.ci95_half_width : 0) ** 2
      + (isNum(b.ci95_half_width) ? b.ci95_half_width : 0) ** 2);
    rows.push({ metric, reference: a, test: b, il, ci, significant: Math.abs(il) > ci });
  }
  return rows;
}

async function loadCompareSide(role, dir) {
  const key = COMPARE_ROLES[role];
  if (!dir) {
    state.compare[key] = null;
    state.compare[`${key}Dir`] = '';
    evaluateCompare();
    return;
  }
  state.compare[`${key}Dir`] = dir;
  state.compare.status = 'loading';
  commit();

  try {
    const response = await fetch(`/api/results?dir=${encodeURIComponent(dir)}`, {
      headers: { Accept: 'application/json' },
    });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try { const body = await response.json(); if (body && body.message) detail = body.message; } catch { /* keep */ }
      throw new Error(detail);
    }
    const payload = await response.json();
    if (!payload || !payload.metadata) throw new Error('That directory does not contain a completed analysis.');
    state.compare[key] = {
      dir: payload.outputDir || dir,
      metadata: payload.metadata,
      aggregate: aggregateFrom(payload.metadata, payload.outputDir || dir),
    };
    state.compare.error = null;
  } catch (err) {
    state.compare[key] = null;
    state.compare.status = 'error';
    state.compare.error = `The ${role} analysis could not be loaded: ${err.message}`;
    commit();
    toast({ title: 'Comparison side failed to load', text: state.compare.error, tone: 'danger' });
    return;
  }
  evaluateCompare();
}

function evaluateCompare() {
  const ref = state.compare.ref;
  const test = state.compare.test;

  if (!ref || !test) {
    state.compare.status = state.compare.status === 'error' ? 'error' : 'empty';
    state.compare.rows = [];
    state.compare.blockers = [];
    commit();
    return;
  }

  const objections = compareObjections(ref, test);
  state.compare.blockers = objections;
  const fatal = objections.some(o => o.fatal);
  state.compare.status = fatal ? 'refused' : 'loaded';
  state.compare.rows = fatal ? [] : insertionLossRows(ref, test);

  const refBands = ref.aggregate.band_exposure_mean_dB;
  const testBands = test.aggregate.band_exposure_mean_dB;
  const comparable = !fatal && refBands.length > 0 && refBands.length === testBands.length;
  state.compare.bands = comparable ? {
    frequencies: ref.aggregate.band_frequencies_Hz,
    reference: refBands,
    test: testBands,
    il: refBands.map((v, i) => (isNum(v) && isNum(testBands[i]) ? v - testBands[i] : NaN)),
  } : null;

  commit();
  announce(fatal
    ? 'Comparison refused: the two measurements cannot be differenced.'
    : `Insertion loss computed from ${ref.aggregate.n_included} reference and ${test.aggregate.n_included} test shots.`);
}

/* ---- compare rendering -------------------------------------------------- */

function compareOptionLabel(item) {
  const record = (item.meta && item.meta.test_metadata) || {};
  const analysis = (item.meta && item.meta.analysis) || {};
  const parts = [
    basename(analysis.input_file) || item.name,
    record.weapon || null,
    record.suppressor || null,
    analysis.timestamp ? fmtTimestamp(analysis.timestamp) : null,
  ].filter(Boolean);
  return parts.join(' · ');
}

function ensureCompareOptions() {
  if (state.history.status === 'idle') loadHistory();
  commit();
}

function fillCompareSelect(selectId, wanted) {
  const select = $(selectId);
  if (!select) return;
  const candidates = state.history.items.filter(item => {
    const configuration = ((item.meta && item.meta.test_metadata) || {}).configuration;
    return configuration === wanted || !configuration;
  });
  const signature = candidates.map(item => item.path).join('|');
  if (select.dataset.signature === signature) return;
  select.dataset.signature = signature;

  const current = select.value;
  while (select.options.length > 1) select.remove(1);
  for (const item of candidates) {
    const option = document.createElement('option');
    option.value = item.path;
    const configuration = ((item.meta && item.meta.test_metadata) || {}).configuration;
    option.textContent = configuration
      ? compareOptionLabel(item)
      : `${compareOptionLabel(item)} — configuration not recorded`;
    select.appendChild(option);
  }
  if (candidates.some(item => item.path === current)) select.value = current;
}

function renderCompareSummary(listId, side) {
  const list = $(listId);
  if (!list) return;
  list.textContent = '';
  if (!side) return;

  const record = (side.metadata.test_metadata) || {};
  const calibration = (side.metadata.calibration) || {};
  const source = (side.metadata.source) || {};
  const rows = [
    ['Configuration', record.configuration || 'not recorded'],
    ['Weapon', record.weapon || '—'],
    ['Ammunition', record.ammunition || '—'],
    ['Suppressor', record.suppressor || '—'],
    ['Microphone', `${record.mic_model || '—'}${record.mic_serial ? ` (${record.mic_serial})` : ''}`],
    ['Geometry', `${isNum(record.mic_distance_m) ? `${record.mic_distance_m} m` : '—'} · `
      + `${isNum(record.mic_angle_deg) ? `${record.mic_angle_deg}°` : '—'} · `
      + `${isNum(record.mic_height_m) ? `${record.mic_height_m} m` : '—'}`],
    ['Level unit', calibration.level_unit || '—'],
    ['Calibration', calibration.calibrated === true
      ? `${calibration.method || 'calibrated'} · ${fmt(calibration.Pa_per_FS, 4)} Pa/FS`
      : 'Uncalibrated'],
    ['Sample rate', isNum(source.sample_rate) ? `${fmt(source.sample_rate / 1000, 1)} kHz` : '—'],
    ['Shots included', `${side.aggregate.n_included} of ${side.aggregate.n_total}`],
  ];
  for (const [key, value] of rows) {
    const frag = fromTemplate('tpl-kv-row');
    if (!frag) break;
    setRaw(slot(frag, 'key'), key);
    setRaw(slot(frag, 'value'), String(value));
    list.appendChild(frag);
  }
}

function renderComparePill(pillId, textId, side) {
  const pill = $(pillId);
  if (!pill) return;
  show(pill, Boolean(side));
  if (!side) return;
  const verdict = quickVerdict(side.metadata);
  setTone(pill, verdict.tone, verdict.icon);
  setRaw($(textId), verdict.label);
}

function renderCompare() {
  fillCompareSelect('compare-ref-select', 'unsuppressed');
  fillCompareSelect('compare-test-select', 'suppressed');

  const ref = state.compare.ref;
  const test = state.compare.test;

  renderComparePill('compare-ref-pill', 'compare-ref-pill-text', ref);
  renderComparePill('compare-test-pill', 'compare-test-pill-text', test);
  renderCompareSummary('compare-ref-summary', ref);
  renderCompareSummary('compare-test-summary', test);
  setDisabled($('btn-compare-ref-open'), !ref);
  setDisabled($('btn-compare-test-open'), !test);

  const status = state.compare.status;
  const showResult = status === 'loaded' || status === 'refused';
  show($('compare-empty'), !showResult);
  show($('compare-loaded'), showResult);
  ['btn-compare-print', 'btn-compare-csv'].forEach(id => setDisabled($(id), status !== 'loaded'));

  // The empty panel carries the reason when a load failed.
  const emptyTitle = qs('#compare-empty .empty-state-title');
  const emptyText = qs('#compare-empty .empty-state-text');
  if (status === 'error') {
    setRaw(emptyTitle, 'One side could not be loaded');
    setRaw(emptyText, state.compare.error || 'The local service did not return a usable analysis.');
  } else if (status === 'loading') {
    setRaw(emptyTitle, 'Loading…');
    setRaw(emptyText, 'Reading both analyses from the local results area.');
  } else {
    setRaw(emptyTitle, 'Choose two measurements');
    setRaw(emptyText, 'Insertion loss is only meaningful when both recordings were made on the same '
      + 'microphone, at the same distance and angle, with the calibration unchanged between them.');
  }

  if (!showResult) return;

  // ---- objections ----
  const objections = state.compare.blockers;
  const fatal = objections.filter(o => o.fatal);
  const banner = $('compare-validity');
  show(banner, objections.length > 0);
  if (objections.length > 0) {
    setTone(banner, fatal.length > 0 ? 'danger' : 'warn');
    setRaw($('compare-validity-title'), fatal.length > 0
      ? 'These two measurements must not be differenced'
      : 'Comparison is possible, but qualify it');
    fillList($('compare-validity-list'), objections.map(o => (o.fatal ? `Refused: ${o.text}` : o.text)), '');
  }

  const hero = $('il-hero');
  const rows = state.compare.rows;

  if (status === 'refused') {
    setRaw($('il-hero-value'), '—');
    setAttr(hero, 'data-tone', 'danger');
    setAttr(hero, 'data-state', 'empty');
    setRaw($('il-hero-caption'), 'No insertion loss is reported: see the reasons above.');
    const metrics = $('compare-metrics');
    if (metrics) metrics.textContent = '';
    const tbody = $('compare-tbody');
    if (tbody) tbody.textContent = '';
    setText($('caption-compare-bands'), 'Not computed — the comparison was refused.');
    chartRegistry.set('compare-band-canvas', drawCompareBandChart);
    chartMessage($('compare-band-canvas'), 'Comparison refused');
    return;
  }

  // ---- hero ----
  const headline = rows.find(r => r.metric.key === 'Lpeak_Z_dB') || rows[0] || null;
  setAttr(hero, 'data-state', headline ? null : 'empty');
  if (headline) {
    setRaw($('il-hero-value'), fmtSigned(headline.il, 1));
    setAttr(hero, 'data-tone', headline.il > 0 ? 'ok' : 'danger');
    setRaw($('il-hero-caption'),
      `${headline.metric.plain}: ${fmt(headline.reference.mean, 1)} → ${fmt(headline.test.mean, 1)} `
      + `${(ref.metadata.calibration || {}).level_unit || 'dB'} · 95% CI ±${fmt(headline.ci, 2)} dB · `
      + `${ref.aggregate.n_included} reference and ${test.aggregate.n_included} test shot(s) · `
      + `${headline.significant ? 'the difference exceeds its confidence interval' : 'the difference is INSIDE its confidence interval and is not resolved'}.`);
  } else {
    setRaw($('il-hero-value'), '—');
    setRaw($('il-hero-caption'), 'No metric is available on both sides.');
  }

  // ---- tiles ----
  const metrics = $('compare-metrics');
  if (metrics) {
    metrics.textContent = '';
    for (const row of rows) {
      const frag = fromTemplate('tpl-metric-tile');
      if (!frag) break;
      const node = frag.firstElementChild;
      node.dataset.metric = row.metric.key;
      node.setAttribute('data-tone', row.il > 0 ? 'ok' : 'warn');
      setLabel(slot(node, 'label'), row.metric.label);
      slot(node, 'label').setAttribute('title', `Insertion loss — ${row.metric.plain}`);
      setRaw(slot(node, 'value'), fmtSigned(row.il, row.metric.digits));
      setRaw(slot(node, 'unit'), 'dB');
      setRaw(slot(node, 'caption'),
        `${fmt(row.reference.mean, row.metric.digits)} → ${fmt(row.test.mean, row.metric.digits)}`
        + ` · 95% CI ±${fmt(row.ci, 2)}`
        + (row.significant ? '' : ' · not resolved'));
      const delta = slot(node, 'delta');
      const direction = Math.abs(row.il) < 0.05 ? 'flat' : (row.il > 0 ? 'down' : 'up');
      delta.setAttribute('data-direction', direction);
      setIcon(slot(node, 'delta-icon'),
        direction === 'down' ? 'i-arrow-down' : direction === 'up' ? 'i-arrow-up' : 'i-minus');
      setRaw(slot(node, 'delta-value'), direction === 'down' ? 'quieter' : direction === 'up' ? 'louder' : 'no change');
      show(delta, true);
      metrics.appendChild(node);
    }
  }

  // ---- table ----
  const tbody = $('compare-tbody');
  if (tbody) {
    tbody.textContent = '';
    const unit = (ref.metadata.calibration || {}).level_unit || 'dB';
    for (const row of rows) {
      const frag = fromTemplate('tpl-compare-row');
      if (!frag) break;
      const tr = frag.firstElementChild;
      tr.dataset.metric = row.metric.key;
      setRaw(slot(tr, 'name'), row.metric.plain);
      setRaw(slot(tr, 'unit'), row.metric.unit === 'level' ? unit : unitFor(row.metric));
      setRaw(slot(tr, 'reference'), fmt(row.reference.mean, row.metric.digits));
      setRaw(slot(tr, 'test'), fmt(row.test.mean, row.metric.digits));
      setRaw(slot(tr, 'delta'), fmtSigned(row.il, row.metric.digits));
      setRaw(slot(tr, 'ci95'), fmt(row.ci, 2));
      const mark = tr.querySelector('.delta-mark');
      if (mark) {
        const direction = Math.abs(row.il) < 0.05 ? 'flat' : (row.il > 0 ? 'down' : 'up');
        mark.setAttribute('data-direction', direction);
        setIcon(slot(tr, 'delta-icon'),
          direction === 'down' ? 'i-arrow-down' : direction === 'up' ? 'i-arrow-up' : 'i-minus');
        mark.setAttribute('title', row.significant
          ? 'The difference is larger than its 95 % confidence interval.'
          : 'The difference is within its 95 % confidence interval and is not resolved.');
      }
      tbody.appendChild(tr);
    }
    if (tbody.childElementCount === 0) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 6;
      td.textContent = 'No metric is present in both records.';
      tr.appendChild(td);
      tbody.appendChild(tr);
    }
  }

  // ---- band chart ----
  chartRegistry.set('compare-band-canvas', drawCompareBandChart);
  drawCompareBandChart();
  const bands = state.compare.bands;
  setText($('caption-compare-bands'), bands
    ? `Reference and test band exposure with their difference, energy-averaged over `
      + `${ref.aggregate.n_included} and ${test.aggregate.n_included} included shot(s). `
      + `A positive insertion loss means the suppressed configuration is quieter in that band.`
    : 'Per-band insertion loss is unavailable: at least one record has no band analysis.');
}

/** The level unit both compared records are in, or null if they disagree. */
function compareLevelUnit() {
  const units = ['ref', 'test']
    .map(side => ((state.compare[side] || {}).metadata || {}))
    .map(meta => ((meta.calibration || {}).level_unit) || null)
    .filter(Boolean);
  if (units.length === 2 && units[0] === units[1]) return units[0];
  // Two records in different units cannot be differenced at all; the comparison
  // is refused upstream, so this only has to avoid asserting one of them.
  return 'dB';
}

function drawCompareBandChart() {
  const canvas = $('compare-band-canvas');
  if (!canvas) return;
  // Registered so a theme change, a resize, or a zoom reset can repaint it
  // through the same path as every other chart.
  chartRegistry.set('compare-band-canvas', drawCompareBandChart);
  const bands = state.compare.bands;
  if (!bands) {
    chartMessage(canvas, 'No comparable band data');
    return;
  }
  const palette = chartPalette();
  drawBandedChart(canvas, {
    frequencies: bands.frequencies,
    unit: 'dB',
    zeroLine: true,
    series: [
      // The bars are a DIFFERENCE and carry a sign. The two lines are the
      // ABSOLUTE levels it was computed from: they are not a change, and they
      // are in whatever unit the records are in, which is "dB re FS" when the
      // pair was measured uncalibrated.
      { label: 'Insertion loss', color: palette.series[0], kind: 'bar',
        values: bands.il, unit: 'dB', signed: true },
      { label: 'Reference', color: palette.series[4], kind: 'line',
        values: bands.reference, unit: compareLevelUnit(), signed: false },
      { label: 'Test', color: palette.series[5], kind: 'line',
        values: bands.test, unit: compareLevelUnit(), signed: false },
    ],
  });
  canvas.setAttribute('aria-label',
    'Insertion loss against one-third-octave band centre frequency, with the reference and test levels.');
}

function downloadCompareCsv() {
  const ref = state.compare.ref;
  const test = state.compare.test;
  if (!ref || !test || state.compare.status !== 'loaded') {
    toast({ title: 'Nothing to export', text: 'Load two comparable measurements first.', tone: 'warn' });
    return;
  }
  const unit = (ref.metadata.calibration || {}).level_unit || 'dB';
  const rows = [
    ['SASA insertion loss'],
    ['reference_dir', ref.dir],
    ['test_dir', test.dir],
    ['level_unit', unit],
    ['reference_shots_included', ref.aggregate.n_included, 'of', ref.aggregate.n_total],
    ['test_shots_included', test.aggregate.n_included, 'of', test.aggregate.n_total],
    [],
    ['metric', 'unit', 'reference_mean', 'reference_ci95', 'test_mean', 'test_ci95',
      'insertion_loss', 'insertion_loss_ci95', 'resolved'],
  ];
  for (const row of state.compare.rows) {
    rows.push([
      row.metric.key,
      row.metric.unit === 'level' ? unit : unitFor(row.metric),
      row.reference.mean, row.reference.ci95_half_width,
      row.test.mean, row.test.ci95_half_width,
      row.il, row.ci, row.significant ? 'yes' : 'no',
    ]);
  }
  const bands = state.compare.bands;
  if (bands) {
    rows.push([]);
    rows.push(['band_Hz', 'reference_dB', 'test_dB', 'insertion_loss_dB']);
    bands.frequencies.forEach((hz, i) => {
      rows.push([hz, bands.reference[i], bands.test[i], bands.il[i]]);
    });
  }
  if (state.compare.blockers.length > 0) {
    rows.push([]);
    rows.push(['qualifications']);
    for (const objection of state.compare.blockers) {
      rows.push([objection.fatal ? 'refused' : 'caution', objection.text]);
    }
  }
  downloadText(`${basename(test.dir)}-insertion-loss.csv`, csvRows(rows));
  toast({ title: 'Comparison exported', tone: 'ok' });
}

function wireCompare() {
  const ref = $('compare-ref-select');
  if (ref) ref.addEventListener('change', () => loadCompareSide('reference', ref.value));
  const test = $('compare-test-select');
  if (test) test.addEventListener('change', () => loadCompareSide('test', test.value));

  const openRef = $('btn-compare-ref-open');
  if (openRef) openRef.addEventListener('click', () => {
    if (state.compare.ref) loadResults(state.compare.ref.dir);
  });
  const openTest = $('btn-compare-test-open');
  if (openTest) openTest.addEventListener('click', () => {
    if (state.compare.test) loadResults(state.compare.test.dir);
  });

  const csv = $('btn-compare-csv');
  if (csv) csv.addEventListener('click', downloadCompareCsv);
}


/* ===========================================================================
   13. HISTORY
   =========================================================================== */

async function loadHistory() {
  state.history.status = 'loading';
  state.history.error = null;
  commit();

  try {
    const response = await fetch('/api/analyses', { headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const items = await response.json();
    state.history.items = Array.isArray(items) ? items : [];
    state.history.status = 'loaded';
  } catch (err) {
    state.history.items = [];
    state.history.status = 'error';
    state.history.error = `The analysis history could not be read: ${err.message}`;
    toast({ title: 'History unavailable', text: state.history.error, tone: 'danger' });
  }
  commit();
}

function historyMatches(item) {
  const record = (item.meta && item.meta.test_metadata) || {};
  const analysis = (item.meta && item.meta.analysis) || {};
  if (state.history.config !== 'all' && record.configuration !== state.history.config) return false;
  const needle = state.history.filter.trim().toLowerCase();
  if (!needle) return true;
  return [
    item.name, analysis.input_file, record.weapon, record.suppressor,
    record.operator, record.ammunition, record.location, record.configuration,
  ].filter(Boolean).join(' ').toLowerCase().includes(needle);
}

function renderHistory() {
  const list = $('history-list');
  const empty = $('history-empty');
  if (!list) return;

  const items = state.history.items.filter(historyMatches);
  const signature = `${state.history.status}|${state.history.filter}|${state.history.config}|`
    + items.map(i => i.path).join('|');
  if (list.dataset.signature !== signature) {
    list.dataset.signature = signature;
    list.textContent = '';
    for (const item of items) {
      const frag = fromTemplate('tpl-history-item');
      if (!frag) break;
      const node = frag.firstElementChild;
      const metadata = item.meta || {};
      const record = metadata.test_metadata || {};
      const analysis = metadata.analysis || {};
      const aggregate = metadata.aggregate || {};
      const statistics = aggregate.statistics || {};
      const peak = statistics.Lpeak_Z || statistics.Lpeak_Z_dB || null;
      const unit = (metadata.calibration || {}).level_unit || 'dB';

      node.dataset.dir = item.path;
      setRaw(slot(node, 'name'), basename(analysis.input_file) || item.name);
      setRaw(slot(node, 'path'), item.path);
      setRaw(slot(node, 'configuration'), record.configuration || 'not recorded');
      setRaw(slot(node, 'shots'), isNum(aggregate.n_valid)
        ? `${aggregate.n_valid} / ${isNum(aggregate.n_shots) ? aggregate.n_shots : '—'}`
        : '—');
      setRaw(slot(node, 'peak'), peak && isNum(peak.mean) ? `${fmt(peak.mean, 1)} ${unit}` : '—');
      setRaw(slot(node, 'date'), analysis.timestamp ? fmtTimestamp(analysis.timestamp) : '—');

      const verdict = quickVerdict(metadata);
      const pill = slot(node, 'validity');
      if (pill) {
        pill.setAttribute('data-tone', verdict.tone);
        setIcon(slot(node, 'validity-icon'), verdict.icon);
        setRaw(slot(node, 'validity-label'), verdict.label);
      }
      node.setAttribute('aria-label',
        `Open ${basename(analysis.input_file) || item.name}, ${record.configuration || 'configuration not recorded'}, ${verdict.label}`);
      list.appendChild(node);
    }
  }

  const nothing = items.length === 0;
  show(list, !nothing);
  show(empty, nothing);
  if (nothing) {
    const title = qs('#history-empty .empty-state-title');
    const text = qs('#history-empty .empty-state-text');
    if (state.history.status === 'error') {
      setRaw(title, 'History could not be read');
      setRaw(text, state.history.error || '');
    } else if (state.history.status === 'loading') {
      setRaw(title, 'Reading history…');
      setRaw(text, 'Listing the analyses this machine has produced.');
    } else if (state.history.items.length > 0) {
      setRaw(title, 'Nothing matches that filter');
      setRaw(text, 'Clear the filter, or choose a different configuration.');
    } else {
      setRaw(title, 'No analyses yet');
      setRaw(text, 'Completed analyses appear here with their key metadata.');
    }
  }
}

function wireHistory() {
  const list = $('history-list');
  if (list) list.addEventListener('click', (ev) => {
    const item = ev.target.closest('.history-item[data-dir]');
    if (!item) return;
    loadResults(item.dataset.dir);
  });

  const refresh = $('btn-history-refresh');
  if (refresh) refresh.addEventListener('click', () => {
    loadHistory();
    announce('Refreshing the analysis history');
  });

  const filter = $('history-filter');
  if (filter) filter.addEventListener('input', () => {
    state.history.filter = filter.value;
    commit();
  });

  const configuration = $('history-config-filter');
  if (configuration) configuration.addEventListener('change', () => {
    state.history.config = configuration.value;
    commit();
  });
}


/* ===========================================================================
   14. SETTINGS AND CALIBRATION PROFILES
   =========================================================================== */

function renderPrefs() {
  const output = $('setting-output-dir');
  // Never write into a control the operator is typing in.
  if (output && document.activeElement !== output && output.value !== state.prefs.outputDir) {
    output.value = state.prefs.outputDir;
  }
  const open = $('setting-open-on-complete');
  if (open && open.checked !== state.prefs.openOnComplete) open.checked = state.prefs.openOnComplete;

  setRaw($('app-version'), state.app.version || '—');
  setText($('about-version'), state.app.version);
  setText($('about-engine'), state.app.engine);
}

function renderProfiles() {
  setRaw($('profile-count'), String(state.profiles.length));

  // The picker on the Analyze page.
  const select = $('cal-profile-select');
  if (select) {
    const signature = state.profiles.map(p => `${p.id}:${p.name}`).join('|');
    if (select.dataset.signature !== signature) {
      select.dataset.signature = signature;
      while (select.options.length > 1) select.remove(1);
      for (const profile of state.profiles) {
        const option = document.createElement('option');
        option.value = profile.id;
        option.textContent = `${profile.name} — ${fmt(profile.paPerFS, 4)} Pa/FS`;
        select.appendChild(option);
      }
    }
    if (select.value !== state.calibration.profileId) {
      select.value = state.profiles.some(p => p.id === state.calibration.profileId)
        ? state.calibration.profileId : '';
    }
  }

  // The management table.
  const tbody = $('profiles-tbody');
  if (!tbody) return;
  const signature = state.profiles.map(p => `${p.id}:${p.paPerFS}:${p.verified}`).join('|');
  if (tbody.dataset.signature === signature) return;
  tbody.dataset.signature = signature;
  tbody.textContent = '';

  if (state.profiles.length === 0) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 5;
    td.textContent = 'No profiles saved. Complete a chain or tone calibration, then use "Save as profile".';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  for (const profile of state.profiles) {
    const frag = fromTemplate('tpl-profile-row');
    if (!frag) break;
    const row = frag.firstElementChild;
    row.dataset.profile = profile.id;
    setRaw(slot(row, 'name'), profile.name);
    setRaw(slot(row, 'pafs'), fmt(profile.paPerFS, 4));
    setRaw(slot(row, 'method'), profile.method || '—');
    setRaw(slot(row, 'verified'), profile.verified || '—');
    const use = row.querySelector('[data-action="use-profile"]');
    if (use) use.setAttribute('aria-label', `Use the ${profile.name} profile`);
    const remove = row.querySelector('[data-action="delete-profile"]');
    if (remove) remove.setAttribute('aria-label', `Delete the ${profile.name} profile`);
    tbody.appendChild(row);
  }
}

function persistProfiles() {
  store.set(STORAGE.profiles, state.profiles);
  renderProfiles();
  commit();
}

function openProfiles(opener) {
  renderProfiles();
  openDialog($('modal-profiles'), { opener });
}

async function saveCurrentAsProfile() {
  const paPerFS = resolvedPaPerFS();
  if (!isNum(paPerFS)) {
    toast({
      title: 'Nothing to save yet',
      text: 'A profile stores a known Pa/FS. Complete the chain figures, or run a tone calibration first.',
      tone: 'warn',
    });
    return;
  }

  let input = null;
  const accepted = await confirmDialog({
    title: 'Save calibration profile',
    acceptLabel: 'Save profile',
    acceptTone: 'primary',
    build: (body) => {
      addParagraph(body,
        `This stores ${fmt(paPerFS, 4)} Pa/FS (full scale = ${fmt(fullScaleDb(paPerFS), 1)} dB SPL) `
        + 'with today\'s date, so the same chain can be reused without re-entering it. '
        + 'A profile is a record of a verified chain, not a substitute for verifying it.');
      const field = document.createElement('div');
      field.className = 'field';
      const label = document.createElement('label');
      label.className = 'field-label';
      label.setAttribute('for', 'profile-name-input');
      label.textContent = 'Profile name';
      input = document.createElement('input');
      input.type = 'text';
      input.className = 'input';
      input.id = 'profile-name-input';
      input.maxLength = 64;
      input.placeholder = 'e.g. GRAS 46BE + Zoom F6 @ 0 dB';
      const description = $('cal-description');
      input.value = description && description.value.trim() !== ''
        ? description.value.trim().slice(0, 64) : '';
      field.append(label, input);
      body.appendChild(field);
    },
  });
  if (!accepted) return;

  const name = input && input.value.trim() !== '' ? input.value.trim() : `Chain ${new Date().toISOString().slice(0, 10)}`;
  const profile = {
    id: uid('cal-'),
    name: name.slice(0, 64),
    paPerFS,
    method: state.calibration.method === 'tone' ? 'Calibrator tone' : 'Recording chain',
    verified: new Date().toISOString().slice(0, 10),
  };
  state.profiles = [profile, ...state.profiles].slice(0, 50);
  persistProfiles();
  toast({ title: 'Profile saved', text: `${profile.name} · ${fmt(paPerFS, 4)} Pa/FS`, tone: 'ok' });
  announce(`Calibration profile ${profile.name} saved`);
}

function wireProfiles() {
  const tbody = $('profiles-tbody');
  if (tbody) tbody.addEventListener('click', async (ev) => {
    const button = ev.target.closest('[data-action]');
    if (!button) return;
    const row = button.closest('tr[data-profile]');
    if (!row) return;
    const profile = state.profiles.find(p => p.id === row.dataset.profile);
    if (!profile) return;

    if (button.dataset.action === 'use-profile') {
      state.calibration.method = 'profile';
      state.calibration.profileId = profile.id;
      const radio = $('cal-method-profile');
      if (radio) radio.checked = true;
      renderProfiles();
      closeDialog($('modal-profiles'));
      commit();
      toast({ title: 'Profile selected', text: `${profile.name} · ${fmt(profile.paPerFS, 4)} Pa/FS`, tone: 'ok' });
      return;
    }

    if (button.dataset.action === 'delete-profile') {
      const yes = await confirmDialog({
        title: 'Delete calibration profile',
        acceptLabel: 'Delete',
        build: (body) => addParagraph(body,
          `"${profile.name}" (${fmt(profile.paPerFS, 4)} Pa/FS, verified ${profile.verified}) will be removed `
          + 'from this machine. Analyses already run with it keep their recorded calibration.'),
      });
      if (!yes) return;
      state.profiles = state.profiles.filter(p => p.id !== profile.id);
      if (state.calibration.profileId === profile.id) state.calibration.profileId = '';
      persistProfiles();
      toast({ title: 'Profile deleted', tone: 'ok' });
    }
  });

  const open = $('btn-open-profiles');
  if (open) open.addEventListener('click', () => openProfiles(open));
}

function wirePrefs() {
  const output = $('setting-output-dir');
  if (output) output.addEventListener('input', () => {
    state.prefs.outputDir = output.value.trim();
    store.set(STORAGE.prefs, state.prefs);
  });

  const choose = $('btn-choose-output-dir');
  if (choose) choose.addEventListener('click', () => {
    // A browser cannot hand a page an absolute filesystem path, and inventing
    // one would be worse than saying so.
    if (output) { output.focus(); output.select(); }
    toast({
      title: 'Type the path',
      text: 'The browser cannot disclose a folder path. Enter an absolute path inside the '
        + 'analysis area; the local service rejects anything outside it. Leave it blank for the default.',
      tone: 'info',
      ttl: 10000,
    });
  });

  const openOnComplete = $('setting-open-on-complete');
  if (openOnComplete) openOnComplete.addEventListener('change', () => {
    state.prefs.openOnComplete = openOnComplete.checked;
    store.set(STORAGE.prefs, state.prefs);
  });
}

/** Version and backend readiness, for the About panel and the sidebar. */
async function loadHealth() {
  try {
    const response = await fetch('/api/health', { headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    state.app.version = health.version || null;
    state.app.engine = health.backend
      ? `Python (${health.backend.interpreter || 'unknown'})`
        + `${health.backend.scriptPresent === false ? ' — main.py NOT FOUND' : ''}`
        + `${health.node ? ` · Node ${health.node}` : ''}`
      : null;
    if (health.backend && health.backend.scriptPresent === false) {
      toast({
        title: 'Analysis engine missing',
        text: 'The local service cannot find main.py. No analysis can be run until it is restored.',
        tone: 'danger',
      });
    }
  } catch {
    state.app.engine = 'The local service did not answer /api/health';
  }
  commit();
}


/* ===========================================================================
   15. THE GUIDED FLOW

   Five stops instead of one long form. The rail below is the only component
   that knows the whole route; each panel only has to be one step's worth of
   work. Nothing is hidden that the operator has not already dealt with — the
   rail states every stop's condition at all times, so "one at a time" never
   becomes "you cannot see what is left".
   =========================================================================== */

function stepIndex(id) {
  const i = STEP_IDS.indexOf(id);
  return i < 0 ? 0 : i;
}

/**
 * The condition of one stop.
 *
 *   current  — where the operator is
 *   done     — nothing outstanding
 *   blocked  — outstanding items, and the operator has already been here
 *   todo     — outstanding items, not yet visited
 *
 * The visited distinction is not cosmetic. Painting an untouched form amber
 * accuses the operator of a mistake they have not had the chance to make; the
 * same mark after they have been and gone is a genuine reminder.
 */
function stepCondition(id) {
  if (id === state.ui.step) return 'current';
  const outstanding = blockersForStep(id).length > 0;
  if (id === 'run') {
    if (state.run.status === 'complete') return 'done';
    return outstanding ? (state.ui.visited.has(id) ? 'blocked' : 'todo') : 'todo';
  }
  if (!outstanding) return 'done';
  return state.ui.visited.has(id) ? 'blocked' : 'todo';
}

const CAL_METHOD_LABEL = {
  tone: 'Calibrator tone',
  chain: 'Sensitivity chain',
  profile: 'Saved profile',
  none: 'Uncalibrated',
};

/** A short, true statement of what each stop currently holds. */
function stepStatusText(id) {
  switch (id) {
    case 'recording': {
      if (state.input.uploading) return 'Uploading…';
      if (state.input.error) return 'Unreadable';
      return state.input.name ? basename(state.input.name) : 'Not chosen';
    }
    case 'calibration': {
      const method = state.calibration.method;
      if (!method) return 'No method';
      const label = CAL_METHOD_LABEL[method] || method;
      const outstanding = blockersForStep('calibration').length;
      return outstanding ? `${label} — incomplete` : label;
    }
    case 'metadata': {
      const missing = missingRequiredMetadata().length;
      return missing === 0 ? 'Complete' : `${missing} missing`;
    }
    case 'settings': {
      const current = settingsValues();
      let changed = 0;
      for (const [key, value] of Object.entries(SETTING_DEFAULTS)) {
        if (!(key in current)) continue;
        if (String(current[key]) !== String(value)) changed += 1;
      }
      const outstanding = blockersForStep('settings').length;
      if (outstanding) return `${outstanding} to fix`;
      return changed === 0 ? 'Defaults' : `${changed} changed`;
    }
    case 'run': {
      if (state.run.status === 'running' || state.run.status === 'starting') return 'Running…';
      if (state.run.status === 'complete') return 'Complete';
      if (state.run.status === 'error') return 'Failed';
      const outstanding = detailedBlockers().length;
      return outstanding === 0 ? 'Ready' : `${outstanding} outstanding`;
    }
    default: return '';
  }
}

function setStep(id, { focus = true } = {}) {
  if (!STEP_IDS.includes(id)) return;
  if (state.ui.step === id) return;
  // Leaving the test record counts as having seen all of it, so its empty
  // required fields start reporting themselves rather than waiting to be
  // individually focused and abandoned.
  if (state.ui.step === 'metadata') touchMetadataFields();
  state.ui.stepDir = stepIndex(id) >= stepIndex(state.ui.step) ? 'fwd' : 'back';
  state.ui.step = id;
  state.ui.visited.add(id);
  if (state.view !== 'analyze') setView('analyze', { focus: false });
  commit();

  if (focus) {
    // Focus the panel's heading, not the first field: a keyboard user needs to
    // be told where they have arrived before they are asked for a value.
    const step = STEPS[stepIndex(id)];
    const heading = qs(`#${step.panel} .flow-step-title`);
    if (heading) {
      heading.setAttribute('tabindex', '-1');
      heading.focus({ preventScroll: true });
    }
    const panel = $(step.panel);
    if (panel) panel.scrollIntoView({ block: 'start', behavior: 'auto' });
  }
  announce(`Step ${stepIndex(id) + 1} of ${STEPS.length}: ${STEPS[stepIndex(id)].name}`);
}

/** The first stop with anything outstanding, or null when the route is clear. */
function firstIncompleteStep() {
  const blockers = detailedBlockers();
  for (const step of STEP_IDS) {
    if (blockers.some(b => b.step === step)) return step;
  }
  return null;
}

function renderStepper() {
  const flow = $('analyze-flow');
  if (flow) {
    flow.dataset.step = state.ui.step;
    flow.dataset.dir = state.ui.stepDir;
  }

  for (const step of STEPS) {
    const node = qs(`.stepper-node[data-step="${step.id}"]`);
    if (!node) continue;
    const item = node.closest('.stepper-item');
    const condition = stepCondition(step.id);
    if (item) item.dataset.state = condition;

    if (condition === 'current') node.setAttribute('aria-current', 'step');
    else node.removeAttribute('aria-current');

    const status = slot(node, 'status');
    setRaw(status, stepStatusText(step.id));

    // The visible label is short; the accessible name carries the whole story,
    // because a screen-reader user cannot see the disc's colour.
    const count = blockersForStep(step.id).length;
    node.setAttribute('aria-label',
      `Step ${stepIndex(step.id) + 1}, ${step.name}. ${stepStatusText(step.id)}.`
      + (count > 0 ? ` ${count} item${count === 1 ? '' : 's'} outstanding.` : ''));

    const panel = $(step.panel);
    if (panel) {
      panel.dataset.active = step.id === state.ui.step ? 'true' : 'false';
      // hidden would defeat the entrance animation and the print rule that
      // shows every step; data-active drives display instead.
      panel.removeAttribute('hidden');
    }
  }
}

/**
 * On the last step the bar carries the run action itself.
 *
 * The buttons are MOVED, not duplicated: a second Run button that looked
 * identical to the first would be two things claiming to be the same action,
 * and moving the node keeps its listeners, its id and its aria-describedby
 * pointing at the blocker list. The review sheet is long enough to scroll,
 * and the action that ends the flow should not scroll away from it.
 */
function placeRunControls(onLast) {
  const bar = $('flow-bar');
  const home = $('run-action-home');
  const run = $('btn-run');
  const cancel = $('btn-cancel-run');
  if (!bar || !home || !run) return;

  const target = onLast ? bar : home;
  if (run.parentElement !== target) {
    target.appendChild(run);
    if (cancel) target.appendChild(cancel);
  }
}

function renderFlowBar() {
  const index = stepIndex(state.ui.step);
  const back = $('btn-step-back');
  const next = $('btn-step-next');
  const status = $('flow-bar-status');

  setDisabled(back, index === 0);

  const mine = blockersForStep(state.ui.step);
  const onLast = index === STEPS.length - 1;
  placeRunControls(onLast);

  if (next) {
    show(next, !onLast);
    // "Continue anyway" rather than a disabled button: the flow never locks
    // the operator into a step. An unfinished step is reported, on the rail
    // and here, and the run itself is what refuses — a step is a place to
    // work, not a gate.
    next.textContent = '';
    const label = document.createElement('span');
    label.textContent = mine.length > 0 ? 'Continue anyway' : 'Continue';
    next.appendChild(label);
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'icon');
    svg.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', '#i-chevron-right');
    svg.appendChild(use);
    next.appendChild(svg);
  }

  if (status) {
    if (mine.length === 1) {
      setRaw(status, mine[0]);
      status.dataset.tone = 'warn';
    } else if (mine.length > 1) {
      setRaw(status, `${mine.length} items outstanding on this step`);
      status.dataset.tone = 'warn';
    } else if (onLast) {
      const all = detailedBlockers().length;
      setRaw(status, all === 0 ? 'Ready to run' : `${all} outstanding on earlier steps`);
      status.dataset.tone = all === 0 ? 'ok' : 'warn';
    } else {
      setRaw(status, 'Nothing outstanding here');
      status.dataset.tone = 'ok';
    }
  }
}

/* ---- the review sheet ---------------------------------------------------- */

/** One row of the review sheet. */
function reviewRow(label, value, { tone = 'ok', step = null } = {}) {
  return { label, value, tone, step };
}

/**
 * The configuration restated in the operator's own words. Everything here is
 * read back from the controls rather than from a parallel copy, so the sheet
 * cannot drift from what will actually be sent.
 */
function reviewRows() {
  const rows = [];
  const md = metadataValues();
  const value = (id) => {
    const raw = md[id];
    return raw === undefined || raw === null || String(raw).trim() === '' ? null : String(raw).trim();
  };

  rows.push(reviewRow('Recording',
    state.input.name ? basename(state.input.name) : 'Not chosen',
    { tone: state.input.path ? 'ok' : 'warn', step: 'recording' }));

  const method = state.calibration.method;
  const paPerFS = resolvedPaPerFS();
  let calValue;
  if (!method) calValue = 'Not chosen';
  else if (method === 'none') calValue = 'None — output is dB re FS';
  else if (isNum(paPerFS)) calValue = `${CAL_METHOD_LABEL[method]} · ${fmt(fullScaleDb(paPerFS), 1)} dB SPL at full scale`;
  else calValue = `${CAL_METHOD_LABEL[method]} — incomplete`;
  rows.push(reviewRow('Calibration', calValue, {
    tone: !method ? 'warn'
      : method === 'none' ? 'info'
      : isNum(paPerFS) ? 'ok' : 'warn',
    step: 'calibration',
  }));

  rows.push(reviewRow('Level scale', levelUnitForMethod() === 'dB SPL'
    ? 'dB SPL — absolute'
    : 'dB re FS — relative only', { tone: 'info', step: 'calibration' }));

  const missing = missingRequiredMetadata();
  rows.push(reviewRow('Test record',
    missing.length === 0 ? 'All required fields present'
      : `${missing.length} missing: ${missing.slice(0, 3).map(m => m.name).join(', ')}${missing.length > 3 ? '…' : ''}`,
    { tone: missing.length === 0 ? 'ok' : 'warn', step: 'metadata' }));

  const weapon = value('md-weapon');
  const ammo = value('md-ammunition');
  const suppressor = value('md-suppressor');
  if (weapon || ammo) {
    rows.push(reviewRow('Configuration',
      [weapon, ammo, suppressor ? `with ${suppressor}` : null].filter(Boolean).join(' · '),
      { tone: 'info', step: 'metadata' }));
  }

  const distance = value('md-distance');
  const angle = value('md-angle');
  const height = value('md-height');
  if (distance || angle || height) {
    rows.push(reviewRow('Microphone',
      [distance ? `${distance} m` : null,
       angle ? `${angle}°` : null,
       height ? `${height} m high` : null].filter(Boolean).join(' · '),
      { tone: 'info', step: 'metadata' }));
  }

  const temp = value('md-temp');
  const humidity = value('md-humidity');
  const pressure = value('md-pressure');
  if (temp || humidity || pressure) {
    rows.push(reviewRow('Atmosphere',
      [temp ? `${temp} °C` : null,
       humidity ? `${humidity} % RH` : null,
       pressure ? `${pressure} kPa` : null].filter(Boolean).join(' · '),
      { tone: 'info', step: 'metadata' }));
  }

  const mode = thresholdMode();
  const threshold = readNumber('threshold-value');
  rows.push(reviewRow('Detection',
    `${mode === 'relative' ? 'Relative' : 'Absolute'} threshold ${threshold.empty ? '—' : threshold.value} dB`
    + ` · ${$('refractory-ms') ? $('refractory-ms').value : '—'} ms refractory`,
    { tone: 'info', step: 'settings' }));

  const nperseg = $('stft-nperseg');
  rows.push(reviewRow('Spectral',
    `${nperseg ? nperseg.value : '—'}-point window · ${$('stft-overlap') ? $('stft-overlap').value : '—'} % overlap`,
    { tone: 'info', step: 'settings' }));

  return rows;
}

function renderReviewSheet() {
  const list = $('config-review-list');
  if (!list) return;
  list.textContent = '';

  for (const row of reviewRows()) {
    const frag = fromTemplate('tpl-review-row');
    if (!frag) break;
    const node = frag.firstElementChild;
    const stateCell = slot(node, 'state');
    if (stateCell) {
      stateCell.dataset.tone = row.tone;
      setIcon(stateCell.querySelector('svg'),
        row.tone === 'warn' ? 'i-alert' : row.tone === 'info' ? 'i-info' : 'i-check');
    }
    setRaw(slot(node, 'label'), row.label);
    setRaw(slot(node, 'value'), row.value);

    const edit = slot(node, 'edit');
    if (edit) {
      if (row.step) {
        edit.dataset.step = row.step;
        edit.setAttribute('aria-label', `Change ${row.label.toLowerCase()}`);
      } else {
        edit.remove();
      }
    }
    list.appendChild(node);
  }

  const outstanding = detailedBlockers().length;
  const pill = $('config-review-pill');
  if (pill) {
    setTone(pill, outstanding === 0 ? 'ok' : 'warn');
    setRaw($('config-review-pill-text'), outstanding === 0 ? 'Ready' : `${outstanding} outstanding`);
  }
}

function renderFlow() {
  renderStepper();
  renderFlowBar();
  renderReviewSheet();
}

function wireFlow() {
  const stepper = $('analyze-stepper');
  if (stepper) {
    stepper.addEventListener('click', (ev) => {
      const node = ev.target.closest('.stepper-node');
      if (!node) return;
      setStep(node.dataset.step);
    });
    // Left/right walk the rail, the same gesture as the result tabs.
    stepper.addEventListener('keydown', (ev) => {
      const index = stepIndex(state.ui.step);
      let next = null;
      if (ev.key === 'ArrowRight') next = Math.min(STEPS.length - 1, index + 1);
      else if (ev.key === 'ArrowLeft') next = Math.max(0, index - 1);
      else if (ev.key === 'Home') next = 0;
      else if (ev.key === 'End') next = STEPS.length - 1;
      if (next === null) return;
      ev.preventDefault();
      setStep(STEP_IDS[next]);
      const node = qs(`.stepper-node[data-step="${STEP_IDS[next]}"]`);
      if (node) node.focus();
    });
  }

  const back = $('btn-step-back');
  if (back) back.addEventListener('click', () => setStep(STEP_IDS[Math.max(0, stepIndex(state.ui.step) - 1)]));

  const next = $('btn-step-next');
  if (next) {
    next.addEventListener('click', () =>
      setStep(STEP_IDS[Math.min(STEPS.length - 1, stepIndex(state.ui.step) + 1)]));
  }

  // The review sheet's pencils walk back to the step that owns the row.
  const list = $('config-review-list');
  if (list) {
    list.addEventListener('click', (ev) => {
      const button = ev.target.closest('[data-step]');
      if (!button) return;
      setStep(button.dataset.step);
    });
  }
}


/* ===========================================================================
   16. MENUS

   One implementation for every dropdown: the trigger owns aria-expanded, the
   surface owns role="menu", and exactly one menu is open at a time.
   =========================================================================== */

function closeMenu({ restoreFocus = false } = {}) {
  if (!state.ui.menu) return;
  const menu = $(state.ui.menu);
  const trigger = qs(`[aria-controls="${state.ui.menu}"]`);
  show(menu, false);
  if (trigger) {
    trigger.setAttribute('aria-expanded', 'false');
    if (restoreFocus) trigger.focus();
  }
  state.ui.menu = null;
}

function openMenu(id) {
  if (state.ui.menu === id) { closeMenu(); return; }
  closeMenu();
  const menu = $(id);
  const trigger = qs(`[aria-controls="${id}"]`);
  if (!menu) return;
  show(menu, true);
  if (trigger) trigger.setAttribute('aria-expanded', 'true');
  state.ui.menu = id;
  const first = qs('.menu-item:not([disabled])', menu);
  if (first) first.focus();
}

function menuItems(menu) {
  return qsa('.menu-item:not([disabled])', menu);
}

function wireMenus() {
  document.addEventListener('click', (ev) => {
    const trigger = ev.target.closest('[aria-haspopup="menu"]');
    if (trigger) {
      ev.preventDefault();
      openMenu(trigger.getAttribute('aria-controls'));
      return;
    }
    // A click inside the open menu runs its item, then closes; a click
    // anywhere else just closes.
    const item = ev.target.closest('.menu-item');
    if (item) { closeMenu(); return; }
    if (state.ui.menu && !ev.target.closest('.menu')) closeMenu();
  });

  document.addEventListener('keydown', (ev) => {
    if (!state.ui.menu) return;
    const menu = $(state.ui.menu);
    if (!menu) return;
    const items = menuItems(menu);
    const index = items.indexOf(document.activeElement);

    if (ev.key === 'Escape') { ev.preventDefault(); closeMenu({ restoreFocus: true }); return; }
    if (ev.key === 'Tab') { closeMenu(); return; }
    if (ev.key === 'ArrowDown') {
      ev.preventDefault();
      items[(index + 1 + items.length) % items.length]?.focus();
    } else if (ev.key === 'ArrowUp') {
      ev.preventDefault();
      items[(index - 1 + items.length) % items.length]?.focus();
    } else if (ev.key === 'Home') { ev.preventDefault(); items[0]?.focus(); }
    else if (ev.key === 'End') { ev.preventDefault(); items[items.length - 1]?.focus(); }
  });
}


/* ===========================================================================
   17. COMMAND PALETTE

   Every action in the application has a name here. That is what lets the
   header keep three controls instead of fifteen, and it is the fastest route
   for someone who already knows where they are going.
   =========================================================================== */

/** Built fresh on every open so availability is the truth, not a cached guess. */
function paletteCommands() {
  const commands = [];
  const add = (command) => commands.push(command);
  const loaded = state.results.status === 'loaded';
  const busy = state.run.status === 'running' || state.run.status === 'starting';

  add({ group: 'Go', name: 'Analyze', hint: 'Set up a measurement', icon: 'i-analyze', key: '1',
        run: () => setView('analyze') });
  add({ group: 'Go', name: 'Results', hint: loaded ? 'The loaded measurement' : 'Nothing loaded yet',
        icon: 'i-results', key: '2', disabled: !loaded, run: () => setView('results') });
  add({ group: 'Go', name: 'Compare', hint: 'Insertion loss between two runs', icon: 'i-compare', key: '3',
        run: () => setView('compare') });
  add({ group: 'Go', name: 'History', hint: 'Every analysis on this machine', icon: 'i-history', key: '4',
        run: () => setView('history') });
  add({ group: 'Go', name: 'Settings', hint: 'Preferences and calibration profiles', icon: 'i-settings', key: '5',
        run: () => setView('settings') });

  STEPS.forEach((step, i) => {
    add({
      group: 'Step',
      name: step.name,
      hint: `Step ${i + 1} — ${stepStatusText(step.id)}`,
      icon: 'i-arrow-right',
      run: () => setStep(step.id),
    });
  });

  add({ group: 'Run', name: 'Run analysis', hint: 'Start the measurement', icon: 'i-play', key: 'R',
        disabled: detailedBlockers().length > 0, run: () => startRun() });
  add({ group: 'Run', name: 'Cancel analysis', hint: 'Stop the engine', icon: 'i-stop',
        disabled: !busy, run: () => cancelRun() });

  if (loaded) {
    const TAB_NAMES = {
      overview: 'Overview', spectrogram: 'Spectrogram', bands: 'Bands', shots: 'Shots',
      string: 'String analysis', table: 'Metrics table', hazard: 'Hazard',
    };
    for (const tab of TABS) {
      add({
        group: 'Results',
        name: TAB_NAMES[tab] || tab,
        hint: 'Result tab',
        icon: 'i-results',
        run: () => { setView('results'); setTab(tab); },
      });
    }
    add({ group: 'Results', name: 'Export metrics as CSV', hint: 'Per-shot values', icon: 'i-download',
          run: () => downloadMetricsCsv() });

    // Individual shots, so a long string is navigable by number instead of by
    // clicking along the strip. Capped at what a search can usefully rank.
    shotList().slice(0, 60).forEach((shot, index) => {
      add({
        group: 'Shot',
        name: `Shot ${shot.shot_number}`,
        hint: isRejected(shot) ? 'Excluded from the aggregate' : 'Go to this shot',
        icon: 'i-target',
        run: () => { setView('results'); setTab('shots'); selectShot(index, { focusStrip: true }); },
      });
    });
  }

  add({ group: 'Setup', name: 'Calibration profiles', hint: 'Saved sensitivity chains', icon: 'i-mic',
        run: () => openProfiles(null) });
  add({ group: 'Setup', name: 'Restore default settings', hint: 'Detection and analysis only', icon: 'i-restore',
        run: () => {
          applySettings(SETTING_DEFAULTS);
          // The defaults are valid by construction, so any error still shown
          // against these controls is about a value that no longer exists.
          for (const id of Object.keys(SETTING_DEFAULTS)) setFieldError(id, null);
          persistSettings();
          commit();
        } });

  add({ group: 'View', name: 'Light theme', icon: 'i-sun', run: () => applyTheme('light') });
  add({ group: 'View', name: 'Dark theme', icon: 'i-moon', run: () => applyTheme('dark') });
  add({ group: 'View', name: 'Match the system theme', icon: 'i-monitor', run: () => applyTheme('system') });
  add({ group: 'View', name: state.ui.sidebar === 'rail' ? 'Expand the sidebar' : 'Collapse the sidebar',
        icon: 'i-sidebar', run: () => toggleSidebar() });

  add({ group: 'Help', name: 'Keyboard shortcuts', icon: 'i-keyboard', key: '?', run: () => showShortcuts() });
  add({ group: 'Help', name: 'About SASA', hint: 'Version, engine, connection', icon: 'i-question',
        run: () => showAbout() });
  add({ group: 'Help', name: 'Print this view', icon: 'i-print', run: () => window.print() });

  return commands;
}

/**
 * Ordered subsequence match, scored so that a name match always outranks a
 * hint match and an earlier match outranks a later one. Deliberately not a
 * fuzzy ranker with tuned weights: on a list this size, predictable beats
 * clever — the same query must always produce the same first result.
 */
function paletteScore(command, query) {
  if (!query) return 0;
  const name = command.name.toLowerCase();
  const hint = `${command.group} ${command.hint || ''}`.toLowerCase();

  if (name.startsWith(query)) return 0;
  const direct = name.indexOf(query);
  if (direct > 0) return 10 + direct;

  // Subsequence within the name.
  let cursor = 0;
  for (const character of query) {
    const at = name.indexOf(character, cursor);
    if (at === -1) { cursor = -1; break; }
    cursor = at + 1;
  }
  if (cursor !== -1) return 100 + cursor;

  const inHint = hint.indexOf(query);
  if (inHint !== -1) return 500 + inHint;

  return null;
}

function paletteMatches() {
  const query = state.ui.paletteQuery.trim().toLowerCase();
  const scored = [];
  for (const command of paletteCommands()) {
    const score = paletteScore(command, query);
    if (score === null) continue;
    scored.push({ command, score });
  }
  scored.sort((a, b) => a.score - b.score);
  return scored.slice(0, 40).map(entry => entry.command);
}

let paletteVisible = [];

function renderPalette() {
  const list = $('palette-list');
  if (!list) return;
  paletteVisible = paletteMatches();
  if (state.ui.paletteIndex >= paletteVisible.length) state.ui.paletteIndex = 0;

  list.textContent = '';
  paletteVisible.forEach((command, index) => {
    const frag = fromTemplate('tpl-palette-item');
    if (!frag) return;
    const node = frag.firstElementChild;
    node.id = `palette-option-${index}`;
    node.dataset.index = String(index);
    const selected = index === state.ui.paletteIndex;
    node.setAttribute('aria-selected', selected ? 'true' : 'false');
    if (command.disabled) node.setAttribute('aria-disabled', 'true');

    setIcon(node.querySelector('.palette-item-icon'), command.icon || 'i-arrow-right');
    setRaw(slot(node, 'name'), command.name);
    const hint = slot(node, 'hint');
    setRaw(hint, command.hint || '');
    show(hint, Boolean(command.hint));
    setRaw(slot(node, 'group'), command.group);
    const key = slot(node, 'key');
    if (key) {
      setRaw(key, command.key || '');
      show(key, Boolean(command.key));
    }
    list.appendChild(node);
  });

  show($('palette-empty'), paletteVisible.length === 0);

  const input = $('palette-input');
  if (input) {
    const active = paletteVisible.length > 0 ? `palette-option-${state.ui.paletteIndex}` : '';
    if (active) input.setAttribute('aria-activedescendant', active);
    else input.removeAttribute('aria-activedescendant');
  }

  const selectedNode = list.children[state.ui.paletteIndex];
  if (selectedNode) selectedNode.scrollIntoView({ block: 'nearest' });
}

function openPalette() {
  const dialog = $('palette');
  const input = $('palette-input');
  if (!dialog) return;
  closeMenu();
  state.ui.paletteQuery = '';
  state.ui.paletteIndex = 0;
  if (input) input.value = '';
  renderPalette();
  openDialog(dialog, { focus: input });
}

function closePalette() {
  closeDialog($('palette'));
}

function runPaletteCommand(index) {
  const command = paletteVisible[index];
  if (!command || command.disabled) return;
  closePalette();
  // After the dialog has given focus back, so the command's own focus wins.
  setTimeout(() => { try { command.run(); } catch (err) { console.error(err); } }, 0);
}

function wirePalette() {
  const dialog = $('palette');
  const input = $('palette-input');
  const list = $('palette-list');
  const trigger = $('btn-command');
  if (!dialog || !input) return;

  const hint = $('cmdk-hint');
  if (hint) hint.textContent = `${MOD_LABEL}K`;

  if (trigger) trigger.addEventListener('click', openPalette);

  input.addEventListener('input', () => {
    state.ui.paletteQuery = input.value;
    state.ui.paletteIndex = 0;
    renderPalette();
  });

  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowDown') {
      ev.preventDefault();
      if (paletteVisible.length === 0) return;
      state.ui.paletteIndex = (state.ui.paletteIndex + 1) % paletteVisible.length;
      renderPalette();
    } else if (ev.key === 'ArrowUp') {
      ev.preventDefault();
      if (paletteVisible.length === 0) return;
      state.ui.paletteIndex = (state.ui.paletteIndex - 1 + paletteVisible.length) % paletteVisible.length;
      renderPalette();
    } else if (ev.key === 'Home') { ev.preventDefault(); state.ui.paletteIndex = 0; renderPalette(); }
    else if (ev.key === 'End') {
      ev.preventDefault();
      state.ui.paletteIndex = Math.max(0, paletteVisible.length - 1);
      renderPalette();
    } else if (ev.key === 'Enter') {
      ev.preventDefault();
      runPaletteCommand(state.ui.paletteIndex);
    }
  });

  if (list) {
    list.addEventListener('click', (ev) => {
      const item = ev.target.closest('.palette-item');
      if (!item) return;
      runPaletteCommand(Number(item.dataset.index));
    });
    list.addEventListener('mousemove', (ev) => {
      const item = ev.target.closest('.palette-item');
      if (!item) return;
      const index = Number(item.dataset.index);
      if (index === state.ui.paletteIndex) return;
      state.ui.paletteIndex = index;
      renderPalette();
    });
  }

  dialog.addEventListener('cancel', (ev) => { ev.preventDefault(); closePalette(); });
  dialog.addEventListener('click', (ev) => { if (ev.target === dialog) closePalette(); });
}


/* ===========================================================================
   18. SIDEBAR COLLAPSE
   =========================================================================== */

function applySidebar(mode) {
  const rail = mode === 'rail';
  state.ui.sidebar = rail ? 'rail' : 'full';
  document.documentElement.dataset.sidebar = state.ui.sidebar;
  const toggle = $('btn-sidebar-toggle');
  if (toggle) {
    toggle.setAttribute('aria-pressed', rail ? 'true' : 'false');
    toggle.setAttribute('aria-label', rail ? 'Expand the sidebar' : 'Collapse the sidebar');
  }
  // Every canvas chart is sized from its container, so a width change has to
  // be followed by a redraw or the plots stay at the old width.
  setTimeout(redrawCharts, 320);
}

function toggleSidebar() {
  const next = state.ui.sidebar === 'rail' ? 'full' : 'rail';
  applySidebar(next);
  store.setRaw(STORAGE.sidebar, next);
}

function wireSidebar() {
  const toggle = $('btn-sidebar-toggle');
  if (toggle) toggle.addEventListener('click', toggleSidebar);
}


/* ===========================================================================
   19. BOOT

   Order matters: hydrate the controls from storage BEFORE wiring, so the
   listeners do not fire on the hydration writes; then wire; then connect.
   =========================================================================== */

function applyMetadataValues(values) {
  for (const id of METADATA_IDS) {
    const el = $(id);
    if (!el) continue;
    const value = values && Object.prototype.hasOwnProperty.call(values, id) ? values[id] : '';
    el.value = value === null || value === undefined ? '' : String(value);
  }
}

function hydrate() {
  // Light unless the operator has chosen otherwise. See boot-theme.js: the
  // report is printed, so the screen should match the paper by default.
  applyTheme(store.getRaw(STORAGE.theme, 'light'));
  applySidebar(store.getRaw(STORAGE.sidebar, 'full'));

  applySettings({ ...SETTING_DEFAULTS, ...(store.get(STORAGE.settings, {}) || {}) });
  applyMetadataValues(store.get(STORAGE.metadata, {}) || {});

  const date = $('md-date');
  if (date && date.value === '') date.value = new Date().toISOString().slice(0, 10);

  state.profiles = (store.get(STORAGE.profiles, []) || []).filter(p => p && p.id && isNum(p.paPerFS));
  state.recent = (store.get(STORAGE.recent, []) || []).filter(r => r && typeof r.path === 'string');
  state.prefs = { outputDir: '', openOnComplete: true, ...(store.get(STORAGE.prefs, {}) || {}) };

  /* The calibration METHOD is deliberately not restored. The values above are a
     convenience; the choice of method is a measurement decision and must be
     made again, explicitly, for every session. That is what "no silent default"
     means here. */
  state.calibration.method = '';
  qsa('input[name="cal-method"]').forEach(radio => { radio.checked = false; });
  const ack = $('cal-uncal-ack');
  if (ack) ack.checked = false;
}

/**
 * Three controls on the Analyze page have no counterpart in the backend's
 * argument list. Rather than let them look as though they change the analysis,
 * each says what it actually does. Reported upstream so the flags can be added.
 */
const UNBOUND_CONTROLS = [
  ['min-snr-hint', 'Used in this interface to flag low-SNR shots; the engine does not take a minimum-SNR argument.'],
  ['band-low-hint', 'Filters the band table and charts in this interface; the engine always computes its full band range.'],
  ['band-high-hint', 'Filters the band table and charts in this interface; the engine always computes its full band range.'],
  ['stft-nperseg-hint', null],
  ['hazard-rounds-hint', 'Applied to the dose recomputed in the Hazard tab; the engine records its own figure as well.'],
];

function annotateUnboundControls() {
  for (const [id, note] of UNBOUND_CONTROLS) {
    if (!note) continue;
    const hint = $(id);
    if (!hint) continue;
    const span = document.createElement('span');
    span.textContent = ` ${note}`;
    hint.appendChild(span);
  }
  // The window function and the A/C weighting switches likewise have no flag;
  // they are marked where they sit rather than in a note nobody reads.
  const windowSelect = $('stft-window');
  if (windowSelect) {
    windowSelect.setAttribute('title',
      'Recorded with the request. The engine currently applies its own default window.');
  }
}

function boot() {
  hydrate();
  annotateUnboundControls();

  wireTheme();
  wireDialogs();
  wireNavigation();
  wireTabs();
  wireKeyboard();
  wireInput();
  wireCalibration();
  wireMetadata();
  wireSettings();
  wireRun();
  wireShotNav();
  wireMetricsTable();
  wireResultsActions();
  wireCharts();
  wireWaveform();
  wireCompare();
  wireHistory();
  wireProfiles();
  wirePrefs();
  wireFlow();
  wireMenus();
  wirePalette();
  wireSidebar();

  setTab('overview');
  renderProfiles();
  renderRecent();
  commit();

  loadHealth();
  loadHistory();
  connect({ manual: true });

  // A run in flight must not be abandoned silently by a page reload.
  window.addEventListener('beforeunload', (ev) => {
    if (state.run.status !== 'running' && state.run.status !== 'starting') return;
    ev.preventDefault();
    ev.returnValue = '';
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}

})();
