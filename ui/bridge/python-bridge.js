/**
 * SASA Python Bridge
 * Spawns main.py, streams its output, and turns a validated UI config into a
 * command line. Also owns the path-containment helpers shared with server.js,
 * so there is exactly one implementation of "is this path allowed".
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const EventEmitter = require('events');

// ── Markers emitted by main.py on stdout ──
// Progress:  [SASA-PROGRESS] <percent> <message>
// Output dir:[SASA-OUTPUT] <absolute path>        (legacy prose forms also parsed)
const PROGRESS_RE = /^\[SASA-PROGRESS\]\s+(-?\d+(?:\.\d+)?)\s*(.*)$/;
const OUTPUT_RE = /^\[SASA-OUTPUT\]\s+(.+)$/;
const LEGACY_OUTPUT_RE = /^(?:Output directory|Results saved to|Saving results to)\s*:?\s*(.+)$/i;

// A single stdout/stderr line longer than this is flushed as-is rather than
// buffered forever (protects against a backend that never emits a newline).
const MAX_LINE_BYTES = 1024 * 1024;
// Only the tail of stderr is retained, so a chatty failure cannot exhaust memory.
const MAX_STDERR_TAIL = 64 * 1024;

// main.py's exit codes. 2 and 3 are diagnoses, not crashes: the run produced an
// output directory and a reason, and the UI must say which.
const EXIT_CODES = {
  0: { code: 'ok' },
  1: { code: 'nonzero-exit', message: 'The analysis failed.' },
  2: { code: 'no-shots', message: 'No shots were detected, so nothing was measured.' },
  3: { code: 'inadmissible', message: 'The measurement is inadmissible and must not be reported.' },
};

class AnalysisError extends Error {
  constructor(message, code, details = {}) {
    super(message);
    this.name = 'AnalysisError';
    this.code = code;
    Object.assign(this, details);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Path containment
// ────────────────────────────────────────────────────────────────────────────

/**
 * Resolve `target` and prove it lives inside one of `roots`.
 *
 * Containment is decided with path.relative() on the *real* (symlink-resolved)
 * paths, so neither "../.." segments nor a symlink pointing outside a root can
 * escape. Returns null when the path is not allowed.
 *
 * @param {string} target        Path to check (absolute, or relative to `base`).
 * @param {string[]} roots       Allowed root directories (absolute).
 * @param {object} [opts]
 * @param {string} [opts.base]   Base for relative targets.
 * @param {boolean} [opts.mustExist=true]  Require the path to exist.
 * @returns {string|null} The resolved absolute path, or null if disallowed.
 */
function resolveWithinRoots(target, roots, opts = {}) {
  const { base = null, mustExist = true } = opts;

  if (typeof target !== 'string' || target.length === 0) return null;
  // A NUL byte truncates the path at the syscall boundary.
  if (target.includes('\0')) return null;

  const resolved = path.resolve(base || process.cwd(), target);

  // Lexical containment first — cheap, and it rejects "../" before any I/O.
  // A root may itself sit behind a symlink, so accept either spelling here; the
  // authoritative check below is always done on real paths.
  const lexicalRoot = roots.find(root =>
    isInside(path.resolve(root), resolved) || isInside(realRoot(root), resolved));
  if (!lexicalRoot) return null;

  if (!fs.existsSync(resolved)) {
    if (mustExist) return null;
    // The path does not exist yet (e.g. an output directory to be created).
    // Verify the nearest existing ancestor is still inside a root once symlinks
    // are resolved, so a symlinked parent cannot place the new path elsewhere.
    const anchor = nearestExistingAncestor(resolved);
    if (!anchor) return null;
    const realAnchor = safeRealpath(anchor);
    if (!realAnchor) return null;
    const tail = path.relative(anchor, resolved);
    const realTarget = path.resolve(realAnchor, tail);
    if (!roots.some(root => isInside(realRoot(root), realTarget))) return null;
    return resolved;
  }

  // The path exists: compare real paths so symlink escapes are caught.
  const realTarget = safeRealpath(resolved);
  if (!realTarget) return null;
  if (!roots.some(root => isInside(realRoot(root), realTarget))) return null;

  return resolved;
}

function isInside(root, target) {
  if (root === target) return true;
  const rel = path.relative(root, target);
  return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
}

function realRoot(root) {
  return safeRealpath(path.resolve(root)) || path.resolve(root);
}

function safeRealpath(p) {
  try { return fs.realpathSync(p); } catch { return null; }
}

function nearestExistingAncestor(p) {
  let cur = p;
  for (;;) {
    const parent = path.dirname(cur);
    if (parent === cur) return null;
    if (fs.existsSync(parent)) return parent;
    cur = parent;
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Config validation
//
// Every field is checked for type, finiteness and range before it can become a
// CLI argument. Values are REJECTED, never coerced: a string "12" is an error,
// not the number 12. Presence is tested with explicit null/undefined checks so
// a legitimate 0 (e.g. --threshold-dB 0) is passed through rather than dropped.
// ────────────────────────────────────────────────────────────────────────────

const NUMBER_FIELDS = {
  // Calibration
  calibratorLevelDb: { flag: '--calibrator-level-dB', min: 0,    max: 200 },
  calibratorFreqHz:  { flag: '--calibrator-freq-Hz',  min: 1,    max: 200000 },
  sensitivityMv:     { flag: '--sensitivity-mV',      min: 0,    max: 1e6, exclusiveMin: true },
  preampGainDb:      { flag: '--preamp-gain-dB',      min: -100, max: 200 },
  adcFullScaleV:     { flag: '--adc-full-scale-V',    min: 0,    max: 1e6, exclusiveMin: true },
  vPerFS:            { flag: '--V-per-FS',            min: 0,    max: 1e6, exclusiveMin: true },
  paPerFS:           { flag: '--Pa-per-FS',           min: 0,    max: 1e9, exclusiveMin: true },
  // Detection
  thresholdDb:         { flag: '--threshold-dB',          min: -200, max: 300 },
  thresholdRelativeDb: { flag: '--threshold-relative-dB', min: 0,    max: 200 },
  refractoryMs:        { flag: '--refractory-ms',         min: 0,    max: 600000 },
  preMs:               { flag: '--pre-ms',                min: 0,    max: 600000 },
  postMs:              { flag: '--post-ms',               min: 0,    max: 600000 },
  // Analysis
  overlapFraction: { flag: '--overlap-fraction', min: 0,     max: 0.99 },
  bandHopMs:       { flag: '--band-hop-ms',      min: 0,     max: 10000, exclusiveMin: true },
  nrrDb:           { flag: '--nrr-dB',           min: 0,     max: 100 },
};

const INTEGER_FIELDS = {
  channel:   { flag: '--channel',   min: 0, max: 1024 },
  noverlap:  { flag: '--noverlap',  min: 0, max: 1048576 },
  minShots:  { flag: '--min-shots', min: 0, max: 100000 },
  maxShots:  { flag: '--max-shots', min: 1, max: 100000 },
};

const BOOLEAN_FLAGS = {
  uncalibrated: '--uncalibrated',
  monoMix:      '--mono-mix',
  noBands:      '--no-bands',
  noTimeSeries: '--no-time-series',
  noPerShot:    '--no-per-shot',
  noPlots:      '--no-plots',
  verbose:      '--verbose',
  quiet:        '--quiet',
};

const ENUM_FIELDS = {
  dtype:         { flag: '--dtype',          values: ['float32', 'float64'] },
  bandWeighting: { flag: '--band-weighting', values: ['fast', 'slow', 'impulse'] },
  channels:      { flag: '--channels',       values: ['all'] },
};

const STRING_FIELDS = {
  calDesc: { flag: '--cal-desc', maxLength: 200 },
  // A saved calibration profile name. Restricted charset: it names a key in the
  // operator's profile store and appears in the report.
  preset:  { flag: '--preset', maxLength: 64, pattern: /^[A-Za-z0-9][A-Za-z0-9 ._-]*$/ },
};

// --nperseg takes "auto" or a window size.
const NPERSEG_MIN = 64;
const NPERSEG_MAX = 1048576;

// Paths the client may name. Each is confined to an allow-list of roots exactly
// like the input file — a flag that takes a path is a file-read primitive.
const FILE_FIELDS = {
  calibratorTone: { flag: '--calibrator-tone', kind: 'audio' },
  calibratorPost: { flag: '--calibrator-post', kind: 'audio' },
  metadataFile:   { flag: '--metadata',        kind: 'json' },
};

const DIRECTORY_FIELDS = {
  // A previous UNSUPPRESSED analysis directory; this is what makes the run
  // report insertion loss.
  reference: { flag: '--reference' },
};

// Test conditions recorded with the result. Sent as config.metadata = { ... }.
const METADATA_FIELDS = {
  operator:          { flag: '--operator',           type: 'string', maxLength: 120 },
  date:              { flag: '--date',               type: 'string', maxLength: 40 },
  location:          { flag: '--location',           type: 'string', maxLength: 200 },
  testId:            { flag: '--test-id',            type: 'string', maxLength: 120 },
  weapon:            { flag: '--weapon',             type: 'string', maxLength: 200 },
  barrelLengthIn:    { flag: '--barrel-length-in',   type: 'number', min: 0,    max: 200 },
  ammunition:        { flag: '--ammunition',         type: 'string', maxLength: 200 },
  suppressor:        { flag: '--suppressor',         type: 'string', maxLength: 200 },
  configuration:     { flag: '--configuration',      type: 'enum',   values: ['suppressed', 'unsuppressed'] },
  micModel:          { flag: '--mic-model',          type: 'string', maxLength: 120 },
  micSerial:         { flag: '--mic-serial',         type: 'string', maxLength: 120 },
  micDistanceM:      { flag: '--mic-distance-m',     type: 'number', min: 0,    max: 1000 },
  micAngleDeg:       { flag: '--mic-angle-deg',      type: 'number', min: -360, max: 360 },
  micHeightM:        { flag: '--mic-height-m',       type: 'number', min: 0,    max: 100 },
  groundSurface:     { flag: '--ground-surface',     type: 'string', maxLength: 120 },
  windscreen:        { flag: '--windscreen',         type: 'string', maxLength: 120 },
  temperatureC:      { flag: '--temperature-C',      type: 'number', min: -100, max: 100 },
  humidityPct:       { flag: '--humidity-pct',       type: 'number', min: 0,    max: 100 },
  pressureKPa:       { flag: '--pressure-kPa',       type: 'number', min: 0,    max: 200 },
  windMps:           { flag: '--wind-mps',           type: 'number', min: 0,    max: 200 },
  calibratorModel:   { flag: '--calibrator-model',   type: 'string', maxLength: 120 },
  calibrationPreDb:  { flag: '--calibration-pre-dB',  type: 'number', min: 0, max: 200 },
  calibrationPostDb: { flag: '--calibration-post-dB', type: 'number', min: 0, max: 200 },
  notes:             { flag: '--notes',              type: 'string', maxLength: 4000, multiline: true },
};

// Deliberately NOT exposed to the client: --save-preset, --delete-preset and
// --profiles-file (they mutate the operator's stored calibration profiles) and
// --config (it reads an arbitrary JSON file and rewrites every other setting).

const ALLOWED_FORMATS = ['png', 'pdf', 'svg', 'html'];

const DEFAULT_AUDIO_EXTENSIONS = [
  '.wav', '.wave', '.flac', '.aif', '.aiff', '.aifc',
  '.w64', '.rf64', '.bwf', '.caf', '.ogg', '.mp3', '.m4a',
  '.mp4', '.mov', '.mkv',
];

const KNOWN_FIELDS = new Set([
  'filePath', 'outputDir', 'formats', 'nperseg', 'metadata',
  ...Object.keys(NUMBER_FIELDS),
  ...Object.keys(INTEGER_FIELDS),
  ...Object.keys(BOOLEAN_FLAGS),
  ...Object.keys(ENUM_FIELDS),
  ...Object.keys(STRING_FIELDS),
  ...Object.keys(FILE_FIELDS),
  ...Object.keys(DIRECTORY_FIELDS),
]);

const JSON_EXTENSIONS = ['.json'];

// Control characters would corrupt the log stream and any downstream report.
// (MULTILINE_ allows newline and tab, for operator notes.)
const CONTROL_CHARS_RE = /[\u0000-\u001F\u007F]/;
const MULTILINE_CONTROL_CHARS_RE = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/;

/** Shared type/length/charset check for every free-text field. */
function checkString(raw, spec, field, errors) {
  if (typeof raw !== 'string') {
    errors.push({ field, message: 'Must be a string.' });
    return undefined;
  }
  if (raw.length > spec.maxLength) {
    errors.push({ field, message: `Must be at most ${spec.maxLength} characters.` });
    return undefined;
  }
  const forbidden = spec.multiline ? MULTILINE_CONTROL_CHARS_RE : CONTROL_CHARS_RE;
  if (forbidden.test(raw)) {
    errors.push({ field, message: 'Must not contain control characters.' });
    return undefined;
  }
  if (spec.pattern && !spec.pattern.test(raw)) {
    errors.push({ field, message: 'Contains characters that are not allowed.' });
    return undefined;
  }
  return raw;
}

/**
 * Validate a run-analysis config from the client.
 *
 * @param {*} config
 * @param {object} opts
 * @param {string[]} opts.inputRoots     Roots an input file may live under.
 * @param {string[]} opts.outputRoots    Roots an output directory may live under.
 * @param {string[]} [opts.allowedExtensions]
 * @param {string} [opts.defaultOutputDir] Used when the client sends none.
 * @returns {{ok: boolean, errors: {field: string, message: string}[], config?: object}}
 */
function validateConfig(config, opts) {
  const errors = [];
  const out = {};

  const inputRoots = opts.inputRoots || [];
  const outputRoots = opts.outputRoots || [];
  const extensions = (opts.allowedExtensions || DEFAULT_AUDIO_EXTENSIONS)
    .map(e => e.toLowerCase());

  if (config === null || typeof config !== 'object' || Array.isArray(config)) {
    return { ok: false, errors: [{ field: 'config', message: 'Config must be an object.' }] };
  }

  // Unknown fields are rejected rather than ignored, so a renamed field in the
  // renderer surfaces immediately instead of silently using a backend default.
  for (const key of Object.keys(config)) {
    if (!KNOWN_FIELDS.has(key)) {
      errors.push({ field: key, message: 'Unknown configuration field.' });
    }
  }

  // ── Input file (required) ──
  if (config.filePath === undefined || config.filePath === null) {
    errors.push({ field: 'filePath', message: 'An input file is required.' });
  } else if (typeof config.filePath !== 'string') {
    errors.push({ field: 'filePath', message: 'Input file must be a string.' });
  } else {
    const resolved = resolveWithinRoots(config.filePath, inputRoots, { mustExist: true });
    if (!resolved) {
      errors.push({ field: 'filePath', message: 'Input file is not an accessible uploaded recording.' });
    } else if (!isFile(resolved)) {
      errors.push({ field: 'filePath', message: 'Input file is not a regular file.' });
    } else if (!extensions.includes(path.extname(resolved).toLowerCase())) {
      errors.push({ field: 'filePath', message: 'Input file type is not supported.' });
    } else {
      out.filePath = resolved;
    }
  }

  // ── Numbers ──
  for (const [field, spec] of Object.entries(NUMBER_FIELDS)) {
    const raw = config[field];
    if (raw === undefined || raw === null) continue;   // explicit: 0 is kept
    if (typeof raw !== 'number' || !Number.isFinite(raw)) {
      errors.push({ field, message: 'Must be a finite number.' });
      continue;
    }
    if (spec.exclusiveMin ? raw <= spec.min : raw < spec.min) {
      errors.push({ field, message: `Must be ${spec.exclusiveMin ? 'greater than' : 'at least'} ${spec.min}.` });
      continue;
    }
    if (raw > spec.max) {
      errors.push({ field, message: `Must be at most ${spec.max}.` });
      continue;
    }
    out[field] = raw;
  }

  // ── Integers ──
  for (const [field, spec] of Object.entries(INTEGER_FIELDS)) {
    const raw = config[field];
    if (raw === undefined || raw === null) continue;
    if (typeof raw !== 'number' || !Number.isInteger(raw)) {
      errors.push({ field, message: 'Must be an integer.' });
      continue;
    }
    if (raw < spec.min || raw > spec.max) {
      errors.push({ field, message: `Must be between ${spec.min} and ${spec.max}.` });
      continue;
    }
    out[field] = raw;
  }

  // ── Booleans ──
  for (const field of Object.keys(BOOLEAN_FLAGS)) {
    const raw = config[field];
    if (raw === undefined || raw === null) continue;
    if (typeof raw !== 'boolean') {
      errors.push({ field, message: 'Must be true or false.' });
      continue;
    }
    out[field] = raw;
  }

  // ── Enums ──
  for (const [field, spec] of Object.entries(ENUM_FIELDS)) {
    const raw = config[field];
    if (raw === undefined || raw === null) continue;
    if (typeof raw !== 'string' || !spec.values.includes(raw)) {
      errors.push({ field, message: `Must be one of: ${spec.values.join(', ')}.` });
      continue;
    }
    out[field] = raw;
  }

  // ── Free-text ──
  for (const [field, spec] of Object.entries(STRING_FIELDS)) {
    const raw = config[field];
    if (raw === undefined || raw === null) continue;
    const value = checkString(raw, spec, field, errors);
    if (value !== undefined) out[field] = value;
  }

  // ── STFT window: "auto" or a window size ──
  if (config.nperseg !== undefined && config.nperseg !== null) {
    const raw = config.nperseg;
    if (raw === 'auto') {
      out.nperseg = 'auto';
    } else if (typeof raw === 'number' && Number.isInteger(raw)
               && raw >= NPERSEG_MIN && raw <= NPERSEG_MAX) {
      out.nperseg = raw;
    } else {
      errors.push({
        field: 'nperseg',
        message: `Must be "auto" or an integer between ${NPERSEG_MIN} and ${NPERSEG_MAX}.`,
      });
    }
  }

  // ── Paths the client may name ──
  // A flag that takes a path is a file-read primitive, so each one is confined
  // to the same roots as the input recording.
  for (const [field, spec] of Object.entries(FILE_FIELDS)) {
    const raw = config[field];
    if (raw === undefined || raw === null) continue;
    if (typeof raw !== 'string') {
      errors.push({ field, message: 'Must be a string.' });
      continue;
    }
    const resolved = resolveWithinRoots(raw, inputRoots, { mustExist: true });
    const allowed = spec.kind === 'json' ? JSON_EXTENSIONS : extensions;
    if (!resolved || !isFile(resolved)) {
      errors.push({ field, message: 'File is not an accessible uploaded file.' });
    } else if (!allowed.includes(path.extname(resolved).toLowerCase())) {
      errors.push({ field, message: 'File type is not supported.' });
    } else {
      out[field] = resolved;
    }
  }

  // ── Directories the client may name (the unsuppressed reference) ──
  const referenceRoots = opts.referenceRoots || outputRoots;
  for (const [field] of Object.entries(DIRECTORY_FIELDS)) {
    const raw = config[field];
    if (raw === undefined || raw === null) continue;
    if (typeof raw !== 'string') {
      errors.push({ field, message: 'Must be a string.' });
      continue;
    }
    const resolved = resolveWithinRoots(raw, referenceRoots, { mustExist: true });
    if (!resolved || !isDirectory(resolved)) {
      errors.push({ field, message: 'Not an accessible analysis directory.' });
    } else {
      out[field] = resolved;
    }
  }

  // ── Test metadata (recorded with the result) ──
  if (config.metadata !== undefined && config.metadata !== null) {
    const meta = config.metadata;
    if (typeof meta !== 'object' || Array.isArray(meta)) {
      errors.push({ field: 'metadata', message: 'Must be an object.' });
    } else {
      const cleaned = {};
      for (const key of Object.keys(meta)) {
        if (!Object.prototype.hasOwnProperty.call(METADATA_FIELDS, key)) {
          errors.push({ field: `metadata.${key}`, message: 'Unknown metadata field.' });
        }
      }
      for (const [key, spec] of Object.entries(METADATA_FIELDS)) {
        const raw = meta[key];
        if (raw === undefined || raw === null || raw === '') continue;
        const field = `metadata.${key}`;
        if (spec.type === 'number') {
          if (typeof raw !== 'number' || !Number.isFinite(raw)) {
            errors.push({ field, message: 'Must be a finite number.' });
          } else if (raw < spec.min || raw > spec.max) {
            errors.push({ field, message: `Must be between ${spec.min} and ${spec.max}.` });
          } else {
            cleaned[key] = raw;
          }
        } else if (spec.type === 'enum') {
          if (typeof raw !== 'string' || !spec.values.includes(raw)) {
            errors.push({ field, message: `Must be one of: ${spec.values.join(', ')}.` });
          } else {
            cleaned[key] = raw;
          }
        } else {
          const value = checkString(raw, spec, field, errors);
          if (value !== undefined) cleaned[key] = value;
        }
      }
      if (Object.keys(cleaned).length > 0) out.metadata = cleaned;
    }
  }

  // ── Plot formats ──
  if (config.formats !== undefined && config.formats !== null) {
    const raw = config.formats;
    const list = Array.isArray(raw) ? raw : (typeof raw === 'string' ? raw.split(',') : null);
    if (!list) {
      errors.push({ field: 'formats', message: 'Must be a string or an array of strings.' });
    } else {
      const cleaned = [];
      let bad = false;
      for (const item of list) {
        if (typeof item !== 'string') { bad = true; break; }
        const fmt = item.trim().toLowerCase();
        if (fmt === '') continue;
        if (!ALLOWED_FORMATS.includes(fmt)) { bad = true; break; }
        if (!cleaned.includes(fmt)) cleaned.push(fmt);
      }
      if (bad || cleaned.length === 0) {
        errors.push({ field: 'formats', message: `Must be a subset of: ${ALLOWED_FORMATS.join(', ')}.` });
      } else {
        out.formats = cleaned.join(',');
      }
    }
  }

  // ── Output directory ──
  if (config.outputDir !== undefined && config.outputDir !== null) {
    if (typeof config.outputDir !== 'string') {
      errors.push({ field: 'outputDir', message: 'Must be a string.' });
    } else {
      const resolved = resolveWithinRoots(config.outputDir, outputRoots, { mustExist: false });
      if (!resolved) {
        errors.push({ field: 'outputDir', message: 'Output directory is outside the permitted results area.' });
      } else {
        out.outputDir = resolved;
      }
    }
  } else if (opts.defaultOutputDir) {
    out.outputDir = opts.defaultOutputDir;
  }

  if (errors.length > 0) return { ok: false, errors };
  return { ok: true, errors: [], config: out };
}

function isFile(p) {
  try { return fs.statSync(p).isFile(); } catch { return false; }
}

function isDirectory(p) {
  try { return fs.statSync(p).isDirectory(); } catch { return false; }
}

/**
 * Turn an already-validated config into argv for main.py.
 * Presence is tested with !== undefined/null so 0 and false survive.
 */
function buildArgs(mainScript, config) {
  const args = [mainScript, config.filePath];

  for (const [field, spec] of Object.entries(NUMBER_FIELDS)) {
    if (config[field] === undefined || config[field] === null) continue;
    args.push(spec.flag, String(config[field]));
  }
  for (const [field, spec] of Object.entries(INTEGER_FIELDS)) {
    if (config[field] === undefined || config[field] === null) continue;
    args.push(spec.flag, String(config[field]));
  }
  for (const [field, spec] of Object.entries(ENUM_FIELDS)) {
    if (config[field] === undefined || config[field] === null) continue;
    args.push(spec.flag, config[field]);
  }
  for (const [field, spec] of Object.entries(STRING_FIELDS)) {
    if (config[field] === undefined || config[field] === null) continue;
    args.push(spec.flag, config[field]);
  }
  for (const [field, spec] of Object.entries(FILE_FIELDS)) {
    if (config[field] === undefined || config[field] === null) continue;
    args.push(spec.flag, config[field]);
  }
  for (const [field, spec] of Object.entries(DIRECTORY_FIELDS)) {
    if (config[field] === undefined || config[field] === null) continue;
    args.push(spec.flag, config[field]);
  }
  for (const [field, flag] of Object.entries(BOOLEAN_FLAGS)) {
    if (config[field] === true) args.push(flag);
  }
  if (config.nperseg !== undefined && config.nperseg !== null) {
    args.push('--nperseg', String(config.nperseg));
  }
  if (config.metadata !== undefined && config.metadata !== null) {
    for (const [field, spec] of Object.entries(METADATA_FIELDS)) {
      const value = config.metadata[field];
      if (value === undefined || value === null) continue;
      args.push(spec.flag, String(value));
    }
  }
  if (config.outputDir !== undefined && config.outputDir !== null) {
    args.push('-o', config.outputDir);
  }
  if (config.formats !== undefined && config.formats !== null) {
    args.push('--formats', config.formats);
  }

  return args;
}

// ────────────────────────────────────────────────────────────────────────────
// Bridge
// ────────────────────────────────────────────────────────────────────────────

class PythonBridge extends EventEmitter {
  /**
   * @param {string} pythonDir
   * @param {object} [options]
   * @param {number} [options.killGraceMs=5000] SIGTERM → SIGKILL grace period.
   */
  constructor(pythonDir, options = {}) {
    super();
    this.pythonDir = pythonDir;
    this.process = null;
    this.killGraceMs = options.killGraceMs ?? 5000;
    this.cancelRequested = false;
    this.finished = false;
    this._killTimer = null;
  }

  /** Locate the interpreter. Returns { path, source }. */
  findPython() {
    const candidates = [
      { p: path.join(this.pythonDir, '.venv', 'bin', 'python'), source: 'venv' },
      { p: path.join(this.pythonDir, '.venv', 'bin', 'python3'), source: 'venv' },
      { p: path.join(this.pythonDir, '.venv', 'Scripts', 'python.exe'), source: 'venv' },
    ];
    for (const c of candidates) {
      if (fs.existsSync(c.p)) return { path: c.p, source: c.source };
    }
    return { path: process.env.SASA_PYTHON || 'python3', source: 'system' };
  }

  /** Back-compat alias. */
  _findPython() {
    return this.findPython().path;
  }

  /**
   * Run main.py with an already-validated config.
   * Resolves { outputDir, outputDirs, exitCode, elapsedMs, stderrTail }.
   * Rejects with AnalysisError, code one of:
   *   'busy' | 'backend-missing' | 'spawn-failed' | 'cancelled' |
   *   'no-shots' (exit 2) | 'inadmissible' (exit 3) | 'nonzero-exit'.
   * The last three carry outputDir/outputDirs: the backend wrote results even
   * though the run did not succeed.
   */
  runAnalysis(config) {
    return new Promise((resolve, reject) => {
      if (this.process) {
        reject(new AnalysisError('An analysis is already running on this connection.', 'busy'));
        return;
      }

      const python = this.findPython();
      const mainScript = path.join(this.pythonDir, 'main.py');
      if (!fs.existsSync(mainScript)) {
        reject(new AnalysisError('The analysis backend could not be located.', 'backend-missing'));
        return;
      }

      const args = buildArgs(mainScript, config);
      const startedAt = Date.now();

      let child;
      try {
        child = spawn(python.path, args, {
          cwd: this.pythonDir,
          env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
          windowsHide: true,
        });
      } catch (err) {
        reject(new AnalysisError(`Failed to start the analysis backend: ${err.message}`, 'spawn-failed'));
        return;
      }

      this.process = child;
      this.cancelRequested = false;
      this.finished = false;

      // --channels all analyses each channel into its own directory, so every
      // announced path is kept, in order, de-duplicated.
      const outputDirs = [];
      const noteOutputDir = (raw) => {
        const resolved = path.resolve(this.pythonDir, raw.trim());
        if (!outputDirs.includes(resolved)) outputDirs.push(resolved);
      };
      let stderrTail = '';
      let settled = false;

      const onLine = (stream, line) => {
        if (stream === 'stdout') {
          const progress = PROGRESS_RE.exec(line);
          if (progress) {
            let percent = Number(progress[1]);
            if (!Number.isFinite(percent)) percent = 0;
            percent = Math.min(100, Math.max(0, percent));
            this.emit('progress', { percent, message: progress[2].trim() });
            return;   // control line — not surfaced as log output
          }

          const marker = OUTPUT_RE.exec(line);
          if (marker) {
            noteOutputDir(marker[1]);
            return;   // control line — not surfaced as log output
          }

          const legacy = LEGACY_OUTPUT_RE.exec(line);
          if (legacy) {
            noteOutputDir(legacy[1]);
          }
        } else {
          stderrTail += line + '\n';
          if (stderrTail.length > MAX_STDERR_TAIL) {
            stderrTail = stderrTail.slice(-MAX_STDERR_TAIL);
          }
        }
        this.emit(stream, line);
      };

      // Remainder-buffered line splitting: a line spanning two chunks is
      // reassembled instead of being torn in half.
      const attach = (streamName, readable) => {
        let remainder = '';
        readable.setEncoding('utf8');
        readable.on('data', (chunk) => {
          remainder += chunk;
          let index;
          while ((index = remainder.indexOf('\n')) !== -1) {
            const line = remainder.slice(0, index).replace(/\r$/, '');
            remainder = remainder.slice(index + 1);
            if (line.length > 0) onLine(streamName, line);
          }
          if (remainder.length > MAX_LINE_BYTES) {
            onLine(streamName, remainder);
            remainder = '';
          }
        });
        readable.on('end', () => {
          const line = remainder.replace(/\r$/, '');
          remainder = '';
          if (line.length > 0) onLine(streamName, line);
        });
        readable.on('error', () => { /* surfaced by the close/error handlers */ });
      };

      attach('stdout', child.stdout);
      attach('stderr', child.stderr);

      const cleanup = () => {
        if (this._killTimer) {
          clearTimeout(this._killTimer);
          this._killTimer = null;
        }
        this.process = null;
        this.finished = true;
      };

      child.on('error', (err) => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(new AnalysisError(`Failed to start the analysis backend: ${err.message}`, 'spawn-failed'));
      });

      child.on('close', (code, signal) => {
        if (settled) return;
        settled = true;
        const wasCancelled = this.cancelRequested;
        cleanup();

        const elapsedMs = Date.now() - startedAt;

        if (wasCancelled) {
          reject(new AnalysisError('Analysis cancelled.', 'cancelled', { exitCode: code, signal, elapsedMs }));
          return;
        }
        const outputDir = outputDirs.length > 0 ? outputDirs[0] : null;

        if (code === 0) {
          resolve({ outputDir, outputDirs, exitCode: 0, elapsedMs, stderrTail: stderrTail.trim() });
          return;
        }

        // 2 (no shots) and 3 (inadmissible) are verdicts about the recording,
        // not backend failures, so they carry their output directory along.
        const known = signal ? null : EXIT_CODES[code];
        const how = signal ? `terminated by ${signal}` : `exit code ${code}`;
        reject(new AnalysisError(
          known ? known.message : `Analysis failed (${how}).`,
          known ? known.code : 'nonzero-exit',
          { exitCode: code, signal, elapsedMs, outputDir, outputDirs, stderrTail: stderrTail.trim() },
        ));
      });
    });
  }

  /**
   * Terminate the child cleanly: SIGTERM, then SIGKILL after the grace period.
   * @returns {boolean} true if a process was signalled.
   */
  cancel() {
    if (!this.process) return false;
    this.cancelRequested = true;
    const child = this.process;

    try { child.kill('SIGTERM'); } catch { /* already gone */ }

    if (this._killTimer) clearTimeout(this._killTimer);
    this._killTimer = setTimeout(() => {
      this._killTimer = null;
      if (this.process === child && child.exitCode === null) {
        try { child.kill('SIGKILL'); } catch { /* already gone */ }
      }
    }, this.killGraceMs);
    if (typeof this._killTimer.unref === 'function') this._killTimer.unref();

    return true;
  }

  get isRunning() {
    return this.process !== null;
  }
}

module.exports = {
  PythonBridge,
  AnalysisError,
  validateConfig,
  buildArgs,
  resolveWithinRoots,
  isInside,
  DEFAULT_AUDIO_EXTENSIONS,
  ALLOWED_FORMATS,
  METADATA_FIELDS,
  EXIT_CODES,
  PROGRESS_RE,
  OUTPUT_RE,
};
