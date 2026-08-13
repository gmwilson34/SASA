#!/usr/bin/env node
/**
 * SASA UI Server
 * Serves the web UI and bridges to the Python analysis backend.
 * Launches the user's default browser automatically.
 *
 * Security posture
 * ────────────────
 * This process reads files from disk on behalf of a browser, so it is treated
 * as a local-only appliance:
 *   • it binds loopback by default (SASA_HOST) and never 0.0.0.0 implicitly;
 *   • every served path is confined to an allow-list of roots and re-checked
 *     after symlink resolution (see resolveWithinRoots in the bridge);
 *   • the Host header is verified on every request, which is what actually
 *     defeats DNS rebinding against a loopback service;
 *   • the Origin header is verified on API requests and on the WebSocket
 *     upgrade, so a page on another origin cannot drive an analysis;
 *   • analysis HTML artifacts are served into a sandboxed opaque origin with a
 *     restrictive CSP, so a poisoned artifact cannot become stored XSS against
 *     the app's own origin.
 */

const http = require('http');
const path = require('path');
const fs = require('fs');
const os = require('os');
const crypto = require('crypto');
const { execFile } = require('child_process');
const express = require('express');
const multer = require('multer');
const { WebSocketServer } = require('ws');
const {
  PythonBridge,
  validateConfig,
  resolveWithinRoots,
  DEFAULT_AUDIO_EXTENSIONS,
} = require('./bridge/python-bridge');

const pkg = require('./package.json');

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

function envInt(name, fallback, min, max) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || !Number.isInteger(value) || value < min || value > max) {
    console.warn(`  ! ${name}="${raw}" is not an integer in [${min}, ${max}] — using ${fallback}.`);
    return fallback;
  }
  return value;
}

function envNumber(name, fallback, min, max) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < min || value > max) {
    console.warn(`  ! ${name}="${raw}" is not a number in [${min}, ${max}] — using ${fallback}.`);
    return fallback;
  }
  return value;
}

function envDir(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  return path.resolve(raw);
}

const PYTHON_DIR = path.resolve(__dirname, '..');
const RENDERER_DIR = path.join(__dirname, 'renderer');
const AUDIO_DIR = envDir('SASA_AUDIO_DIR', path.join(PYTHON_DIR, 'Audio'));
const UPLOAD_DIR = envDir('SASA_UPLOAD_DIR', path.join(AUDIO_DIR, 'uploads'));
const ANALYSIS_DIR = envDir('SASA_ANALYSIS_DIR', path.join(AUDIO_DIR, 'analysis'));

// Loopback by default. Setting SASA_HOST to a routable address exposes a
// file-reading server to the network and is never done implicitly.
const HOST = process.env.SASA_HOST || '127.0.0.1';
const PORT = envInt('SASA_PORT', 3847, 1, 65535);

// Long high-rate recordings are legitimately huge: 192 kHz / 24-bit / stereo is
// ~2 GB per hour. Default ceiling 16 GiB, configurable.
const MAX_UPLOAD_BYTES = envInt('SASA_MAX_UPLOAD_BYTES', 16 * 1024 * 1024 * 1024,
  1024 * 1024, 1024 * 1024 * 1024 * 1024);
const UPLOAD_TTL_HOURS = envNumber('SASA_UPLOAD_TTL_HOURS', 24 * 7, 0, 24 * 365);
const UPLOAD_SWEEP_MINUTES = envNumber('SASA_UPLOAD_SWEEP_MINUTES', 60, 1, 24 * 60);

const MAX_WS_MESSAGE_BYTES = envInt('SASA_MAX_WS_MESSAGE_BYTES', 256 * 1024, 1024, 8 * 1024 * 1024);
// A backend that prints per-shot detail on a long recording can emit tens of
// thousands of lines. Log forwarding yields once this much is already queued,
// so a chatty run cannot exhaust memory or starve the UI's event loop; control
// messages (progress, complete, error) are never dropped.
const MAX_LOG_BACKLOG_BYTES = envInt('SASA_MAX_LOG_BACKLOG_BYTES', 512 * 1024, 4096, 64 * 1024 * 1024);
const MAX_JSON_BYTES = 256 * 1024;
const MAX_METADATA_BYTES = envInt('SASA_MAX_METADATA_BYTES', 64 * 1024 * 1024, 1024, 1024 * 1024 * 1024);
const MAX_CSV_BYTES = envInt('SASA_MAX_CSV_BYTES', 32 * 1024 * 1024, 1024, 1024 * 1024 * 1024);

const OPEN_BROWSER = process.env.SASA_OPEN_BROWSER !== '0';

const ALLOWED_UPLOAD_EXTENSIONS = (process.env.SASA_UPLOAD_EXTENSIONS
  ? process.env.SASA_UPLOAD_EXTENSIONS.split(',')
  : DEFAULT_AUDIO_EXTENSIONS)
  .map(e => e.trim().toLowerCase())
  .filter(e => /^\.[a-z0-9]{1,10}$/.test(e));

// Roots a request may read from, and roots an input recording may live under.
const READ_ROOTS = [ANALYSIS_DIR, UPLOAD_DIR];
const INPUT_ROOTS = [UPLOAD_DIR, AUDIO_DIR];

// WebSocket wire protocol version — bump on any breaking change.
const PROTOCOL_VERSION = 1;

for (const dir of [UPLOAD_DIR, ANALYSIS_DIR]) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

// ────────────────────────────────────────────────────────────────────────────
// Content policy for served artifacts
// ────────────────────────────────────────────────────────────────────────────

// An analysis artifact is attacker-influenced data (it embeds the input file
// name, operator notes and so on). Rendering it needs inline script — Plotly is
// inlined into the document — so containment comes from `sandbox` instead: the
// document lands in an opaque origin with no access to the app's origin, no
// network egress (connect-src 'none') and no form submission.
const CSP_ARTIFACT = [
  "default-src 'none'",
  "script-src 'unsafe-inline' 'unsafe-eval' blob:",
  "style-src 'unsafe-inline'",
  "img-src data: blob:",
  "font-src data:",
  "connect-src 'none'",
  "form-action 'none'",
  "base-uri 'none'",
  "frame-ancestors 'self'",
  'sandbox allow-scripts allow-downloads',
].join('; ');

// For formats that can carry script but never need it (SVG, PDF).
const CSP_INERT = [
  "default-src 'none'",
  "style-src 'unsafe-inline'",
  "img-src data:",
  "frame-ancestors 'self'",
  'sandbox',
].join('; ');

const APP_CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  // 'self' is what covers the WebSocket: CSP Level 3 matches a ws:// URL
  // against 'self' when the host and port are the page's own, which includes
  // the IPv6 loopback the app is reached on when localhost resolves to ::1.
  // The two named hosts are belt and braces for the IPv4 spellings.
  //
  // There was a third entry, `ws://[::1]:PORT`. A bracketed IPv6 literal is
  // not in CSP's host-source grammar, so every browser rejected the WHOLE
  // source expression as invalid, logged it to the console on every page
  // load, and ignored it. It never granted anything.
  `connect-src 'self' ws://localhost:${PORT} ws://127.0.0.1:${PORT}`,
  "frame-src 'self'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join('; ');

const ARTIFACT_TYPES = {
  '.png':  { type: 'image/png' },
  '.jpg':  { type: 'image/jpeg' },
  '.jpeg': { type: 'image/jpeg' },
  '.webp': { type: 'image/webp' },
  '.svg':  { type: 'image/svg+xml', csp: CSP_INERT },
  '.pdf':  { type: 'application/pdf', csp: CSP_INERT },
  '.html': { type: 'text/html; charset=utf-8', csp: CSP_ARTIFACT },
  '.json': { type: 'application/json; charset=utf-8', csp: CSP_INERT },
  '.csv':  { type: 'text/csv; charset=utf-8', csp: CSP_INERT, disposition: 'attachment' },
  '.txt':  { type: 'text/plain; charset=utf-8', csp: CSP_INERT },
};

// Extensions /api/results advertises as viewable renderings of an analysis.
const RESULT_IMAGE_EXTENSIONS = ['.png', '.svg', '.pdf', '.html'];

// ────────────────────────────────────────────────────────────────────────────
// Small helpers
// ────────────────────────────────────────────────────────────────────────────

const HOME_DIR = os.homedir();

/**
 * Strip machine-identifying absolute paths out of anything sent to the client.
 * The configured data roots become named placeholders, backend paths become
 * "./…" and any other path under the user's home becomes "~/…", so a Python
 * traceback stays useful without disclosing the filesystem.
 *
 * Longest prefix first: UPLOAD_DIR and ANALYSIS_DIR normally sit inside
 * AUDIO_DIR, which normally sits inside PYTHON_DIR, and the most specific
 * substitution has to win. Without this, pointing SASA_AUDIO_DIR outside the
 * program directory puts full absolute paths back into every log line.
 */
const REDACTIONS = [
  [UPLOAD_DIR, '<uploads>'],
  [ANALYSIS_DIR, '<analyses>'],
  [AUDIO_DIR, '<audio>'],
  [PYTHON_DIR, '.'],
]
  .filter(([dir]) => typeof dir === 'string' && dir.length > 1 && dir !== '/')
  .sort((a, b) => b[0].length - a[0].length);

function redact(text) {
  if (typeof text !== 'string' || text.length === 0) return text;
  let out = text;
  for (const [dir, label] of REDACTIONS) out = out.split(dir).join(label);
  if (HOME_DIR && HOME_DIR !== '/') out = out.split(HOME_DIR).join('~');
  return out;
}

/** A query parameter must be a single string; repeated keys arrive as arrays. */
function queryString(value) {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

/** One path segment: no separators, no NUL, not "." or "..". */
function safeSegment(value) {
  if (typeof value !== 'string' || value.length === 0 || value.length > 255) return null;
  if (value === '.' || value === '..') return null;
  if (value.includes('/') || value.includes('\\') || value.includes('\0')) return null;
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001F\u007F]/.test(value)) return null;
  return value;
}

function isDirectory(p) {
  try { return fs.statSync(p).isDirectory(); } catch { return false; }
}

function isFile(p) {
  try { return fs.statSync(p).isFile(); } catch { return false; }
}

/** Read and parse JSON, refusing anything implausibly large. */
function readJsonFile(file, maxBytes) {
  try {
    const stat = fs.statSync(file);
    if (!stat.isFile() || stat.size > maxBytes) return null;
    return JSON.parse(fs.readFileSync(file, 'utf-8'));
  } catch {
    return null;
  }
}

function sendError(res, status, code, message) {
  res.status(status).json({ error: message, code });
}

/** Log the real cause locally; hand the client a code and a safe sentence. */
function failRequest(res, status, code, message, err) {
  if (err) console.error(`  ! ${code}: ${err && err.message ? err.message : err}`);
  sendError(res, status, code, message);
}

// ────────────────────────────────────────────────────────────────────────────
// Path confinement for request parameters
// ────────────────────────────────────────────────────────────────────────────

/**
 * Resolve a client-supplied analysis directory. Absolute paths (as handed out
 * by /api/analyses and the WebSocket "complete" message) and bare directory
 * names relative to the analysis root are both accepted; everything else — and
 * anything that resolves outside the allowed roots, symlinks included — is
 * rejected.
 */
function resolveResultsDir(raw) {
  const dir = resolveWithinRoots(raw, READ_ROOTS, { base: ANALYSIS_DIR, mustExist: true });
  if (!dir || !isDirectory(dir)) return null;
  return dir;
}

/**
 * Resolve a fixed, code-chosen file name inside an already-validated analysis
 * directory, and prove it is still inside a root once symlinks are followed.
 * Without this a symlink dropped into an analysis directory turns a plain read
 * (analysis_metadata.json, metrics_summary.csv) into an arbitrary-file read.
 */
function resolveInside(dir, name) {
  const resolved = resolveWithinRoots(path.join(dir, name), READ_ROOTS, { mustExist: true });
  return resolved && isFile(resolved) ? resolved : null;
}

/** Same, for a subdirectory. */
function resolveDirInside(dir, name) {
  const resolved = resolveWithinRoots(path.join(dir, name), READ_ROOTS, { mustExist: true });
  return resolved && isDirectory(resolved) ? resolved : null;
}

/** Resolve <dir>/[sub/]<file> and prove the result is still inside a root. */
function resolveArtifact(dir, sub, file) {
  const safeFile = safeSegment(file);
  if (!safeFile) return null;
  const safeSub = sub === null || sub === undefined ? null : safeSegment(sub);
  if (sub !== null && sub !== undefined && !safeSub) return null;

  const candidate = safeSub ? path.join(dir, safeSub, safeFile) : path.join(dir, safeFile);
  const resolved = resolveWithinRoots(candidate, READ_ROOTS, { mustExist: true });
  if (!resolved || !isFile(resolved)) return null;
  return resolved;
}

// ────────────────────────────────────────────────────────────────────────────
// Express app
// ────────────────────────────────────────────────────────────────────────────

const app = express();
app.disable('x-powered-by');

const ALLOWED_HOSTNAMES = new Set(['localhost', '127.0.0.1', '::1', '[::1]', HOST, `[${HOST}]`]);

/**
 * The port a URL or Host header actually addresses. An omitted port is the
 * scheme's default, NOT "any port": http://localhost is port 80, which is a
 * different origin from this server and must not be trusted just because the
 * hostname matches.
 */
function effectivePort(port, protocol) {
  if (port !== '') return port;
  return protocol === 'https:' ? '443' : '80';
}

/** Host header check — the real defence against DNS rebinding on loopback. */
function isAllowedHost(hostHeader) {
  if (typeof hostHeader !== 'string' || hostHeader.length === 0 || hostHeader.length > 255) return false;
  let hostname = hostHeader;
  let port = '';
  if (hostname.startsWith('[')) {
    const end = hostname.indexOf(']');
    if (end === -1) return false;
    port = hostname.slice(end + 1).replace(/^:/, '');
    hostname = hostname.slice(0, end + 1);
  } else {
    const colon = hostname.lastIndexOf(':');
    if (colon !== -1) {
      port = hostname.slice(colon + 1);
      hostname = hostname.slice(0, colon);
    }
  }
  // This server always speaks cleartext HTTP, so a Host without a port means 80.
  if (effectivePort(port, 'http:') !== String(PORT)) return false;
  return ALLOWED_HOSTNAMES.has(hostname.toLowerCase());
}

/** Origin header check — blocks another origin from driving this server. */
function isAllowedOrigin(origin) {
  if (typeof origin !== 'string' || origin.length === 0) return false;
  if (origin === 'null') return false;
  let url;
  try { url = new URL(origin); } catch { return false; }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return false;
  // Port must match exactly. Anything else — including another local service on
  // port 80 or 443 — is a different origin and gets no access to this API.
  if (effectivePort(url.port, url.protocol) !== String(PORT)) return false;
  return ALLOWED_HOSTNAMES.has(url.hostname.toLowerCase());
}

app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');

  if (!isAllowedHost(req.headers.host)) {
    return sendError(res, 421, 'bad-host', 'This server only answers requests addressed to the local machine.');
  }
  // Origin is absent on same-origin navigations and simple GETs; when a browser
  // does send one it must match. Non-GET always has to carry a trusted Origin.
  const origin = req.headers.origin;
  if (origin !== undefined && !isAllowedOrigin(origin)) {
    return sendError(res, 403, 'bad-origin', 'Cross-origin requests are not accepted.');
  }
  if (req.method !== 'GET' && req.method !== 'HEAD' && origin === undefined) {
    return sendError(res, 403, 'missing-origin', 'This request must carry an Origin header.');
  }
  return next();
});

app.use(express.json({ limit: MAX_JSON_BYTES }));

// ── Static Files ──
app.use(express.static(RENDERER_DIR, {
  index: 'index.html',
  dotfiles: 'ignore',
  setHeaders: (res) => {
    res.setHeader('Content-Security-Policy', APP_CSP);
    res.setHeader('Cache-Control', 'no-cache');
  },
}));

// ────────────────────────────────────────────────────────────────────────────
// Uploads
// ────────────────────────────────────────────────────────────────────────────

/**
 * The stored name is generated here, never taken from the client: separators,
 * "..", NULs and shell metacharacters in file.originalname cannot influence
 * where the bytes land. The original name is kept only as display metadata.
 */
function storedName(originalName) {
  const ext = path.extname(typeof originalName === 'string' ? originalName : '').toLowerCase();
  const safeExt = ALLOWED_UPLOAD_EXTENSIONS.includes(ext) ? ext : '.bin';
  const base = path.basename(typeof originalName === 'string' ? originalName : '', ext)
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .replace(/^[._-]+/, '')
    .slice(0, 64) || 'recording';
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  return `${stamp}_${crypto.randomBytes(6).toString('hex')}_${base}${safeExt}`;
}

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOAD_DIR),
    filename: (req, file, cb) => cb(null, storedName(file.originalname)),
  }),
  limits: {
    fileSize: MAX_UPLOAD_BYTES,
    files: 1,
    fields: 16,
    parts: 20,
    fieldSize: 64 * 1024,
    headerPairs: 64,
  },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname || '').toLowerCase();
    if (!ALLOWED_UPLOAD_EXTENSIONS.includes(ext)) {
      const err = new multer.MulterError('LIMIT_UNEXPECTED_FILE', file.fieldname);
      err.sasaMessage = `Unsupported file type. Accepted: ${ALLOWED_UPLOAD_EXTENSIONS.join(', ')}.`;
      return cb(err);
    }
    return cb(null, true);
  },
});

// ── API: Upload a recording and return its path ──
app.post('/api/upload', (req, res, next) => {
  upload.single('file')(req, res, (err) => {
    if (err) return next(err);
    if (!req.file) return sendError(res, 400, 'no-file', 'No file was uploaded.');

    // Belt and braces: multer generated the name, but prove the bytes landed
    // inside the upload root before telling anyone the path.
    const stored = resolveWithinRoots(req.file.path, [UPLOAD_DIR], { mustExist: true });
    if (!stored) {
      safeUnlink(req.file.path);
      return failRequest(res, 500, 'upload-failed', 'The upload could not be stored.', 'escaped upload root');
    }

    return res.json({
      path: stored,
      name: path.basename(req.file.originalname || stored),
      storedName: path.basename(stored),
      size: req.file.size,
    });
  });
});

// Multer's own errors, translated. Registered as error middleware so it also
// catches aborted or oversized bodies.
app.use((err, req, res, next) => {
  if (!(err instanceof multer.MulterError)) return next(err);
  if (req.file && req.file.path) safeUnlink(req.file.path);

  if (err.code === 'LIMIT_FILE_SIZE') {
    return sendError(res, 413, 'file-too-large',
      `That recording is larger than the ${formatBytes(MAX_UPLOAD_BYTES)} upload limit.`);
  }
  if (err.code === 'LIMIT_UNEXPECTED_FILE') {
    return sendError(res, 400, 'unsupported-file', err.sasaMessage || 'Unexpected file field.');
  }
  return sendError(res, 400, 'upload-rejected', 'The upload was rejected.');
});

function safeUnlink(p) {
  try { fs.unlinkSync(p); } catch { /* already gone */ }
}

function formatBytes(n) {
  const gib = n / (1024 * 1024 * 1024);
  if (gib >= 1) return `${Number(gib.toFixed(gib < 10 ? 1 : 0))} GB`;
  return `${Math.round(n / (1024 * 1024))} MB`;
}

// ────────────────────────────────────────────────────────────────────────────
// Results
// ────────────────────────────────────────────────────────────────────────────

// ── API: List previous analyses ──
app.get('/api/analyses', (req, res) => {
  if (!isDirectory(ANALYSIS_DIR)) return res.json([]);

  try {
    const entries = fs.readdirSync(ANALYSIS_DIR, { withFileTypes: true })
      .filter(d => d.isDirectory())               // isDirectory() is false for symlinks
      .map(d => {
        const dir = path.join(ANALYSIS_DIR, d.name);
        // Re-checked after symlink resolution, exactly like every other read.
        const metaPath = resolveInside(dir, 'analysis_metadata.json');
        const meta = metaPath ? readJsonFile(metaPath, MAX_METADATA_BYTES) : null;
        return { id: d.name, name: d.name, path: dir, meta };
      })
      .filter(d => d.meta)
      .sort((a, b) => analysisTimestamp(b.meta).localeCompare(analysisTimestamp(a.meta)));

    res.set('Cache-Control', 'no-store');
    res.json(entries);
  } catch (err) {
    failRequest(res, 500, 'list-failed', 'The analysis history could not be read.', err);
  }
});

/** schema 2.0 puts the timestamp under analysis.timestamp; 1.x had it at root. */
function analysisTimestamp(meta) {
  if (!meta || typeof meta !== 'object') return '';
  if (meta.analysis && typeof meta.analysis.timestamp === 'string') return meta.analysis.timestamp;
  return typeof meta.timestamp === 'string' ? meta.timestamp : '';
}

// ── API: Load results for a specific analysis ──
app.get('/api/results', (req, res) => {
  const requested = queryString(req.query.dir);
  if (!requested) return sendError(res, 400, 'missing-dir', 'No analysis was specified.');

  const outputDir = resolveResultsDir(requested);
  if (!outputDir) return sendError(res, 404, 'unknown-analysis', 'That analysis could not be found.');

  try {
    const metadataPath = resolveInside(outputDir, 'analysis_metadata.json');
    const metadata = metadataPath ? readJsonFile(metadataPath, MAX_METADATA_BYTES) : null;
    if (!metadata) {
      return sendError(res, 404, 'no-metadata', 'That directory does not contain a completed analysis.');
    }

    // Collect renderings, keyed by base name: { waveform_full: { png, html } }
    const images = {};
    for (const entry of fs.readdirSync(outputDir, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      const ext = path.extname(entry.name).toLowerCase();
      if (!RESULT_IMAGE_EXTENSIONS.includes(ext)) continue;
      const key = entry.name.slice(0, -ext.length);
      if (!images[key]) images[key] = {};
      images[key][ext.slice(1)] = entry.name;
    }

    // Per-shot images. The subdirectory is re-checked after symlink resolution:
    // a "shots" symlink must not turn this into a directory listing of anywhere.
    const shotImages = [];
    const shotsDir = resolveDirInside(outputDir, 'shots');
    if (shotsDir) {
      for (const entry of fs.readdirSync(shotsDir, { withFileTypes: true })) {
        if (entry.isFile() && path.extname(entry.name).toLowerCase() === '.png') {
          shotImages.push(entry.name);
        }
      }
      shotImages.sort();
    }

    // CSV, if it is small enough to hand over inline
    let csv = null;
    let csvTruncated = false;
    const csvPath = resolveInside(outputDir, 'metrics_summary.csv');
    if (csvPath) {
      if (fs.statSync(csvPath).size <= MAX_CSV_BYTES) {
        csv = fs.readFileSync(csvPath, 'utf-8');
      } else {
        csvTruncated = true;
      }
    }

    res.set('Cache-Control', 'no-store');
    res.json({
      metadata,
      images,
      shotImages,
      csv,
      csvTruncated,
      outputDir,
      outputId: path.basename(outputDir),
    });
  } catch (err) {
    failRequest(res, 500, 'results-failed', 'The analysis results could not be read.', err);
  }
});

// ── API: Serve one artifact from an analysis output dir ──
app.get('/api/image', (req, res) => {
  const requestedDir = queryString(req.query.dir);
  const requestedFile = queryString(req.query.file);
  const requestedSub = req.query.sub === undefined ? null : queryString(req.query.sub);

  if (!requestedDir || !requestedFile) {
    return sendError(res, 400, 'missing-parameter', 'Both an analysis and a file must be specified.');
  }
  if (req.query.sub !== undefined && requestedSub === null) {
    return sendError(res, 400, 'bad-parameter', 'Invalid subdirectory.');
  }

  const dir = resolveResultsDir(requestedDir);
  if (!dir) return sendError(res, 404, 'unknown-analysis', 'That analysis could not be found.');

  const filePath = resolveArtifact(dir, requestedSub, requestedFile);
  if (!filePath) return sendError(res, 404, 'unknown-artifact', 'That file could not be found.');

  const ext = path.extname(filePath).toLowerCase();
  const spec = ARTIFACT_TYPES[ext];
  if (!spec) return sendError(res, 415, 'unsupported-artifact', 'That file type is not served.');

  let size;
  try { size = fs.statSync(filePath).size; } catch (err) {
    return failRequest(res, 404, 'unknown-artifact', 'That file could not be found.', err);
  }

  res.setHeader('Content-Type', spec.type);
  res.setHeader('Content-Length', size);
  res.setHeader('Cache-Control', 'private, max-age=300');
  // Every artifact gets a policy; the HTML ones get a sandbox as well, so an
  // artifact rendered inline cannot reach the app's origin.
  res.setHeader('Content-Security-Policy', spec.csp || CSP_INERT);
  res.setHeader('Content-Disposition',
    `${spec.disposition || 'inline'}; filename="${path.basename(filePath).replace(/["\\]/g, '_')}"`);

  const stream = fs.createReadStream(filePath);
  stream.on('error', (err) => {
    console.error(`  ! artifact stream failed: ${err.message}`);
    if (res.headersSent) return res.destroy();
    res.removeHeader('Content-Length');
    res.removeHeader('Content-Disposition');
    return sendError(res, 500, 'read-failed', 'That file could not be read.');
  });
  res.on('close', () => stream.destroy());
  stream.pipe(res);
});

// ── API: Health ──
app.get('/api/health', (req, res) => {
  const probe = new PythonBridge(PYTHON_DIR).findPython();
  res.set('Cache-Control', 'no-store');
  res.json({
    status: 'ok',
    name: 'SASA',
    version: pkg.version,
    protocolVersion: PROTOCOL_VERSION,
    uptime_s: Math.round(process.uptime()),
    node: process.versions.node,
    backend: {
      // Never the absolute path — only whether the pieces are in place.
      interpreter: probe.source,
      scriptPresent: isFile(path.join(PYTHON_DIR, 'main.py')),
    },
    // Where results land when the operator has not chosen somewhere else.
    // Shown as the Output directory field's placeholder, so "blank" reads as
    // "here" rather than as "nowhere decided yet". Not a secret: the Results
    // view already offers to open this directory and copy its path.
    defaultOutputDir: ANALYSIS_DIR,
    limits: {
      maxUploadBytes: MAX_UPLOAD_BYTES,
      allowedExtensions: ALLOWED_UPLOAD_EXTENSIONS,
      uploadTtlHours: UPLOAD_TTL_HOURS,
      maxWsMessageBytes: MAX_WS_MESSAGE_BYTES,
    },
    activeRuns,
  });
});

// ── API: Describe a chosen recording, without analysing it ──
//
// The interface asks for this the moment a file is picked. Everything it
// answers — bin spacing, the top of the band range, whether the rate can
// resolve a rise time at all, which channel will be measured — is in the
// file's header, and the alternative to reading it here is letting the
// operator find out from the verdict after a run they did not need to spend.
//
// It reads headers only: `main.py --probe` decodes nothing.
const PROBE_TIMEOUT_MS = 20000;

app.get('/api/probe', (req, res) => {
  const requested = typeof req.query.path === 'string' ? req.query.path : '';
  const resolved = resolveWithinRoots(requested, INPUT_ROOTS, { mustExist: true });
  if (!resolved) {
    return sendError(res, 400, 'bad-path',
      'That file is not inside the SASA workspace, so it cannot be read.');
  }
  if (!isFile(resolved)) {
    return sendError(res, 404, 'not-found', 'No such file.');
  }

  const python = new PythonBridge(PYTHON_DIR).findPython();
  execFile(
    python.path,
    [path.join(PYTHON_DIR, 'main.py'), '--probe', resolved],
    { timeout: PROBE_TIMEOUT_MS, maxBuffer: 1 << 20, cwd: PYTHON_DIR },
    (err, stdout) => {
      res.set('Cache-Control', 'no-store');
      // A non-zero exit is expected for an unreadable file: main.py still
      // prints the JSON that says why, so stdout is parsed either way.
      let parsed = null;
      try { parsed = JSON.parse(String(stdout || '').trim()); } catch { parsed = null; }
      if (parsed && typeof parsed === 'object') return res.json(parsed);

      const reason = err && err.killed ? 'the probe timed out' : 'the probe produced no answer';
      return res.json({
        readable: false,
        problem: `That file could not be described: ${reason}.`,
        notes: [],
      });
    },
  );
});

// ── Detection preview ──
//
// Runs detection and nothing else, so the operator can move a setting against
// the answer instead of against a guess. It needs no calibration and writes no
// output; every level it returns is dB re FS.
//
// Longer than the probe's budget because this one decodes the file: extraction
// of a video is cached after the first call, but the first call pays for it.
const DETECT_TIMEOUT_MS = 120000;

/* Query parameters the preview accepts, with the flag each becomes and the
   range it must lie in. Values are REJECTED rather than clamped: a threshold
   silently changed to fit is a different measurement from the one asked for. */
const DETECT_PARAMS = [
  { query: 'thresholdRelativeDb', flag: '--threshold-relative-dB', min: 0, max: 200 },
  { query: 'refractoryMs', flag: '--refractory-ms', min: 0, max: 600000 },
  { query: 'preMs', flag: '--pre-ms', min: 0, max: 600000 },
  { query: 'postMs', flag: '--post-ms', min: 0, max: 600000 },
  { query: 'expectedShots', flag: '--expected-shots', min: 1, max: 100000, integer: true },
  { query: 'channel', flag: '--channel', min: 0, max: 1024, integer: true },
];

app.get('/api/detect', (req, res) => {
  const requested = typeof req.query.path === 'string' ? req.query.path : '';
  const resolved = resolveWithinRoots(requested, INPUT_ROOTS, { mustExist: true });
  if (!resolved) {
    return sendError(res, 400, 'bad-path',
      'That file is not inside the SASA workspace, so it cannot be read.');
  }
  if (!isFile(resolved)) {
    return sendError(res, 404, 'not-found', 'No such file.');
  }

  const args = [path.join(PYTHON_DIR, 'main.py'), '--detect-only', resolved];
  for (const spec of DETECT_PARAMS) {
    const raw = req.query[spec.query];
    if (raw === undefined || raw === null || String(raw).trim() === '') continue;
    const value = Number(raw);
    if (!Number.isFinite(value) || value < spec.min || value > spec.max
        || (spec.integer && !Number.isInteger(value))) {
      return sendError(res, 400, 'invalid-config',
        `${spec.query} must be a number between ${spec.min} and ${spec.max}.`);
    }
    args.push(spec.flag, String(value));
  }
  if (String(req.query.autoDetect) === 'false') args.push('--no-auto-detect');

  const python = new PythonBridge(PYTHON_DIR).findPython();
  execFile(
    python.path, args,
    { timeout: DETECT_TIMEOUT_MS, maxBuffer: 8 << 20, cwd: PYTHON_DIR },
    (err, stdout) => {
      res.set('Cache-Control', 'no-store');
      // As with the probe, a non-zero exit still carries the JSON that says
      // why, so stdout is parsed whatever the exit code was.
      let parsed = null;
      try { parsed = JSON.parse(String(stdout || '').trim()); } catch { parsed = null; }
      if (parsed && typeof parsed === 'object') return res.json(parsed);

      const reason = err && err.killed
        ? 'detection took longer than two minutes and was stopped'
        : 'detection produced no answer';
      return res.json({ readable: false, problem: `That file could not be examined: ${reason}.`, shots: [] });
    },
  );
});

// ── Unmatched API routes ──
app.use('/api', (req, res) => sendError(res, 404, 'not-found', 'No such endpoint.'));

// ── Last-resort error handler ──
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  console.error(`  ! unhandled request error: ${err && err.stack ? err.stack : err}`);
  if (res.headersSent) return res.destroy();
  return sendError(res, 500, 'internal-error', 'The request could not be completed.');
});

// ────────────────────────────────────────────────────────────────────────────
// HTTP server & WebSocket
// ────────────────────────────────────────────────────────────────────────────

const server = http.createServer(app);
const wss = new WebSocketServer({ noServer: true, maxPayload: MAX_WS_MESSAGE_BYTES });

/** Sockets in flight, so shutdown and the heartbeat can reach them. */
const sessions = new Map();
let activeRuns = 0;

function refuseUpgrade(socket, status, reason) {
  socket.write(`HTTP/1.1 ${status} ${reason}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`);
  socket.destroy();
}

server.on('upgrade', (req, socket, head) => {
  if (!isAllowedHost(req.headers.host)) return refuseUpgrade(socket, 421, 'Misdirected Request');
  // A browser always sends Origin on a WebSocket handshake, so requiring it
  // costs nothing and stops a page on another origin from opening a channel.
  if (!isAllowedOrigin(req.headers.origin)) return refuseUpgrade(socket, 403, 'Forbidden');

  let pathname;
  try { pathname = new URL(req.url, 'http://localhost').pathname; } catch { pathname = null; }
  if (pathname !== '/ws') return refuseUpgrade(socket, 404, 'Not Found');

  wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
});

function send(ws, payload) {
  if (ws.readyState !== ws.OPEN) return;
  try { ws.send(JSON.stringify(payload)); } catch (err) {
    console.error(`  ! websocket send failed: ${err.message}`);
  }
}

function sendFailure(ws, requestId, code, message, extra = {}) {
  send(ws, { type: 'error', requestId, code, message, ...extra });
}

wss.on('connection', (ws) => {
  const session = { bridge: null, requestId: null, inputPath: null, alive: true };
  sessions.set(ws, session);

  send(ws, {
    type: 'ready',
    protocolVersion: PROTOCOL_VERSION,
    version: pkg.version,
    limits: { maxUploadBytes: MAX_UPLOAD_BYTES, maxWsMessageBytes: MAX_WS_MESSAGE_BYTES },
  });

  ws.on('pong', () => { session.alive = true; });

  ws.on('message', (raw, isBinary) => {
    if (isBinary) return sendFailure(ws, null, 'invalid-message', 'Binary frames are not accepted.');

    let msg;
    try { msg = JSON.parse(raw.toString('utf-8')); } catch {
      return sendFailure(ws, null, 'invalid-message', 'Message was not valid JSON.');
    }
    if (msg === null || typeof msg !== 'object' || Array.isArray(msg)) {
      return sendFailure(ws, null, 'invalid-message', 'Message must be a JSON object.');
    }

    const requestId = typeof msg.requestId === 'string' && msg.requestId.length > 0 && msg.requestId.length <= 64
      ? msg.requestId
      : null;

    switch (msg.type) {
      case 'run-analysis':
        // A rejection here would otherwise be an unhandled rejection (fatal on
        // modern Node) AND would leave the client waiting on a run that will
        // never report. Every failure has to come back as a message.
        return void startAnalysis(ws, session, msg.config, requestId).catch((err) => {
          console.error(`  ! run-analysis failed unexpectedly: ${err && err.stack ? err.stack : err}`);
          session.bridge = null;
          session.requestId = null;
          session.inputPath = null;
          sendFailure(ws, requestId, 'internal-error', 'The analysis could not be started.');
        });
      case 'cancel':
        return cancelAnalysis(ws, session, requestId);
      case 'ping':
        return send(ws, { type: 'pong', requestId });
      default:
        return sendFailure(ws, requestId, 'invalid-message', 'Unrecognised message type.');
    }
  });

  ws.on('close', () => {
    // A closed tab must not leave a Python process running.
    if (session.bridge) session.bridge.cancel();
    sessions.delete(ws);
  });

  ws.on('error', (err) => console.error(`  ! websocket error: ${err.message}`));
});

const heartbeat = setInterval(() => {
  for (const [ws, session] of sessions) {
    if (!session.alive) { ws.terminate(); continue; }
    session.alive = false;
    try { ws.ping(); } catch { /* closing */ }
  }
}, 30000);
if (typeof heartbeat.unref === 'function') heartbeat.unref();

async function startAnalysis(ws, session, rawConfig, requestId) {
  if (session.bridge) {
    return sendFailure(ws, requestId, 'busy', 'An analysis is already running on this connection.');
  }

  // Every field is type-, range- and path-checked before it can become a CLI
  // argument; nothing is coerced, so a bad value is an error, not a default.
  const validation = validateConfig(rawConfig, {
    inputRoots: INPUT_ROOTS,
    outputRoots: [ANALYSIS_DIR],
    allowedExtensions: ALLOWED_UPLOAD_EXTENSIONS,
    defaultOutputDir: ANALYSIS_DIR,
  });
  if (!validation.ok) {
    return sendFailure(ws, requestId, 'invalid-config', 'The analysis settings were rejected.', {
      fields: validation.errors,
    });
  }

  const bridge = new PythonBridge(PYTHON_DIR);
  session.bridge = bridge;
  session.requestId = requestId;
  session.inputPath = validation.config.filePath;
  activeRuns += 1;

  // Log forwarding with backpressure: when the socket is already behind, lines
  // are counted instead of queued and the count rides along on the next one.
  let dropped = 0;
  const forward = (stream) => (line) => {
    if (ws.bufferedAmount > MAX_LOG_BACKLOG_BYTES) { dropped += 1; return; }
    const payload = { type: 'log', requestId, stream, line: redact(line) };
    if (dropped > 0) { payload.dropped = dropped; dropped = 0; }
    send(ws, payload);
  };
  bridge.on('stdout', forward('stdout'));
  bridge.on('stderr', forward('stderr'));
  bridge.on('progress', p => send(ws, {
    type: 'progress', requestId, percent: p.percent, message: p.message,
  }));

  const flushDropped = () => {
    if (dropped === 0) return;
    send(ws, {
      type: 'log', requestId, stream: 'stdout', dropped,
      line: `[${dropped} log ${dropped === 1 ? 'line' : 'lines'} omitted to keep the interface responsive]`,
    });
    dropped = 0;
  };

  send(ws, { type: 'started', requestId, startedAt: new Date().toISOString() });

  try {
    const result = await bridge.runAnalysis(validation.config);
    flushDropped();

    // The backend prints where it wrote; confirm that is somewhere we may read
    // from before handing the path to the UI.
    const outputDirs = readableOutputDirs(result.outputDirs);
    if (outputDirs.length === 0) {
      return sendFailure(ws, requestId, 'output-missing',
        'The analysis finished but did not report a readable results directory.',
        { exitCode: result.exitCode });
    }

    send(ws, {
      type: 'complete',
      requestId,
      outputDir: outputDirs[0],
      outputDirs,                                  // one per channel with --channels all
      outputId: path.basename(outputDirs[0]),
      exitCode: result.exitCode,
      elapsedMs: result.elapsedMs,
    });
  } catch (err) {
    flushDropped();
    const code = err && err.code ? err.code : 'internal-error';
    if (code === 'cancelled') {
      send(ws, { type: 'cancelled', requestId, elapsedMs: err.elapsedMs ?? null });
    } else {
      console.error(`  ! analysis ${code}: ${err && err.message}`);
      // "no shots" and "inadmissible" are verdicts: the backend still wrote a
      // directory, and the UI needs it to explain why the run is unusable.
      const outputDirs = readableOutputDirs(err && err.outputDirs);
      sendFailure(ws, requestId, code, redact(err && err.message ? err.message : 'The analysis failed.'), {
        exitCode: err && err.exitCode !== undefined ? err.exitCode : null,
        signal: err && err.signal !== undefined ? err.signal : null,
        outputDir: outputDirs.length > 0 ? outputDirs[0] : null,
        outputDirs,
        stderr: err && err.stderrTail ? redact(err.stderrTail).slice(-8000) : null,
      });
    }
  } finally {
    session.bridge = null;
    session.requestId = null;
    session.inputPath = null;
    activeRuns -= 1;
  }
}

function cancelAnalysis(ws, session, requestId) {
  if (!session.bridge) {
    return sendFailure(ws, requestId, 'not-running', 'There is no analysis to cancel.');
  }
  session.bridge.cancel();
  send(ws, { type: 'cancelling', requestId: requestId ?? session.requestId });
}

/**
 * Keep only the directories the backend reported that this server is actually
 * allowed to read, so a path from the child process still cannot widen access.
 */
function readableOutputDirs(dirs) {
  if (!Array.isArray(dirs)) return [];
  const out = [];
  for (const dir of dirs) {
    const resolved = resolveResultsDir(dir);
    if (resolved && !out.includes(resolved)) out.push(resolved);
  }
  return out;
}

/** Input files of running analyses, so the sweeper never deletes one in use. */
function inputPathsInUse() {
  const inUse = new Set();
  for (const session of sessions.values()) {
    if (session.inputPath) inUse.add(session.inputPath);
  }
  return inUse;
}

// ────────────────────────────────────────────────────────────────────────────
// Upload housekeeping
// ────────────────────────────────────────────────────────────────────────────

/** Delete uploads older than the TTL. Regular files only, never recursive. */
function sweepUploads() {
  if (UPLOAD_TTL_HOURS <= 0) return;
  const cutoff = Date.now() - UPLOAD_TTL_HOURS * 3600 * 1000;
  const inUse = inputPathsInUse();
  let removed = 0;

  let entries;
  try { entries = fs.readdirSync(UPLOAD_DIR, { withFileTypes: true }); } catch { return; }

  for (const entry of entries) {
    if (!entry.isFile()) continue;                   // skips dirs and symlinks
    const full = path.join(UPLOAD_DIR, entry.name);
    if (inUse.has(full)) continue;
    try {
      const stat = fs.lstatSync(full);
      if (!stat.isFile() || stat.mtimeMs >= cutoff) continue;
      fs.unlinkSync(full);
      removed += 1;
    } catch { /* raced with another delete */ }
  }

  if (removed > 0) console.log(`  · removed ${removed} ${removed === 1 ? 'upload' : 'uploads'} older than ${UPLOAD_TTL_HOURS}h`);
}

const sweeper = setInterval(sweepUploads, UPLOAD_SWEEP_MINUTES * 60 * 1000);
if (typeof sweeper.unref === 'function') sweeper.unref();

// ────────────────────────────────────────────────────────────────────────────
// Start
// ────────────────────────────────────────────────────────────────────────────

function openBrowser(url) {
  // execFile, not exec: no shell, so nothing in the URL can be interpreted.
  if (process.platform === 'darwin') return execFile('open', [url], () => {});
  if (process.platform === 'win32') return execFile('cmd', ['/c', 'start', '', url], () => {});
  return execFile('xdg-open', [url], () => {});
}

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n  ! Port ${PORT} is already in use. Set SASA_PORT to choose another.\n`);
  } else {
    console.error(`\n  ! Server error: ${err.message}\n`);
  }
  process.exit(1);
});

server.listen(PORT, HOST, () => {
  const displayHost = HOST === '127.0.0.1' || HOST === '::1' ? 'localhost' : HOST;
  const url = `http://${displayHost}:${PORT}`;
  console.log(`\n  ╔══════════════════════════════════════════╗`);
  console.log(`  ║   SASA — Shot Acoustic Spectral Analysis ║`);
  console.log(`  ║   Ridgeback Defense                      ║`);
  console.log(`  ╠══════════════════════════════════════════╣`);
  console.log(`  ║   UI running at: ${url}          ║`);
  console.log(`  ╚══════════════════════════════════════════╝\n`);
  if (HOST !== '127.0.0.1' && HOST !== 'localhost' && HOST !== '::1') {
    console.warn(`  ! Bound to ${HOST}: this server reads local files and is now reachable`);
    console.warn(`    from the network. Use SASA_HOST=127.0.0.1 unless that is intended.\n`);
  }

  sweepUploads();
  if (OPEN_BROWSER) openBrowser(url);
});

let shuttingDown = false;
function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`\n  · ${signal} received — shutting down.`);
  for (const [ws, session] of sessions) {
    if (session.bridge) session.bridge.cancel();
    try { ws.close(1001, 'Server shutting down'); } catch { /* already closed */ }
  }
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 8000).unref();
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
