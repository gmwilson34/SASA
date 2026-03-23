#!/usr/bin/env node
/**
 * SASA UI Server
 * Serves the web UI and bridges to the Python analysis backend.
 * Launches the user's default browser automatically.
 */

const http = require('http');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const express = require('express');
const multer = require('multer');
const { WebSocketServer } = require('ws');
const { PythonBridge } = require('./bridge/python-bridge');

const PYTHON_DIR = path.resolve(__dirname, '..');
const PORT = process.env.SASA_PORT || 3847;
const UPLOAD_DIR = path.join(PYTHON_DIR, 'Audio', 'uploads');

const app = express();
app.use(express.json());

// Ensure upload directory exists
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// Configure multer for file uploads — preserve original name in Audio/uploads/
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOAD_DIR),
    filename: (req, file, cb) => {
      // Avoid collisions: prefix with timestamp if file already exists
      let name = file.originalname;
      if (fs.existsSync(path.join(UPLOAD_DIR, name))) {
        const ext = path.extname(name);
        const base = path.basename(name, ext);
        name = `${base}_${Date.now()}${ext}`;
      }
      cb(null, name);
    },
  }),
});

// ── Static Files ──
app.use(express.static(path.join(__dirname, 'renderer')));

// ── MIME types for images served from analysis output ──
const MIME_MAP = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf',
  '.html': 'text/html',
  '.csv': 'text/csv',
  '.json': 'application/json',
};

// ── API: Upload a file and return its absolute path ──
app.post('/api/upload', upload.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
  res.json({ path: req.file.path, name: req.file.originalname });
});

// ── API: List previous analyses ──
app.get('/api/analyses', (req, res) => {
  const analysisDir = path.join(PYTHON_DIR, 'Audio', 'analysis');
  if (!fs.existsSync(analysisDir)) return res.json([]);

  try {
    const entries = fs.readdirSync(analysisDir, { withFileTypes: true })
      .filter(d => d.isDirectory())
      .map(d => {
        const metaPath = path.join(analysisDir, d.name, 'analysis_metadata.json');
        let meta = null;
        if (fs.existsSync(metaPath)) {
          try { meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8')); } catch {}
        }
        return { name: d.name, path: path.join(analysisDir, d.name), meta };
      })
      .filter(d => d.meta)
      .sort((a, b) => (b.meta.timestamp || '').localeCompare(a.meta.timestamp || ''));

    res.json(entries);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── API: Load results for a specific analysis ──
app.get('/api/results', (req, res) => {
  const outputDir = req.query.dir;
  if (!outputDir || !fs.existsSync(outputDir)) {
    return res.status(400).json({ error: 'Invalid output directory' });
  }

  try {
    const metadataPath = path.join(outputDir, 'analysis_metadata.json');
    if (!fs.existsSync(metadataPath)) {
      return res.status(404).json({ error: 'No analysis_metadata.json found' });
    }
    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));

    // Collect images
    const images = {};
    const files = fs.readdirSync(outputDir);
    for (const file of files) {
      if (file.endsWith('.png') || file.endsWith('.html')) {
        const key = file.replace(/\.(png|html)$/, '');
        if (!images[key]) images[key] = {};
        images[key][file.endsWith('.html') ? 'html' : 'png'] = file;
      }
    }

    // Per-shot images
    const shotImages = [];
    const shotsDir = path.join(outputDir, 'shots');
    if (fs.existsSync(shotsDir)) {
      const shotFiles = fs.readdirSync(shotsDir).filter(f => f.endsWith('.png')).sort();
      for (const f of shotFiles) shotImages.push(f);
    }

    // CSV
    let csv = null;
    const csvPath = path.join(outputDir, 'metrics_summary.csv');
    if (fs.existsSync(csvPath)) {
      csv = fs.readFileSync(csvPath, 'utf-8');
    }

    res.json({ metadata, images, shotImages, csv, outputDir });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── API: Serve an image from an analysis output dir ──
app.get('/api/image', (req, res) => {
  const dir = req.query.dir;
  const file = req.query.file;
  const sub = req.query.sub; // optional subdirectory like "shots"

  if (!dir || !file) return res.status(400).send('Missing dir or file');

  // Security: prevent path traversal
  const safeName = path.basename(file);
  const safeSub = sub ? path.basename(sub) : null;
  const filePath = safeSub
    ? path.join(dir, safeSub, safeName)
    : path.join(dir, safeName);

  if (!fs.existsSync(filePath)) return res.status(404).send('Not found');

  const ext = path.extname(safeName).toLowerCase();
  res.setHeader('Content-Type', MIME_MAP[ext] || 'application/octet-stream');
  fs.createReadStream(filePath).pipe(res);
});

// ── Create HTTP Server & WebSocket ──
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

// ── WebSocket: handle analysis runs ──
wss.on('connection', (ws) => {
  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }

    if (msg.type === 'run-analysis') {
      runAnalysis(ws, msg.config);
    }
  });
});

async function runAnalysis(ws, config) {
  const bridge = new PythonBridge(PYTHON_DIR);

  bridge.on('stdout', (line) => {
    ws.send(JSON.stringify({ type: 'log', line }));
  });
  bridge.on('stderr', (line) => {
    ws.send(JSON.stringify({ type: 'log', line }));
  });
  bridge.on('progress', (pct) => {
    ws.send(JSON.stringify({ type: 'progress', pct }));
  });

  try {
    const result = await bridge.runAnalysis(config);
    ws.send(JSON.stringify({ type: 'complete', outputDir: result.outputDir }));
  } catch (err) {
    ws.send(JSON.stringify({ type: 'error', message: err.message }));
  }
}

// ── Start ──
server.listen(PORT, () => {
  const url = `http://localhost:${PORT}`;
  console.log(`\n  ╔══════════════════════════════════════════╗`);
  console.log(`  ║   SASA — Shot Acoustic Spectral Analysis ║`);
  console.log(`  ║   Ridgeback Defense                      ║`);
  console.log(`  ╠══════════════════════════════════════════╣`);
  console.log(`  ║   UI running at: ${url}          ║`);
  console.log(`  ╚══════════════════════════════════════════╝\n`);

  // Open in default browser
  const cmd = process.platform === 'darwin' ? 'open'
    : process.platform === 'win32' ? 'start'
    : 'xdg-open';
  exec(`${cmd} ${url}`);
});
