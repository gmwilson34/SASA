/* ═══════════════════════════════════════════════════════════
   SASA UI — Renderer Application Logic
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  const state = {
    selectedFilePath: null,
    isRunning: false,
    currentOutputDir: null,
    ws: null,
    // Results data (kept for per-shot browsing)
    metadata: null,
    shotImages: [],
    currentShotIndex: 0,
  };

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const fileDrop = $('#file-drop');
  const fileInput = $('#file-input');
  const fileSelected = $('#file-selected');
  const fileName = $('#file-name');
  const filePath = $('#file-path');
  const fileClear = $('#file-clear');
  const filePathInput = $('#file-path-input');
  const btnSetPath = $('#btn-set-path');
  const btnRun = $('#btn-run');
  const progressSection = $('#progress-section');
  const progressFill = $('#progress-fill');
  const progressPct = $('#progress-pct');
  const logOutput = $('#log-output');
  const logContainer = $('#log-container');
  const calMode = $('#cal-mode');

  // ── Navigation ──
  $$('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  function switchView(viewName) {
    $$('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.view === viewName));
    $$('.view').forEach(v => v.classList.remove('active'));
    $(`#view-${viewName}`).classList.add('active');
    if (viewName === 'history') loadHistory();
  }

  // ── File Upload ──
  async function uploadFile(file) {
    showFileSelected(file.name, 'Uploading...');
    btnRun.disabled = true;
    const form = new FormData();
    form.append('file', file);
    try {
      const resp = await fetch('/api/upload', { method: 'POST', body: form });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Upload failed');
      state.selectedFilePath = data.path;
      showFileSelected(data.name, data.path);
      btnRun.disabled = false;
    } catch (err) {
      toast(`Upload failed: ${err.message}`, 'error');
      clearFileSelection();
    }
  }

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
  });

  fileDrop.addEventListener('dragover', e => { e.preventDefault(); fileDrop.classList.add('drag-over'); });
  fileDrop.addEventListener('dragleave', () => fileDrop.classList.remove('drag-over'));
  fileDrop.addEventListener('drop', e => {
    e.preventDefault();
    fileDrop.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
  });

  btnSetPath.addEventListener('click', () => setPathFromInput());
  filePathInput.addEventListener('keydown', e => { if (e.key === 'Enter') setPathFromInput(); });

  function setPathFromInput() {
    const p = filePathInput.value.trim();
    if (!p) return;
    state.selectedFilePath = p;
    showFileSelected(p.split('/').pop() || p, p);
    btnRun.disabled = false;
  }

  function showFileSelected(name, pathStr) {
    fileName.textContent = name;
    filePath.textContent = pathStr;
    fileDrop.classList.add('hidden');
    fileSelected.classList.remove('hidden');
  }

  function clearFileSelection() {
    state.selectedFilePath = null;
    fileInput.value = '';
    filePathInput.value = '';
    fileDrop.classList.remove('hidden');
    fileSelected.classList.add('hidden');
    btnRun.disabled = true;
  }

  fileClear.addEventListener('click', clearFileSelection);

  // ── Calibration Toggle ──
  calMode.addEventListener('change', () => {
    $('#cal-direct').classList.add('hidden');
    $('#cal-sensitivity').classList.add('hidden');
    if (calMode.value === 'direct') $('#cal-direct').classList.remove('hidden');
    else if (calMode.value === 'sensitivity') $('#cal-sensitivity').classList.remove('hidden');
  });

  // ── WebSocket ──
  function connectWS() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${location.host}/ws`);
    ws.onmessage = event => {
      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }
      if (msg.type === 'log') {
        logOutput.textContent += msg.line + '\n';
        logContainer.scrollTop = logContainer.scrollHeight;
      } else if (msg.type === 'progress') {
        const pct = Math.min(100, Math.max(0, msg.pct));
        progressFill.style.width = `${pct}%`;
        progressPct.textContent = `${pct}%`;
      } else if (msg.type === 'complete') {
        progressFill.style.width = '100%';
        progressPct.textContent = '100%';
        logOutput.textContent += '\n--- Analysis complete ---\n';
        if (msg.outputDir) {
          state.currentOutputDir = msg.outputDir;
          loadResults(msg.outputDir).then(() => switchView('results'));
        }
        analysisFinished();
      } else if (msg.type === 'error') {
        logOutput.textContent += `\nError: ${msg.message}\n`;
        toast(msg.message, 'error');
        analysisFinished();
      }
    };
    ws.onclose = () => setTimeout(connectWS, 2000);
    state.ws = ws;
  }
  connectWS();

  // ── Run Analysis ──
  btnRun.addEventListener('click', () => {
    if (state.isRunning || !state.selectedFilePath) return;
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      toast('Reconnecting...', 'error');
      connectWS();
      return;
    }

    state.isRunning = true;
    btnRun.classList.add('running');
    btnRun.querySelector('span').textContent = 'Running...';
    btnRun.disabled = true;
    progressSection.classList.remove('hidden');
    progressFill.style.width = '0%';
    progressPct.textContent = '0%';
    logOutput.textContent = '';

    const config = {
      filePath: state.selectedFilePath,
      thresholdDb: parseFloat($('#threshold-db').value) || 120,
      refractoryMs: parseFloat($('#refractory-ms').value) || 200,
      preMs: parseFloat($('#pre-ms').value) || 50,
      postMs: parseFloat($('#post-ms').value) || 200,
      nperseg: parseInt($('#nperseg').value) || 2048,
      formats: $('#formats').value,
      noBands: !$('#compute-bands').checked,
      noPerShot: !$('#per-shot-plots').checked,
    };
    if (calMode.value === 'direct') config.paPerFS = parseFloat($('#pa-per-fs').value);
    else if (calMode.value === 'sensitivity') {
      config.sensitivityMv = parseFloat($('#sensitivity-mv').value);
      config.vPerFS = parseFloat($('#v-per-fs').value);
    }
    const calDesc = $('#cal-desc').value.trim();
    if (calDesc) config.calDesc = calDesc;

    state.ws.send(JSON.stringify({ type: 'run-analysis', config }));
  });

  function analysisFinished() {
    state.isRunning = false;
    btnRun.classList.remove('running');
    btnRun.querySelector('span').textContent = 'Run Analysis';
    btnRun.disabled = false;
  }

  // ════════════════════════════════════════════
  // RESULTS
  // ════════════════════════════════════════════

  async function loadResults(outputDir) {
    try {
      const resp = await fetch(`/api/results?dir=${encodeURIComponent(outputDir)}`);
      const data = await resp.json();
      if (!resp.ok) return console.error('Load failed:', data.error);

      state.currentOutputDir = data.outputDir;
      state.metadata = data.metadata;
      state.shotImages = data.shotImages || [];

      $('#results-empty').classList.add('hidden');
      $('#results-loaded').classList.remove('hidden');

      const inputFile = data.metadata.input_file || 'Unknown';
      const nShots = data.metadata.n_shots || 0;
      $('#results-subtitle').textContent = `${inputFile} — ${nShots} shots detected`;

      renderMetrics(data.metadata);

      // Render plots — prefer interactive HTML, fall back to PNG
      renderPlot('plot-waveform', data.outputDir, data.images, 'waveform_full');
      renderPlot('plot-spectrogram-z', data.outputDir, data.images, 'spectrogram_z_full');

      // Handle both old (A-weighted) and new (C-weighted) spectrogram naming
      const specKey2 = data.images['spectrogram_c_full'] ? 'spectrogram_c_full'
                     : data.images['spectrogram_a_full'] ? 'spectrogram_a_full' : null;
      const specLabel2 = specKey2 === 'spectrogram_a_full' ? 'Spectrogram (A)' : 'Spectrogram (C)';
      // Update tab label to match what data exists
      const specTab = document.querySelector('[data-tab="spectrogram-c"]');
      if (specTab) specTab.textContent = specLabel2;
      renderPlot('plot-spectrogram-c', data.outputDir, data.images, specKey2);

      renderPlot('plot-bands', data.outputDir, data.images, 'bands_full');

      // Per-shot setup
      renderShotNav(data.metadata, data.shotImages);
      renderTable(data.csv);
    } catch (err) {
      console.error('Load results error:', err);
    }
  }

  // ── Aggregate Metric Cards ──
  function renderMetrics(meta) {
    const agg = meta.aggregate || {};
    // Correct keys: Lpeak_Z_max (not _mean), LAFmax_mean, LAE_mean
    const cards = [
      { cls: 'peak', label: 'Peak SPL (Z)', value: formatDb(agg.Lpeak_Z_max), unit: 'dB' },
      { cls: 'rms',  label: 'LAFmax (mean)', value: formatDb(agg.LAFmax_mean), unit: 'dB' },
      { cls: 'sel',  label: 'LAE (mean)', value: formatDb(agg.LAE_mean), unit: 'dB' },
      { cls: 'shots', label: 'Shots Detected', value: meta.n_shots ?? '—', unit: '' },
    ];
    $('#metrics-row').innerHTML = cards.map(c => `
      <div class="metric-card ${c.cls}">
        <div class="metric-label">${c.label}</div>
        <div class="metric-value">${c.value}<span class="metric-unit">${c.unit}</span></div>
      </div>
    `).join('');
  }

  function formatDb(val) {
    if (val == null || isNaN(val)) return '—';
    return val.toFixed(1);
  }

  // ── Plot Renderer (HTML iframe or PNG fallback) ──
  function renderPlot(containerId, outputDir, images, key) {
    const container = $(`#${containerId}`);
    const imgData = images[key];

    if (!imgData) {
      container.innerHTML = '<span class="no-image">No data available</span>';
      return;
    }

    // Prefer interactive HTML (Plotly)
    if (imgData.html) {
      const src = `/api/image?dir=${encodeURIComponent(outputDir)}&file=${encodeURIComponent(imgData.html)}`;
      container.innerHTML = `<iframe src="${src}" title="${key}"></iframe>`;
      return;
    }

    // Fall back to PNG
    if (imgData.png) {
      const src = `/api/image?dir=${encodeURIComponent(outputDir)}&file=${encodeURIComponent(imgData.png)}`;
      container.innerHTML = `<img src="${src}" alt="${key}" />`;
      container.querySelector('img').addEventListener('click', () => openModal(src));
      return;
    }

    container.innerHTML = '<span class="no-image">No image available</span>';
  }

  // ── Per-Shot Navigator ──
  function renderShotNav(meta, shotImages) {
    const agg = meta.aggregate || {};
    const shots = agg.shots || [];
    const selector = $('#shot-selector');

    if (shots.length === 0) {
      selector.innerHTML = '<span class="no-image">No per-shot data</span>';
      $('#shot-metrics-row').innerHTML = '';
      $('#plot-shot-detail').innerHTML = '';
      return;
    }

    // Build shot pills
    selector.innerHTML = shots.map((_, i) => `
      <button class="shot-pill ${i === 0 ? 'active' : ''}" data-idx="${i}">${i + 1}</button>
    `).join('');

    selector.querySelectorAll('.shot-pill').forEach(pill => {
      pill.addEventListener('click', () => selectShot(parseInt(pill.dataset.idx)));
    });

    $('#shot-prev').onclick = () => selectShot(Math.max(0, state.currentShotIndex - 1));
    $('#shot-next').onclick = () => selectShot(Math.min(shots.length - 1, state.currentShotIndex + 1));

    selectShot(0);
  }

  function selectShot(idx) {
    const agg = state.metadata?.aggregate || {};
    const shots = agg.shots || [];
    if (idx < 0 || idx >= shots.length) return;

    state.currentShotIndex = idx;
    const shot = shots[idx];

    // Update pill active state
    $$('.shot-pill').forEach((p, i) => p.classList.toggle('active', i === idx));

    // Shot metrics
    const metricsRow = $('#shot-metrics-row');
    const fields = [
      { label: 'Lpeak Z', value: shot.Lpeak_Z, unit: 'dB' },
      { label: 'Lpeak A', value: shot.Lpeak_A, unit: 'dB' },
      { label: 'Lpeak C', value: shot.Lpeak_C, unit: 'dB' },
      { label: 'LAE',     value: shot.LAE,     unit: 'dB' },
      { label: 'LAFmax',  value: shot.LAFmax,  unit: 'dB' },
      { label: 'LASmax',  value: shot.LASmax,  unit: 'dB' },
      { label: 'LZFmax',  value: shot.LZFmax,  unit: 'dB' },
      { label: 'Duration', value: shot.duration_s ? (shot.duration_s * 1000).toFixed(0) : null, unit: 'ms' },
    ];
    metricsRow.innerHTML = fields.map(f => `
      <div class="shot-metric-card">
        <div class="shot-metric-label">${f.label}</div>
        <div class="shot-metric-value">${f.value != null ? (typeof f.value === 'number' ? f.value.toFixed(1) : f.value) : '—'}<span class="metric-unit">${f.unit}</span></div>
      </div>
    `).join('');

    // Draw band exposure chart
    drawBandChart(shot);

    // Shot summary image
    const detail = $('#plot-shot-detail');
    if (state.shotImages.length > idx) {
      const imgFile = state.shotImages[idx];
      const src = `/api/image?dir=${encodeURIComponent(state.currentOutputDir)}&sub=shots&file=${encodeURIComponent(imgFile)}`;
      detail.innerHTML = `<img src="${src}" alt="Shot ${idx + 1}" />`;
      detail.querySelector('img').addEventListener('click', () => openModal(src));
    } else {
      detail.innerHTML = '<span class="no-image">No summary plot for this shot</span>';
    }
  }

  // ── Interactive Band Exposure Chart (Canvas) ──
  function drawBandChart(shot) {
    const canvas = $('#shot-band-canvas');
    if (!canvas) return;
    const freqs = shot.band_frequencies;
    const levels = shot.band_exposure_dB;
    if (!freqs || !levels || freqs.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width - 40;
    const H = 280;
    // Skip drawing if container isn't visible yet (zero width)
    if (W <= 0) return;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const pad = { top: 24, right: 20, bottom: 44, left: 50 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const minDb = Math.floor(Math.min(...levels) / 5) * 5 - 5;
    const maxDb = Math.ceil(Math.max(...levels) / 5) * 5 + 5;
    const barW = Math.max(4, plotW / freqs.length - 2);

    // Grid lines
    ctx.strokeStyle = '#1e1e2a';
    ctx.lineWidth = 1;
    ctx.font = '10px -apple-system, sans-serif';
    ctx.fillStyle = '#555568';
    ctx.textAlign = 'right';
    for (let db = minDb; db <= maxDb; db += 10) {
      const y = pad.top + plotH - ((db - minDb) / (maxDb - minDb)) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillText(`${db}`, pad.left - 6, y + 3);
    }

    // Bars
    for (let i = 0; i < freqs.length; i++) {
      const x = pad.left + (i / freqs.length) * plotW + 1;
      const val = levels[i];
      const barH = ((val - minDb) / (maxDb - minDb)) * plotH;
      const y = pad.top + plotH - barH;

      // Color gradient based on dB level
      const t = (val - minDb) / (maxDb - minDb);
      const r = Math.round(59 + t * 190);
      const g = Math.round(130 - t * 80);
      const b = Math.round(246 - t * 100);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x, y, barW, barH);
    }

    // X-axis labels
    ctx.fillStyle = '#555568';
    ctx.textAlign = 'center';
    ctx.font = '9px -apple-system, sans-serif';
    const labelEvery = freqs.length > 20 ? 3 : freqs.length > 10 ? 2 : 1;
    for (let i = 0; i < freqs.length; i += labelEvery) {
      const x = pad.left + (i / freqs.length) * plotW + barW / 2;
      const label = freqs[i] >= 1000 ? `${(freqs[i] / 1000).toFixed(freqs[i] % 1000 === 0 ? 0 : 1)}k` : `${freqs[i]}`;
      ctx.save();
      ctx.translate(x, pad.top + plotH + 10);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(label, 0, 0);
      ctx.restore();
    }

    // Title
    ctx.fillStyle = '#8888a0';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`1/3-Octave Band Exposure — Shot ${shot.shot_number}`, W / 2, 14);

    // Y-axis label
    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('dB SPL', 0, 0);
    ctx.restore();

    // ── Tooltip on hover ──
    canvas.onmousemove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left - pad.left;
      const idx = Math.floor((mx / plotW) * freqs.length);
      if (idx >= 0 && idx < freqs.length) {
        canvas.title = `${freqs[idx]} Hz: ${levels[idx].toFixed(1)} dB SPL`;
        canvas.style.cursor = 'crosshair';
      } else {
        canvas.title = '';
        canvas.style.cursor = 'default';
      }
    };
  }

  // ── Tabs ──
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      $$('.tab-panel').forEach(p => p.classList.remove('active'));
      $(`#tab-${btn.dataset.tab}`).classList.add('active');
      // Re-draw the band chart when the per-shot tab becomes visible
      // (fixes first-render bug where canvas has zero width)
      if (btn.dataset.tab === 'shots') {
        const agg = state.metadata?.aggregate || {};
        const shots = agg.shots || [];
        if (shots.length > 0 && state.currentShotIndex < shots.length) {
          requestAnimationFrame(() => drawBandChart(shots[state.currentShotIndex]));
        }
      }
    });
  });

  // ── CSV Table ──
  function renderTable(csv) {
    const container = $('#metrics-table');
    if (!csv) {
      container.innerHTML = '<span class="no-image" style="padding:20px;display:block">No metrics CSV available</span>';
      return;
    }
    const lines = csv.trim().split('\n');
    if (lines.length < 2) {
      container.innerHTML = '<span class="no-image" style="padding:20px;display:block">Empty CSV</span>';
      return;
    }
    const headers = parseCSVLine(lines[0]);
    const rows = lines.slice(1).map(parseCSVLine);
    let html = '<table class="metrics-tbl"><thead><tr>';
    for (const h of headers) html += `<th>${esc(h)}</th>`;
    html += '</tr></thead><tbody>';
    for (const row of rows) {
      html += '<tr>';
      for (const cell of row) html += `<td>${esc(cell)}</td>`;
      html += '</tr>';
    }
    html += '</tbody></table>';
    container.innerHTML = html;
  }

  function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    for (const ch of line) {
      if (ch === '"') inQuotes = !inQuotes;
      else if (ch === ',' && !inQuotes) { result.push(current.trim()); current = ''; }
      else current += ch;
    }
    result.push(current.trim());
    return result;
  }

  function esc(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ── Image Modal ──
  function openModal(src) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `<img src="${src}" />`;
    overlay.addEventListener('click', () => overlay.remove());
    const handler = e => { if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', handler); } };
    document.addEventListener('keydown', handler);
    document.body.appendChild(overlay);
  }

  // ── Copy Path ──
  $('#btn-copy-path').addEventListener('click', () => {
    if (state.currentOutputDir) {
      navigator.clipboard.writeText(state.currentOutputDir).then(() => toast('Copied', 'success'));
    }
  });

  // ── History ──
  async function loadHistory() {
    const list = $('#history-list');
    list.innerHTML = '<span class="no-image" style="padding:40px;display:block;text-align:center;color:var(--text-muted)">Loading...</span>';
    try {
      const resp = await fetch('/api/analyses');
      const analyses = await resp.json();
      if (!analyses || analyses.length === 0) {
        list.innerHTML = '<span class="no-image" style="padding:40px;display:block;text-align:center;color:var(--text-muted)">No previous analyses found</span>';
        return;
      }
      list.innerHTML = analyses.map(entry => {
        const meta = entry.meta;
        const nShots = meta.n_shots ?? '—';
        const duration = meta.duration_s ? `${meta.duration_s.toFixed(1)}s` : '—';
        const sr = meta.sample_rate ? `${(meta.sample_rate / 1000).toFixed(0)} kHz` : '';
        const peakDb = meta.aggregate?.Lpeak_Z_max;
        return `
          <div class="history-item" data-path="${esc(entry.path)}">
            <div class="history-info">
              <div class="history-name">${esc(entry.name)}</div>
              <div class="history-detail">${esc(meta.input_file || '')} · ${sr} · ${duration}</div>
            </div>
            <div class="history-meta">
              <div class="history-stat">
                <span class="history-stat-value">${nShots}</span>
                <span class="history-stat-label">Shots</span>
              </div>
              ${peakDb != null ? `
              <div class="history-stat">
                <span class="history-stat-value">${peakDb.toFixed(1)}</span>
                <span class="history-stat-label">Peak dB</span>
              </div>` : ''}
            </div>
          </div>
        `;
      }).join('');
      list.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', async () => {
          await loadResults(item.dataset.path);
          switchView('results');
        });
      });
    } catch {
      list.innerHTML = '<span class="no-image" style="padding:40px;display:block;text-align:center;color:var(--text-muted)">Error loading history</span>';
    }
  }

  // ── Toast ──
  function toast(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3000);
  }
});
