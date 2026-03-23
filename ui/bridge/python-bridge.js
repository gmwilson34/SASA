const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const EventEmitter = require('events');

class PythonBridge extends EventEmitter {
  constructor(pythonDir) {
    super();
    this.pythonDir = pythonDir;
    this.process = null;
  }

  _findPython() {
    // Check for virtual environment first
    const venvPython = path.join(this.pythonDir, '.venv', 'bin', 'python');
    if (fs.existsSync(venvPython)) return venvPython;

    const venvPython3 = path.join(this.pythonDir, '.venv', 'bin', 'python3');
    if (fs.existsSync(venvPython3)) return venvPython3;

    // Windows venv
    const venvPythonWin = path.join(this.pythonDir, '.venv', 'Scripts', 'python.exe');
    if (fs.existsSync(venvPythonWin)) return venvPythonWin;

    // Fallback to system python
    return 'python3';
  }

  runAnalysis(config) {
    return new Promise((resolve, reject) => {
      const python = this._findPython();
      const mainScript = path.join(this.pythonDir, 'main.py');

      const args = [mainScript, config.filePath];

      // Calibration
      if (config.paPerFS) {
        args.push('--Pa-per-FS', String(config.paPerFS));
      }
      if (config.sensitivityMv) {
        args.push('--sensitivity-mV', String(config.sensitivityMv));
      }
      if (config.vPerFS) {
        args.push('--V-per-FS', String(config.vPerFS));
      }
      if (config.calDesc) {
        args.push('--cal-desc', config.calDesc);
      }

      // Shot detection
      if (config.thresholdDb) {
        args.push('--threshold-dB', String(config.thresholdDb));
      }
      if (config.refractoryMs) {
        args.push('--refractory-ms', String(config.refractoryMs));
      }
      if (config.preMs) {
        args.push('--pre-ms', String(config.preMs));
      }
      if (config.postMs) {
        args.push('--post-ms', String(config.postMs));
      }

      // STFT
      if (config.nperseg) {
        args.push('--nperseg', String(config.nperseg));
      }

      // Analysis options
      if (config.noBands) {
        args.push('--no-bands');
      }
      if (config.noPerShot) {
        args.push('--no-per-shot');
      }

      // Output
      if (config.outputDir) {
        args.push('-o', config.outputDir);
      }

      // Formats
      if (config.formats) {
        args.push('--formats', config.formats);
      }

      this.process = spawn(python, args, {
        cwd: this.pythonDir,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      let stdout = '';
      let stderr = '';
      let lastOutputDir = null;

      this.process.stdout.on('data', (data) => {
        const text = data.toString();
        stdout += text;
        const lines = text.split('\n').filter(Boolean);
        for (const line of lines) {
          this.emit('stdout', line);

          // Parse progress hints
          const progressMatch = line.match(/\[(\d+)%\]/);
          if (progressMatch) {
            this.emit('progress', parseInt(progressMatch[1]));
          }

          // Detect output directory
          const dirMatch = line.match(/Results saved to:\s*(.+)/i) ||
                          line.match(/Output directory:\s*(.+)/i) ||
                          line.match(/Saving results to\s+(.+)/i);
          if (dirMatch) {
            lastOutputDir = dirMatch[1].trim();
          }
        }
      });

      this.process.stderr.on('data', (data) => {
        const text = data.toString();
        stderr += text;
        const lines = text.split('\n').filter(Boolean);
        for (const line of lines) {
          this.emit('stderr', line);
        }
      });

      this.process.on('close', (code) => {
        this.process = null;
        if (code === 0) {
          resolve({ outputDir: lastOutputDir, stdout, stderr });
        } else {
          reject(new Error(`Analysis failed (exit code ${code}):\n${stderr || stdout}`));
        }
      });

      this.process.on('error', (err) => {
        this.process = null;
        reject(new Error(`Failed to start Python: ${err.message}`));
      });
    });
  }

  cancel() {
    if (this.process) {
      this.process.kill('SIGTERM');
      this.process = null;
    }
  }
}

module.exports = { PythonBridge };
