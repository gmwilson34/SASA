# AHAAH in SASA — what it is, and what this build will and will not tell you

**Status of this build: NOT VALIDATED. SASA produces no Auditory Risk Unit figure.**

`ahaah.py` implements the US Army Research Laboratory's Auditory Hazard Assessment
Algorithm for Humans as far as the public release allows. It does not reproduce ARL's own
reference case, so it emits nothing. `compute_ahaah()` returns a result with
`valid = False`, every hazard field `NaN`, and an explanation. There is no configuration
flag, no override and no "advanced mode" that turns a number on.

If you need an impulse-noise hazard figure today, use the **A-weighted energy metrics**
(peak level, L<sub>Aeq</sub>, L<sub>Aeq8hr</sub>, A-weighted energy, allowed rounds
N<sub>a</sub>). MIL-STD-1474E approves those alongside AHAAH, SASA computes them exactly,
and they are unaffected by anything on this page.

---

## 1. What AHAAH is

AHAAH is an electroacoustic model of the human ear, developed at ARL over roughly three
decades, that predicts hearing hazard from a single impulse by simulating what the ear
physically does with it:

```
free-field pressure
  → outer ear (head diffraction, concha, ear canal)
  → middle ear: an electroacoustic network with two nonlinearities
        (1) annular-ligament peak limiting of stapes motion
        (2) the middle-ear-muscle (acoustic reflex) attenuator
  → stapes volume velocity
  → cochlea: a 1-D travelling-wave model, evaluated at 23 places along the
    basilar membrane (11.76 kHz down to 0.38 kHz)
  → basilar-membrane displacement
  → hazard: square each upward displacement peak, in microns, and sum them
    at each of the 23 places. The LARGEST of the 23 sums is the total.
```

That total is the **Auditory Risk Unit (ARU)**. Derived quantities:

| Quantity | Relation |
|---|---|
| Allowed exposures, occasional (≤ 1 session/week) | N = 500 / ARU |
| Allowed exposures, occupational (≥ 2 sessions/week) | N = 200 / ARU |
| Compound threshold shift | CTS dB = 26.6 · ln(ARU) − 140.1 |
| Predicted permanent shift | ≈ 0.6 × CTS once 500 ARU is exceeded |

500 ARU corresponds to about 25 dB of threshold shift. All four relations are reproduced
exactly by this implementation and are verified in the test suite; they are not the
problem. The **ARU itself** is.

AHAAH always models the **95th-percentile (most susceptible) ear**, which ARL achieves by
raising the test impulse by 10 dB before the model runs. Published ARL figures, including
the reference case below, are 95th-percentile figures.

---

## 2. Warned vs unwarned — the crux, and why unwarned leads

The acoustic reflex tightens the middle-ear muscles and attenuates transmission by roughly
20 dB below 1 kHz. AHAAH models two conditions:

- **Warned** — the listener knows the shot is coming, so the contraction is already in
  place when the impulse arrives.
- **Unwarned** — the contraction has to be evoked by the impulse itself, arriving about
  9 ms late, i.e. after the peak has already passed.

For ARL's own reference case the two answers are **391.0 ARU warned** and
**2237 ARU unwarned** — a factor of **5.72**. Which switch you set is worth more than any
other decision in the whole analysis.

**SASA leads with unwarned, always, and computes both.** `compute_ahaah_both()` returns
`(unwarned, warned)` in that order and every report format puts unwarned first. The reason
is in section 5: the field evidence that the protective reflex can be assumed is weak, and
the unwarned case is the conservative one.

---

## 3. Achieved validation — the actual numbers

The public AHAAH v2.1 release contains exactly one reference waveform with published
results: `160F.AHA` (2048 samples at 125 kHz, peak 186.936 dB, free field / human /
normal incidence), with a companion 23-band table `160F.HAZ`. That is the entire oracle.

Run `python ahaah.py <path>/160F.AHA --research --haz <path>/160F.HAZ` to reproduce this.

### Headline

| Check | This implementation | ARL reference | Error | Tolerance | Result |
|---|---|---|---|---|---|
| Warned total | 387.703 | 391.0 | **−0.8 %** | ±5 % | see note |
| **Unwarned total** | 1085.736 | 2237 | **−51.5 %** | ±5 % | **FAIL** |
| Unwarned/warned ratio | 2.80 | 5.72 | **−51.1 %** | ±10 % | **FAIL** |
| Warned peak band | 8 (3.94 kHz) | 9 (3.37 kHz) | — | exact | **FAIL** |
| Bands within ±10 % | **3 of 23** | — | — | 23 of 23 | **FAIL** |
| Allowed exposures, warned | 1.3 | 1.3 | — | — | pass |
| Allowed exposures, unwarned | 0.5 | 0.2 | — | — | **FAIL** |
| Peak stapes displacement | 24.70 µm | "a few tens of microns" | — | 10–60 µm | plausible |

### Per band, warned condition (the condition `160F.HAZ` reports)

| Band | f (kHz) | Model (µm²) | ARL (AHU) | Ratio |
|---:|---:|---:|---:|---:|
| 1 | 11.76 | 140.45 | 185.05 | 0.76 |
| 2 | 10.06 | 196.69 | 238.24 | 0.83 |
| 3 | 8.60 | 220.23 | 218.45 | **1.01** |
| 4 | 7.36 | 258.23 | 226.66 | 1.14 |
| 5 | 6.29 | 305.16 | 293.46 | **1.04** |
| 6 | 5.38 | 330.54 | 287.91 | 1.15 |
| 7 | 4.60 | 324.43 | 269.02 | 1.21 |
| 8 | 3.94 | 387.70 | 335.17 | 1.16 |
| 9 | 3.37 | 383.64 | 390.99 | **0.98** |
| 10 | 2.88 | 307.79 | 380.82 | 0.81 |
| 11 | 2.46 | 351.64 | 302.05 | 1.16 |
| 12 | 2.11 | 297.05 | 245.65 | 1.21 |
| 13 | 1.80 | 325.70 | 187.75 | 1.74 |
| 14 | 1.54 | 379.77 | 141.06 | 2.69 |
| 15 | 1.32 | 356.13 | 105.86 | **3.36** |
| 16 | 1.13 | 251.51 | 76.21 | 3.30 |
| 17 | 0.97 | 107.97 | 53.78 | 2.01 |
| 18 | 0.83 | 32.79 | 36.90 | 0.89 |
| 19 | 0.71 | 14.23 | 24.58 | 0.58 |
| 20 | 0.60 | 8.46 | 15.65 | 0.54 |
| 21 | 0.52 | 5.19 | 9.15 | 0.57 |
| 22 | 0.44 | 3.54 | 4.72 | 0.75 |
| 23 | 0.38 | 2.39 | 2.09 | 1.14 |

Bold entries are the three bands inside ±10 %.

### The warned figure is a coincidence, not a partial success

−0.8 % on the warned total looks like the model nearly works. It does not, and quoting it
that way would be the exact failure this codebase was rebuilt to remove.

- It is the **maximum of a curve that is wrong**. Twenty of the 23 bands miss by more than
  10 %, the worst by a factor of 3.4, and the curve peaks in the wrong band.
- Sweeping only the modelling switches the code already exposes as uncertain
  (`CochlearGainFactor`, `cf_alignment`, `extra_wkb_taper`, `peak_rule`, `bm_sign`,
  `outer_ear`) produces 96 configurations whose warned totals span **10.3 to 2617** — a
  factor of 255. **Four** of the 96 land within ±5 % of 391.0. **None** of those four is
  also within ±5 % of 2237 unwarned. Hitting the warned figure is roughly a 4 %
  coincidence in this parameter space.

### What is actually wrong

The reference constrains a *ratio* as well as a *level*, and the ratio fails structurally.
Across all 96 configurations above, the unwarned/warned ratio stays between **2.59 and
3.17** against a reference of **5.72**, while the absolute level moves by 255×. Two
separate errors are present and they partly cancel in the warned case — the worst state a
model can be in.

Probing further:

- Scaling the annular-ligament nonlinearity (`Lgap` from 10 to 300) moves the level by a
  factor of 33 000 and the ratio only from 2.78 to 3.16. **The ligament sets the level, not
  the ratio.**
- The middle-ear-muscle attenuator does move the ratio: `MemMagK`/`MemMagR` near 40 on a
  single element reproduces 5.54. The implementation uses 12, on one element.
- `man.coe` gives `MemMagK` as `12 1 6` and `MemMagR` as `12 1 12` — **three-element
  vectors**. Which circuit elements receive which of the three is documented nowhere, and
  this implementation applies only the leading 12, to the annular ligament alone.

So the first question to put to ARL is the **MEM element mapping**, then the
**annular-ligament functional form**. Both are listed in `docs/AHAAH-SPEC.md` §12.

### Two linear checks that do pass

These confirm the outer/middle-ear linear chain is built correctly. They say nothing about
the hazard figure.

- Free field → eardrum agrees with ARL's own `Dat/FFEDM90.DAT` (Mehrgardt & Mellert) to
  within 3.0 dB max / 1.4 dB rms from 0.2 to 8 kHz, with the ear-canal resonance at
  2663 Hz against a published 2660 Hz.
- The modelled middle-ear-muscle contraction attenuates 20.6 / 21.5 / 18.9 dB at
  125 / 500 / 1000 Hz, falling monotonically to 4.5 dB at 11.8 kHz — matching ARL's
  description, "on the order of 20 dB below about 1.0 kHz and progressively less at higher
  frequencies".

---

## 4. What this build guarantees, and what it does not

### It guarantees

- **No ARU is emitted.** `compute_ahaah()` and `compute_ahaah_both()` always return
  `valid = False` with `status = "not_validated"` and all-`NaN` hazard fields.
- **A bad recording is diagnosed as a bad recording first.** The input gates run before
  the model's own refusal, so you are told about the measurement problem, not just the
  model. The gates refuse: uncalibrated input; sample rate below 96 kHz; a clipped
  waveform; a peak below 130 dB; non-finite samples; an all-zero record; fewer than 16
  samples; and grazing incidence requested through the wrong outer-ear route.
- **No silent geometry guess.** An `.AHA` file whose "Microphone relative to ear"
  calculation code is missing or unrecognised is rejected, not defaulted. That code decides
  whether the head-diffraction and ear-canal gain (about 19 dB at 2.7 kHz) is applied.
- **No silent numerical runaway.** Above roughly 200 dB the stiffening annular ligament
  destabilises the time-domain solve. It used to return a large finite number and call it
  valid — at 214 dB, a hazard index of 6.7 × 10³⁶ from a stapes displacement of
  7 × 10¹⁴ metres. It now raises `ModelDivergedError`.
- **Both reflex conditions, unwarned first,** everywhere.
- **Every coefficient is traceable** (section 6) and every modelling inference is declared
  in `ahaah.DECLARED_ASSUMPTIONS`, categorised (a) documented / (b) derivable /
  (c) inferred. There are currently 14 category-(c) inferences.

### It does not guarantee

- **Any correspondence with a real AHAAH ARU.** That is the whole point of this page.
- **Anything about the four undocumented choices.** The functional form of the
  annular-ligament nonlinearity, the WKB cochlear equations and their "taper" term, which
  of the three listed `CochlearGainFactor` values is active (0.0724 vs 0.025 is a factor of
  8.4 in the answer), and the MEM element mapping are each capable of moving the result by
  an order of magnitude, and none of them appears anywhere in the public release.
- **Sufficiency of a single reference case.** Even a pass on 160F would validate a model
  with that many free choices only weakly. More published cases are needed.

### The research path

`run_unvalidated_model()` runs the model anyway, for development and for the validation
test. It must be called with `acknowledge_not_validated=True`, and it deliberately has no
field named ARU: the numbers come back as `band_sum_sq_displacement_um2` and
`max_band_sum_sq_um2`, the physical quantity actually computed (microns squared of summed
upward basilar-membrane peaks). The CLI exposes it behind `--research`. **Nothing
customer-facing may consume it, and nothing in SASA does.**

---

## 5. Scientific standing — AHAAH is not settled

MIL-STD-1474E (15 April 2015) approves **two** impulse-noise metrics: the ARU computed with
AHAAH, **and** the A-weighted energy method. They are alternatives within one standard and
**should be read together**. Where they disagree, the disagreement is information, not an
error to be resolved by picking the friendlier number.

AHAAH is not undisputed, and NATO has not adopted it:

- **2003, NATO** (RTO Task Group 021 / TR-017 work on the "Reconsideration of the effects
  of impulse noise") reported unsatisfactory results for several exposure conditions.
- **2010, American Institute of Biological Sciences** — the peer review commissioned for
  the standard. MIL-STD-1474E Table B-II records that it "generally supported the AHAAH
  model although recommended that several critical assumptions embedded in the model —
  especially the influence of the middle ear reflex under various conditions — need further
  research while a measure based on Leq8 can be used as an interim metric."
- **2012, NIOSH** review raised further concerns about the model's assumptions and its
  validation basis.
- **2016–2017**, work on acoustic-reflex prevalence found the middle-ear-muscle reflex is
  not pervasive enough to be assumed. In one live-fire study, early middle-ear-muscle
  contraction was **absent in 18 of 19 subjects** firing M4 rifles.

That last point is why SASA defaults to unwarned. The warned condition assumes a protective
reflex that, in the field, most shooters were found not to produce in time. For the ARL
reference case that assumption is worth a factor of 5.72 — the difference between "1.3
rounds allowed" and "0.2 rounds allowed".

---

## 6. Measurement geometry — required, and not guessable

AHAAH is defined on a pressure history at a stated place relative to the head, and the
model applies a different transfer for each. Getting it wrong is a ~19 dB error at
2.7 kHz, which the hazard rule then squares.

| Code | Meaning | SASA `input_location` |
|---|---|---|
| 1 | Free field, human, **normal** incidence (source toward the side of the head) | `free_field_normal` |
| 5 | Free field, human, **grazing** incidence (source ahead — a shooter firing a rifle) | `free_field_grazing` |
| 2 | Ear-canal entrance | `ear_canal_entrance` |
| 3 | Eardrum | `eardrum` |
| 4, 6 | Free field, **manikin**, normal / grazing — referred to the eardrum | `eardrum` |

Requirements for a measurement SASA will accept:

- **Calibrated.** ARU is an absolute-level metric. A recording in dB re full scale cannot
  produce one. Use a calibrator tone if at all possible — it captures the whole chain as
  configured.
- **Sample rate ≥ 96 kHz.** Band 1 sits at 11.76 kHz and the ARL reference case is sampled
  at 125 kHz. Below 96 kHz the blast rise and the high-frequency bands are not represented,
  and upsampling cannot restore them. (48–96 kHz can be forced with `allow_low_rate=True`,
  which attaches a permanent warning to the result. Below 48 kHz, nothing is computed.)
- **Not clipped.** The metric is dominated by the squared peak. A flat-topped peak
  understates the hazard by an unknown amount, so SASA refuses rather than guess.
- **Peak at or above 130 dB.** Below that ARL states the model does not apply; the middle
  ear is linear and the figure is meaningless.
- **Geometry recorded.** Which of the codes above applies, written down at the time of
  measurement. SASA will not assume it.

---

## 7. Provenance of every coefficient

All model coefficients come from a single file in the public ARL release:

```
AHAAH v2.1 public release
  AHAAH_ver_2_1/AHAAH_MIL-STD-1474E_defaultHPD/man.coe
  md5 8f21f4316def7dbcb1bd5f4c9ef5fed0
```

A byte-identical copy is checked in at `tests/data/ahaah/man.coe` and its md5 is asserted
by the test suite, so the values the code was written against cannot drift.

`man.coe` is **positional**: each line carries the value first and the parameter name only
as part of a trailing comment, so the original program identifies parameters by line index.
Rather than parse it at run time — where a shifted line would silently change the answer —
the values are embedded in `ahaah.py` as named constants with units and source, so that
they appear in the diff and can be reviewed. **Every one of them has been checked, value by
value, against the released file.**

Where a line carries several numbers, the file's own comment for `CochlearGainFactor`
("0.0724 .025 0.15 … (0.15 for WKBOrig) (0.025 for WKBTaper)") establishes that the first
number is active and the rest are documented alternates. That convention is followed
throughout, and the alternates are retained as `*_ALT` constants. It is also, note, an
inference: the active model *is* WKBTaper, for which the comment names 0.025, so
first-value-active and select-by-type disagree by a factor of 8.4 in the final figure.

Coefficient groups embedded: integration (`Alphar`…`Betai`); outer ear (`Rdf`, `Ldf`, `L1`,
`L2`, `S1`, `S2`); middle-ear cavity (`Lh`, `Cb`, `Rh`, `Cm`); eardrum (`Lds`, `Rds`,
`Cds`, `Ldm`, `Cdc`, `Rdc`); ossicles (`Nt`, `Rmi`, `Cmi`, `Li`, `Ris`, `Cis`, `Ls`, `Lv`,
`Lo`, `Ral`, `Rc`, `Ro`, `Cal`, `Crw`, `Astapes`); annular-ligament nonlinearity (`Lgap`,
`Eo`, `Eb`, `Ramp` — named in the file and defined nowhere in the release); cochlea (`So`,
`Mo`, `Rvo`, `Bo`, `Ao`, `Fo`, `Delo`, `D1`–`D5`, `Dc`, `Nw`, `Ca`, `Bcoef`, `XbApex`,
`XbmFrom`, `XbmTo`, `XbmNo`, `Sweight*`, `DamageThreshold`, `CochlearGainFactor`,
`WkbScalaBMwidthDecay`); MEM reflex (`MemDelay`, `MemTimeConst`, `MemMagK`, `MemMagR`,
`AdaptFactor`); and the build switches. The hearing-protector elements (`Ccush`…`Cmat`) are
deliberately not used — this build models the unprotected ear, `Earplug = 0`,
`HeadPhone = 0`.

Other released data used:

- `Dat/F11_12D.DAT` — Shaw's tabulated free-field to ear-canal-entrance transfer by
  azimuth, used for the outer ear (0° and 90° rows).
- `Dat/FFEDM90.DAT` — Mehrgardt & Mellert free-field-to-eardrum magnitude, used as a
  validation curve only, never as a filter. Checked in at `tests/data/ahaah/FFEDM90.DAT`.
- `160F.AHA`, `160F.HAZ` — the reference case and its band table. Checked in at
  `tests/data/ahaah/`.

The whole model is a **CGS acoustic-impedance analogue**: pressure as voltage in
dyne/cm², volume velocity as current in cm³/s, inertance in g/cm⁴, compliance in cm⁵/dyne,
resistance in dyne·s/cm⁵ — `man.coe`'s own unit comments. Input pressure in Pascals is
converted once, at a single point in the code, by ×10.

---

## 8. What would change this verdict

In rough order of value:

1. **The MEM element mapping** — which circuit elements receive `MemMagK = 12, 1, 6` and
   `MemMagR = 12, 1, 12`. This is what the failing warned/unwarned ratio points at.
2. **The annular-ligament functional form** and the meaning of `Lgap`, `Eo`, `Eb`, `Ramp`.
   Price & Kalb (1991) *JASA* **90**, 219–227; Price (1974) *JASA* **56**, 195–197; and the
   Appendix A that `Weapon_noise_AHAAH.pdf` refers to but does not contain.
3. **The WKB cochlear equations** and the meaning of the taper term. Kalb & Price (1987,
   2002); Price & Kalb (1998, 2000).
4. **Which `CochlearGainFactor` is active** — worth ×8.4 in the answer on its own.
5. **More reference cases with published ARU.** One case cannot validate a model with this
   many free choices even when it passes. The Albuquerque waveform set (Johnson 1994, 1998;
   Patterson et al. 1997) is the obvious source.

Until then the status stays `not_validated` and SASA prints no ARU. That is the correct
outcome, not a gap to be papered over.

---

## 9. Related files

| File | Contents |
|---|---|
| `ahaah.py` | The implementation, its declared assumptions, and the CLI |
| `tests/test_ahaah.py` | 54 tests; the validation gate is `test_VALIDATION_against_ARL_160F_reference` and is **expected to fail** |
| `docs/AHAAH-SPEC.md` | The full specification, the honesty map (§10), the validation protocol (§11) and what is missing (§12) |
| `tests/data/ahaah/` | The checked-in oracle: `man.coe`, `160F.AHA`, `160F.HAZ`, `FFEDM90.DAT` |
| `metrics.py` | The A-weighted energy metrics — the other MIL-STD-1474E method, computed exactly |

The validation gate is marked `@pytest.mark.validation`. A build that needs a green board
can deselect it with `-m "not validation"`. **Do not loosen its tolerances.** Its failure is
what keeps `compute_ahaah()` silent, and a silent AHAAH is this build working as designed.
