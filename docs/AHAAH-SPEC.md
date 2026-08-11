# AHAAH Implementation Specification

**Auditory Hazard Assessment Algorithm for Humans (AHAAH), US Army Research Laboratory**
Specification for a Python re-implementation (`ahaah.py`) inside SASA.

Status: **specification only. No validated implementation exists.**
Author: derived from the public ARL v2.1 release (MIL-STD-1474E default-HPD build) and the
ARL/MIL-STD documentation set. Date of analysis: 2026-08-11.

---

## 0. The governing rule for this document

A number that looks authoritative and is wrong is worse than no number.

This document distinguishes, for every part of the algorithm, between what the sources
**state**, what is **derivable** from them, and what is an **inference**. Section 10 (the
honesty map) is the most important section in this document. Section 11 defines the pass
criteria that must be met before any SASA output may be labelled "AHAAH ARU".

If those criteria are not met, SASA must not present an ARU number as an AHAAH result. It may
present it as an unvalidated internal estimate, clearly marked, or not at all. Saying
"not validated" is a success. Shipping a plausible wrong number is a failure.

---

## 1. Scientific standing (must be reflected in code comments and user-facing output)

MIL-STD-1474E (15 Apr 2015) approves **two** metrics for impulse noise: the ARU metric
computed with AHAAH, and an energy metric (L\_IAeq100ms). The standard itself, in Table B-II,
records that:

> "The American Institute of Biological Sciences' review generally supported the AHAAH model
> although recommended that several critical assumptions embedded in the model — especially
> the influence of the middle ear reflex under various conditions — need further research
> while a measure based on Leq8 can be used as an interim metric."

AHAAH is not undisputed and NATO has not adopted it. A 2003 NATO study reported unsatisfactory
results for several exposure conditions; 2010 AIBS and 2012 NIOSH reviews raised concerns;
2016–2017 work found the acoustic (middle-ear-muscle) reflex is not pervasive enough to be
assumed, including a live-fire study in which early middle-ear-muscle contraction was absent
in 18 of 19 subjects firing M4 rifles.

**The warned/unwarned switch is the crux.** For the single reference case in this release it is
the difference between 391.0 ARU and 2237 ARU — a factor of **5.72**. The implementation must
**always compute and report both**, and the **headline figure must default to UNWARNED**
(the conservative case, no reflex assumed).

---

## 2. Source inventory

All paths are under the unpacked public ARL release. Base:

```
<SCRATCH>/ahaah/v21/AHAAH_ver_2_1/AHAAH_MIL-STD-1474E_defaultHPD/
```

| File | md5 | Role |
|---|---|---|
| `man.coe` | `8f21f4316def7dbcb1bd5f4c9ef5fed0` | **All model coefficients.** 99 lines, positional. |
| `EarModel.JPG` | `72ffd24e006cee9572817ad2610bebf7` | Circuit topology (3072×2304 ARL slide). |
| `160F.AHA` | `9d83dc49a091dce86f1bd8dadd6df176` | Reference waveform + reference results in header. |
| `160F.HAZ` | `ebd8f068f3e76e01e6b3084b633d0bdd` | Reference 23-band hazard table. |
| `MAN.FIG` | — | Index of the validation transfer-function datasets. |
| `Dat/*.DAT` | — | Published comparison curves (see §4.6). |
| `AHAAH_MIL-STD-1474E_Default_HPD.exe` | — | Windows binary (Delphi). Inspected via `strings` only. |
| `HPD Atten.txt`, `Dat/Hearing Protectors/` | — | Hearing-protector module data. Out of scope for v1. |

Documentation set:

| Document | Content that matters | Equations? |
|---|---|---|
| `AHAAH_Functional_Description.pdf` (3 pp) | Prose description of the whole chain. Explicitly "non-mathematical". | none |
| `AHAAH_validation.pdf` (2 pp) | Validation history (Albuquerque studies). | none |
| `CalculationProdedure_HearingProt.pdf` (4 pp) | **The 95th-percentile +10 dB statement.** | none |
| `Weapon_noise_AHAAH.pdf` (34 pp) | Validation cases, many warned/unwarned ARU pairs. | none |
| `ahaah-MIL-STD-1474E-Final-15Apr2015.pdf` (123 pp) | Annex A: `ARU = sum(D²)`, CTS formula, max-of-23 rule, 95th percentile. | two (see §8, §9) |

**Extraction note:** all five PDFs extracted cleanly with `pypdf`; no PDF resisted extraction.
**However**: `Weapon_noise_AHAAH.pdf` refers on p.11 to "Appendix A" for the discussion of the
middle-ear non-linearity — *that appendix is not present in the PDF* (the document ends at the
reference list on p.34). **There is no equation for the annular-ligament non-linearity, the
middle-ear-muscle frequency shaping, or the WKB cochlear solution anywhere in the supplied
material.** This is the single largest gap and is the reason §10 exists.

---

## 3. Facts verified by direct computation

These were computed from the release files during preparation of this spec and are the
foundation of everything below. Re-verify them in the test suite.

| Check | Result |
|---|---|
| `F(x) = Fo·exp(−x/Dc)`, Fo = 20000 Hz, Dc = 0.8 cm, against all 23 rows of `160F.HAZ` | **exact match to 2 dp for all 23 bands** (x = 0.425 → 11.76 kHz; x = 3.175 → 0.38 kHz) |
| Band spacing | Δx = 0.125 cm uniform → frequency ratio 1.1691 = **0.2254 octave (4.436 bands/octave)**. The docs say "about 1/3 octave"; it is not. Use the formula, not the description. |
| Max of the 23 `160F.HAZ` values | 390.99399 AHU at band 9 (x = 1.425 cm, 3.37 kHz) → header reports **391.0 A.R.U. (warned)**. **TOTAL ARU = MAX over the 23 bands, confirmed.** |
| Sum of the 23 values | 4031.24 — does **not** match anything in the header. The sum rule is wrong. |
| `N = 500 / ARU` | 500/390.99 = 1.279 → header "1.3"; 500/2237 = 0.2235 → header "0.2". **Confirmed, reported to 1 dp.** |
| `CTS = 26.6·ln(ARU) − 140.1` | at 500 ARU → 25.21 dB, matching ARL's "500 ARUs ≈ 25 dB". At 391.0 → 18.67 dB. At 2237 → 65.06 dB. |
| Unwarned/warned ratio for 160F | 2237/391.0 = **5.72** |
| `160F.AHA` payload | 2048 samples, 125000 Hz, calc code 1, values in **Pascals**, 5 significant figures, range −15020 … +44340 Pa |
| Peak from samples | 44340 Pa at sample 628 = 5.024 ms → **186.915 dB** re 20 µPa. Header says **186.936 dB "at 5.0 milliseconds"**. Δ = 0.021 dB, unexplained (see §11.1). Parabolic 3-point interpolation gives 187.042 dB — also not a match. |
| A-weighted energy | IEC 61672 A-weighting (bilinear-transformed analogue prototype), ∫p\_A²dt = 174946.7 Pa²·s over the full record. Header value 422.641 J/m² implies **ρc = 413.9 rayl**. With ρc = 415 → 421.56 (−0.26 %). |
| Leq (unweighted, 13.55 ms from 2.50 ms) | computed 165.825 dB, header 165.848 dB (Δ 0.023 dB) |
| LAeq (same window) | computed 165.088 dB, header 165.123 dB (Δ 0.035 dB) |
| LAeq8hr | computed 101.815 dB, header 101.846 dB (Δ 0.031 dB) |
| `160F.HAZ` provenance | its maximum equals the **warned** figure, so the `.HAZ` table is the warned run. There is no unwarned band table in the release. |

Derived circuit sanity checks (all consistent with anatomy, supporting the netlist reading in §5):

| Quantity | Value | Interpretation |
|---|---|---|
| `Rdf/(2π·Ldf)` | 802 Hz | head-diffraction corner (c/2πa for a 8.75 cm head ≈ 626 Hz) |
| `1/(2π√(Lh·Cb))` | 596 Hz | middle-ear cavity / mastoid Helmholtz resonance |
| `Cm · ρc²` | 0.49 cm³ | tympanic cavity volume |
| `Cb · ρc²` | 7.14 cm³ | mastoid air-cell volume |
| `Astapes²/Cal` | 9.0×10⁵ dyne/cm | annular ligament mechanical stiffness → 20 µm displacement needs ≈8.6×10⁴ dyne/cm² at the stapes ≈ 147 dB at the drum after the 1:20 transformer. Matches the documented "non-linear above 130–140 dB". |
| `L1·S1` | 0.975 cm³ | ear canal volume (textbook ≈1.0 cm³) |
| `√(So/Mo)/2π` | 66.1 kHz | **does not equal Fo = 20 kHz** (factor 3.30). See §10.5. |
| `Rvo/√(So·Mo)` | 0.0379 | vs `Delo` = 0.03 (fwhm/f₀). Close but not equal. |
| Exponential-length self-consistency | `D1 = 1/(2/Dc − 1/D2)` → 1/(2.5−1) = 0.6667 vs stated 0.666 ✓; `D2 = D3 = D4/2 = 1` ✓; `D5 = D4 = 2` ✓; `1/Dc = ½(1/D1 + 1/D2) = 1.25` ✓ | the cochlear exponents are mutually consistent |

---

## 4. Input handling

### 4.1 `.AHA` file format

Tab-separated three-column ASCII header (`PARAMETER`, `VALUE`, `COMMENT`), a variable number of
header rows, then one sample per line in `%.4E` format, in **Pascals**. Header fields observed:

```
Title for waveform plot
Sampling rate                                     125000
Microphone relative to ear                        1        <- calc code
Number of Samples                                 2048
Number of exposures with no protector, warned     1.3      (comment: "391.0 A.R.U.")
Number of exposures with no protector, unwarned   0.2      (comment: "2237 A.R.U.")
Peak Pressure Level                               186.936  dB at 5.0 milliseconds
LAeq and Leq                                      165.123  dB ... Leq = 165.848 dB
LAeq8hr                                           101.846  dB
Na                                                0.020
A-weighted energy                                 422.641  joules/m^2
```

Parse defensively: header row count and label wording differ between the five `.AHA` files in the
release. Locate the numeric payload by pattern, not by line count.

### 4.2 Calculation ("microphone relative to ear") codes

Verbatim from the `160F.AHA` header comment:

> `Calc Code: 1: FFHN 5:FFHG -> 2: Ear-canal entrance  4: FFMN, 6:FFMG -> 3: Eardrum where FF=Free-field, H=human, M=manikin, N=Normal, G=Grazing`

| Code | Meaning | Entry point into the model |
|---|---|---|
| 1 | Free-field, human, normal incidence | converted to ear-canal-entrance pressure |
| 5 | Free-field, human, grazing incidence | converted to ear-canal-entrance pressure |
| 4 | Free-field, manikin, normal incidence | converted to eardrum pressure |
| 6 | Free-field, manikin, grazing incidence | converted to eardrum pressure |
| 2 | Ear-canal entrance pressure | direct |
| 3 | Eardrum pressure | direct |

"Normal incidence" means normal to the ear-canal axis, i.e. **90° azimuth, sound travelling
toward the side of the head**. MIL-STD-1474E B-A.2.1 confirms this is the default worst case:
"AHAAH evaluates auditory hazard for the 95th percentile (most susceptible) human ear for sounds
travelling toward the side of the head (a worst-case condition). Additionally, AHAAH can evaluate
auditory hazard for grazing incidence (such as experienced by a human firing a rifle)."
`160F.AHA` uses code 1. `Dat/FFEDM90.DAT` is the 90°-azimuth validation curve, consistent.

### 4.3 The free-field-to-ear transfer

The binary references `\Dat\F11_12d.dat` immediately adjacent to the string
`NorCE:10 NorED:11 GrED:21 ShED31 NoED:00` and the string `, incident angle = `.

`Dat/F11_12D.DAT` is a 22-line tab-separated matrix:
- line 1: 77 frequencies in kHz (0.00 placeholder, then 0.20 … 15.00, 1/12-octave-ish);
- lines 2–21: azimuth 0°, 18°, 36° … 342° (20 rows), each a gain in dB at those frequencies;
- line 22: a further gain row (0.00 … 13.48 peak near 3 kHz), consistent with an
  ear-canal-entrance-to-eardrum transfer.

This is Shaw's data (figures 11 and 12), i.e. **free-field→ear-canal-entrance vs azimuth** and
**ear-canal-entrance→eardrum**. The two-digit codes decode plausibly as
`<incidence><destination>`: destination 0 = canal entrance, 1 = eardrum; incidence 1 = normal,
2 = grazing, 3 = "Sh", 0 = none.

**Ambiguity (see §10.1):** it is not established whether, for calc code 1, AHAAH applies this
tabulated transfer and then starts the circuit at the canal-entrance node, or drives the circuit
from the `2P`/`P` diffraction sources and computes the outer ear itself. Both mechanisms are
present in the release and they are alternatives, not a chain — applying both would double-count
the outer ear.

### 4.4 Pre-processing performed by the AHAAH GUI (MIL-STD B.5.2.2.1.1)

Before analysis, a raw recording is: baseline-corrected ("Set Baseline"), trimmed
("Select Segment"), calibrated to Pascals ("Calibrate"), given a start point
("Establish Start"), and **tapered at both ends** ("Taper Ends") "to minimize artifactual
effects". Sampling rate is entered by hand. The taper window shape and length are **not
documented**. `160F.AHA` is already pre-processed, so v1 of `ahaah.py` should consume `.AHA`
files as-is and must not re-taper them. For SASA's own recordings the taper must be specified
explicitly and reported as an implementation choice.

### 4.5 The 95th-percentile question — RESOLVED

This materially changes the answer, so the evidence is quoted in full.

`CalculationProdedure_HearingProt.pdf`, p.3:

> "To arrive at 95 percentile susceptibility, AHAAH assumes that susceptibility is normally
> distributed, with a standard deviation of 6 dB. It argues further, that a susceptible ear is
> like a normal ear except that it is effectively being driven harder. By coupling these two
> concepts, **the model achieves the prediction for the 95 percentile ear by artificially raising
> the SPL on the test impulse by 10 dB (1.64 standard deviations) and doing the hazard
> calculation.**"

`Weapon_noise_AHAAH.pdf`, footnote 1, p.32:

> "The model adjusts for susceptibility by asserting that the susceptible ear behaves like a
> median ear that is being driven by a more intense waveform. This means that **if the peak
> pressure of the waveform is reduced by 10 dB (1.64 SDs) and run through the model, loss in the
> 50 percentile ear can be predicted** and so forth for other quartiles."

MIL-STD-1474E, B-A.1.4 and B-A.2.1:

> "AHAAH provides an analysis representative of the 95th percentile ear (most susceptible)."
> "AHAAH evaluates auditory hazard for the 95th percentile (most susceptible) human ear…"

The v2.1 binary's Analyze menu offers exactly four hazard actions —
`ARU(W) Auditory Risk Units, Warned` and `ARU(U) Auditory Risk Units, Unwarned`, plus protected
variants. **There is no percentile selector anywhere in the UI strings.**

**Conclusion (category (b), unambiguously derivable):** the +10 dB susceptibility adjustment is
applied *internally and unconditionally*. The reference values 391.0 / 2237 ARU are
**95th-percentile figures for an input waveform whose stated peak is 186.936 dB**. An
implementation that omits the +10 dB will under-report. The magnitude of the effect is large and
non-linear: `Weapon_noise_AHAAH.pdf` gives paired 95th/median figures for the spark-gap impulses
— 6-gap: 176.1 vs 56 AHU warned (×3.14) and 995.4 vs 342 unwarned (×2.91); 3-gap: 31.6 vs 6.5
warned (×4.86). So the +10 dB is worth roughly ×3–5 in ARU, i.e. it cannot be applied as a fixed
post-hoc factor. It must be applied to the input waveform.

**Implementation requirement:** apply a gain of 10^(10/20) = 3.16228 to the pressure waveform
before it enters the model, and report the multiplier in provenance. Provide a
`percentile="median"` option that omits it, clearly labelled as *not* the MIL-STD figure.

### 4.6 The `Dat/*.DAT` reference curves are NOT filters

`MAN.FIG` heads the list with "5 Number of transfer function picks to follow", naming the five
transfer functions AHAAH can plot, then "10 Number of transfer function types and file names to
follow". These files are **published measurement data plotted against the model's own computed
transfer functions** — they are validation targets, not signal-path filters. Do not convolve
with them.

Common format: 3 title lines; a `±1` axis-direction flag; a frequency-range/tick line; a
value-range/tick line; then rows of `value  frequency_in_kHz`.

| File | Content | Source |
|---|---|---|
| `FFEDM90.DAT` | Free-field→eardrum pressure ratio, magnitude dB, 90° az. 38 points, 0.21–15.00 kHz. Peak **+18.94 dB at 2.66 kHz**; −5.97 dB at 15 kHz. | Mehrgardt & Mellert |
| `FFEDP90.DAT` | Same, **phase in periods** (not radians, not degrees). 38 points. +0.10 at 0.42 kHz falling to −1.11 at 15 kHz. | Mehrgardt & Mellert |
| `RBM.DAT` / `RBP.DAT` | Eardrum acoustic input impedance, dB re 1 dyne·s/cm⁵ and phase in periods. 33 points, 0.099–3.94 kHz. | Rabinowitz |
| `FIG5CMH/CPH.DAT` | External-ear radiation impedance at the eardrum. | Rosowski |
| `FIG5EMH/EPH.DAT` | Eardrum pressure → stapes volume velocity (transfer admittance, dB re 1 cm⁵/dyne·s). Peak −87.4 dB at 0.79–1.0 kHz. | Kringlebotn & Gundersen |
| `FIG5DMH/DPH.DAT` | Alternative eardrum impedance. | Zwislocki |
| `AWT.DAT` | **The A-weighting curve**, 61 points at 1/6-octave spacing, 0.02–19.95 kHz. Verified standard: −50.45 dB @ 20 Hz, 0.00 @ 1 kHz, −2.49 @ 10 kHz, −9.32 @ 20 kHz. (Its first three header lines are a stale copy of the eardrum-to-stapes template — ignore them.) | IEC A-weighting |
| `EARM.DAT` | E-A-R foam plug insertion loss, 8 octave points. | — |
| `BK*.DAT` | Békésy's cochlear envelope data at 25–1600 Hz, for plotting the cochlear model. | Békésy |

---

## 5. The middle-ear network — netlist

### 5.1 Analogue and units

The model is a **CGS acoustic-impedance analogue**. This is confirmed by `man.coe`'s own unit
comments (`cm^5/dyne`, `dyne-sec/cm^5`, `g/cm^4`, `dyne/cm^3`, `g/cm^3`, `cm^2`).

| Circuit quantity | Physical quantity | Unit |
|---|---|---|
| voltage | pressure `p` | dyne/cm² (= 0.1 Pa) |
| current | volume velocity `U` | cm³/s |
| inductance `L` | acoustic inertance | g/cm⁴ |
| capacitance `C` | acoustic compliance | cm⁵/dyne |
| resistance `R` | acoustic resistance | dyne·s/cm⁵ |

Convert the input waveform from Pa to dyne/cm² by ×10.

### 5.2 Nodes

```
GND    reference (free field / atmosphere)
Nff    diffraction node (output of the source network)
Npl    after the air plug
Ned    eardrum, lateral (canal) surface        -- pressure Pe
Nmec   middle-ear cavity                        -- the "bulla" node, return rail for the
                                                   eardrum and ossicular shunt branches
Nmal   transformer secondary / malleus-incus
Ninc   after Li
Nsta   after Ls (stapes)
Nves   after Cal+Ral (vestibule)
Ncoc   cochlear input                           -- pressure Pc, volume velocity Uc
Nst    scala tympani return rail
```

### 5.3 Netlist (topology read from `EarModel.JPG`, values from `man.coe`)

| # | Element(s) | Value | Unit | Topology | Anatomy |
|---|---|---|---|---|---|
| 1 | source `2P` | 2×p(t) | dyne/cm² | GND → (series `Rdf`) → `Nff` | pressure doubling at a rigid head surface, high-frequency limit |
| 2 | `Rdf` | 1.29e-1 | dyne·s/cm⁵ | series, `2P` → `Nff` | diffraction loss |
| 3 | source `P` | p(t) | dyne/cm² | GND → (series `Ldf`) → `Nff` | free-field pressure, low-frequency limit |
| 4 | `Ldf` | 2.56e-5 | g/cm⁴ | series, `P` → `Nff` | head diffraction inertance; corner `Rdf/2πLdf` = 802 Hz |
| 5 | `Rpl ∥ Lpl` | **absent from man.coe** | — | series `Nff` → `Npl` | "air plug" — the residual air path when an earplug is fitted. With `Earplug 0` this branch is a short. |
| 6 | `L1/A1`, `L2/A2`, `L3/A3` | see below | — | series `Npl` → `Ned` | concha + ear canal, drawn as three cylinders |
| 7 | `Lds` + `Cds` + `Rds` | 4.6e-3 / 0.395e-6 / 1.147e3 | g/cm⁴, cm⁵/dyne, dyne·s/cm⁵ | **series triple, shunt** `Ned` → `Nmec` | eardrum **independent part** (drum motion that does not drive the ossicles) |
| 8 | `Ldm` + `Cdc` + `Rdc` | 2.2e-2 / 2.31e-6 / 60.6 | " | **series triple, in the main line** `Ned` → transformer primary | eardrum **conductive part** |
| 9 | transformer `1:Nt` | Nt = 20 | — | primary between (8) and `Nmec`; secondary `Nmal`→`Nmec` | lever ratio × effective-area ratio. `p_out = Nt·p_in`, `U_out = U_in/Nt` |
| 10 | `Cmi` + `Rmi` | 4.33e-10 / 8.391e5 | cm⁵/dyne, dyne·s/cm⁵ | series pair, **shunt** `Nmal` → `Nmec` | malleo-incudal joint |
| 11 | `Li` | 2.44 | g/cm⁴ | series `Nmal` → `Ninc` | incus mass |
| 12 | `Cis` + `Ris` | 4.57e-10 / 0.475e5 | cm⁵/dyne, dyne·s/cm⁵ | series pair, **shunt** `Ninc` → `Nmec` | incudo-stapedial joint |
| 13 | `Ls` | 2.44 | g/cm⁴ | series `Ninc` → `Nsta` | stapes mass |
| 14 | `Cal` + `Ral` | 4.9e-10 / 2.06e5 | cm⁵/dyne, dyne·s/cm⁵ | **series pair, in the main line** `Nsta` → `Nves` | **annular ligament — the site of nonlinearity 1** |
| 15 | `Lv` | 6.25 | g/cm⁴ | series `Nves` → `Ncoc` | vestibular volume |
| 16 | `Rc` | 2.64e6 | dyne·s/cm⁵ | shunt `Ncoc` → `Nst` | cochlear input resistance |
| 17 | `Lo` + `Ro` | 52.4 / 1.8e5 | g/cm⁴, dyne·s/cm⁵ | series pair, shunt `Ncoc` → `Nst` | **helicotrema** (low-frequency shunt) |
| 18 | `Crw` | 1.38e-8 | cm⁵/dyne | **series in the return rail**, `Nmec` → `Nst` | round window membrane |
| 19 | `(Lh + Cb + Rh) ∥ Cm` | 1.4e-2 / 5.1e-6 / 91.85 ∥ 0.35e-6 | — | **series in the return rail**, GND → `Nmec` | `Cm` = tympanic cavity (0.49 cm³); `Lh+Cb+Rh` = aditus neck + mastoid air cells (7.14 cm³), resonance 596 Hz |
| — | `Astapes` | 2.1e-2 | cm² | — | stapes footplate area; stapes displacement `d(t) = (1/Astapes)·∫Uc dt` |

Notes on the netlist:

- **The bulla and the round window sit in the return rail, not in the forward path.** This is
  physically correct and is the part of the diagram most easily misread. The eardrum branches
  and the ossicular shunts return to `Nmec` (the middle-ear cavity), not to true ground, so the
  drum correctly sees (canal pressure − cavity pressure). `Crw` then separates the cavity from
  the scala tympani.
- **Stapes volume velocity = the series current through `Cal`/`Ral`/`Lv` = `Uc`.**
- **The outer ear:** `man.coe` contains `L1 = 2.215`, `L2 = 6.962e-1`, `S1 = 0.44`, `S2 = 4.3`
  and **no `L3`, no `A1/A2/A3`, no `Rpl`, no `Lpl`**. `CochlearModelType`-adjacent switch
  `HornExtEar 1` selects `1=Horn` over `0=Three-cylinders`. Since the three-cylinder option
  would require an `L3`/`A3` that the file does not contain, **the default configuration is the
  horn, and the diagram depicts the alternative three-cylinder configuration.** Pairing by
  magnitude, `(L1 = 2.215 cm, S1 = 0.44 cm²)` is the ear canal proper (volume 0.975 cm³,
  quarter-wave resonance 3.88 kHz — both textbook) and `(L2 = 0.696 cm, S2 = 4.3 cm²)` is the
  concha (volume 3.0 cm³). **The horn profile between S2 and S1 is not specified** — see §10.1.
- `Tm3Piston 1 = Two piston eardrum` is the default: two drum branches (independent and
  conductive), matching the diagram.
- `TerminateCochlea 0` = finite termination.

### 5.4 `man.coe` parsing

Positional, whitespace-delimited: the value(s) come first, everything after is a comment
including the parameter name. **The names are comments — the program identifies parameters by
line position.** Any re-implementation must therefore either hard-code the values or parse by
line index, and must assert the file's md5 before doing so.

Where a line holds several numbers, the file's own comment for `CochlearGainFactor`
(`0.0724 .025 0.15 CochlearGainFactor (0.15 for WKBOrig) (0.025 for WKBTaper)`) shows the
convention: **the first number is the active value; the remainder are documented alternates.**
This convention is critical and dangerous — see §10.5.

Multi-valued lines: `Betar 2, 1, 0`; `MemDelay 9.0E-3 -50E-3`; `MemMagK 12 1 6`;
`MemMagR 12 1 12`; `Nw 2.0 1.23`; `Ca 0.5 0.316`; `SweightApex 1.0 0.2`;
`WkbScalaBMwidthDecay 0.75 0.9`; `CochlearGainFactor 0.0724 .025 0.15`;
`Lleak 2e-3 2e-3 2e-1`; `Rleak 3 3 7.3e1`.

---

## 6. Network solution

AHAAH integrates the network **in the time domain** — this is forced by the two nonlinearities
(§7), which are state-dependent and cannot be expressed as a fixed transfer function.

`man.coe` lines 1–4 are:

```
1         Alphar
0         Alphai
2         Betar 2, 1, 0
0         Betai
```

These are the only candidates for integration-scheme coefficients (complex α and β, with the
`2, 1, 0` list following the "first is active, rest are alternates" convention). Their meaning is
**not documented** — see §10.2.

**Recommended implementation** (an inference, to be declared as such):

1. Write the network as a state-space system in the state vector
   `x = [U_L (all inertance currents), p_C (all compliance pressures)]`.
2. Build it by modified nodal analysis over the node list in §5.2, treating the transformer as an
   ideal 1:Nt two-port.
3. Integrate with the trapezoidal rule (companion-model / Tustin), which is what `Betar = 2`
   would produce under the mapping `s → (Betar/T)·(1−z⁻¹)/(1+z⁻¹)`; re-form the companion
   conductances each step only for the two nonlinear elements.
4. Time step = the sample interval (8 µs at 125 kHz). Verify by internal oversampling (×4, ×8)
   that the ARU result is converged to <1 %; if it is not, oversample the input by band-limited
   interpolation and report the factor.

---

## 7. The two nonlinearities

### 7.1 Annular-ligament peak limiting

`AHAAH_Functional_Description.pdf` p.2:

> "Because the annular ligament that holds the stapes in position in the oval window has a finite
> width and is very tough, it stops the stapes from displacing more than a few tens of microns.
> Were the middle ear linear, at very high SPLs it would try to displace 1000 microns or more.
> So at high SPLs the annular ligament represents a strong peak-limiting element in the ear."

That is the entire documentation of this element. The candidate parameters are the block of
`man.coe` values that immediately follows `Astapes`:

```
2.1e-2    Astapes
30        Lgap
0.01      Eo
0.20      Eb
6         Ramp
```

`Lgap`, `Eo`, `Eb` and `Ramp` are **named but never defined** in any supplied document, and carry
no units in the file. Plausible readings (all unverified): `Lgap` = ligament gap/width;
`Eo`, `Eb` = elastic moduli or strain thresholds at onset and at the hard bound;
`Ramp` = the exponent of the stiffening transition.

**This is the single most important undocumented item in the model** — it is a hard limiter
sitting directly in the path that determines stapes displacement, hence basilar-membrane
displacement, hence ARU (which is displacement *squared*). See §10.3.

Whatever form is adopted, the following are constrained by the sources and must hold:

- the element is `Cal` (in series with `Ral`) in the main line, so the nonlinearity is a
  **displacement-dependent compliance**, not a clipper on pressure;
- it is essentially linear below ~130–140 dB peak at the ear;
- it limits stapes displacement to "a few tens of microns" where a linear model would give
  ≥1000 µm.

### 7.2 Middle-ear-muscle (MEM) reflex

`AHAAH_Functional_Description.pdf` p.2:

> "When they contract, they attenuate on the order of 20 dB at frequencies below about 1.0 kHz
> and progressively less at higher frequencies. The middle ear muscles can contract reflexively
> in response to intense sounds, which means that the contractions have a latency of about
> 10 msec and a rise in their effect."

MIL-STD-1474E B-A.1.3:

> "Options in AHAAH include accommodating middle ear muscle contractions occurring either before
> the onset of the stimulus being analyzed (a warned exposure) or a muscle contraction that is
> elicited by the impulse noise and includes a latency and a growth to full effect (unwarned
> exposure)."

Parameters:

```
9.0E-3 -50E-3  sec MemDelay for evoked response     <- 9 ms evoked (unwarned); -50 ms (warned)
11.7e-3        sec MemTimeConst
12 1 6         MemMagK
12 1 12        MemMagR
2.5            AdaptFactor
```

**Time course (category (b), derivable):**
- *Unwarned*: contraction begins at `t = t_trigger + 9 ms` and grows toward full effect with time
  constant 11.7 ms. What constitutes `t_trigger` (waveform start, or a level crossing) is not
  stated.
- *Warned*: the contraction began 50 ms **before** t = 0, so at t = 0 it is already at
  `1 − exp(−50/11.7)` = 98.6 % of full effect — effectively fully contracted throughout.

**Frequency shaping (category (c), inferred):** `MemMagK` and `MemMagR` are multipliers on
stiffness (`K = 1/C`) and resistance of ossicular elements. The leading value 12 in both is
strongly consistent with the documented behaviour: multiplying the ossicular compliances'
stiffness by 12 raises the stiffness-controlled (low-frequency) impedance by 20·log₁₀(12) =
**21.6 dB**, with progressively less effect above the resonance where mass dominates — exactly
"about 20 dB below 1 kHz and progressively less at higher frequencies". **Which** elements the
three values apply to (candidates: `Cds`/`Cdc`, `Cmi`, `Cis`, `Cal`) is not stated. `AdaptFactor
2.5` is undocumented; the name suggests decay/adaptation of the contraction over time.

---

## 8. Cochlea

`CochlearModelType 2 = WKBTaper` is the default (`3=WKBNoTaper 1=WKBOrig 0=DifEqn`).

The parameter set is a complete, internally consistent specification of a **1-D
transmission-line (long-wave) cochlea with exponentially graded properties**:

| Parameter | Value | Meaning (file's own comment) |
|---|---|---|
| `So` | 1e9 | dyne/cm³, BM stiffness per unit area at base |
| `Mo` | 5.8e-3 | g/cm³, BM + vertically moving fluid mass per unit area at base |
| `Rvo` | 91.2 | dyne·s/cm³, BM viscous resistance per unit area at base |
| `Bo` | 0.008 | cm, width of BM at base |
| `Ao` | 1.25e-2 | cm², effective scala cross-section `Sv·St/(Sv+St)` |
| `Fo` | 20000 | Hz, max resonant freq at base |
| `Delo` | 0.03 | loss constant at base, fwhm/charfreq |
| `D1` | 0.666 | `S(x) = So·exp(−x/D1)` |
| `D2` | 1 | `M(x) = Mo·exp(+x/D2)` |
| `D3` | 1 | `R(x) = Rvo·exp(+x/D3)` |
| `D4` | 2 | `B(x) = Bo·exp(+x/D4)` |
| `D5` | 2 | `A(x) = Ao·exp(−x/D5)` |
| `Dc` | 0.8 | `F(x) = Fo·exp(−x/Dc)` |
| `XbApex` | 3.5 | cm, physical length base→apex |
| `XbmFrom/To/No` | 0.425 / 3.175 / 23 | the 23 evaluation points |
| `CochlearGainFactor` | **0.0724** (alt 0.025 WKBTaper, 0.15 WKBOrig) | scaling to microns |
| `WkbScalaBMwidthDecay` | 0.75, 0.9 | cm⁻¹, "decay length based on tapering factors" |
| `Nw` | 2.0, 1.23 | undocumented |
| `Ca` | 0.5, 0.316 | "Cochlear Amplifier Gain" |
| `Bcoef` | 2.0 | undocumented |
| `SweightBase/Apex` | 1 / 1.0 (alt 0.2) | "stress weight" at base / apex |
| `DamageThreshold` | 0 | no threshold — every upward peak counts |

The exponents are mutually consistent (§3), which is strong circumstantial evidence that the
intended formulation is the standard one:

- partition impedance per unit area `Z(x,ω) = R(x) + jωM(x) + S(x)/(jω)`
- local resonance `ω₀(x) = √(S(x)/M(x))`, decaying as `exp(−x/Dc)` with `1/Dc = ½(1/D1 + 1/D2)`
- long-wave wavenumber `k²(x,ω) = 2jωρ·B(x) / (A(x)·Z(x,ω))`
- WKB solution `p(x) ∝ [k(x)]^(−1/2)·exp(−j∫₀ˣ k dξ)`, with the "taper" being the
  `A(x)`/`B(x)` grading carried in the amplitude factor.

**But this formulation is nowhere written down in the supplied material.** See §10.5. In
particular `√(So/Mo)/2π = 66.1 kHz ≠ Fo = 20 kHz` (factor 3.30), so either the effective mass is
~10.9× `Mo`, or `Fo` is a labelling convention independent of the mechanics. The 23 band
frequencies in `160F.HAZ` follow `Fo·exp(−x/Dc)` exactly, so **`Fo`/`Dc` is definitely what
labels the bands** — but that does not settle what drives the mechanics.

**Output:** basilar-membrane displacement in **microns** at the 23 locations,
`x = 0.425 … 3.175 cm`, step 0.125 cm.

---

## 9. Hazard accumulation

MIL-STD-1474E Annex A, B-A.1.1 — the authoritative statement:

> "It keeps track of the displacements at 23 locations (roughly one-third octave intervals) and
> derives a dose at each location by **squaring the peak amplitude of each upward displacement of
> the basilar membrane (in microns) and summing them** for the analysis interval. The result (at
> each location) is in auditory risk units (ARU):
>
> **ARU = sum (D²)**
>
> where D is the **upward** basilar membrane displacement (in microns)."

MIL-STD-1474E B-A.2.1 — the reduction rule:

> "The calculation is accomplished at 23 evenly spaced locations along the basilar membrane and
> **the location with the largest value is reported.**"

Algorithm:

```
for each location j in 1..23:
    d_j(t) = BM displacement in microns
    find every local maximum (peak) of d_j with d_j > DamageThreshold (= 0)
    AHU_j = sum over those peaks of (peak value)^2
ARU_total = max_j AHU_j
```

Verified against the reference: `max(160F.HAZ) = 390.99399` at band 9 → header "391.0 A.R.U.".
The sum (4031.24) matches nothing.

**Unresolved details of peak detection** (see §10.6): "each upward displacement" is ambiguous
between (a) every local maximum of `d_j(t)` with `d_j > 0`, (b) every positive-going
zero-crossing-to-zero-crossing excursion, counted once at its maximum, and (c) every local
maximum regardless of sign, taking only the positive ones. These give different totals for a
waveform with ripple. Interpretation (b) is the most defensible reading of "as each peak passes"
and is recommended as the primary, with (a) and (c) as documented alternatives to be tested
against the reference.

Sign convention: "upward" means toward the scala vestibuli. The sign of the model's BM
displacement relative to the input pressure polarity must be pinned down by test, not assumed —
inverting it changes the answer.

### 9.1 Derived quantities

| Quantity | Formula | Verified against |
|---|---|---|
| Compound threshold shift | `CTS_dB = 26.6·ln(ARU) − 140.1` | MIL-STD B-A.1.2; self-consistent at 500 ARU → 25.2 dB |
| Allowed exposures (occasional, ≤1×/week) | `N = 500 / ARU`, reported to 1 dp | 500/390.99 = 1.3 ✓, 500/2237 = 0.2 ✓ |
| Allowed exposures (occupational, ≥2×/week) | `N = 200 / ARU` | MIL-STD B.5.2.1.1.1 |
| Multiple impulses | doses **add**: total = n × ARU\_per\_impulse | MIL-STD B-A.2.2 |
| Predicted PTS | ≈ 0.6 × CTS, when the 500-ARU limit is exceeded | MIL-STD B-A.1.2 |

---

## 10. THE HONESTY MAP

Category **(a)** = explicitly documented in the sources.
Category **(b)** = not stated, but unambiguously derivable from the sources or from the data.
Category **(c)** = an inference. **Every (c) item is a risk to the final number.**

### 10.1 Outer-ear handling per input-location code — **(a) partially, (c) for the mechanism**

- (a) The six calc codes and their destinations are stated verbatim in the `.AHA` header.
- (a) MIL-STD states the default is 90° azimuth (side of head), with grazing incidence available.
- (b) `Dat/F11_12D.DAT` is an azimuth-indexed (20 × 77) Shaw HRTF table plus a canal-entrance→
  eardrum row; the binary references it alongside an `incident angle` prompt.
- **(c) RISK:** it is *not established* whether, for a free-field input, AHAAH (i) applies the
  tabulated Shaw transfer and enters the circuit at the canal-entrance node, or (ii) drives the
  `2P`/`P` diffraction sources and computes the concha and canal from the circuit. Both
  mechanisms exist in the release. Choosing wrongly, or applying both, misstates the eardrum
  pressure by up to ~19 dB at 2.7 kHz (the peak of `FFEDM90.DAT`) — which, squared and
  accumulated, is an order of magnitude in ARU.
- **(c) RISK:** with `HornExtEar 1`, the horn profile joining `S2 = 4.3 cm²` to `S1 = 0.44 cm²`
  over `L1`, `L2` is not specified (exponential? conical? how are the two length/area pairs
  arranged?). `man.coe` lacks `L3`, `A1`, `A2`, `A3`, so the three-cylinder configuration shown
  in `EarModel.JPG` cannot be reconstructed from this coefficient file at all.
- **(c) RISK:** `Rpl` and `Lpl` (the "air plug") have no values in `man.coe`. Assumed to be a
  short circuit when `Earplug 0`. Unverified.

### 10.2 Middle-ear network solution method — **(b) for the topology, (c) for the integrator**

- (a) The chain is a time-domain electroacoustic analogue (all documents).
- (b) The topology in §5.3 is read directly from `EarModel.JPG` and every label on it has a value
  in `man.coe` (except `Rpl`, `Lpl`, `L3`, `A1–A3`). The netlist is high confidence: nine
  independent anatomical sanity checks (§3) come out right.
- (b) CGS acoustic units, confirmed by `man.coe`'s own unit comments.
- **(c) RISK:** `Alphar/Alphai/Betar/Betai` (1, 0, 2, 0) are undocumented. The trapezoidal
  interpretation is a guess. A different integrator (backward Euler, or a different `Betar`)
  changes the damping of the high-Q ossicular resonances and hence the peak displacements. Effect
  size: probably small (a few %) if the time step is short relative to the resonances, but
  unquantified.
- **(c) RISK:** the time step is assumed equal to the sample interval. Not stated.

### 10.3 Annular-ligament nonlinearity — **(c) ENTIRELY. HIGHEST RISK ITEM.**

- (a) Prose only: "it stops the stapes from displacing more than a few tens of microns"; linear
  below 130–140 dB.
- (b) It is the compliance `Cal` (`4.9e-10 cm⁵/dyne`, mechanical stiffness 9.0×10⁵ dyne/cm) in
  the main line; a 20 µm displacement corresponds to ~147 dB at the drum, consistent with the
  stated onset.
- **(c) The functional form is not documented anywhere in the supplied material.** The parameters
  `Lgap 30`, `Eo 0.01`, `Eb 0.20`, `Ramp 6` are named but never defined, and have no units. The
  `Weapon_noise_AHAAH.pdf` text points to "Appendix A" for this discussion and **that appendix is
  not in the PDF.**
- **Consequence:** this element sets the peak stapes displacement for exactly the impulses SASA
  cares about (gunshots, 150–190 dB). ARU is displacement squared and summed. An error of ×1.5 in
  the clipping level is ×2.25 in ARU. **Without this, the model cannot be reproduced.**
- **Resolution options, in order of preference:** (1) obtain Price & Kalb (1991), *J. Acoust.
  Soc. Am.* **90**, 219–227, and Price (1974), *JASA* **56**, 195–197, which are the cited
  primary sources; (2) obtain the missing Appendix A from ARL; (3) obtain the AHAAH source code
  or a decompilation of `AHAAH_MIL-STD-1474E_Default_HPD.exe`; (4) fit the form to the single
  reference case — **which is not validation and must never be presented as such.**

### 10.4 MEM reflex time course and frequency shaping — **(b) for timing, (c) for shaping**

- (a) ~20 dB below 1 kHz, progressively less above; ~10 ms latency and a growth to full effect;
  warned = contraction already in place.
- (b) Latency 9 ms, time constant 11.7 ms, warned pre-activation at −50 ms → 98.6 % contracted at
  t = 0. Read directly from `man.coe`.
- **(c) RISK:** the mapping of `MemMagK 12 1 6` / `MemMagR 12 1 12` onto specific circuit elements
  is inferred. The leading 12 reproduces the documented 21.6 dB low-frequency attenuation, which
  is good corroboration, but which of `Cds`, `Cdc`, `Cmi`, `Cis`, `Cal` receive the 12, the 1 and
  the 6 is unknown, and that determines the *frequency shape* of the attenuation.
- **(c) RISK:** `AdaptFactor 2.5` is entirely undocumented.
- **(c) RISK:** the trigger for the unwarned latency (waveform start? a level crossing? which
  level?) is not stated. For a 16.4 ms record with the peak at 5.0 ms, a 9 ms latency measured
  from t = 0 versus from the peak moves the onset from 9.0 ms to 14.0 ms and changes how much of
  the impulse is attenuated.
- **Mitigating factor:** this item only affects the *warned* number. The unwarned number — which
  is SASA's headline — depends on the MEM only through whatever contraction develops after 9 ms,
  which for a rifle impulse is largely after the damage is done. That is precisely ARL's own
  argument, and it means **the unwarned figure is the more robust of the two**, an additional
  reason to lead with it.

### 10.5 WKB cochlear formulation and taper — **(b) for the parameters, (c) for the equations**

- (a) 23 locations, roughly one-third octave (actually 0.2254 octave), BM displacement in microns.
- (b) The graded-parameter set is complete and internally consistent: `D1 = 1/(2/Dc − 1/D2)`,
  `D2 = D3 = D4/2`, `D5 = D4` all check out exactly, and `F(x)` reproduces all 23 reference band
  frequencies exactly. This is strong evidence for the standard 1-D long-wave formulation.
- **(c) RISK:** the actual WKB expression, the taper term, and the boundary/termination handling
  are not written down anywhere in the supplied material.
- **(c) RISK — the largest single numerical unknown:** `CochlearGainFactor` is listed as
  `0.0724 .025 0.15` with the comment `(0.15 for WKBOrig) (0.025 for WKBTaper)`. The active model
  is `WKBTaper`. Under the "first value is active" convention the program uses **0.0724**; under a
  "select by model type" convention it uses **0.025**. **The ratio is 2.90 in displacement and
  8.39 in ARU.** This one ambiguity spans an order of magnitude in the final answer and must be
  resolved by the reference case, not by preference.
- **(c) RISK:** `WkbScalaBMwidthDecay 0.75 0.9`, `Nw 2.0 1.23`, `Ca 0.5 0.316` ("Cochlear
  Amplifier Gain") and `Bcoef 2.0` are undocumented. `Ca` implies an **active** (cochlear
  amplifier) term in the model that is not mentioned in any of the prose descriptions.
- **(c) RISK:** `√(So/Mo)/2π = 66.1 kHz` versus `Fo = 20 kHz`, a factor of 3.30 — unexplained.
- (b) `SweightBase 1` / `SweightApex 1.0` (first values) means the "stress weight" is uniform, so
  no positional weighting is applied before squaring. The alternate `0.2` would taper it.
  `DamageThreshold 0` means no displacement threshold.

### 10.6 Hazard accumulation and peak detection — **(a) for the rule, (c) for the details**

- (a) `ARU = sum(D²)` over **upward** peaks, in microns, per location; total = **max** over the 23
  locations. Both quoted verbatim from MIL-STD Annex A and both verified numerically against
  `160F.HAZ`/`160F.AHA`.
- **(c) RISK:** what counts as "a peak" (§9) — every local maximum, or one per positive
  excursion. For a ringing BM response the difference can be tens of percent.
- **(c) RISK:** the sign convention for "upward" relative to input pressure polarity.
- **(c) RISK:** the analysis interval. The header reports the A-weighted window as
  "13.55 msec starting at 2.50 msec" out of a 16.384 ms record; whether the ARU accumulation uses
  the same window or the whole record is not stated.

### 10.7 The 95th-percentile +10 dB — **(b), RESOLVED**

See §4.5. The +10 dB is applied internally and unconditionally; the reference values are
95th-percentile figures. Not a guess: three independent documents state it and the binary has no
percentile control. Residual risk is only *where* in the chain the gain is applied — before the
outer ear (stated: "raising the SPL on the test impulse") versus later. Because the middle ear is
nonlinear, the placement matters; "on the test impulse" is explicit enough to place it at the
input.

### 10.8 Summary of risk

| Item | Category | Effect on ARU if wrong |
|---|---|---|
| Annular-ligament nonlinearity form | (c) | **unbounded — blocks reproduction entirely** |
| `CochlearGainFactor` 0.0724 vs 0.025 | (c) | **×8.4** |
| WKB formulation details | (c) | large, unquantified |
| Free-field→ear path (which mechanism) | (c) | up to ×10 |
| Peak-detection rule | (c) | tens of percent |
| MEM element mapping | (c) | warned figure only |
| Integrator scheme | (c) | few percent (est.) |
| +10 dB susceptibility | (b) | ×3–5 — **resolved, must be applied** |
| max-not-sum, `sum(D²)`, `F(x)`, `N = 500/ARU`, `CTS` | (a)/(b) verified | — |

**Bottom line: four category-(c) items are each individually capable of moving the answer by an
order of magnitude, and one of them (the annular ligament) has no documented form at all. The
supplied public release is not sufficient to reproduce AHAAH by construction.** It is sufficient
to *attempt* a reproduction and to *test* it, which is what §11 is for.

---

## 11. Validation protocol

There is exactly **one** fully-specified reference case in the release: `160F.AHA` with its
header results and `160F.HAZ`. The other four `.AHA` files carry **no** reference results. This
scarcity is itself a finding: a single reference case cannot validate a model with four
order-of-magnitude free choices. The protocol below is therefore built to make over-fitting
visible rather than to hide it.

### 11.0 Anti-overfitting rules (mandatory)

1. **No parameter may be tuned to the 160F result.** All values come from `man.coe` as written.
2. Any choice among alternates (e.g. `CochlearGainFactor` 0.0724 vs 0.025), and any category-(c)
   inference, must be **declared in writing before the run**, in a machine-readable
   `assumptions.json`, with a `category` field of `a`/`b`/`c`.
3. The count of category-(c) choices exercised is recorded in the run provenance and shown in any
   report. A pass achieved after N ≥ 2 category-(c) choices were varied to obtain it is a **fit,
   not a validation**, and must be reported as `status: fitted`, never `status: validated`.
4. Every attempt is logged, including failures. The log is part of the deliverable.

### 11.1 Tier 0 — input pipeline (no ear model required; must pass first)

Run against `160F.AHA`. All of these are computable today and half of them are already confirmed.

| # | Assertion | Tolerance | Current status |
|---|---|---|---|
| T0.1 | Parse: 2048 samples, fs = 125000 Hz, calc code 1 | exact | ✓ verified |
| T0.2 | `max|p|` = 44340 Pa at sample 628 (5.024 ms) | exact | ✓ verified |
| T0.3 | Peak pressure level = 186.936 dB | **≤ 0.05 dB** | ⚠ raw samples give 186.915 (Δ 0.021); parabolic interpolation gives 187.042. **The exact peak-level definition is unresolved.** Passing at 0.05 dB is acceptable; the discrepancy must be recorded, not papered over. |
| T0.4 | Leq = 165.848 dB over 13.55 ms from 2.50 ms | ≤ 0.05 dB | ✓ 165.825 (Δ 0.023) |
| T0.5 | LAeq = 165.123 dB, same window | ≤ 0.05 dB | ✓ 165.088 (Δ 0.035) |
| T0.6 | LAeq8hr = 101.846 dB | ≤ 0.05 dB | ✓ 101.815 (Δ 0.031) |
| T0.7 | A-weighted energy = 422.641 J/m² | ≤ 1 % | ✓ 421.56 with ρc = 415 (−0.26 %); implies ρc = 413.9 |
| T0.8 | `Na` = 0.020 (allowed rounds on A-weighted energy) | exact to 3 dp | derived: 8.7 J/m² ÷ 422.641 = 0.0206 → 0.020 ✓ |
| T0.9 | All 23 band frequencies from `Fo·exp(−x/Dc)` match `160F.HAZ` | exact to 2 dp | ✓ verified |
| T0.10 | A-weighting from `Dat/AWT.DAT` and from IEC 61672 agree | ≤ 0.2 dB, 20 Hz–20 kHz | to do |

Tier 0 failing means the file reader or the metrics are wrong and nothing downstream can be
trusted.

### 11.2 Tier 1 — linear transfer functions (middle ear, MEM off, nonlinearity off)

`MAN.FIG` names exactly the five transfer functions ARL themselves use to validate the model.
Drive the network with a low-amplitude swept sine or impulse (peak ≤ 100 dB SPL, guaranteeing
linearity), and compare:

| # | Model output | Reference | Band | Tolerance |
|---|---|---|---|---|
| T1.1 | free-field → eardrum pressure, magnitude | `FFEDM90.DAT` | 0.21–8 kHz | **± 3 dB**, and the main peak within ± ⅓ octave of 2.66 kHz |
| T1.2 | free-field → eardrum pressure, phase (periods) | `FFEDP90.DAT` | 0.21–8 kHz | ± 0.1 period |
| T1.3 | eardrum acoustic input impedance, magnitude | `RBM.DAT` | 0.1–4 kHz | ± 3 dB |
| T1.4 | eardrum impedance, phase | `RBP.DAT` | 0.1–4 kHz | ± 0.05 period |
| T1.5 | eardrum pressure → stapes volume velocity | `FIG5EMH/EPH.DAT` | 0.2–10 kHz | ± 3 dB / ± 0.1 period |
| T1.6 | external-ear radiation impedance | `FIG5CMH/CPH.DAT` | 0.2–8 kHz | ± 5 dB (this curve is spiky) |
| T1.7 | free-field → stapes volume velocity vs the A-weighting curve | `AWT.DAT` | 0.1–8 kHz | shape agreement ± 5 dB — ARL's own claim: "the transfer function from the free field to the stapes looks very much like the A-weighting curve" |

Tier 1 is where the §10.1 outer-ear ambiguity gets settled empirically: run both candidate
mechanisms and keep whichever matches `FFEDM90.DAT`. Applying both will overshoot by ~19 dB at
2.7 kHz and will be obvious.

Tier 1 passing establishes the netlist and units. Tier 1 failing means §5 is misread.

### 11.3 Tier 2 — the full nonlinear model against the reference case

Input: `160F.AHA`, calc code 1, no hearing protector, +10 dB susceptibility applied.

| # | Assertion | Reference | Pass tolerance |
|---|---|---|---|
| T2.1 | Total ARU, **warned** | 391.0 | **± 5 %** (371.5 – 410.6) |
| T2.2 | Total ARU, **unwarned** | 2237 | **± 5 %** (2125 – 2349) |
| T2.3 | Argmax band (warned) | band 9 (x = 1.425 cm, 3.37 kHz) | exact |
| T2.4 | All 23 warned band values | `160F.HAZ` | **± 10 % each** |
| T2.5 | Rank order of the 23 warned band values | `160F.HAZ` | Spearman ρ ≥ 0.98 |
| T2.6 | Unwarned/warned ratio | 5.72 | ± 10 % |
| T2.7 | Allowed exposures | 1.3 warned / 0.2 unwarned | exact after rounding to 1 dp |
| T2.8 | Peak stapes displacement | — | must lie in 10–60 µm ("a few tens of microns"), and be ≥ 10× below the value a linear model gives |

Rationale for ± 5 %: ARU maps to threshold shift as `26.6·ln(ARU)`, so ± 5 % in ARU is ± 1.3 dB
of predicted CTS — below audiometric test-retest variability. ± 10 % per band is ± 2.5 dB. These
are tight enough to be meaningful and loose enough to tolerate integrator and interpolation
differences from a 1990s Delphi program.

### 11.4 Tier 3 — plausibility cross-checks (weak, but free)

`Weapon_noise_AHAAH.pdf` reports warned/unwarned ARU pairs for waveforms we do **not** have:

| Case | Warned | Unwarned | Ratio |
|---|---|---|---|
| 7.62 mm rifle @ 155 dB | 16.6 | 118 | 7.1 |
| 7.62 mm rifle @ 158 dB | 22.3 | 153 | 6.9 |
| M-72 LAW @ 161 dB | 105 | 516 | 4.9 |
| M-72 LAW @ 179 dB | 921 | 4217 | 4.6 |
| AT-4 @ 189.8 dB | 856 | 5774 | 6.7 |
| Spark gap, 6-gap @ 166 dB | 176.1 | 995.4 | 5.7 |
| Spark gap, 3-gap @ 166 dB | 31.6 | 186 | 5.9 |
| 120 mm mortar, at eardrum under plug | 7.2 | 36.8 | 5.1 |
| **160F (our reference)** | **391.0** | **2237** | **5.72** |

These cannot validate anything (no waveforms), but they bound the expected behaviour: the
unwarned/warned ratio for impulsive events is **4.6 – 7.1**. An implementation producing a ratio
outside ~3–8 is wrong somewhere in the MEM, regardless of whether it hits T2.1/T2.2.

Also usable as regression fixtures (no reference values, stability only):
`0110F.AHA` (172.1 dB peak), `0930F.AHA` (139.2 dB), `0839336.AHA` (194.5 dB, 8192 samples),
`0110F_DEFAULT 03 (MUFF)_90.AHA` (158.5 dB, calc code 3 = eardrum).

### 11.5 Gating rule for SASA output

```
if Tier0 and Tier1 and Tier2 all pass, with the number of category-(c) choices
   varied to achieve the pass == 0:
       label = "AHAAH ARU (MIL-STD-1474E), validated against ARL 160F reference"
elif Tier0 and Tier1 and Tier2 pass but category-(c) choices were varied:
       label = "AHAAH-style ARU, FITTED to the single ARL reference case — not independently
                validated"
       and the fitted parameters must be listed in the output
else:
       DO NOT EMIT AN ARU NUMBER.
       Emit the A-weighted metrics (Tier 0, which are exactly reproducible) and state that
       AHAAH analysis is unavailable in this build.
```

In every case:
- **both** warned and unwarned figures are computed and shown;
- the **unwarned** figure is the headline;
- the scientific-standing note of §1 accompanies the number;
- the provenance record carries: `man.coe` md5, the reference-case pass/fail table, the
  `assumptions.json` with its (a)/(b)/(c) counts, and the +10 dB susceptibility flag.

---

## 12. What is missing and how to get it

| Missing | Why it matters | How to resolve |
|---|---|---|
| Annular-ligament nonlinearity functional form | blocks reproduction | Price & Kalb (1991) *JASA* **90**, 219–227; Price (1974) *JASA* **56**, 195–197; missing Appendix A of `Weapon_noise_AHAAH.pdf`; ARL directly |
| Meaning of `Lgap`, `Eo`, `Eb`, `Ramp` | same | same; or decompilation of the released binary |
| WKB equations and taper term | large scaling uncertainty | Kalb & Price (1987, 2002); Price & Kalb (1998, 2000) |
| Which `CochlearGainFactor` is active | ×8.4 in ARU | settled empirically by Tier 2, or by decompilation |
| `MemMagK`/`MemMagR` element mapping | warned figure shape | ARL; or fit to the "20 dB below 1 kHz" curve |
| `Alphar/Betar` integration scheme | few percent | ARL; or demonstrate insensitivity by oversampling |
| Horn profile for the external ear | up to ×10 via eardrum pressure | settled empirically by Tier 1 against `FFEDM90.DAT` |
| Additional reference cases with published ARU | single-case validation is weak | ARL AHAAH distribution has further sample files; the Albuquerque waveform set (Johnson 1994, 1998; Patterson et al. 1997) |
| Exact peak-pressure-level definition | 0.02 dB, cosmetic | low priority |

---

## 13. Recommended build order

1. `.AHA` reader + A-weighted metrics → **Tier 0**. Cheap, exact, immediately useful to SASA
   even if the ear model never validates.
2. `man.coe` reader keyed to md5, with positional parsing and an explicit alternates policy.
3. Linear network in state-space, MEM off, nonlinearity off → **Tier 1**. Settles the outer-ear
   ambiguity and the units.
4. MEM (warned/unwarned) → re-run Tier 1 with the contraction in place and confirm ~20 dB of
   low-frequency attenuation, "progressively less" above 1 kHz.
5. WKB cochlea → check the BM-displacement/stapes-displacement ratio and the 23 band
   frequencies; compare the envelope against `BK*.DAT` (Békésy) qualitatively.
6. Annular-ligament nonlinearity → last, because it is the least constrained, and because
   everything before it can be validated without it.
7. Hazard accumulation → **Tier 2**.
8. Apply the §11.5 gate. Do not skip it.

Until step 8 passes, `ahaah.py` must expose its output only under a name that cannot be mistaken
for a MIL-STD result, and the SASA UI must not display an ARU figure.
