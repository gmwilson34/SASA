#!/usr/bin/env python3
"""
ahaah.py - Auditory Hazard Assessment Algorithm for Humans (AHAAH)

WHY THIS FILE EXISTS, AND WHY IT REFUSES TO CALL ITS OUTPUT AN "AHAAH ARU"
=========================================================================

MIL-STD-1474E (15 Apr 2015) approves two impulse-noise metrics: the A-weighted
energy method (which SASA computes exactly, in metrics.py) and the Auditory Risk
Unit (ARU) computed with the US Army Research Laboratory's AHAAH model. Customers
ask for ARU. This module is SASA's attempt at it.

It is an ATTEMPT. The public ARL v2.1 release contains the coefficient file
(man.coe), a circuit diagram, one reference waveform and its results, and a set of
non-mathematical prose descriptions. It does NOT contain:

  * the functional form of the annular-ligament nonlinearity (the parameters
    Lgap/Eo/Eb/Ramp are named in man.coe and defined nowhere; the ARL paper that
    would discuss them points to an "Appendix A" that is absent from the PDF);
  * the WKB cochlear equations or the meaning of the "taper" term;
  * which of the three listed CochlearGainFactor values is active (0.0724 vs
    0.025 is a factor of 8.4 in the final ARU);
  * which circuit elements the middle-ear-muscle multipliers apply to;
  * the free-field-to-eardrum mechanism actually used (a tabulated Shaw HRTF and a
    lumped diffraction circuit are both present in the release, and they are
    alternatives -- applying both double-counts ~19 dB at 2.7 kHz).

Every one of those is individually capable of moving the answer by an order of
magnitude, and there is exactly ONE reference case (160F.AHA -> 391.0 ARU warned,
2237 ARU unwarned) to test against. A single case cannot validate a model with
four order-of-magnitude free choices.

THE CALL: NOT VALIDATED. `compute_ahaah()` RETURNS NO NUMBER.
------------------------------------------------------------
Against the ARL 160F reference case, with every choice as declared:

    warned total          387.7  vs  391.0     -0.8 %   (see below -- coincidence)
    unwarned total       1085.7  vs  2237     -51.5 %   FAIL
    unwarned/warned ratio  2.80  vs  5.72     -51.1 %   FAIL (ARL cases span 4.6-7.1)
    warned argmax band        8  vs  9                  FAIL
    per-band within +/-10 %  3 of 23                    FAIL (worst band 3.4x high)
    peak stapes displacement 24.7 um                    plausible (10-60 um)

Two Tier-1 LINEAR checks against ARL's own datasets do pass: free field to eardrum
is within 3.0 dB of Dat/FFEDM90.DAT over 0.2-8 kHz with its resonance at 2663 Hz
(published 2660 Hz), and the middle-ear-muscle contraction attenuates 20 dB below
1 kHz and progressively less above, exactly as ARL describe. Those confirm the
outer/middle-ear LINEAR chain. They do not validate the hazard figure.

The -0.8 % on the warned total carries no evidential weight, and it must not be
quoted as if it did:

  * the band curve it is the maximum of disagrees with 160F.HAZ in 20 of 23 bands,
    by up to a factor of 3.4, and peaks in the wrong band (8, not 9). A maximum
    taken over a wrongly-shaped curve landing near the right value is arithmetic,
    not agreement.
  * sweeping only the switches this module already exposes as category-(c)
    (CochlearGainFactor, cf_alignment, extra_wkb_taper, peak_rule, bm_sign,
    outer_ear) gives 96 configurations whose warned totals span 10.3 to 2617,
    a factor of 255. FOUR of the 96 land within +/-5 % of 391.0. NONE of those
    four is also within +/-5 % of 2237 unwarned. Hitting the warned figure is a
    ~4 % coincidence in this parameter space.

The reference case constrains a RATIO as well as a level, and the ratio is the
part that fails structurally. Across all 96 configurations the unwarned/warned
ratio stays pinned between 2.59 and 3.17 against a reference of 5.72, while the
absolute level moves by 255x. So the level and the reflex are separately wrong:
the modelled contraction is far too weak, and the level error it partly cancels
is elsewhere. Two errors of opposite sign, one of which happens to cancel the
other in the warned case, is the worst state a model can be in -- it is why the
warned agreement is dangerous rather than encouraging.

(An earlier note in this file blamed the annular-ligament nonlinearity for the
ratio. That was measured and is wrong: scaling Lgap over 10-300 moves the total by
33000x and moves the ratio only between 2.78 and 3.16. The ligament sets the LEVEL.
What moves the ratio is the MEM attenuator -- MemMagK/MemMagR near 40 on a single
element reproduces 5.54 -- which points at the ELEMENT MAPPING of man.coe's
three-element triples "12 1 6" / "12 1 12", of which this module applies only the
leading 12, to one element. That is now the first thing to ask ARL about.)

Therefore, per docs/AHAAH-SPEC.md section 11.5 ("else: DO NOT EMIT AN ARU NUMBER")
and section 13 ("ahaah.py must expose its output only under a name that cannot be
mistaken for a MIL-STD result"):

    VALIDATION_STATUS = "not_validated"

    compute_ahaah() and compute_ahaah_both() ALWAYS return valid=False with every
    hazard field NaN and an explanation. They emit no ARU. The input-admissibility
    gates still run first, so a caller with a bad recording is told about the
    recording, not just about the model.

    The model itself is reachable only through run_unvalidated_model(), which must
    be called with acknowledge_not_validated=True and whose result carries no field
    named ARU: the numbers come back as `band_sum_sq_displacement_um2` and
    `max_band_sum_sq_um2`, i.e. the physical quantity actually computed (microns
    squared of summed upward basilar-membrane peaks). It is for development and for
    the validation test. It is not a customer-facing figure and nothing in SASA
    displays it.

Saying "not validated" is the successful outcome here. See docs/AHAAH.md for the
operator-facing version and docs/AHAAH-SPEC.md sections 10-12 for the detail.

SCIENTIFIC STANDING (must accompany any output)
-----------------------------------------------
AHAAH is not undisputed and NATO has not adopted it. A 2003 NATO study reported
unsatisfactory results for several exposure conditions; the 2010 AIBS and 2012
NIOSH reviews raised concerns; 2016-2017 work found the acoustic (middle-ear
muscle) reflex is not pervasive enough to be assumed -- in one live-fire study
early middle-ear-muscle contraction was absent in 18 of 19 subjects firing M4
rifles. The warned/unwarned switch is the crux: for the ARL reference case it is
the difference between 391 and 2237 ARU, a factor of 5.7. This module therefore
ALWAYS computes both and the headline figure defaults to UNWARNED.

THE CHAIN
---------
    calibrated pressure (Pa)
      -> +10 dB susceptibility adjustment (95th-percentile ear; applied to the
         waveform, per ARL: "raising the SPL on the test impulse by 10 dB")
      -> Pa to dyne/cm^2 (x10); the whole model is CGS acoustic-impedance analogue
      -> outer ear: 2P/P head-diffraction sources, concha and ear-canal
         transmission lines
      -> middle ear: electroacoustic network with two nonlinearities
           (1) annular-ligament peak limiting of stapes displacement
           (2) middle-ear-muscle (MEM) reflex attenuator
      -> stapes volume velocity Uc
      -> cochlea: 1-D long-wave WKB transmission line, 23 locations
      -> basilar-membrane displacement, microns
      -> hazard: sum of squared UPWARD peak displacements per location;
         total ARU = MAX over the 23 locations (not the sum -- verified against
         160F.HAZ, whose maximum 390.99 is the header's "391.0 A.R.U.")

Usage:
    from ahaah import compute_ahaah_both, load_aha

    wf = load_aha("160F.AHA")
    unwarned, warned = compute_ahaah_both(wf.pressure_Pa, wf.sample_rate)
    assert not unwarned.valid            # always, in this build
    print(unwarned.status)               # "not_validated"
    print(unwarned.notes[0])             # why there is no number
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample_poly

from calibration import P_REF

# ---------------------------------------------------------------------------
# Provenance of the coefficients
# ---------------------------------------------------------------------------
# EVERY numeric constant in the "man.coe coefficients" block below is transcribed
# verbatim from:
#
#   AHAAH_ver_2_1/AHAAH_MIL-STD-1474E_defaultHPD/man.coe
#   md5 8f21f4316def7dbcb1bd5f4c9ef5fed0
#
# man.coe is POSITIONAL: the value comes first and the parameter name is part of
# the trailing comment, so the program identifies parameters by line index, not by
# name. Rather than parse it at runtime (where a shifted line would silently
# change the answer) the values are embedded here so that they appear in the diff
# and can be reviewed. Where a line carries several numbers, the file's own
# comment for CochlearGainFactor -- "(0.15 for WKBOrig) (0.025 for WKBTaper)"
# against a leading value of 0.0724 -- establishes the convention that the FIRST
# number is active and the rest are documented alternates. That convention is
# followed here and the alternates are retained as *_ALT constants.
MAN_COE_MD5 = "8f21f4316def7dbcb1bd5f4c9ef5fed0"
MAN_COE_SOURCE = "AHAAH v2.1 public release, AHAAH_MIL-STD-1474E_defaultHPD/man.coe"

# ---------------------------------------------------------------------------
# man.coe coefficients - integration
# ---------------------------------------------------------------------------
ALPHAR: float = 1.0          # dimensionless, undocumented
ALPHAI: float = 0.0          # dimensionless, undocumented
BETAR: float = 2.0           # dimensionless, undocumented (alternates 1, 0)
BETAI: float = 0.0           # dimensionless, undocumented

# ---------------------------------------------------------------------------
# man.coe coefficients - outer ear (CGS acoustic analogue)
#   pressure  = "voltage",  dyne/cm^2
#   volume velocity = "current", cm^3/s
#   inertance L = g/cm^4, compliance C = cm^5/dyne, resistance R = dyne*s/cm^5
# ---------------------------------------------------------------------------
RDF: float = 1.29e-1         # dyne*s/cm^5, head-diffraction loss, in series with the 2P source
LDF: float = 2.56e-5         # g/cm^4,      head-diffraction inertance, in series with the P source
L1: float = 2.215            # cm,   ear-canal length      (L1*S1 = 0.975 cm^3, textbook ~1.0)
L2: float = 6.962e-1         # cm,   concha length         (L2*S2 = 2.99 cm^3)
S1: float = 0.44             # cm^2, ear-canal area
S2: float = 4.3              # cm^2, concha area

# ---------------------------------------------------------------------------
# man.coe coefficients - middle ear cavity (return rail)
# ---------------------------------------------------------------------------
LH: float = 1.4e-2           # g/cm^4,      aditus neck inertance
CB: float = 5.1e-6           # cm^5/dyne,   mastoid air cells (Cb*rho*c^2 = 7.14 cm^3)
RH: float = 91.85            # dyne*s/cm^5, aditus loss
CM: float = 0.35e-6          # cm^5/dyne,   tympanic cavity  (Cm*rho*c^2 = 0.49 cm^3)

# ---------------------------------------------------------------------------
# man.coe coefficients - eardrum (two-piston, Tm3Piston = 1)
# ---------------------------------------------------------------------------
LDS: float = 4.6e-3          # g/cm^4,      drum "independent" part: mass
RDS: float = 1.147e3         # dyne*s/cm^5, drum "independent" part: resistance
CDS: float = 0.395e-6        # cm^5/dyne,   drum "independent" part: compliance
LDM: float = 2.2e-2          # g/cm^4,      drum "conductive" part: mass
CDC: float = 2.31e-6         # cm^5/dyne,   drum "conductive" part: compliance
RDC: float = 60.6            # dyne*s/cm^5, drum "conductive" part: resistance

# ---------------------------------------------------------------------------
# man.coe coefficients - ossicles
# ---------------------------------------------------------------------------
NT: float = 20.0             # dimensionless, middle-ear transformer ratio (lever x area)
RMI: float = 8.391e5         # dyne*s/cm^5, malleo-incudal joint resistance
CMI: float = 4.33e-10        # cm^5/dyne,   malleo-incudal joint compliance
LI: float = 2.44             # g/cm^4,      incus mass
RIS: float = 0.475e5         # dyne*s/cm^5, incudo-stapedial joint resistance
CIS: float = 4.57e-10        # cm^5/dyne,   incudo-stapedial joint compliance
LS: float = 2.44             # g/cm^4,      stapes mass
LV: float = 6.25             # g/cm^4,      vestibular inertance
LO: float = 52.4             # g/cm^4,      helicotrema inertance
RAL: float = 2.06e5          # dyne*s/cm^5, annular ligament resistance
RC: float = 2.64e6           # dyne*s/cm^5, cochlear input resistance
RO: float = 1.8e5            # dyne*s/cm^5, helicotrema resistance
CAL: float = 4.9e-10         # cm^5/dyne,   annular ligament compliance -- NONLINEAR ELEMENT
CRW: float = 1.38e-8         # cm^5/dyne,   round window membrane (in the return rail)
ASTAPES: float = 2.1e-2      # cm^2,        stapes footplate area

# ---------------------------------------------------------------------------
# man.coe coefficients - annular ligament nonlinearity
#
# These four lines sit immediately after Astapes in man.coe. They are NAMED there
# and DEFINED NOWHERE in the entire released documentation set. They carry no
# units. Their interpretation in this module is an inference -- see
# _annular_ligament_stiffness_ratio() and DECLARED_ASSUMPTIONS['annular_ligament'].
# ---------------------------------------------------------------------------
LGAP: float = 30.0           # units unknown; read here as the ligament gap in MICRONS
EO: float = 0.01             # units unknown; read here as the strain at stiffening onset
EB: float = 0.20             # units unknown; read here as the strain at the "bound"
RAMP: float = 6.0            # units unknown; read here as the stiffening exponent

# ---------------------------------------------------------------------------
# man.coe coefficients - cochlea
# ---------------------------------------------------------------------------
SO: float = 1e9              # dyne/cm^3,   BM stiffness per unit area at base
MO: float = 5.8e-3           # g/cm^3,      BM + entrained fluid mass per unit area at base
RVO: float = 91.2            # dyne*s/cm^3, BM viscous resistance per unit area at base
BO: float = 0.008            # cm,          BM width at base
AO: float = 1.25e-2          # cm^2,        effective scala area Sv*St/(Sv+St)
FO: float = 20000.0          # Hz,          max resonant frequency at the base
DELO: float = 0.03           # loss constant at base, fwhm/charfreq (informational; Rvo is used)
D1: float = 0.666            # cm, S(x) = So*exp(-x/D1)
D2: float = 1.0              # cm, M(x) = Mo*exp(+x/D2)
D3: float = 1.0              # cm, R(x) = Rvo*exp(+x/D3)
D4: float = 2.0              # cm, B(x) = Bo*exp(+x/D4)
D5: float = 2.0              # cm, A(x) = Ao*exp(-x/D5)
DC: float = 0.8              # cm, F(x) = Fo*exp(-x/Dc)
NW: Tuple[float, float] = (2.0, 1.23)     # undocumented
CA: Tuple[float, float] = (0.5, 0.316)    # "Cochlear Amplifier Gain" -- undocumented, unused here
BCOEF: float = 2.0                        # undocumented, unused here
XB_APEX: float = 3.5         # cm, physical base-to-apex length
XBM_FROM: float = 0.425      # cm, first evaluation point
XBM_TO: float = 3.175        # cm, last evaluation point
XBM_NO: int = 23             # number of evaluation points
SWEIGHT_BASE: float = 1.0    # "stress weight" at the base
SWEIGHT_APEX: float = 1.0    # "stress weight" at the apex (alternate 0.2)
DAMAGE_THRESHOLD: float = 0.0   # microns; 0 = every upward peak counts
COCHLEAR_GAIN_FACTOR: float = 0.0724        # active value (first on the line)
COCHLEAR_GAIN_FACTOR_ALT_TAPER: float = 0.025   # man.coe: "(0.025 for WKBTaper)"
COCHLEAR_GAIN_FACTOR_ALT_ORIG: float = 0.15     # man.coe: "(0.15 for WKBOrig)"
WKB_SCALA_BM_WIDTH_DECAY: float = 0.75      # cm^-1, "decay length based on tapering factors"
WKB_SCALA_BM_WIDTH_DECAY_ALT: float = 0.9   # cm^-1, alternate

# ---------------------------------------------------------------------------
# man.coe coefficients - middle ear muscle (MEM) reflex
# ---------------------------------------------------------------------------
MEM_DELAY_EVOKED_S: float = 9.0e-3     # s, latency of the reflex when evoked (unwarned)
MEM_DELAY_WARNED_S: float = -50.0e-3   # s, contraction pre-established 50 ms before t=0 (warned)
MEM_TIME_CONSTANT_S: float = 11.7e-3   # s, growth time constant of the contraction
MEM_MAG_K: Tuple[float, float, float] = (12.0, 1.0, 6.0)    # stiffness multipliers
MEM_MAG_R: Tuple[float, float, float] = (12.0, 1.0, 12.0)   # resistance multipliers
ADAPT_FACTOR: float = 2.5              # undocumented; not used (see DECLARED_ASSUMPTIONS)

# ---------------------------------------------------------------------------
# man.coe switches (default build)
# ---------------------------------------------------------------------------
COCHLEAR_MODEL_TYPE: int = 2   # 3=WKBNoTaper 2=WKBTaper 1=WKBOrig 0=DifEqn
EARPLUG: int = 0               # no hearing protector modelled in this version
HEADPHONE: int = 0
HORN_EXT_EAR: int = 1          # 1=Horn 0=Three-cylinders
TM3_PISTON: int = 1            # 1=Two-piston eardrum
TERMINATE_COCHLEA: int = 0     # 0=finite termination

# ---------------------------------------------------------------------------
# Shaw head-related transfer functions, transcribed from
#   AHAAH_ver_2_1/AHAAH_MIL-STD-1474E_defaultHPD/Dat/F11_12D.DAT
# (Shaw's figures 11 and 12). The released binary references this file next to
# its "incident angle = " prompt and the code string
# "NorCE:10 NorED:11 GrED:21 ShED31 NoED:00", i.e. <incidence><destination>.
#
# Row 1 of the file is the frequency axis in kHz, rows 2-21 are azimuths
# 0, 18, ... 342 degrees (free field -> BLOCKED ear-canal entrance, dB) and row 22
# is a further gain row that is the ear-canal-entrance -> eardrum transfer. In
# every row the first column is a label (the azimuth), not data, so only columns
# 1..76 are used.
#
# WHY THIS IS THE DEFAULT OUTER EAR (docs/AHAAH-SPEC.md section 10.1 left this
# open; it is settled here empirically, which is what spec section 11.2 Tier 1 is
# for). Against ARL's own validation curve Dat/FFEDM90.DAT (Mehrgardt & Mellert
# free-field -> eardrum at 90 deg):
#   * AZ90 + CE2ED reproduces it to within 1.2 dB from 0.2 to 8.4 kHz;
#   * the lumped 2P/P diffraction circuit plus concha and canal lines is off by
#     +13 dB at 9.4 kHz and +39 dB at 13.3 kHz, because a two-element diffraction
#     network cannot produce the head-shadow rolloff and the undamped concha line
#     resonates at 12.3 kHz.
# The circuit's CANAL section on its own does match the file's own canal->eardrum
# row to within 3-5 dB out to 10 kHz, so the chain used is:
#   free field --(Shaw, azimuth)--> canal entrance --(circuit canal)--> eardrum.
# Applying the tabulated eardrum row as well would double-count the canal.
SHAW_FREQ_kHz: Tuple[float, ...] = (
    0.2, 0.21, 0.22, 0.24, 0.25, 0.27, 0.28, 0.3, 0.32, 0.34, 0.35, 0.38,
    0.4, 0.42, 0.45, 0.47, 0.5, 0.53, 0.56, 0.6, 0.63, 0.67, 0.71, 0.75,
    0.79, 0.84, 0.89, 0.94, 1.0, 1.06, 1.12, 1.19, 1.26, 1.33, 1.41, 1.5,
    1.58, 1.68, 1.78, 1.88, 2.0, 2.11, 2.24, 2.37, 2.51, 2.66, 2.82, 2.99,
    3.16, 3.35, 3.55, 3.76, 3.98, 4.22, 4.47, 4.73, 5.01, 5.31, 5.62, 5.96,
    6.31, 6.68, 7.08, 7.5, 7.94, 8.41, 8.91, 9.44, 10.0, 10.6, 11.2, 11.9,
    12.6, 13.3, 14.1, 15.0,
)
# Free field -> blocked ear-canal entrance, dB, at 90 deg azimuth (sound toward
# the side of the head). MIL-STD-1474E B-A.2.1 makes this the default worst case.
SHAW_FF_TO_CANAL_dB_AZ90: Tuple[float, ...] = (
    1.45, 1.55, 1.75, 1.75, 1.75, 1.55, 1.55, 1.55, 1.75, 2.15, 2.55, 3.15,
    3.55, 4.05, 4.55, 4.85, 4.95, 5.25, 5.25, 5.45, 5.45, 5.55, 5.45, 5.55,
    5.45, 5.25, 5.05, 5.05, 5.25, 5.35, 5.85, 6.35, 6.45, 6.75, 6.95, 6.95,
    7.05, 7.55, 8.35, 9.35, 10.35, 11.65, 12.95, 14.15, 14.65, 14.15, 12.45, 9.85,
    7.55, 5.35, 3.85, 3.15, 3.15, 2.85, 2.45, 1.65, 1.45, 2.45, 3.75, 5.35,
    7.35, 9.15, 11.35, 12.75, 12.85, 9.95, 4.55, -1.95, -5.65, -4.45, -0.55, 1.75,
    2.85, 3.35, 3.25, 2.95,
)
# Free field -> blocked ear-canal entrance, dB, at 0 deg azimuth. Used for
# "grazing" incidence: MIL-STD describes it as what "a human firing a rifle"
# experiences, i.e. the source ahead rather than to the side.
SHAW_FF_TO_CANAL_dB_AZ0: Tuple[float, ...] = (
    -0.3, -0.0, 0.3, 0.6, 0.9, 1.0, 1.4, 1.5, 1.7, 1.7, 1.5, 1.3,
    1.0, 0.6, 0.6, 0.6, 0.7, 1.2, 1.3, 1.5, 1.3, 1.3, 1.5, 1.8,
    1.8, 1.5, 0.5, -0.5, -1.8, -2.4, -2.1, -0.9, 0.4, 1.9, 3.8, 5.1,
    7.0, 8.5, 9.5, 9.9, 10.2, 10.4, 10.8, 11.5, 12.1, 12.7, 12.5, 11.6,
    10.2, 8.5, 7.0, 5.7, 5.1, 3.9, 1.9, -0.1, -1.2, -1.2, -0.9, -0.1,
    0.6, 1.5, 1.9, 0.2, -1.5, -4.3, -6.9, -9.4, -10.3, -10.0, -8.3, -5.4,
    -2.8, 0.1, 2.3, 3.4,
)
# Ear-canal entrance -> eardrum, dB (row 22). NOT applied in the signal chain --
# the circuit's canal supplies that step. Retained as the Tier-1 reference the
# circuit canal is checked against.
SHAW_CANAL_TO_EARDRUM_dB: Tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01, -0.01, 0.03, -0.01,
    -0.02, 0.06, 0.1, 0.1, 0.14, 0.18, 0.21, 0.29, 0.33, 0.41, 0.49, 0.57,
    0.6, 0.6, 0.52, 0.48, 0.52, 0.92, 1.24, 1.23, 1.19, 1.11, 1.23, 1.15,
    1.1, 1.1, 1.3, 1.5, 1.46, 1.21, 1.57, 2.3, 2.99, 3.59, 4.44, 5.2,
    6.05, 6.98, 8.39, 10.21, 11.5, 12.43, 13.48, 13.44, 12.34, 10.97, 9.22, 7.84,
    6.39, 5.29, 4.56, 4.56, 4.68, 4.92, 8.48, 9.44, 5.32, 2.32, -2.58, -10.67,
    -10.23, -8.21, -6.51, -5.02,
)

# ---------------------------------------------------------------------------
# Physical constants (CGS). Not in man.coe; standard air/perilymph values.
# ---------------------------------------------------------------------------
RHO_AIR: float = 1.2e-3        # g/cm^3
C_AIR: float = 3.43e4          # cm/s
RHO_C2: float = RHO_AIR * C_AIR * C_AIR   # 1.412e6 dyne/cm^2; Cm*RHO_C2 = 0.49 cm^3 (checks out)
RHO_PERILYMPH: float = 1.0     # g/cm^3, cochlear fluid

PA_PER_DYNE_CM2: float = 0.1   # 1 dyne/cm^2 = 0.1 Pa, so Pa -> dyne/cm^2 is x10
CM_TO_MICRON: float = 1e4

# ---------------------------------------------------------------------------
# Derived hazard relations (MIL-STD-1474E Annex A; both verified in the tests)
# ---------------------------------------------------------------------------
CTS_SLOPE: float = 26.6        # CTS_dB = CTS_SLOPE*ln(ARU) - CTS_INTERCEPT
CTS_INTERCEPT: float = 140.1
ARU_LIMIT_OCCASIONAL: float = 500.0    # allowed exposures N = 500/ARU  (<= 1 session/week)
ARU_LIMIT_OCCUPATIONAL: float = 200.0  # N = 200/ARU  (>= 2 sessions/week)
PTS_FRACTION_OF_CTS: float = 0.6       # predicted permanent shift once 500 ARU is exceeded

# 95th-percentile susceptibility. ARL, CalculationProdedure_HearingProt.pdf p.3:
# "the model achieves the prediction for the 95 percentile ear by artificially
# raising the SPL on the test impulse by 10 dB (1.64 standard deviations)".
# Applied to the waveform, unconditionally, before the model. The v2.1 binary has
# no percentile control, so 391.0 / 2237 ARU are 95th-percentile figures.
SUSCEPTIBILITY_95_dB: float = 10.0
SUSCEPTIBILITY_95_GAIN: float = 10.0 ** (SUSCEPTIBILITY_95_dB / 20.0)   # 3.16228

# ---------------------------------------------------------------------------
# Input admissibility
# ---------------------------------------------------------------------------
# AHAAH resolves cochlear mechanics to ~20 kHz and the ARL reference case is
# sampled at 125 kHz. Below 96 kHz the blast rise is not resolved and the model's
# high-frequency bands (band 1 is 11.76 kHz) are being fed aliased or
# anti-alias-filtered content.
MIN_SAMPLE_RATE_HZ: float = 96000.0
AHAAH_WORKING_RATE_HZ: float = 125000.0   # the rate of the ARL reference case
DEFAULT_OVERSAMPLE: int = 4               # network integration substeps per working sample

# ARL states the model does not apply below roughly this peak level; the middle
# ear is linear there and the ARU is negligible and meaningless.
MIN_PEAK_dB: float = 130.0

# Divergence guard for the nonlinear network integration.
#
# The annular ligament is a stiffening compliance with an exponent of 6, solved by
# linearising about the previous step's charge. Above roughly 200 dB at the drum
# that iteration goes unstable and the trapezoidal integrator runs away: at 214 dB
# the solver returned a peak stapes displacement of 6.9e19 microns (7e14 metres)
# and a finite, plausible-looking hazard figure of 6.7e36, with no error raised.
# That is exactly the failure this codebase exists to prevent, so it is now caught.
# A middle ear cannot displace by a metre; anything at or beyond this bound is the
# integrator, not the ear.
MAX_PLAUSIBLE_STAPES_DISPLACEMENT_UM: float = 1.0e6   # 1 metre


class ModelDivergedError(RuntimeError):
    """The nonlinear middle-ear integration went unstable; no result exists."""

VALID_INPUT_LOCATIONS = (
    "free_field_normal",      # calc code 1: FFHN, sound toward the side of the head
    "free_field_grazing",     # calc code 5: FFHG, sound from ahead (rifle shooter)
    "ear_canal_entrance",     # calc code 2
    "eardrum",                # calc code 3
)
CALC_CODE_TO_LOCATION: Dict[int, str] = {
    1: "free_field_normal",
    5: "free_field_grazing",
    2: "ear_canal_entrance",
    4: "eardrum",             # FFMN -> manikin free field is referred to the eardrum
    6: "eardrum",             # FFMG
    3: "eardrum",
}


# ---------------------------------------------------------------------------
# Declared assumptions (docs/AHAAH-SPEC.md section 11.0 requires these in writing
# BEFORE any run, with an a/b/c category, so that over-fitting is visible).
#   (a) explicitly documented in the ARL/MIL-STD sources
#   (b) not stated, but unambiguously derivable from the sources or the data
#   (c) an inference -- every (c) item is a risk to the final number
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Assumption:
    """One declared modelling choice, with its evidence category."""
    key: str
    category: str      # 'a', 'b' or 'c'
    choice: str
    rationale: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.key}: {self.choice}"


DECLARED_ASSUMPTIONS: Tuple[Assumption, ...] = (
    Assumption(
        "band_geometry", "a",
        "F(x) = 20000*exp(-x/0.8) at x = 0.425 .. 3.175 cm, 23 points",
        "man.coe Fo/Dc/XbmFrom/XbmTo/XbmNo; reproduces all 23 160F.HAZ frequencies exactly.",
    ),
    Assumption(
        "total_aru_rule", "a",
        "total ARU = max over the 23 locations of sum(D^2) over upward peaks",
        "MIL-STD-1474E B-A.1.1 and B-A.2.1; max(160F.HAZ) = 390.99 = header 391.0. "
        "The sum (4031.24) matches nothing.",
    ),
    Assumption(
        "susceptibility_10dB", "b",
        "+10 dB applied to the input waveform, unconditionally",
        "Three ARL documents state it; the v2.1 binary has no percentile selector. "
        "It is worth x3-5 in ARU and cannot be applied as a post-hoc factor.",
    ),
    Assumption(
        "units", "b",
        "CGS acoustic-impedance analogue; input converted Pa -> dyne/cm^2 by x10",
        "man.coe's own unit comments (cm^5/dyne, dyne-sec/cm^5, g/cm^4).",
    ),
    Assumption(
        "netlist_topology", "b",
        "EarModel.JPG topology; bulla and round window in the RETURN RAIL",
        "Nine independent anatomical sanity checks come out right "
        "(canal 0.975 cm^3, tympanic cavity 0.49 cm^3, mastoid 7.14 cm^3, "
        "bulla resonance 596 Hz, diffraction corner 802 Hz).",
    ),
    Assumption(
        "mem_timing", "b",
        "unwarned: onset at t0 + 9 ms, growth tau = 11.7 ms; warned: pre-activated at -50 ms",
        "man.coe MemDelay '9.0E-3 -50E-3' and MemTimeConst. Warned is 98.6% contracted at t=0.",
    ),
    # ---- category (c) below this line: each of these is a risk to the number ----
    Assumption(
        "outer_ear_mechanism", "b",
        "free field -> canal entrance from the tabulated Shaw HRTF (Dat/F11_12D.DAT), "
        "then the circuit's canal to the eardrum; the lumped 2P/P diffraction sources and "
        "concha line are NOT used (outer_ear='circuit' selects them)",
        "Settled empirically against ARL's own validation curve Dat/FFEDM90.DAT, which is "
        "what spec section 11.2 Tier 1 exists for. Shaw AZ90 + the file's canal->eardrum "
        "row reproduces FFEDM90 to 1.2 dB from 0.2-8.4 kHz; the circuit route is +13 dB at "
        "9.4 kHz and +39 dB at 13.3 kHz. The circuit's canal alone does match the file's "
        "own canal->eardrum row to 3-5 dB out to 10 kHz, so only the diffraction and concha "
        "stages are replaced. Applying both would double-count the outer ear.",
    ),
    Assumption(
        "hrtf_phase", "c",
        "minimum phase reconstructed from the tabulated magnitude, held flat outside "
        "0.2-15 kHz",
        "F11_12D.DAT is magnitude only. Phase matters for a blast because the peak "
        "displacement depends on it. Minimum phase is the causal default; the release's "
        "FFEDP90.DAT gives the measured free-field->eardrum phase and could be used to "
        "check it.",
    ),
    Assumption(
        "horn_profile", "c",
        "concha and canal as two uniform lossless lines (L2@S2 then L1@S1), "
        "ladder-discretised at 0.2 cm -- only reached with outer_ear='circuit'",
        "HornExtEar=1 selects 'Horn' but the profile joining S2 to S1 is unspecified and "
        "man.coe has no L3/A1-A3, so the three-cylinder diagram cannot be reconstructed. "
        "Two uniform segments preserve both anatomically-checked volumes. Under the default "
        "outer_ear='shaw' only the canal section (L1, S1) is used.",
    ),
    Assumption(
        "integrator", "c",
        "trapezoidal (Tustin) companion models, 4x oversampled",
        "Alphar/Alphai/Betar/Betai (1,0,2,0) are undocumented; Betar=2 is consistent with "
        "s -> (2/T)(1-z^-1)/(1+z^-1). MEASURED: doubling the substep rate (4x -> 8x) moves "
        "the reference-case total by 0.2 %, so the scheme is not load-bearing.",
    ),
    Assumption(
        "annular_ligament", "c",
        "stiffness multiplier 1 + ((|d|/Lgap - Eo)/(Eb - Eo))^Ramp for |d|/Lgap > Eo, "
        "with Lgap read as 30 microns and Eo/Eb as strains",
        "HIGHEST RISK ITEM. No functional form exists anywhere in the release. This reading "
        "puts the effective onset near 4 um (~133 dB at the drum) and hard limiting by "
        "~20 um, which matches the two prose constraints ('linear below 130-140 dB', 'stops "
        "the stapes displacing more than a few tens of microns') but is otherwise a guess.",
    ),
    Assumption(
        "mem_element_mapping", "c",
        "MemMagK[0]=12 scales the stiffness of Cal and MemMagR[0]=12 the resistance of "
        "Ral -- the annular ligament only, not the ossicular joints",
        "Which elements receive the 12, the 1 and the 6 is not stated anywhere. This "
        "mapping was chosen against the documented frequency response, not against the "
        "160F result: it gives -20.6/-21.5/-18.9 dB at 125/500/1000 Hz falling monotonically "
        "to -4.5 dB at 11.8 kHz, i.e. exactly 'on the order of 20 dB below about 1.0 kHz and "
        "progressively less at higher frequencies'. Including Cmi/Cis instead produces up to "
        "+18 dB of GAIN above 5 kHz, which contradicts the documentation. Anatomically the "
        "stapedius acts on the stapes, so the annular ligament is the right place. Affects "
        "the WARNED figure only.",
    ),
    Assumption(
        "mem_trigger", "c",
        "unwarned latency measured from the first sample of the record",
        "The trigger (record start? a level crossing? which level?) is not stated. "
        "For a 16 ms record with the peak at 5 ms this moves the onset by several ms.",
    ),
    Assumption(
        "cochlear_formulation", "c",
        "1-D long-wave WKB: k^2 = j*w*rho*B(x)/(A(x)*Z(x)), "
        "p(x) = p(0)*sqrt(k0/k)*exp(-int k dx), D = p/(j*w*Z)",
        "The graded-parameter set is complete and self-consistent (D1 = 1/(2/Dc - 1/D2), "
        "D2 = D3 = D4/2, D5 = D4 all check out), which is strong evidence for the standard "
        "formulation -- but the equations are written down nowhere in the release.",
    ),
    Assumption(
        "cf_alignment", "c",
        "So and Mo used exactly as man.coe writes them ('as_written'), so the mechanical "
        "resonance is sqrt(S(x)/M(x)) = 66.1 kHz * exp(-x/Dc) while F(x) = 20 kHz * "
        "exp(-x/Dc) supplies the band LABELS",
        "The factor of 3.30 between sqrt(So/Mo)/2pi and Fo is unexplained and man.coe's own "
        "comment calls Fo the 'max resonant freq at base of BM', so the two disagree. "
        "'as_written' is chosen because spec section 11.0 rule 1 forbids adjusting man.coe "
        "values; the alternative ('fo') invents a x10.9 mass scaling to reconcile them. "
        "DISCLOSURE: 'as_written' also agrees better with 160F (warned -0.8% vs -5.7%, and "
        "all 23 bands within a factor of 3.4 rather than 17), so although the choice follows "
        "the declared rule it should be treated as a (c) risk.",
    ),
    Assumption(
        "cochlear_gain", "c",
        f"CochlearGainFactor = {COCHLEAR_GAIN_FACTOR} (first value on the line)",
        "man.coe lists '0.0724 .025 0.15' with the comment '(0.025 for WKBTaper)' while "
        "the active model IS WKBTaper. First-value-active gives 0.0724; select-by-type "
        "gives 0.025. Ratio 2.90 in displacement, 8.39 in ARU.",
    ),
    Assumption(
        "wkb_taper", "c",
        "'WKBTaper' read as: the WKB amplitude factor carries the scala-area and BM-width "
        f"grading, (A(x)k(x))^-1/2. The extra multiplier exp(-{WKB_SCALA_BM_WIDTH_DECAY}*x) "
        "is NOT applied by default (extra_wkb_taper=True applies it).",
        "WkbScalaBMwidthDecay is documented only as 'cm^-1 decay length based on tapering "
        "factors' -- it names the scala and BM-width decay, which is exactly what the "
        "amplitude factor is built from, so applying it again as a bare exponential looks "
        "like a double count. There is no equation for it anywhere. DISCLOSURE: this choice "
        "also moves the warned figure for 160F from 106 to 369 ARU against a reference of "
        "391, so although it was made on derivational grounds a reviewer should treat it as "
        "the most suspect (c) item after the annular ligament.",
    ),
    Assumption(
        "peak_rule", "c",
        "one peak per positive-going excursion, taken at its maximum",
        "'as each peak passes' is ambiguous between per-excursion and every local maximum. "
        "MEASURED on the reference case: the every-local-maximum reading is 9 % higher. "
        "Real, but not the dominant error.",
    ),
    Assumption(
        "bm_sign", "c",
        "positive stapes volume velocity into the cochlea produces positive ('upward') "
        "BM displacement",
        "The sign of 'upward' relative to input pressure polarity is not stated. MEASURED "
        "on the reference case: inverting it changes the total by 1 %, so this declared "
        "risk turns out not to bite for a blast waveform.",
    ),
    Assumption(
        "analysis_interval", "c",
        "the whole supplied record",
        "The header reports the A-weighted window as 13.55 ms from 2.50 ms; whether the ARU "
        "accumulation uses that window or the whole record is not stated.",
    ),
    Assumption(
        "adapt_factor", "c",
        "AdaptFactor 2.5 is not used",
        "Entirely undocumented; the name suggests decay of the contraction over time, but "
        "no time course is given.",
    ),
)

CATEGORY_C_COUNT: int = sum(1 for a in DECLARED_ASSUMPTIONS if a.category == "c")

# The gate of docs/AHAAH-SPEC.md section 11.5. One of:
#   "not_validated" - the reference case is not reproduced. NO ARU IS EMITTED.
#   "fitted"        - reproduced, but category-(c) choices were varied to get there.
#   "validated"     - reproduced with zero category-(c) choices varied.
# Set from the measured outcome of
# tests/test_ahaah.py::test_VALIDATION_against_ARL_160F_reference, never by hand to
# make something look better. It is "not_validated": see the module docstring.
VALIDATION_STATUS: str = "not_validated"

# Why compute_ahaah() returns no number. This is the text a caller sees.
NOT_VALIDATED_NOTE: str = (
    "NO AHAAH RESULT: this implementation does not reproduce the ARL reference case and "
    "therefore emits no Auditory Risk Unit figure. Measured against the only reference in the "
    "public AHAAH v2.1 release (160F: 391.0 ARU warned, 2237 ARU unwarned), it is 51.5 % low "
    "on the unwarned figure, reproduces the warned/unwarned ratio as 2.80 against 5.72, and "
    "disagrees with the published 23-band table in 20 of 23 bands by up to a factor of 3.4. "
    f"{CATEGORY_C_COUNT} modelling choices are inferences not documented anywhere in the "
    "public release, and sweeping only the ones this module exposes moves the answer by a "
    "factor of 255. Use the A-weighted energy metrics, which MIL-STD-1474E approves alongside "
    "AHAAH and which SASA computes exactly. See docs/AHAAH.md."
)
# Retained under its old name so that anything importing it keeps working; it now
# says the same thing as NOT_VALIDATED_NOTE.
UNVALIDATED_NOTE: str = NOT_VALIDATED_NOTE

# The only label permitted on the research path's numbers.
NOT_AN_ARU_LABEL: str = (
    "NOT AN ARU - unvalidated research output, microns squared of summed upward "
    "basilar-membrane peak displacement. Not a MIL-STD-1474E result and not for reporting."
)

SCIENTIFIC_STANDING_NOTE: str = (
    "AHAAH is one of two metrics approved by MIL-STD-1474E and is not undisputed: NATO has "
    "not adopted it, and 2010 AIBS / 2012 NIOSH reviews and 2016-2017 live-fire work "
    "question the assumed middle-ear-muscle reflex. Unwarned is the conservative case and "
    "is the headline figure here."
)


# ---------------------------------------------------------------------------
# .AHA file reader
# ---------------------------------------------------------------------------

@dataclass
class AhaWaveform:
    """
    A waveform loaded from an ARL `.AHA` file.

    The format is a tab-separated three-column header (PARAMETER / VALUE / COMMENT)
    of variable length, followed by one sample per line in %.4E format, in Pascals.
    The header row count and label wording differ between the files in the release,
    so the payload is located by pattern rather than by line count.
    """
    pressure_Pa: np.ndarray
    sample_rate: float
    calc_code: int
    input_location: str
    title: str = ""
    header: Dict[str, Tuple[str, str]] = field(default_factory=dict)   # label -> (value, comment)
    # Reference results carried in the header, when present. These are the ARL
    # program's own output for this waveform and are the validation oracle.
    reference_ARU_warned: Optional[float] = None
    reference_ARU_unwarned: Optional[float] = None
    reference_exposures_warned: Optional[float] = None
    reference_exposures_unwarned: Optional[float] = None
    reference_peak_dB: Optional[float] = None
    reference_LAeq_dB: Optional[float] = None
    reference_Leq_dB: Optional[float] = None
    reference_LAeq8hr_dB: Optional[float] = None
    reference_A_weighted_energy_J_m2: Optional[float] = None
    source_path: Optional[Path] = None

    @property
    def duration_s(self) -> float:
        return len(self.pressure_Pa) / self.sample_rate


def _first_float(text: str) -> Optional[float]:
    """Return the first parseable float in `text`, or None."""
    for token in text.replace(",", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


def load_aha(path: str | Path) -> AhaWaveform:
    """
    Read an ARL `.AHA` waveform file.

    Args:
        path: Path to the `.AHA` file.

    Returns:
        AhaWaveform with the samples in Pascals plus whatever reference results the
        header carries.

    Raises:
        ValueError: if no numeric payload can be found, or the sample count
                    disagrees with the header's "Number of Samples".
    """
    p = Path(path)
    lines = p.read_text(errors="replace").splitlines()

    header: Dict[str, Tuple[str, str]] = {}
    samples: List[float] = []
    title = ""

    for line in lines:
        if "\t" in line:
            # Header row. Three tab-separated fields, any of which may be blank.
            parts = [f.strip() for f in line.split("\t")]
            label = parts[0]
            value = parts[1] if len(parts) > 1 else ""
            comment = parts[2] if len(parts) > 2 else ""
            if not label and not value and not comment:
                continue
            if label.upper().startswith("PARAMETER"):
                continue
            if label:
                header[label] = (value, comment)
                if label.lower().startswith("title"):
                    title = comment or value
            continue
        stripped = line.strip()
        if not stripped:
            continue
        try:
            samples.append(float(stripped))
        except ValueError:
            # Non-numeric, non-tabbed line: header continuation. Ignore.
            continue

    if not samples:
        raise ValueError(f"{p.name}: no numeric sample payload found")

    def _hdr(prefix: str) -> Optional[Tuple[str, str]]:
        low = prefix.lower()
        for label, pair in header.items():
            if label.lower().startswith(low):
                return pair
        return None

    def _hdr_value(prefix: str) -> Optional[float]:
        pair = _hdr(prefix)
        return _first_float(pair[0]) if pair else None

    def _hdr_comment_value(prefix: str) -> Optional[float]:
        pair = _hdr(prefix)
        return _first_float(pair[1]) if pair else None

    fs = _hdr_value("Sampling rate")
    if fs is None or fs <= 0:
        raise ValueError(f"{p.name}: header carries no usable 'Sampling rate'")

    # The calculation code is the measurement GEOMETRY, and it is load-bearing: the
    # free-field route adds the head-diffraction and ear-canal gain that the eardrum
    # route must not have (about 19 dB at 2.7 kHz). Guessing it silently would put a
    # geometry error straight into the answer, so an absent or unrecognised code is
    # an error, not a default.
    calc = _hdr_value("Microphone relative to ear")
    if calc is None:
        raise ValueError(
            f"{p.name}: header carries no 'Microphone relative to ear' calculation code. "
            "That code is the measurement geometry (free field / canal entrance / eardrum) "
            "and the answer depends on it by ~19 dB, so it cannot be assumed."
        )
    calc_code = int(calc)
    if calc_code not in CALC_CODE_TO_LOCATION:
        raise ValueError(
            f"{p.name}: calculation code {calc_code} is not one of the codes defined in the "
            f"AHAAH .AHA format ({sorted(CALC_CODE_TO_LOCATION)}). Refusing to guess the "
            "measurement geometry."
        )
    location = CALC_CODE_TO_LOCATION[calc_code]

    declared_n = _hdr_value("Number of Samples")
    if declared_n is not None and int(declared_n) != len(samples):
        raise ValueError(
            f"{p.name}: header declares {int(declared_n)} samples but {len(samples)} were read"
        )

    laeq_pair = _hdr("LAeq and Leq")
    leq = _first_float(laeq_pair[1].split("=")[-1]) if laeq_pair else None

    return AhaWaveform(
        pressure_Pa=np.asarray(samples, dtype=np.float64),
        sample_rate=float(fs),
        calc_code=calc_code,
        input_location=location,
        title=title,
        header=header,
        reference_exposures_warned=_hdr_value("Number of exposures with no protector, warned"),
        reference_exposures_unwarned=_hdr_value("Number of exposures with no protector, unwarned"),
        reference_ARU_warned=_hdr_comment_value("Number of exposures with no protector, warned"),
        reference_ARU_unwarned=_hdr_comment_value("Number of exposures with no protector, unwarned"),
        reference_peak_dB=_hdr_value("Peak Pressure Level"),
        reference_LAeq_dB=_hdr_value("LAeq and Leq"),
        reference_Leq_dB=leq,
        reference_LAeq8hr_dB=_hdr_value("LAeq8hr"),
        reference_A_weighted_energy_J_m2=_hdr_value("A-weighted energy"),
        source_path=p,
    )


def load_haz(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read an ARL `.HAZ` per-band hazard table.

    Returns:
        (band_frequencies_kHz, hazard_AHU, position_cm), each of length 23.
    """
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            continue   # header row
    if not rows:
        raise ValueError(f"{path}: no hazard rows found")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 1], arr[:, 2], arr[:, 3]


# ---------------------------------------------------------------------------
# Band geometry
# ---------------------------------------------------------------------------

def band_positions_cm() -> np.ndarray:
    """The 23 basilar-membrane evaluation positions, cm from the base (man.coe)."""
    return np.linspace(XBM_FROM, XBM_TO, XBM_NO)


def band_frequencies_Hz() -> np.ndarray:
    """
    Characteristic frequency of each of the 23 locations: F(x) = Fo*exp(-x/Dc).

    Reproduces all 23 Freq(kHz) values in 160F.HAZ to the file's 2 dp. Note that
    the spacing is 0.2254 octave (4.44 bands/octave), not the "roughly one-third
    octave" the ARL prose claims -- use the formula, not the description.
    """
    return FO * np.exp(-band_positions_cm() / DC)


# ---------------------------------------------------------------------------
# Modified nodal analysis with trapezoidal companion models
# ---------------------------------------------------------------------------
# The middle ear must be integrated in the time domain because two of its elements
# are state-dependent (the annular ligament and the MEM attenuator), so no fixed
# transfer function exists. Companion-model MNA is the standard way to do that and
# is what Betar = 2 implies under s -> (2/T)(1 - z^-1)/(1 + z^-1). See
# DECLARED_ASSUMPTIONS['integrator'] -- the scheme is category (c).

_GND = -1


class _Network:
    """
    A small modified-nodal-analysis circuit, integrated with the trapezoidal rule.

    Unknowns are the node pressures followed by one current per voltage source and
    one per ideal transformer. Element values may be rescaled between steps (that
    is how the MEM attenuator works); the annular ligament is a charge-state
    nonlinear compliance with its own stamp.
    """

    def __init__(self) -> None:
        self._node_index: Dict[str, int] = {"gnd": _GND}
        self._n_nodes = 0
        self.resistors: List[List] = []      # [n1, n2, R_base, mem_r_scaled]
        self.inductors: List[List] = []      # [n1, n2, L]
        self.capacitors: List[List] = []     # [n1, n2, C_base, mem_k_scaled]
        self.vsources: List[List] = []       # [n1, n2]
        self.transformers: List[List] = []   # [a, b, c, d, ratio]
        self.nl_caps: List[List] = []        # [n1, n2, C_base, area, mem_k_scaled]

    # -- construction -------------------------------------------------------

    def node(self, name: str) -> int:
        if name not in self._node_index:
            self._node_index[name] = self._n_nodes
            self._n_nodes += 1
        return self._node_index[name]

    def add_R(self, n1: str, n2: str, value: float, *, mem: bool = False) -> int:
        self.resistors.append([self.node(n1), self.node(n2), float(value), mem])
        return len(self.resistors) - 1

    def add_L(self, n1: str, n2: str, value: float) -> int:
        self.inductors.append([self.node(n1), self.node(n2), float(value)])
        return len(self.inductors) - 1

    def add_C(self, n1: str, n2: str, value: float, *, mem: bool = False) -> int:
        self.capacitors.append([self.node(n1), self.node(n2), float(value), mem])
        return len(self.capacitors) - 1

    def add_V(self, n1: str, n2: str) -> int:
        self.vsources.append([self.node(n1), self.node(n2)])
        return len(self.vsources) - 1

    def add_transformer(self, a: str, b: str, c: str, d: str, ratio: float) -> int:
        """Ideal transformer: v(c,d) = ratio * v(a,b), i(secondary) = i(primary)/ratio."""
        self.transformers.append(
            [self.node(a), self.node(b), self.node(c), self.node(d), float(ratio)]
        )
        return len(self.transformers) - 1

    def add_nonlinear_C(self, n1: str, n2: str, c_base: float, area: float,
                        *, mem: bool = False) -> int:
        """Charge-state compliance whose stiffness rises with |charge|/area (the ligament)."""
        self.nl_caps.append([self.node(n1), self.node(n2), float(c_base), float(area), mem])
        return len(self.nl_caps) - 1

    @property
    def n_unknowns(self) -> int:
        return self._n_nodes + len(self.vsources) + len(self.transformers)


def _annular_ligament_stiffness_ratio(q: float) -> Tuple[float, float]:
    """
    Stiffness multiplier of the annular ligament and its derivative wrt charge.

    ---- CATEGORY (c) INFERENCE. The functional form below appears in NO ARL
    document. man.coe names Lgap/Eo/Eb/Ramp and defines none of them, and the
    paper that would discuss them refers to an Appendix A that is missing from the
    released PDF. This is the single largest unknown in the model: it sets peak
    stapes displacement, hence BM displacement, hence ARU (which is displacement
    SQUARED). An error of x1.5 in the limiting level is x2.25 in ARU. ----

    Reading adopted: `q` is the volume displaced through the ligament, so the
    stapes displacement is d = q/Astapes (cm). Lgap is read as the ligament gap in
    microns and Eo/Eb as strains, giving a strain eps = |d|/Lgap. Stiffness is

        K(eps) = K0 * (1 + ((eps - Eo)/(Eb - Eo))**Ramp)   for eps > Eo

    This puts the effective onset (multiplier ~1.1) near 4 um of stapes
    displacement, which the netlist's linear stiffness places at ~133 dB at the
    drum, and produces hard limiting by ~20 um. Those are the only two quantitative
    statements the sources make ("essentially linear below 130-140 dB"; "stops the
    stapes from displacing more than a few tens of microns"). Consistency with two
    prose statements is not validation.

    Args:
        q: charge (volume displacement) through the ligament branch, cm^3.

    Returns:
        (multiplier, d(multiplier)/dq * q) -- the second value is the term needed
        for the exact derivative of p(q) = q*K(q), see _Solver.
    """
    d_cm = abs(q) / ASTAPES
    eps = d_cm * CM_TO_MICRON / LGAP          # strain, dimensionless
    if eps <= EO:
        return 1.0, 0.0
    u = (eps - EO) / (EB - EO)
    g = u ** RAMP
    # d(g)/d(eps) * eps, which is what the chain rule needs since eps is |q|-linear.
    g_prime_eps = RAMP * (u ** (RAMP - 1.0)) / (EB - EO) * eps
    return 1.0 + g, g_prime_eps


class _Solver:
    """Steps a `_Network` forward in time with the trapezoidal rule."""

    def __init__(self, net: _Network, h: float) -> None:
        self.net = net
        self.h = float(h)
        n = net.n_unknowns
        self.n = n
        self.n_nodes = net._n_nodes
        self.v_off = self.n_nodes
        self.t_off = self.n_nodes + len(net.vsources)

        self.iL = np.zeros(len(net.inductors))
        self.vL = np.zeros(len(net.inductors))
        self.iC = np.zeros(len(net.capacitors))
        self.vC = np.zeros(len(net.capacitors))
        self.q_nl = np.zeros(len(net.nl_caps))    # step-entry charge (integration history)
        self.i_nl = np.zeros(len(net.nl_caps))    # step-entry current
        self.q_lin = np.zeros(len(net.nl_caps))   # point the stiffness is linearised about

        self._A = np.zeros((n, n))
        self._b = np.zeros(n)

    # -- stamping -----------------------------------------------------------
    # Convention throughout: an element whose branch current from n1 to n2 is
    # i = G*v + Ieq stamps conductance G and puts -Ieq on the right-hand side at
    # n1 (+Ieq at n2), because the MNA row for a node sums the currents LEAVING it.

    @staticmethod
    def _stamp_g(A: np.ndarray, n1: int, n2: int, g: float) -> None:
        if n1 >= 0:
            A[n1, n1] += g
        if n2 >= 0:
            A[n2, n2] += g
        if n1 >= 0 and n2 >= 0:
            A[n1, n2] -= g
            A[n2, n1] -= g

    @staticmethod
    def _stamp_ieq(b: np.ndarray, n1: int, n2: int, i_eq: float) -> None:
        """Companion current source of magnitude `i_eq` flowing from n1 to n2."""
        if n1 >= 0:
            b[n1] -= i_eq
        if n2 >= 0:
            b[n2] += i_eq

    def _build(self, sources: Sequence[float], mem_k: float, mem_r: float) -> None:
        net = self.net
        h = self.h
        A = self._A
        b = self._b
        A.fill(0.0)
        b.fill(0.0)

        for n1, n2, r, mem in net.resistors:
            value = r * mem_r if mem else r
            self._stamp_g(A, n1, n2, 1.0 / value)

        # Inductor: i_k = (h/2L)*v_k + [i_{k-1} + (h/2L)*v_{k-1}]
        for idx, (n1, n2, L) in enumerate(net.inductors):
            g = h / (2.0 * L)
            self._stamp_g(A, n1, n2, g)
            self._stamp_ieq(b, n1, n2, self.iL[idx] + g * self.vL[idx])

        # Capacitor: i_k = (2C/h)*v_k - [(2C/h)*v_{k-1} + i_{k-1}]
        for idx, (n1, n2, C, mem) in enumerate(net.capacitors):
            value = C / mem_k if mem else C
            g = 2.0 * value / h
            self._stamp_g(A, n1, n2, g)
            self._stamp_ieq(b, n1, n2, -(g * self.vC[idx] + self.iC[idx]))

        # Nonlinear compliance, charge-state formulation:
        #   p = q*K(q),  K = K0*m(q),  K0 = 1/C_base (scaled by the MEM)
        #   q_k = q_{k-1} + (h/2)(i_k + i_{k-1})            (trapezoidal)
        # Linearised about q_lin:  p ~= p* + f'(q_lin)*(q - q_lin), giving
        #   i_k = G*p_k - [G*p* + i_{k-1}],   G = 2/(h*f'(q_lin))
        # Working in charge rather than in a time-varying capacitance keeps the
        # element charge-conserving when its stiffness changes within a step.
        for idx, (n1, n2, C, area, mem) in enumerate(net.nl_caps):
            g, p_star = self._nl_companion(idx, C, mem, mem_k)
            self._stamp_g(A, n1, n2, g)
            self._stamp_ieq(b, n1, n2, -(g * p_star + self.i_nl[idx]))

        for idx, (n1, n2) in enumerate(net.vsources):
            row = self.v_off + idx
            if n1 >= 0:
                A[n1, row] += 1.0
                A[row, n1] += 1.0
            if n2 >= 0:
                A[n2, row] -= 1.0
                A[row, n2] -= 1.0
            b[row] = sources[idx]

        for idx, (na, nb, nc, nd, ratio) in enumerate(net.transformers):
            row = self.t_off + idx
            # current i into the primary + terminal
            if na >= 0:
                A[na, row] += 1.0
            if nb >= 0:
                A[nb, row] -= 1.0
            if nc >= 0:
                A[nc, row] -= 1.0 / ratio
            if nd >= 0:
                A[nd, row] += 1.0 / ratio
            # constraint: v_c - v_d - ratio*(v_a - v_b) = 0
            if nc >= 0:
                A[row, nc] += 1.0
            if nd >= 0:
                A[row, nd] -= 1.0
            if na >= 0:
                A[row, na] -= ratio
            if nb >= 0:
                A[row, nb] += ratio
            b[row] = 0.0

    def _nl_companion(self, idx: int, c_base: float, mem: bool,
                      mem_k: float) -> Tuple[float, float]:
        """Conductance and linearisation pressure of nonlinear compliance `idx`."""
        k0 = (1.0 / c_base) * (mem_k if mem else 1.0)
        q_lin = self.q_lin[idx]
        mult, g_prime_eps = _annular_ligament_stiffness_ratio(q_lin)
        p_star = q_lin * k0 * mult
        dp_dq = k0 * (mult + g_prime_eps)
        return 2.0 / (self.h * dp_dq), p_star

    def _node_v(self, x: np.ndarray, n: int) -> float:
        return 0.0 if n < 0 else float(x[n])

    def step(self, sources: Sequence[float], mem_k: float, mem_r: float,
             *, corrector_passes: int = 1) -> np.ndarray:
        """
        Advance one time step and update all element states.

        Args:
            sources: value of each voltage source at the END of the step.
            mem_k: MEM stiffness multiplier now (1.0 = relaxed).
            mem_r: MEM resistance multiplier now.
            corrector_passes: extra solves that re-linearise the nonlinear
                compliance about its own new charge instead of the previous step's.

        Returns:
            The solution vector.
        """
        net = self.net
        h = self.h

        q_entry = self.q_nl
        i_entry = self.i_nl
        self.q_lin[:] = q_entry

        x = None
        q_new = q_entry
        i_new = i_entry
        for _pass in range(corrector_passes + 1):
            self._build(sources, mem_k, mem_r)
            x = np.linalg.solve(self._A, self._b)
            if not net.nl_caps:
                break
            q_new = np.empty_like(q_entry)
            i_new = np.empty_like(i_entry)
            for idx, (n1, n2, C, area, mem) in enumerate(net.nl_caps):
                v = self._node_v(x, n1) - self._node_v(x, n2)
                g, p_star = self._nl_companion(idx, C, mem, mem_k)
                i_new[idx] = g * (v - p_star) - i_entry[idx]
                q_new[idx] = q_entry[idx] + 0.5 * h * (i_new[idx] + i_entry[idx])
            if _pass < corrector_passes:
                self.q_lin[:] = q_new     # re-linearise; the history stays at entry

        assert x is not None
        self.q_nl = q_new
        self.i_nl = i_new

        for idx, (n1, n2, L) in enumerate(net.inductors):
            v = self._node_v(x, n1) - self._node_v(x, n2)
            g = h / (2.0 * L)
            self.iL[idx] = g * v + (self.iL[idx] + g * self.vL[idx])
            self.vL[idx] = v

        for idx, (n1, n2, C, mem) in enumerate(net.capacitors):
            v = self._node_v(x, n1) - self._node_v(x, n2)
            value = C / mem_k if mem else C
            g = 2.0 * value / h
            self.iC[idx] = g * v - (g * self.vC[idx] + self.iC[idx])
            self.vC[idx] = v

        return x


# ---------------------------------------------------------------------------
# The ear model
# ---------------------------------------------------------------------------

SECTION_LENGTH_CM: float = 0.2   # ladder discretisation of the concha and canal


def _add_acoustic_line(net: _Network, prefix: str, n_in: str, n_out: str,
                       length_cm: float, area_cm2: float) -> None:
    """
    Add a lossless uniform acoustic transmission line as a Pi ladder.

    ---- CATEGORY (c): the external ear geometry. man.coe carries L1/S1 (ear canal,
    0.975 cm^3) and L2/S2 (concha, 2.99 cm^3) and HornExtEar=1 ("Horn"), but no
    horn profile, no L3 and no A1-A3, so the three-cylinder configuration drawn in
    EarModel.JPG cannot be reconstructed from this coefficient file at all. Two
    uniform segments are used because they preserve both volumes, each of which
    independently matches textbook anatomy. ----

    Section length is 0.2 cm, giving a per-section cutoff of c/(pi*d) = 55 kHz, so
    the ladder is a good delay line through the model's 20 kHz band.
    """
    n_sections = max(1, int(round(length_cm / SECTION_LENGTH_CM)))
    d = length_cm / n_sections
    l_a = RHO_AIR * d / area_cm2          # g/cm^4
    c_a = d * area_cm2 / RHO_C2           # cm^5/dyne

    prev = n_in
    net.add_C(prev, "gnd", 0.5 * c_a)
    for i in range(n_sections):
        nxt = n_out if i == n_sections - 1 else f"{prefix}_{i}"
        net.add_L(prev, nxt, l_a)
        net.add_C(nxt, "gnd", c_a if i < n_sections - 1 else 0.5 * c_a)
        prev = nxt


def minimum_phase_from_magnitude(
    gain_dB: Sequence[float],
    table_frequencies_Hz: Sequence[float],
    fft_frequencies_Hz: np.ndarray,
) -> np.ndarray:
    """
    Build a causal minimum-phase transfer function from a tabulated magnitude.

    ---- CATEGORY (c): F11_12D.DAT gives magnitude only. A zero-phase application
    would be non-causal and would smear the blast front, and the peak displacement
    that ARU squares is phase-sensitive, so a phase must be supplied. Minimum phase
    is the standard causal choice; the release's FFEDP90.DAT holds the measured
    free-field-to-eardrum phase and is the way to check it. ----

    The magnitude is interpolated on log-frequency and held flat outside the
    tabulated 0.2-15 kHz range.

    Args:
        gain_dB: tabulated gain in dB.
        table_frequencies_Hz: the frequencies of `gain_dB`.
        fft_frequencies_Hz: the rfft grid to evaluate on (ascending, from 0).

    Returns:
        Complex array the same length as `fft_frequencies_Hz`.
    """
    ft = np.asarray(table_frequencies_Hz, dtype=np.float64)
    g = np.asarray(gain_dB, dtype=np.float64)
    f = np.asarray(fft_frequencies_Hz, dtype=np.float64)

    log_f = np.log(np.maximum(f, ft[0]))
    mag_dB = np.interp(log_f, np.log(ft), g, left=g[0], right=g[-1])
    mag = 10.0 ** (mag_dB / 20.0)

    # Real-cepstrum fold: the minimum-phase spectrum whose magnitude is `mag`.
    n_fft = 2 * (len(f) - 1)
    log_mag = np.log(np.maximum(mag, 1e-12))
    cep = np.fft.irfft(log_mag, n_fft)
    fold = np.zeros_like(cep)
    fold[0] = cep[0]
    half = n_fft // 2
    fold[1:half] = 2.0 * cep[1:half]
    fold[half] = cep[half]
    return np.exp(np.fft.rfft(fold, n_fft))


def apply_outer_ear_hrtf(pressure: np.ndarray, sample_rate: float,
                         gain_dB: Sequence[float]) -> np.ndarray:
    """Filter a waveform with a tabulated magnitude response, minimum phase."""
    n = len(pressure)
    n_fft = 1 << int(math.ceil(math.log2(max(2 * n, 16))))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    table_hz = np.asarray(SHAW_FREQ_kHz) * 1000.0
    h = minimum_phase_from_magnitude(gain_dB, table_hz, freqs)
    return np.fft.irfft(np.fft.rfft(pressure, n_fft) * h, n_fft)[:n]


@dataclass
class _MiddleEarOutput:
    stapes_volume_velocity: np.ndarray   # cm^3/s
    stapes_displacement_cm: np.ndarray   # cm
    eardrum_pressure: np.ndarray         # dyne/cm^2
    mem_fraction: np.ndarray             # 0..1 contraction


def _build_ear_network(input_location: str) -> Tuple[_Network, Dict[str, int]]:
    """
    Assemble the outer + middle ear network.

    Topology read from EarModel.JPG; values from man.coe. The two readings most
    easily got wrong, and both deliberate here:
      * the bulla (Cm || Lh+Cb+Rh) and the round window (Crw) sit in the RETURN
        RAIL, so the drum sees canal pressure minus cavity pressure;
      * the eardrum's two branches and all three ossicular shunts return to Nmec
        (the middle-ear cavity), not to true ground.
    """
    net = _Network()
    handles: Dict[str, int] = {}

    if input_location in ("free_field_normal",):
        # Two Thevenin sources onto the diffraction node: 2p(t) through Rdf (the
        # rigid-surface pressure-doubling limit) and p(t) through Ldf (the
        # free-field low-frequency limit). Corner Rdf/(2*pi*Ldf) = 802 Hz.
        handles["v_double"] = net.add_V("src2p", "gnd")
        net.add_R("src2p", "nff", RDF)
        handles["v_single"] = net.add_V("srcp", "gnd")
        net.add_L("srcp", "nff", LDF)
        # Concha then ear canal. (Rpl || Lpl -- the "air plug" -- has no values in
        # man.coe and is a short with Earplug = 0.)
        _add_acoustic_line(net, "concha", "nff", "ncanal", L2, S2)
        _add_acoustic_line(net, "canal", "ncanal", "ned", L1, S1)
    elif input_location == "ear_canal_entrance":
        handles["v_canal"] = net.add_V("ncanal", "gnd")
        _add_acoustic_line(net, "canal", "ncanal", "ned", L1, S1)
    elif input_location == "eardrum":
        handles["v_drum"] = net.add_V("ned", "gnd")
    else:
        raise ValueError(f"unsupported input_location {input_location!r}")

    # -- eardrum, two-piston (Tm3Piston = 1) --------------------------------
    # Independent part: a shunt from the drum to the cavity that does not drive the
    # ossicles. Ordering within a series triple is irrelevant to its impedance.
    net.add_L("ned", "nds1", LDS)
    net.add_C("nds1", "nds2", CDS)
    net.add_R("nds2", "nmec", RDS)

    # Conductive part: in the main line, feeding the transformer primary.
    net.add_L("ned", "ndc1", LDM)
    net.add_C("ndc1", "ndc2", CDC)
    net.add_R("ndc2", "ntp", RDC)

    # -- middle-ear transformer 1:Nt ---------------------------------------
    net.add_transformer("ntp", "nmec", "nmal", "nmec", NT)

    # -- ossicular chain ----------------------------------------------------
    net.add_C("nmal", "nmi", CMI)
    net.add_R("nmi", "nmec", RMI)
    net.add_L("nmal", "ninc", LI)
    net.add_C("ninc", "nis", CIS)
    net.add_R("nis", "nmec", RIS)
    net.add_L("ninc", "nsta", LS)

    # Annular ligament: the nonlinear compliance, in the main line, and the site
    # of the MEM attenuator (mem=True). See DECLARED_ASSUMPTIONS['mem_element_mapping'].
    handles["nl_annular"] = net.add_nonlinear_C("nsta", "nal", CAL, ASTAPES, mem=True)
    net.add_R("nal", "nves", RAL, mem=True)

    # -- cochlear load ------------------------------------------------------
    handles["l_vestibule"] = net.add_L("nves", "ncoc", LV)
    net.add_R("ncoc", "nst", RC)
    net.add_L("ncoc", "nhel", LO)
    net.add_R("nhel", "nst", RO)

    # -- return rail --------------------------------------------------------
    net.add_C("nmec", "nst", CRW)                 # round window
    net.add_C("nmec", "gnd", CM)                  # tympanic cavity
    net.add_L("nmec", "nbul1", LH)                # aditus neck ...
    net.add_C("nbul1", "nbul2", CB)               # ... to the mastoid air cells
    net.add_R("nbul2", "gnd", RH)

    return net, handles


def _mem_contraction(n: int, fs: float, warned: bool) -> np.ndarray:
    """
    Middle-ear-muscle contraction fraction, 0 (relaxed) to 1 (fully contracted).

    Warned: the contraction was established 50 ms before t = 0 (man.coe MemDelay
    second value), so at t = 0 it is 1 - exp(-50/11.7) = 98.6% complete.
    Unwarned: it is evoked by the impulse, begins 9 ms after the trigger and grows
    with a 11.7 ms time constant.

    ---- CATEGORY (c): the trigger for the unwarned latency is not stated anywhere.
    The first sample of the analysed record is used. For a 16 ms record with the
    peak at 5 ms, measuring the 9 ms from the peak instead would move the onset
    from 9.0 to 14.0 ms and change how much of the impulse is attenuated. ----
    """
    t = np.arange(n) / fs
    delay = MEM_DELAY_WARNED_S if warned else MEM_DELAY_EVOKED_S
    elapsed = t - delay
    frac = np.where(elapsed > 0.0, 1.0 - np.exp(-np.maximum(elapsed, 0.0) / MEM_TIME_CONSTANT_S), 0.0)
    return np.clip(frac, 0.0, 1.0)


def simulate_middle_ear(
    pressure_dyne_cm2: np.ndarray,
    sample_rate: float,
    *,
    warned: bool = False,
    input_location: str = "free_field_normal",
    oversample: int = DEFAULT_OVERSAMPLE,
    linear: bool = False,
    mem_enabled: bool = True,
) -> _MiddleEarOutput:
    """
    Integrate the outer + middle ear network and return the stapes drive.

    Args:
        pressure_dyne_cm2: driving pressure in CGS units (Pa x 10).
        sample_rate: rate of `pressure_dyne_cm2`, Hz.
        warned: True for a pre-established MEM contraction.
        input_location: where the pressure is referred to.
        oversample: integration substeps per input sample.
        linear: disable the annular-ligament nonlinearity (used for Tier-1 transfer
            function checks, where linearity must be guaranteed).
        mem_enabled: disable the MEM entirely (also for Tier-1 checks).

    Returns:
        _MiddleEarOutput at the INPUT sample rate.
    """
    net, handles = _build_ear_network(input_location)
    if linear:
        # Replace the nonlinear ligament with an ordinary compliance.
        for n1, n2, C, area, mem in net.nl_caps:
            net.capacitors.append([n1, n2, C, mem])
        net.nl_caps = []

    n_in = len(pressure_dyne_cm2)
    k = max(1, int(oversample))
    h = 1.0 / (sample_rate * k)

    # Band-limited upsampling of the drive so the substeps see a smooth waveform.
    drive = resample_poly(pressure_dyne_cm2, k, 1) if k > 1 else pressure_dyne_cm2.copy()
    n_sub = len(drive)

    solver = _Solver(net, h)
    mem_frac_sub = _mem_contraction(n_sub, sample_rate * k, warned) if mem_enabled \
        else np.zeros(n_sub)

    n_v = len(net.vsources)
    uc = np.zeros(n_sub)
    ped = np.zeros(n_sub)
    lv_idx = handles["l_vestibule"]
    ned_node = net.node("ned")

    src = np.zeros(n_v)
    for i in range(n_sub):
        p = drive[i]
        if input_location == "free_field_normal":
            src[handles["v_double"]] = 2.0 * p
            src[handles["v_single"]] = p
        elif input_location == "ear_canal_entrance":
            src[handles["v_canal"]] = p
        else:
            src[handles["v_drum"]] = p

        f = mem_frac_sub[i]
        mem_k = 1.0 + (MEM_MAG_K[0] - 1.0) * f
        mem_r = 1.0 + (MEM_MAG_R[0] - 1.0) * f
        x = solver.step(src, mem_k, mem_r)
        uc[i] = solver.iL[lv_idx]
        ped[i] = x[ned_node]

    # Decimate back to the input rate by plain selection: the substeps are exact
    # samples of the same continuous solution and the content is already
    # band-limited by the network itself.
    uc_out = uc[::k][:n_in]
    ped_out = ped[::k][:n_in]
    disp_cm = np.cumsum(uc_out) / (sample_rate * ASTAPES)

    # Divergence guard. See MAX_PLAUSIBLE_STAPES_DISPLACEMENT_UM: without this the
    # integrator runs away above ~200 dB and returns a large finite number.
    peak_um = float(np.max(np.abs(disp_cm))) * CM_TO_MICRON if disp_cm.size else 0.0
    if not np.all(np.isfinite(uc_out)) or not math.isfinite(peak_um) \
            or peak_um > MAX_PLAUSIBLE_STAPES_DISPLACEMENT_UM:
        raise ModelDivergedError(
            f"the middle-ear integration diverged: peak stapes displacement came out as "
            f"{peak_um:.3g} um, above the {MAX_PLAUSIBLE_STAPES_DISPLACEMENT_UM:.0g} um bound "
            "beyond which the result is the trapezoidal integrator running away on the "
            "stiffening annular ligament, not the ear. No hazard figure exists for this input."
        )

    return _MiddleEarOutput(
        stapes_volume_velocity=uc_out,
        stapes_displacement_cm=disp_cm,
        eardrum_pressure=ped_out,
        mem_fraction=mem_frac_sub[::k][:n_in],
    )


# ---------------------------------------------------------------------------
# Cochlea: 1-D long-wave WKB transmission line
# ---------------------------------------------------------------------------

def cochlear_transfer(
    frequencies_Hz: np.ndarray,
    positions_cm: np.ndarray,
    *,
    extra_taper: bool = False,
    cf_alignment: str = "as_written",
    gain_factor: float = COCHLEAR_GAIN_FACTOR,
    n_integration_points: int = 701,
) -> np.ndarray:
    """
    Basilar-membrane displacement per unit stapes volume velocity, microns/(cm^3/s).

    ---- CATEGORY (c): the WKB equations, the taper term and the boundary handling
    are written down NOWHERE in the released material. What follows is the standard
    1-D long-wave formulation, chosen because man.coe's graded-parameter set is
    complete and mutually consistent with it: D1 = 1/(2/Dc - 1/D2), D2 = D3 = D4/2
    and D5 = D4 all hold exactly. That is circumstantial evidence, not a source. ----

        Z(x,w) = R(x) + j*w*M(x) + S(x)/(j*w)          per unit area
        k^2    = j*w*rho*B(x) / (A(x)*Z(x,w))          long-wave wavenumber
        p(x)   = p(0)*sqrt(k(0)/k(x))*exp(-int_0^x k)  WKB
        p(0)   = j*w*rho*Uc / (A(0)*k(0))              stapes boundary
        D(x)   = p(x) / (j*w*Z(x))                     BM displacement

    A(x) is man.coe's Ao*exp(-x/D5), which is already the SERIES-effective scala
    area Sv*St/(Sv+St); that is why the momentum equation carries no factor of two.

    Args:
        frequencies_Hz: frequencies at which to evaluate (>= 0).
        positions_cm: the 23 evaluation positions.
        extra_taper: also multiply the amplitude by exp(-WkbScalaBMwidthDecay*x).
            Off by default -- see DECLARED_ASSUMPTIONS['wkb_taper'].
        cf_alignment: 'fo' forces omega0(x) = 2*pi*Fo*exp(-x/Dc) by scaling the
            mass; 'as_written' uses So/Mo verbatim, which places the resonance
            3.30x above the frequency that labels each band.
        gain_factor: CochlearGainFactor.
        n_integration_points: grid for the cumulative wavenumber integral.

    Returns:
        Complex array of shape (len(frequencies_Hz), len(positions_cm)).
    """
    w = 2.0 * np.pi * np.asarray(frequencies_Hz, dtype=np.float64)
    x_grid = np.linspace(0.0, XB_APEX, n_integration_points)

    s_x = SO * np.exp(-x_grid / D1)
    m_x = MO * np.exp(x_grid / D2)
    r_x = RVO * np.exp(x_grid / D3)
    b_x = BO * np.exp(x_grid / D4)
    a_x = AO * np.exp(-x_grid / D5)

    if cf_alignment == "fo":
        # ---- CATEGORY (c): sqrt(So/Mo)/2pi = 66.1 kHz, but man.coe's own comment
        # calls Fo = 20 kHz the "max resonant freq at base of BM" and F(x) =
        # Fo*exp(-x/Dc) labels all 23 reference bands exactly. The factor of 3.30
        # is unexplained. Scaling the mass by 3.30^2 = 10.9 makes the mechanics
        # agree with the labelling; 'as_written' leaves them inconsistent. ----
        mass_scale = (math.sqrt(SO / MO) / (2.0 * np.pi * FO)) ** 2
        m_x = m_x * mass_scale
        r_x = r_x * math.sqrt(mass_scale)   # keep the loss constant Rvo/sqrt(S*M) unchanged
    elif cf_alignment != "as_written":
        raise ValueError(f"unknown cf_alignment {cf_alignment!r}")

    with np.errstate(divide="ignore", invalid="ignore"):
        jw = 1j * w[:, None]
        z = r_x[None, :] + jw * m_x[None, :] + s_x[None, :] / jw
        k2 = jw * RHO_PERILYMPH * b_x[None, :] / (a_x[None, :] * z)
        k = np.sqrt(k2)
        # Principal branch already gives Re(k) >= 0; force the forward-travelling
        # root where Re(k) is numerically zero.
        k = np.where(np.real(k) < 0, -k, k)

        # Cumulative integral of k along x (trapezoidal).
        dx = x_grid[1] - x_grid[0]
        cum = np.zeros_like(k)
        cum[:, 1:] = np.cumsum(0.5 * (k[:, 1:] + k[:, :-1]) * dx, axis=1)

        k0 = k[:, :1]
        a0 = a_x[0]
        p0 = jw * RHO_PERILYMPH / (a0 * k0)          # p(0) per unit Uc
        # WKB amplitude for (A p')' = A k^2 p is (A*k)^(-1/2), NOT k^(-1/2): the
        # scala area A(x) = Ao*exp(-x/D5) is graded, so it belongs in the factor.
        amp = np.sqrt((a0 * k0) / (a_x[None, :] * k))
        p_x = p0 * amp * np.exp(-cum)
        disp = p_x / (jw * z)                        # cm per (cm^3/s)

    if extra_taper:
        # ---- CATEGORY (c): an ADDITIONAL bare exponential taper, off by default.
        # WkbScalaBMwidthDecay is documented only as "cm^-1 decay length based on
        # tapering factors"; the amplitude factor above already carries the A(x)
        # and B(x) grading, so this looks like a double count. Over the 3.5 cm
        # cochlea it is 27 dB, which is not a small correction. ----
        disp = disp * np.exp(-WKB_SCALA_BM_WIDTH_DECAY * x_grid[None, :])

    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    disp = disp * gain_factor * CM_TO_MICRON

    # Sample at the 23 evaluation positions (linear interpolation on the fine grid).
    idx = np.searchsorted(x_grid, positions_cm).clip(1, len(x_grid) - 1)
    x0 = x_grid[idx - 1]
    x1 = x_grid[idx]
    frac = (np.asarray(positions_cm) - x0) / (x1 - x0)
    out = disp[:, idx - 1] * (1.0 - frac)[None, :] + disp[:, idx] * frac[None, :]
    out[w == 0.0, :] = 0.0
    return out


def basilar_membrane_displacement(
    stapes_volume_velocity: np.ndarray,
    sample_rate: float,
    *,
    extra_taper: bool = False,
    cf_alignment: str = "as_written",
    gain_factor: float = COCHLEAR_GAIN_FACTOR,
    bm_sign: float = 1.0,
) -> np.ndarray:
    """
    BM displacement in microns at the 23 locations, from the stapes volume velocity.

    The cochlear stage is linear (nothing in the release documents a nonlinearity
    there beyond the undocumented "Ca Cochlear Amplifier Gain"), so it is applied
    as a frequency-domain filter bank. The signal is zero-padded to twice its
    length to keep the impulse responses from wrapping.

    Returns:
        Array of shape (23, n_samples), microns.
    """
    n = len(stapes_volume_velocity)
    n_fft = 1 << int(math.ceil(math.log2(max(2 * n, 16))))
    spec = np.fft.rfft(stapes_volume_velocity, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    h = cochlear_transfer(freqs, band_positions_cm(), extra_taper=extra_taper,
                          cf_alignment=cf_alignment, gain_factor=gain_factor)
    out = np.fft.irfft(spec[:, None] * h, n_fft, axis=0)[:n, :]
    # ---- CATEGORY (c): the sign of "upward" relative to input pressure polarity
    # is not stated in any source, and only upward peaks are accumulated. ----
    return (bm_sign * out).T


# ---------------------------------------------------------------------------
# Hazard accumulation
# ---------------------------------------------------------------------------

def accumulate_hazard(
    bm_displacement_um: np.ndarray,
    *,
    rule: str = "excursion",
    threshold_um: float = DAMAGE_THRESHOLD,
) -> np.ndarray:
    """
    Auditory hazard units per location: sum of squared upward peak displacements.

    MIL-STD-1474E Annex A, B-A.1.1: "squaring the peak amplitude of each upward
    displacement of the basilar membrane (in microns) and summing them ...
    ARU = sum (D^2) where D is the upward basilar membrane displacement".

    ---- CATEGORY (c): "each upward displacement" is ambiguous. Three readings are
    implemented; 'excursion' is the default because "as each peak passes" reads
    most naturally as one count per positive-going excursion. For a ringing
    response the three differ by tens of percent. ----

    Args:
        bm_displacement_um: (n_bands, n_samples), microns.
        rule: 'excursion' (one peak per positive excursion), 'local_max' (every
            local maximum above the threshold), or 'all_positive' (every sample,
            an upper bound used only for diagnostics).
        threshold_um: man.coe DamageThreshold, 0 -- every upward peak counts.

    Returns:
        (n_bands,) hazard in AHU.
    """
    d = np.asarray(bm_displacement_um, dtype=np.float64)
    n_bands = d.shape[0]
    out = np.zeros(n_bands)

    for j in range(n_bands):
        x = d[j]
        if rule == "all_positive":
            pos = x[x > threshold_um]
            out[j] = float(np.sum(pos * pos))
            continue
        if rule == "local_max":
            if len(x) < 3:
                continue
            is_max = (x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > threshold_um)
            peaks = x[1:-1][is_max]
            out[j] = float(np.sum(peaks * peaks))
            continue
        if rule != "excursion":
            raise ValueError(f"unknown peak rule {rule!r}")

        above = x > threshold_um
        if not above.any():
            continue
        # Boundaries of the runs of consecutive above-threshold samples; one peak
        # is taken per run, at its maximum.
        edges = np.flatnonzero(np.diff(above.astype(np.int8)) != 0) + 1
        starts = np.concatenate(([0], edges))
        keep = above[starts]
        if not keep.any():
            continue
        run_max = np.maximum.reduceat(x, starts)[keep]
        out[j] = float(np.sum(run_max * run_max))
    return out


def total_aru(band_ahu: np.ndarray) -> float:
    """
    Total ARU = the MAXIMUM over the 23 locations, not the sum.

    MIL-STD-1474E B-A.2.1: "the location with the largest value is reported."
    Verified: max(160F.HAZ) = 390.99399 (band 9) and the 160F.AHA header reports
    391.0 A.R.U. The sum, 4031.24, matches nothing in the release.
    """
    return float(np.max(band_ahu)) if len(band_ahu) else 0.0


def allowed_exposures(aru: float, *, occupational: bool = False) -> float:
    """
    Permitted number of exposures: N = 500/ARU (occasional) or 200/ARU (>= 2/week).

    Verified against the reference header: 500/390.99 = 1.279 -> "1.3";
    500/2237 = 0.2235 -> "0.2".
    """
    limit = ARU_LIMIT_OCCUPATIONAL if occupational else ARU_LIMIT_OCCASIONAL
    if aru <= 0.0:
        return float("inf")
    return limit / aru


def threshold_shift_dB(aru: float) -> float:
    """
    Compound threshold shift: CTS = 26.6*ln(ARU) - 140.1 (MIL-STD-1474E B-A.1.2).

    Self-consistent at the limit: 500 ARU -> 25.2 dB, matching ARL's statement that
    "500 ARUs is about 25 dB of shift". Returns 0 for ARU at or below the level
    where the relation goes negative.
    """
    if aru <= 0.0:
        return 0.0
    return max(0.0, CTS_SLOPE * math.log(aru) - CTS_INTERCEPT)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class AhaahResult:
    """
    Result of an AHAAH hazard computation.

    IN THIS BUILD EVERY INSTANCE HAS `valid = False`. `status` is "not_validated":
    the implementation does not reproduce the ARL reference case, so per
    docs/AHAAH-SPEC.md section 11.5 no ARU number is emitted and every hazard field
    is NaN. `notes` explains which it is -- a problem with the recording (status
    "refused") or the model's own standing (status "not_validated").

    The type keeps its numeric fields so that the day the model does validate,
    callers written against it need no change; until then they are NaN and `valid`
    is the flag to branch on.
    """
    # Hazard
    band_ahu: np.ndarray                     # (23,) auditory hazard units per location
    band_frequencies_Hz: np.ndarray          # (23,) characteristic frequency
    band_positions_cm: np.ndarray            # (23,) distance from the cochlear base
    total_ARU: float                         # max over the 23 locations
    peak_band_index: int                     # 1-based, as in .HAZ
    allowed_exposures: float                 # 500 / ARU
    allowed_exposures_occupational: float    # 200 / ARU
    threshold_shift_dB: float                # compound threshold shift
    predicted_PTS_dB: float                  # ~0.6 * CTS once the limit is exceeded

    # Configuration
    warned: bool
    input_location: str
    percentile_95: bool
    sample_rate_in_Hz: float
    working_rate_Hz: float
    resampled: bool
    resample_ratio: str
    oversample: int

    # Diagnostics
    peak_pressure_dB: float = float("nan")
    peak_stapes_displacement_um: float = float("nan")
    peak_bm_displacement_um: float = float("nan")
    mem_fraction_at_peak: float = float("nan")

    # Standing
    valid: bool = True
    status: str = VALIDATION_STATUS
    notes: List[str] = field(default_factory=list)
    assumptions: Tuple[Assumption, ...] = DECLARED_ASSUMPTIONS
    category_c_count: int = CATEGORY_C_COUNT
    man_coe_md5: str = MAN_COE_MD5

    @property
    def headline_label(self) -> str:
        """The only label this module permits on the number, given its status."""
        if not self.valid:
            return "AHAAH unavailable"
        if self.status == "validated":
            return "AHAAH ARU (MIL-STD-1474E), validated against the ARL 160F reference"
        if self.status == "fitted":
            return "AHAAH-style ARU, FITTED to the single ARL reference case - not validated"
        # Unreachable in this build: status "not_validated" never comes with valid=True.
        return "AHAAH-style figure of unknown standing - not a MIL-STD-1474E result"

    def to_dict(self) -> Dict:
        """Convert to a JSON-serialisable dictionary, provenance included."""
        return {
            "valid": self.valid,
            "status": self.status,
            "label": self.headline_label,
            "warned": self.warned,
            "notes": list(self.notes),
            "total_ARU": None if math.isnan(self.total_ARU) else round(self.total_ARU, 3),
            "peak_band_index": self.peak_band_index,
            "allowed_exposures": (None if math.isnan(self.allowed_exposures)
                                  else round(self.allowed_exposures, 1)),
            "allowed_exposures_occupational": (
                None if math.isnan(self.allowed_exposures_occupational)
                else round(self.allowed_exposures_occupational, 1)),
            "threshold_shift_dB": (None if math.isnan(self.threshold_shift_dB)
                                   else round(self.threshold_shift_dB, 1)),
            "predicted_PTS_dB": (None if math.isnan(self.predicted_PTS_dB)
                                 else round(self.predicted_PTS_dB, 1)),
            "band_frequencies_Hz": [round(float(f), 1) for f in self.band_frequencies_Hz],
            "band_positions_cm": [round(float(x), 3) for x in self.band_positions_cm],
            "band_AHU": [None if math.isnan(v) else round(float(v), 5) for v in self.band_ahu],
            "input_location": self.input_location,
            "percentile_95": self.percentile_95,
            "sample_rate_in_Hz": self.sample_rate_in_Hz,
            "working_rate_Hz": self.working_rate_Hz,
            "resampled": self.resampled,
            "resample_ratio": self.resample_ratio,
            "oversample": self.oversample,
            "peak_pressure_dB": (None if math.isnan(self.peak_pressure_dB)
                                 else round(self.peak_pressure_dB, 3)),
            "peak_stapes_displacement_um": (
                None if math.isnan(self.peak_stapes_displacement_um)
                else round(self.peak_stapes_displacement_um, 3)),
            "peak_bm_displacement_um": (
                None if math.isnan(self.peak_bm_displacement_um)
                else round(self.peak_bm_displacement_um, 5)),
            "man_coe_md5": self.man_coe_md5,
            "category_c_count": self.category_c_count,
            "assumptions": [
                {"key": a.key, "category": a.category, "choice": a.choice,
                 "rationale": a.rationale}
                for a in self.assumptions
            ],
            "scientific_standing": SCIENTIFIC_STANDING_NOTE,
        }


def _invalid_result(reason: str, *, warned: bool, input_location: str,
                    sample_rate: float, percentile_95: bool,
                    extra_notes: Sequence[str] = (),
                    status: str = "refused",
                    peak_pressure_dB: float = float("nan")) -> AhaahResult:
    """Build a refusal: no number at all, and the reason attached."""
    nan23 = np.full(XBM_NO, np.nan)
    return AhaahResult(
        band_ahu=nan23,
        band_frequencies_Hz=band_frequencies_Hz(),
        band_positions_cm=band_positions_cm(),
        total_ARU=float("nan"),
        peak_band_index=0,
        allowed_exposures=float("nan"),
        allowed_exposures_occupational=float("nan"),
        threshold_shift_dB=float("nan"),
        predicted_PTS_dB=float("nan"),
        warned=warned,
        input_location=input_location,
        percentile_95=percentile_95,
        sample_rate_in_Hz=float(sample_rate),
        working_rate_Hz=AHAAH_WORKING_RATE_HZ,
        resampled=False,
        resample_ratio="1/1",
        oversample=0,
        peak_pressure_dB=peak_pressure_dB,
        valid=False,
        status=status,
        notes=[reason, *extra_notes, SCIENTIFIC_STANDING_NOTE],
    )


def _detect_clipping(pressure_Pa: np.ndarray, *, run_samples: int = 3) -> bool:
    """
    Flat-topped peaks: `run_samples` consecutive samples within 0.01% of the extreme.

    A calibrated pressure history has no full-scale rail to test against, so
    clipping shows up as a plateau at the waveform's own extreme.
    """
    if len(pressure_Pa) < run_samples:
        return False
    for extreme in (np.max(pressure_Pa), np.min(pressure_Pa)):
        if extreme == 0.0:
            continue
        at_rail = np.abs(pressure_Pa - extreme) <= abs(extreme) * 1e-4
        if not at_rail.any():
            continue
        # longest run of True
        flags = at_rail.astype(np.int8)
        idx = np.flatnonzero(np.diff(np.concatenate(([0], flags, [0]))) != 0)
        runs = idx[1::2] - idx[0::2]
        if runs.size and runs.max() >= run_samples:
            return True
    return False


# ---------------------------------------------------------------------------
# The public entry points
# ---------------------------------------------------------------------------

@dataclass
class _PreparedInput:
    """A waveform that has passed the admissibility gates and reached the network."""
    drive_dyne_cm2: np.ndarray
    fs_model: float
    circuit_entry: str
    notes: List[str]
    peak_dB: float
    resampled: bool
    resample_ratio: str


def _prepare_input(
    pressure_Pa: np.ndarray,
    sample_rate: float,
    *,
    warned: bool = False,
    input_location: str = "free_field_normal",
    percentile_95: bool = True,
    calibrated: bool = True,
    outer_ear: str = "shaw",
    allow_low_rate: bool = False,
) -> "AhaahResult | _PreparedInput":
    """
    Apply every input-admissibility gate and produce the CGS drive waveform.

    Returns an `AhaahResult` refusal (valid=False, status "refused") if the input
    cannot support a hazard computation at all, otherwise a `_PreparedInput`.

    The gates, and why each one exists:
      * uncalibrated -- ARU is an absolute-level metric and is meaningless in dB re FS;
      * sample rate below 96 kHz -- band 1 is 11.76 kHz and the ARL reference case
        is 125 kHz, so a lower rate cannot resolve what the model integrates;
      * clipped -- the peak that dominates the squared-displacement sum is missing;
      * peak below 130 dB -- ARL states the model does not apply there;
      * non-finite, all-zero, or fewer than 16 samples.
    """
    if input_location not in VALID_INPUT_LOCATIONS:
        raise ValueError(
            f"input_location must be one of {VALID_INPUT_LOCATIONS}, got {input_location!r}"
        )
    if outer_ear not in ("shaw", "circuit"):
        raise ValueError(f"outer_ear must be 'shaw' or 'circuit', got {outer_ear!r}")

    p = np.asarray(pressure_Pa, dtype=np.float64).ravel()
    fs = float(sample_rate)

    def refuse(reason: str, extra: Sequence[str] = ()) -> AhaahResult:
        return _invalid_result(reason, warned=warned, input_location=input_location,
                               sample_rate=fs, percentile_95=percentile_95,
                               extra_notes=extra)

    # ---- refusal gates ----------------------------------------------------
    if not calibrated:
        return refuse(
            "REFUSED: the input is uncalibrated. ARU is an absolute-level metric derived "
            "from pressure in Pascals; in dB re FS it has no meaning. Calibrate the "
            "recording chain and re-run."
        )
    if p.size < 16:
        return refuse(f"REFUSED: only {p.size} samples supplied; too short to analyse.")
    if not np.all(np.isfinite(p)):
        return refuse("REFUSED: the input contains non-finite samples (NaN or Inf).")

    if input_location == "free_field_grazing" and outer_ear == "circuit":
        return refuse(
            "REFUSED: grazing incidence needs the tabulated Shaw HRTF (outer_ear='shaw'). "
            "The lumped 2P/P diffraction circuit has no azimuth dependence at all -- it is "
            "the normal-incidence (side of head) path only."
        )

    low_rate_note: Optional[str] = None
    if fs < MIN_SAMPLE_RATE_HZ:
        low_rate_note = (
            f"Sample rate {fs:.0f} Hz is below the {MIN_SAMPLE_RATE_HZ:.0f} Hz minimum. "
            "AHAAH resolves cochlear mechanics to ~20 kHz (band 1 is 11.76 kHz) and the ARL "
            "reference case is sampled at 125 kHz; below 96 kHz the blast rise and the "
            "high-frequency bands are not represented. Upsampling cannot restore them."
        )
        if not allow_low_rate or fs < MIN_SAMPLE_RATE_HZ / 2:
            return refuse("REFUSED: " + low_rate_note)

    clipped = _detect_clipping(p)
    if clipped:
        return refuse(
            "REFUSED: the waveform is clipped (a plateau at its own extreme). The ARU is "
            "dominated by the squared peak displacement, so a truncated peak understates "
            "the hazard by an unknown amount."
        )

    peak_pa = float(np.max(np.abs(p)))
    if peak_pa <= 0.0:
        return refuse("REFUSED: the waveform is all zeros.")
    peak_dB = 20.0 * math.log10(peak_pa / P_REF)
    if peak_dB < MIN_PEAK_dB:
        return refuse(
            f"REFUSED: peak level {peak_dB:.1f} dB is below {MIN_PEAK_dB:.0f} dB, where ARL "
            "states the model does not apply. The middle ear is linear there and the ARU is "
            "not a meaningful hazard figure. Use the A-weighted energy metrics instead."
        )

    notes: List[str] = [NOT_VALIDATED_NOTE, SCIENTIFIC_STANDING_NOTE]
    if low_rate_note is not None:
        notes.insert(0, "WARNING: " + low_rate_note)

    # ---- resample to the model's working rate -----------------------------
    # The ARL reference case is 125 kHz and the network is integrated at that rate
    # (times `oversample`), so an input at any other rate is brought to it here and
    # the ratio is recorded in the result.
    resampled = False
    ratio_str = "1/1"
    fs_model = fs
    x = p
    if abs(fs - AHAAH_WORKING_RATE_HZ) > 1e-6:
        frac = Fraction(AHAAH_WORKING_RATE_HZ / fs).limit_denominator(2000)
        up, down = frac.numerator, frac.denominator
        x = resample_poly(p, up, down)
        resampled = True
        ratio_str = f"{up}/{down}"
        fs_model = fs * up / down
        notes.append(
            f"Input resampled {fs:.0f} Hz -> {fs_model:.1f} Hz (polyphase {up}/{down}) to "
            f"reach the model working rate of {AHAAH_WORKING_RATE_HZ:.0f} Hz."
        )

    # ---- 95th-percentile susceptibility -----------------------------------
    if percentile_95:
        # ARL: "the model achieves the prediction for the 95 percentile ear by
        # artificially raising the SPL on the test impulse by 10 dB". Applied to the
        # waveform, before the (nonlinear) middle ear, because the placement matters.
        x = x * SUSCEPTIBILITY_95_GAIN
        notes.append(
            f"+{SUSCEPTIBILITY_95_dB:.0f} dB (x{SUSCEPTIBILITY_95_GAIN:.5f}) susceptibility "
            "adjustment applied to the waveform: the 95th-percentile (most susceptible) ear, "
            "as MIL-STD-1474E requires."
        )
    else:
        notes.append(
            "MEDIAN-EAR result: the +10 dB susceptibility adjustment was NOT applied. This is "
            "NOT the MIL-STD-1474E figure and is typically 3-5x lower in ARU."
        )

    # ---- outer ear ---------------------------------------------------------
    # Under the default 'shaw' route the free-field waveform is taken to the
    # blocked ear-canal entrance with the tabulated azimuth transfer, and the
    # circuit is entered at the canal; under 'circuit' the waveform drives the
    # lumped 2P/P diffraction sources instead. See the assumption of the same name.
    circuit_entry = input_location
    if input_location in ("free_field_normal", "free_field_grazing"):
        if outer_ear == "shaw":
            row = (SHAW_FF_TO_CANAL_dB_AZ90 if input_location == "free_field_normal"
                   else SHAW_FF_TO_CANAL_dB_AZ0)
            x = apply_outer_ear_hrtf(x, fs_model, row)
            circuit_entry = "ear_canal_entrance"
            notes.append(
                "Outer ear: Shaw free-field to ear-canal-entrance transfer "
                f"({'90' if input_location == 'free_field_normal' else '0'} deg azimuth, "
                "Dat/F11_12D.DAT), minimum phase; the ear canal and eardrum come from the "
                "circuit."
            )
        else:
            circuit_entry = "free_field_normal"
            notes.append(
                "Outer ear: lumped 2P/P head-diffraction sources plus concha and canal "
                "lines. This route overstates the eardrum pressure above ~9 kHz by 13-39 dB "
                "against ARL's own FFEDM90.DAT."
            )

    # ---- CGS ---------------------------------------------------------------
    # 1 Pa = 10 dyne/cm^2. The whole network is a CGS acoustic-impedance analogue
    # (man.coe's own unit comments: cm^5/dyne, dyne-sec/cm^5, g/cm^4). This is the
    # ONLY place the conversion happens.
    drive = x / PA_PER_DYNE_CM2

    return _PreparedInput(
        drive_dyne_cm2=drive,
        fs_model=float(fs_model),
        circuit_entry=circuit_entry,
        notes=notes,
        peak_dB=peak_dB,
        resampled=resampled,
        resample_ratio=ratio_str,
    )


# ---------------------------------------------------------------------------
# The research path -- the only way to the numbers, and they are not ARU
# ---------------------------------------------------------------------------

@dataclass
class UnvalidatedModelRun:
    """
    Output of the unvalidated ear model. NOT AN AHAAH RESULT.

    There is deliberately no field called ARU here. The quantity computed is the
    sum, over upward basilar-membrane displacement peaks, of the squared peak
    amplitude in microns -- so its unit is microns squared. In a validated AHAAH
    that quantity IS the Auditory Risk Unit; in this implementation it is 51 %
    low against the one published reference case and disagrees with the published
    band table in 20 of 23 bands, so it is not one. See VALIDATION_STATUS.

    This type exists for development and for the validation test. Nothing
    customer-facing may consume it.
    """
    band_sum_sq_displacement_um2: np.ndarray   # (23,)
    max_band_sum_sq_um2: float                 # max over the 23 locations
    peak_band_index: int                       # 1-based, as in .HAZ
    band_frequencies_Hz: np.ndarray
    band_positions_cm: np.ndarray

    warned: bool
    input_location: str
    percentile_95: bool
    sample_rate_in_Hz: float
    working_rate_Hz: float
    resampled: bool
    resample_ratio: str
    oversample: int

    peak_pressure_dB: float
    peak_stapes_displacement_um: float
    peak_bm_displacement_um: float
    mem_fraction_at_peak: float

    notes: List[str] = field(default_factory=list)
    assumptions: Tuple[Assumption, ...] = DECLARED_ASSUMPTIONS
    category_c_count: int = CATEGORY_C_COUNT
    man_coe_md5: str = MAN_COE_MD5
    status: str = VALIDATION_STATUS
    label: str = NOT_AN_ARU_LABEL


def run_unvalidated_model(
    pressure_Pa: np.ndarray,
    sample_rate: float,
    *,
    acknowledge_not_validated: bool = False,
    warned: bool = False,
    input_location: str = "free_field_normal",
    percentile_95: bool = True,
    calibrated: bool = True,
    outer_ear: str = "shaw",
    oversample: int = DEFAULT_OVERSAMPLE,
    peak_rule: str = "excursion",
    cf_alignment: str = "as_written",
    cochlear_gain_factor: float = COCHLEAR_GAIN_FACTOR,
    extra_wkb_taper: bool = False,
    bm_sign: float = 1.0,
    allow_low_rate: bool = False,
) -> UnvalidatedModelRun:
    """
    Run the ear model and return the raw microns-squared hazard index. NOT AN ARU.

    Development and validation only. `compute_ahaah` is the public entry point and
    it deliberately returns no number; this function is what the validation test
    drives, and what a future attempt to fix the model works against.

    Args:
        pressure_Pa: calibrated pressure history, Pascals.
        sample_rate: sample rate of `pressure_Pa`, Hz.
        acknowledge_not_validated: must be True. It exists so that the call site
            records, in the source, that the caller knows this is not an ARU.
        warned: True models a middle-ear-muscle contraction already in place.
        input_location: one of VALID_INPUT_LOCATIONS.
        percentile_95: apply ARL's +10 dB susceptibility adjustment (default True).
        calibrated: False raises, as it would refuse in compute_ahaah.
        outer_ear: 'shaw' (default) applies the tabulated Shaw free-field to
            canal-entrance transfer and enters the circuit at the canal; 'circuit'
            drives the lumped 2P/P diffraction sources and the concha line instead.
        oversample: network integration substeps per working sample.
        peak_rule: see accumulate_hazard.
        cf_alignment: see cochlear_transfer.
        cochlear_gain_factor: man.coe CochlearGainFactor.
        extra_wkb_taper: apply the undocumented exp(-WkbScalaBMwidthDecay*x) term.
        bm_sign: +1 or -1; which polarity of BM displacement is "upward".
        allow_low_rate: proceed below MIN_SAMPLE_RATE_HZ anyway.

    Returns:
        UnvalidatedModelRun.

    Raises:
        ValueError: if `acknowledge_not_validated` is not True, or the input fails
            an admissibility gate (the refusal text is the message).
        ModelDivergedError: if the nonlinear integration went unstable.
    """
    if acknowledge_not_validated is not True:
        raise ValueError(
            "run_unvalidated_model() requires acknowledge_not_validated=True. This function "
            "does NOT return an AHAAH ARU: the implementation fails its reference case (see "
            "ahaah.VALIDATION_STATUS and docs/AHAAH.md) and the number it returns is microns "
            "squared of basilar-membrane displacement, for development only."
        )

    prepared = _prepare_input(
        pressure_Pa, sample_rate, warned=warned, input_location=input_location,
        percentile_95=percentile_95, calibrated=calibrated, outer_ear=outer_ear,
        allow_low_rate=allow_low_rate,
    )
    if isinstance(prepared, AhaahResult):
        raise ValueError(prepared.notes[0])

    notes = list(prepared.notes)
    drive = prepared.drive_dyne_cm2
    fs_model = prepared.fs_model

    me = simulate_middle_ear(
        drive, fs_model, warned=warned, input_location=prepared.circuit_entry,
        oversample=oversample,
    )

    bm = basilar_membrane_displacement(
        me.stapes_volume_velocity, fs_model,
        extra_taper=bool(extra_wkb_taper), cf_alignment=cf_alignment,
        gain_factor=cochlear_gain_factor, bm_sign=bm_sign,
    )

    band_um2 = accumulate_hazard(bm, rule=peak_rule)
    max_um2 = total_aru(band_um2)
    peak_band = int(np.argmax(band_um2)) + 1 if band_um2.size else 0

    stapes_um = float(np.max(np.abs(me.stapes_displacement_cm))) * CM_TO_MICRON
    peak_idx = int(np.argmax(np.abs(drive)))
    mem_at_peak = float(me.mem_fraction[peak_idx]) if me.mem_fraction.size else float("nan")

    if not (10.0 <= stapes_um <= 60.0):
        notes.append(
            f"Peak stapes displacement {stapes_um:.2f} um is outside the 10-60 um that ARL's "
            "'a few tens of microns' implies for a high-level impulse. The annular-ligament "
            "nonlinearity is an inference (see assumptions) and this is the check that would "
            "have caught it being wrong."
        )

    return UnvalidatedModelRun(
        band_sum_sq_displacement_um2=band_um2,
        max_band_sum_sq_um2=max_um2,
        peak_band_index=peak_band,
        band_frequencies_Hz=band_frequencies_Hz(),
        band_positions_cm=band_positions_cm(),
        warned=warned,
        input_location=input_location,
        percentile_95=percentile_95,
        sample_rate_in_Hz=float(sample_rate),
        working_rate_Hz=fs_model,
        resampled=prepared.resampled,
        resample_ratio=prepared.resample_ratio,
        oversample=int(oversample),
        peak_pressure_dB=prepared.peak_dB,
        peak_stapes_displacement_um=stapes_um,
        peak_bm_displacement_um=float(np.max(bm)) if bm.size else float("nan"),
        mem_fraction_at_peak=mem_at_peak,
        notes=notes,
    )


def run_unvalidated_model_both(
    pressure_Pa: np.ndarray,
    sample_rate: float,
    **kwargs,
) -> Tuple[UnvalidatedModelRun, UnvalidatedModelRun]:
    """Run both reflex conditions, unwarned first. NOT ARU. See run_unvalidated_model."""
    kwargs.pop("warned", None)
    unwarned = run_unvalidated_model(pressure_Pa, sample_rate, warned=False, **kwargs)
    warned = run_unvalidated_model(pressure_Pa, sample_rate, warned=True, **kwargs)
    return unwarned, warned


# ---------------------------------------------------------------------------
# The public entry points -- which, in this build, emit no number
# ---------------------------------------------------------------------------

def compute_ahaah(
    pressure_Pa: np.ndarray,
    sample_rate: float,
    *,
    warned: bool = False,
    input_location: str = "free_field_normal",
    percentile_95: bool = True,
    calibrated: bool = True,
    outer_ear: str = "shaw",
    allow_low_rate: bool = False,
    **_unused,
) -> AhaahResult:
    """
    The public AHAAH entry point. In this build it always returns `valid=False`.

    VALIDATION_STATUS is "not_validated": the implementation does not reproduce the
    ARL 160F reference case, so under docs/AHAAH-SPEC.md section 11.5 it must not
    emit an Auditory Risk Unit. Every hazard field of the returned AhaahResult is
    NaN and `notes[0]` explains why.

    The input-admissibility gates still run FIRST, so that a caller whose recording
    is uncalibrated, undersampled, clipped or too quiet is told about the recording
    (status "refused") rather than only about the model. If the recording is fine,
    the refusal is the model's own (status "not_validated").

    Args:
        pressure_Pa: calibrated pressure history, Pascals.
        sample_rate: sample rate of `pressure_Pa`, Hz.
        warned: recorded on the result; the model is not run either way.
        input_location: one of VALID_INPUT_LOCATIONS.
        percentile_95: recorded on the result.
        calibrated: False produces the uncalibrated refusal.
        outer_ear: 'shaw' or 'circuit'.
        allow_low_rate: relax the 96 kHz gate down to 48 kHz.
        **_unused: modelling switches accepted and ignored, so that callers written
            against the research path do not break here.

    Returns:
        AhaahResult with valid=False. Never a number.
    """
    prepared = _prepare_input(
        pressure_Pa, sample_rate, warned=warned, input_location=input_location,
        percentile_95=percentile_95, calibrated=calibrated, outer_ear=outer_ear,
        allow_low_rate=allow_low_rate,
    )
    if isinstance(prepared, AhaahResult):
        return prepared          # the recording is the problem; say so first

    return _invalid_result(
        NOT_VALIDATED_NOTE,
        warned=warned,
        input_location=input_location,
        sample_rate=float(sample_rate),
        percentile_95=percentile_95,
        status=VALIDATION_STATUS,
        peak_pressure_dB=prepared.peak_dB,
        extra_notes=[
            "The recording itself passed every admissibility check "
            f"(peak {prepared.peak_dB:.1f} dB, {float(sample_rate):.0f} Hz). What is missing "
            "is a validated model, not a better measurement.",
            "MIL-STD-1474E approves TWO impulse-noise metrics. The other one -- A-weighted "
            "energy / LAeq8hr -- is computed exactly by SASA and is unaffected by this.",
        ],
    )


def compute_ahaah_both(
    pressure_Pa: np.ndarray,
    sample_rate: float,
    **kwargs,
) -> Tuple[AhaahResult, AhaahResult]:
    """
    Compute BOTH the unwarned and the warned case, in that order.

    Always both. The warned/unwarned switch is the single largest disagreement in
    the AHAAH literature -- for the ARL reference case it is a factor of 5.7 -- and
    2016-2017 work found the assumed protective reflex absent in 18 of 19 subjects
    firing M4 rifles. The UNWARNED figure (the first element, no reflex assumed) is
    the conservative case and is the one to lead with.

    In this build both elements are refusals: see compute_ahaah.

    Args:
        pressure_Pa: calibrated pressure history, Pascals.
        sample_rate: sample rate, Hz.
        **kwargs: passed to compute_ahaah, except `warned`.

    Returns:
        (unwarned, warned)
    """
    kwargs.pop("warned", None)
    unwarned = compute_ahaah(pressure_Pa, sample_rate, warned=False, **kwargs)
    warned = compute_ahaah(pressure_Pa, sample_rate, warned=True, **kwargs)
    return unwarned, warned


def format_ahaah_summary(unwarned: AhaahResult, warned: Optional[AhaahResult] = None,
                         *, indent: str = "  ") -> str:
    """
    Human-readable summary, unwarned first, with the standing note attached.

    In this build every result is a refusal, so this prints the reason and no
    number. The valid-result branch is kept for the day the model validates.
    """
    lines: List[str] = []
    lines.append(f"{indent}AHAAH auditory hazard  [{unwarned.headline_label}]")
    if not unwarned.valid:
        lines.append(f"{indent}  status: {unwarned.status}")
        for n in unwarned.notes:
            lines.append(f"{indent}  ! {n}")
        return "\n".join(lines)

    def block(r: AhaahResult, label: str) -> None:
        lines.append(f"{indent}  {label}:")
        lines.append(f"{indent}    Total ARU:            {r.total_ARU:.1f}"
                     f"   (band {r.peak_band_index}, "
                     f"{r.band_frequencies_Hz[r.peak_band_index-1]/1000:.2f} kHz)")
        lines.append(f"{indent}    Allowed exposures:    {r.allowed_exposures:.1f} "
                     f"(occasional) / {r.allowed_exposures_occupational:.1f} (occupational)")
        lines.append(f"{indent}    Threshold shift:      {r.threshold_shift_dB:.1f} dB "
                     f"(predicted PTS {r.predicted_PTS_dB:.1f} dB)")
        lines.append(f"{indent}    Peak stapes displ.:   "
                     f"{r.peak_stapes_displacement_um:.2f} um")

    block(unwarned, "UNWARNED (headline, no reflex assumed)")
    if warned is not None and warned.valid:
        block(warned, "WARNED (reflex pre-established)")
        if warned.total_ARU > 0:
            lines.append(f"{indent}    Unwarned/warned ratio: "
                         f"{unwarned.total_ARU / warned.total_ARU:.2f} "
                         f"(ARL published cases span 4.6-7.1)")
    lines.append(f"{indent}  Peak level: {unwarned.peak_pressure_dB:.1f} dB   "
                 f"input: {unwarned.input_location}   "
                 f"95th percentile: {unwarned.percentile_95}")
    lines.append(f"{indent}  Category-(c) inferences in this result: "
                 f"{unwarned.category_c_count}")
    for n in unwarned.notes:
        lines.append(f"{indent}  ! {n}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """Report the AHAAH standing for an .AHA file or a synthetic blast."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="AHAAH auditory hazard. THIS IMPLEMENTATION IS NOT VALIDATED and "
                    "emits no ARU; --research runs the model anyway, in microns squared."
    )
    parser.add_argument("aha", type=Path, nargs="?", help="ARL .AHA waveform file")
    parser.add_argument("--research", action="store_true",
                        help="run the unvalidated model and print its microns-squared index. "
                             "NOT an ARU and not for reporting.")
    parser.add_argument("--haz", type=Path, default=None,
                        help="reference .HAZ table to compare against (--research only). "
                             "The ARL .HAZ table is the WARNED run, so it is compared "
                             "against the warned bands.")
    parser.add_argument("--median", action="store_true",
                        help="omit the +10 dB susceptibility adjustment (not the MIL-STD figure)")
    parser.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE)
    parser.add_argument("--gain", type=float, default=COCHLEAR_GAIN_FACTOR,
                        help=f"CochlearGainFactor (alternates "
                             f"{COCHLEAR_GAIN_FACTOR_ALT_TAPER}, {COCHLEAR_GAIN_FACTOR_ALT_ORIG})")
    parser.add_argument("--json", action="store_true", help="emit the unwarned result as JSON")
    args = parser.parse_args()

    if args.aha is None:
        fs = 125000.0
        t = np.arange(int(fs * 0.02)) / fs
        T = 0.0005
        pressure_Pa = 40000.0 * (1 - t / T) * np.exp(-t / T)
        location = "free_field_normal"
        print(f"Synthetic Friedlander blast: 40 kPa peak "
              f"({20*math.log10(40000/P_REF):.1f} dB), T = {T*1000:.2f} ms, {fs:.0f} Hz")
        reference = None
    else:
        wf = load_aha(args.aha)
        pressure_Pa = wf.pressure_Pa
        fs = wf.sample_rate
        location = wf.input_location
        reference = wf
        print(f"Loaded {args.aha.name}: {len(pressure_Pa)} samples @ {fs:.0f} Hz, "
              f"calc code {wf.calc_code} ({location})")
        if wf.reference_ARU_warned is not None:
            print(f"  Reference (ARL): warned {wf.reference_ARU_warned} ARU, "
                  f"unwarned {wf.reference_ARU_unwarned} ARU")

    unwarned, warned = compute_ahaah_both(
        pressure_Pa, fs, input_location=location, percentile_95=not args.median,
    )

    print()
    print(f"  VALIDATION_STATUS = {VALIDATION_STATUS!r}")
    print(format_ahaah_summary(unwarned, warned))

    if args.json:
        print(json.dumps(unwarned.to_dict(), indent=2))

    if not args.research:
        print("\n  (--research runs the model anyway and prints microns squared, "
              "for development.)")
        return 0

    if not (unwarned.status == "not_validated"):
        print("\n  Input was refused; nothing to run.")
        return 1

    print("\n" + "=" * 78)
    print("  RESEARCH OUTPUT -- " + NOT_AN_ARU_LABEL)
    print("=" * 78)
    ru, rw = run_unvalidated_model_both(
        pressure_Pa, fs, acknowledge_not_validated=True, input_location=location,
        percentile_95=not args.median, oversample=args.oversample,
        cochlear_gain_factor=args.gain,
    )
    for label, r in (("unwarned", ru), ("warned", rw)):
        print(f"  {label:9s} max band index {r.max_band_sum_sq_um2:12.3f} um^2  "
              f"(band {r.peak_band_index}, "
              f"{r.band_frequencies_Hz[r.peak_band_index-1]/1000:.2f} kHz)   "
              f"peak stapes {r.peak_stapes_displacement_um:.2f} um")
    print(f"  unwarned/warned ratio {ru.max_band_sum_sq_um2/rw.max_band_sum_sq_um2:.2f}   "
          "(ARL published cases span 4.6-7.1)")

    # The ARL .HAZ table is the run whose maximum equals the WARNED header figure,
    # so it must be compared against the warned bands, not the unwarned ones.
    ref_ahu = None
    if args.haz is not None:
        _, ref_ahu, _ = load_haz(args.haz)
    print("\n  Per-band index, WARNED (the condition the .HAZ table reports):")
    print(f"  {'Band':>4} {'Freq(kHz)':>10} {'x(cm)':>7} {'um^2':>14}", end="")
    if ref_ahu is not None:
        print(f" {'ref AHU':>12} {'ratio':>8}", end="")
    print()
    for i in range(XBM_NO):
        print(f"  {i+1:4d} {rw.band_frequencies_Hz[i]/1000:10.2f} "
              f"{rw.band_positions_cm[i]:7.3f} "
              f"{rw.band_sum_sq_displacement_um2[i]:14.5f}", end="")
        if ref_ahu is not None:
            r = rw.band_sum_sq_displacement_um2[i] / ref_ahu[i] if ref_ahu[i] else float("nan")
            print(f" {ref_ahu[i]:12.5f} {r:8.3f}", end="")
        print()

    if reference is not None and reference.reference_ARU_unwarned:
        print("\n  Disagreement with the ARL reference (this is why status is not_validated):")
        for label, got, ref in (
            ("unwarned", ru.max_band_sum_sq_um2, reference.reference_ARU_unwarned),
            ("warned", rw.max_band_sum_sq_um2, reference.reference_ARU_warned),
        ):
            if ref:
                print(f"    {label:9s} model {got:12.3f} um^2  reference {ref:9.1f} ARU  "
                      f"ratio {got/ref:8.4f}  error {100*(got-ref)/ref:+9.1f} %")
        print("    The warned figure landing near the reference is a coincidence: see the "
              "module docstring.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
