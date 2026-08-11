# AHAAH reference fixtures

Verbatim copies from the public US Army Research Laboratory release
`AHAAH_ver_2_1/AHAAH_MIL-STD-1474E_defaultHPD/`. They are the validation oracle for
`ahaah.py` and are checked in so that `tests/test_ahaah.py` can always run.

| File | Role |
|---|---|
| `160F.AHA` | Reference waveform (2048 samples, 125 kHz, Pascals) with ARL's own results in the header: 391.0 ARU warned, 2237 ARU unwarned. |
| `160F.HAZ` | ARL's per-band hazard table for the warned run: 23 rows of BIndex / Freq(kHz) / Hazard(AHU) / Distance(cm). |
| `man.coe` | The coefficient file. md5 `8f21f4316def7dbcb1bd5f4c9ef5fed0`. Kept so the constants embedded in `ahaah.py` can be diffed against their source. |
| `FFEDM90.DAT` | Mehrgardt & Mellert free-field-to-eardrum pressure ratio at 90 degrees azimuth. ARL's own linear-transfer-function validation curve (spec section 11.2, Tier 1). |

These are not modified. Do not edit them; they only have value as an independent oracle.
