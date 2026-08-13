"""
test_ui_contract.py - the interface and the servers speak the same language.

SASA has two back ends behind one interface: ui/bridge/python-bridge.js, which
the development server uses, and app.py, which is what the packaged desktop
application runs. Both take the same JSON config object from the same function
in ui/renderer/app.js.

They drifted, and nothing noticed, because nothing in the test suite ever read
one against the other. The interface sent `uncalibrated`, a calibrator tone, or
the recording chain's `adcFullScaleV`; app.py read `calMode`, `vPerFS` and
nothing else. In the shipped application that left exactly ONE working
calibration method out of four -- a saved profile -- and every other choice was
refused with "No calibration was supplied", which was true of the request and
completely untrue of what the operator had filled in. The test record went the
same way: sent as `metadata`, read as `testMetadata`, so microphone distance,
angle, temperature, humidity and pressure never reached the engine at all, and
the atmospheric correction ran on nothing.

These tests read the three files and compare the vocabularies. They are
deliberately textual: importing the browser code is not possible, and the
failure being guarded against is a name that exists on one side and not the
other.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RENDERER = _REPO_ROOT / "ui" / "renderer" / "app.js"
_BRIDGE = _REPO_ROOT / "ui" / "bridge" / "python-bridge.js"
_SERVER = _REPO_ROOT / "app.py"

# Sent by the interface but resolved before the config is built, so no server
# reads them by name. Keep this list short and justified.
_NOT_A_SERVER_KEY: set[str] = set()


def interface_keys() -> set[str]:
    """Every key buildRunConfig() can put in the config it sends."""
    source = _RENDERER.read_text(encoding="utf-8")
    start = source.find("function buildRunConfig")
    assert start >= 0, "buildRunConfig has been renamed; this test needs updating"
    # The function ends at the next top-level `function` declaration.
    end = source.find("\nfunction ", start + 1)
    body = source[start:end if end > 0 else len(source)]
    return set(re.findall(r"\bconfig\.([A-Za-z][A-Za-z0-9]*)\s*=", body)) - _NOT_A_SERVER_KEY


def server_keys() -> set[str]:
    """Every key app.py reads out of the client's config object."""
    source = _SERVER.read_text(encoding="utf-8")
    found: set[str] = set()
    for pattern in (
        r"_get_(?:str|int|number|bool)\(config,\s*'([A-Za-z][A-Za-z0-9]*)'",
        r"_present\(config,\s*'([A-Za-z][A-Za-z0-9]*)'",
        r"config\.get\('([A-Za-z][A-Za-z0-9]*)'",
        r"config\['([A-Za-z][A-Za-z0-9]*)'\]",
    ):
        found.update(re.findall(pattern, source))
    return found


def bridge_keys() -> set[str]:
    """
    Every key the development bridge accepts.

    Two sources: the flag tables that map a key to a command-line option, and
    the handful named directly in KNOWN_FIELDS because the bridge handles them
    itself rather than passing them through as a flag.
    """
    source = _BRIDGE.read_text(encoding="utf-8")
    keys = set(re.findall(r"^\s{2}([A-Za-z][A-Za-z0-9]*):\s*(?:\{\s*flag|'--)", source, re.M))

    known = re.search(r"const KNOWN_FIELDS = new Set\(\[(.*?)\]\)", source, re.S)
    if known:
        keys.update(re.findall(r"'([A-Za-z][A-Za-z0-9]*)'", known.group(1)))
    return keys


def test_app_py_reads_every_key_the_interface_sends():
    """
    The packaged application must understand its own interface.

    A key it does not read is not a degraded feature: it is a control the
    operator sets that does nothing, silently, with no way to tell from the
    screen that it was ignored.
    """
    missing = sorted(interface_keys() - server_keys())
    assert not missing, (
        "the interface sends these and app.py never reads them, so the packaged "
        f"app ignores or refuses them: {missing}"
    )


def test_bridge_accepts_every_key_the_interface_sends():
    """The development server must understand it too, or testing proves nothing."""
    missing = sorted(interface_keys() - bridge_keys())
    assert not missing, (
        f"the interface sends these and python-bridge.js does not accept them: {missing}"
    )


@pytest.mark.parametrize("payload,expect_calibrated,expect_method", [
    ({"uncalibrated": True}, False, "uncalibrated"),
    ({"sensitivityMv": 12.5, "preampGainDb": 0.0, "adcFullScaleV": 1.0}, True, "recording_chain"),
    ({"paPerFS": 143.96}, True, "direct"),
])
def test_every_calibration_method_the_interface_offers_resolves(
    payload, expect_calibrated, expect_method
):
    """
    Each branch of the interface's calibration selector, as it sends it.

    The tone method needs a recording and is covered in test_calibration.py;
    what is checked here is that the other three are not refused for speaking
    the wrong dialect.
    """
    import app
    from calibration import Calibration

    calibration, _inputs = app.build_calibration(dict(payload), Calibration)
    assert calibration.calibrated is expect_calibrated
    assert calibration.method == expect_method


def test_a_config_with_no_calibration_is_still_refused():
    """
    The point of the above is not that everything is accepted.

    Widening what counts as a calibration must not widen it to nothing: a
    config that names no method at all is the one case that has to keep
    failing, because the alternative is relative levels reported as dB SPL.
    """
    import app
    from calibration import Calibration

    with pytest.raises(app.ConfigError):
        app.build_calibration({"filePath": "/tmp/x.wav"}, Calibration)

    # An explicitly empty flag is not a choice either.
    with pytest.raises(app.ConfigError):
        app.build_calibration({"uncalibrated": False}, Calibration)


# ---------------------------------------------------------------------------
# The input probe
#
# The probe crosses three layers too — main.py writes the JSON, ui/server.js
# forwards it, ui/renderer/app.js reads fields out of it by name — and it is
# the same kind of contract as the run config, so it is guarded the same way.
# ---------------------------------------------------------------------------

_UI_SERVER = _REPO_ROOT / "ui" / "server.js"


def probe_fields_produced() -> set[str]:
    """Every key main.probe_input() puts in its result."""
    source = (_REPO_ROOT / "main.py").read_text(encoding="utf-8")
    start = source.find("def probe_input(")
    assert start >= 0, "probe_input has been renamed; this test needs updating"
    end = source.find("\ndef ", start + 1)
    body = source[start:end if end > 0 else len(source)]

    name = r"([A-Za-z_][A-Za-z0-9_]*)"
    produced: set[str] = set()
    produced |= set(re.findall(rf'"{name}":', body))          # the initial literal
    produced |= set(re.findall(rf'out\["{name}"\]', body))    # direct assignment
    produced |= set(re.findall(rf"^\s+{name}=", body, re.M))   # out.update(field=...)
    assert {"sample_rate", "nyquist_Hz", "readable"} <= produced, sorted(produced)
    return produced


def probe_fields_read_by_the_interface() -> set[str]:
    """Every field the renderer reads off a probe result."""
    source = _RENDERER.read_text(encoding="utf-8")
    # The whole identifier, not just its lower-case head: `probe.nyquist_Hz`
    # truncated to `nyquist_` would be reported as a field nobody writes.
    return set(re.findall(r"\bprobe\.([A-Za-z_][A-Za-z0-9_]*)", source))


def test_the_interface_reads_no_probe_field_the_engine_does_not_write():
    produced = probe_fields_produced()
    read = probe_fields_read_by_the_interface()
    missing = sorted(read - produced)
    assert not missing, (
        "ui/renderer/app.js reads probe fields main.probe_input() never writes: "
        f"{missing}. They will be undefined at runtime."
    )


def test_the_server_exposes_the_probe_the_interface_calls():
    server = _UI_SERVER.read_text(encoding="utf-8")
    renderer = _RENDERER.read_text(encoding="utf-8")

    called = re.findall(r"fetch\(`(/api/[a-z-]+)", renderer)
    assert "/api/probe" in called, "the renderer no longer calls /api/probe"
    assert "app.get('/api/probe'" in server, "ui/server.js no longer serves /api/probe"
    assert "'--probe'" in server, "the probe endpoint no longer runs main.py --probe"


def test_the_probe_cli_flag_exists_under_that_exact_name():
    """The server spells the flag in a string; argparse has to agree."""
    main_source = (_REPO_ROOT / "main.py").read_text(encoding="utf-8")
    assert 'add_argument("--probe"' in main_source
