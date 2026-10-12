"""Unit tests for heartbeat_monitor.py's staleness logic."""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import phase2.heartbeat_monitor as heartbeat_monitor


def _write_status(tmp_path, **fields):
    tmp_path.write_text(json.dumps(fields))


def test_status_is_stale_when_missing():
    heartbeat_monitor.STATUS_PATH = Path(tempfile.mkdtemp()) / "bot_status.json"
    assert heartbeat_monitor.status_is_stale(time.time()) is True
    print("PASS: a missing status file is treated as stale")


def test_status_is_stale_when_timestamp_too_old():
    path = Path(tempfile.mkdtemp()) / "bot_status.json"
    heartbeat_monitor.STATUS_PATH = path
    _write_status(path, status="running", ts=time.time() - 1000)
    assert heartbeat_monitor.status_is_stale(time.time()) is True
    print("PASS: an old timestamp on a 'running' status is stale")


def test_status_is_not_stale_when_fresh():
    path = Path(tempfile.mkdtemp()) / "bot_status.json"
    heartbeat_monitor.STATUS_PATH = path
    _write_status(path, status="running", ts=time.time())
    assert heartbeat_monitor.status_is_stale(time.time()) is False
    print("PASS: a fresh timestamp is not stale")


def test_status_ended_is_not_stale_within_the_grace_period():
    """Per the client 2026-08-26: a session that ended early (no entry by
    10:30am) leaves the process alive for hours waiting on the 4pm report,
    with no more status writes expected -- this used to flap ALERT/
    RECOVERED every STALE_THRESHOLD_SECONDS the whole quiet stretch."""
    path = Path(tempfile.mkdtemp()) / "bot_status.json"
    heartbeat_monitor.STATUS_PATH = path
    _write_status(path, status="ended", ts=time.time() - 10_000)
    assert heartbeat_monitor.status_is_stale(time.time()) is False
    print("PASS: status='ended' is not stale within ENDED_GRACE_SECONDS")


def test_status_ended_past_the_grace_period_is_stale_again():
    """Audit finding, 2026-08-26: the original unbounded "ended" skip meant
    a genuine hang AFTER writing "ended" (e.g. stuck building the report,
    before ever sending it) would never alert. Bounded to
    ENDED_GRACE_SECONDS -- comfortably longer than the realistic gap from
    an early exit to the 4pm report, but not silent forever."""
    path = Path(tempfile.mkdtemp()) / "bot_status.json"
    heartbeat_monitor.STATUS_PATH = path
    _write_status(path, status="ended", ts=time.time() - heartbeat_monitor.ENDED_GRACE_SECONDS - 1)
    assert heartbeat_monitor.status_is_stale(time.time()) is True
    print("PASS: an 'ended' status older than ENDED_GRACE_SECONDS is stale again")


def _capture_sends(monkeypatched_messages):
    def fake_send(message):
        monkeypatched_messages.append(message)
    return fake_send


def test_a_persisting_incident_realerts_instead_of_going_silent():
    """The 2026-09-08 fix. The live gateway sat unauthenticated from
    2026-08-31 to 2026-09-07; the bot crashed on it every morning, this
    watchdog sent ONE crash alert on day one, latched, and never spoke
    again -- a week of no live connection was indistinguishable from
    silence. An unresolved incident must keep saying so."""
    sent = []
    original = heartbeat_monitor.send_telegram
    heartbeat_monitor.send_telegram = _capture_sends(sent)
    try:
        incident = {"crash": None}
        t0 = 1_000_000.0
        heartbeat_monitor._track(incident, "crash", True, t0, "ALERT", "RECOVERED")
        assert len(sent) == 1, "first cycle alerts"

        # Still broken an hour later -- inside the re-alert interval, stays quiet.
        heartbeat_monitor._track(incident, "crash", True, t0 + 3600, "ALERT", "RECOVERED")
        assert len(sent) == 1, "must not alert every cycle"

        # Still broken past the re-alert interval -- says so again.
        heartbeat_monitor._track(
            incident, "crash", True, t0 + heartbeat_monitor.REALERT_INTERVAL_SECONDS + 1,
            "ALERT", "RECOVERED")
        assert len(sent) == 2, "an unresolved incident must re-alert"
        assert "still unresolved" in sent[1]
    finally:
        heartbeat_monitor.send_telegram = original
    print("PASS: a persisting incident re-alerts on the interval instead of latching silent")


def test_an_incident_that_clears_sends_exactly_one_recovered():
    sent = []
    original = heartbeat_monitor.send_telegram
    heartbeat_monitor.send_telegram = _capture_sends(sent)
    try:
        incident = {"gateway": None}
        t0 = 1_000_000.0
        heartbeat_monitor._track(incident, "gateway", True, t0, "ALERT", "RECOVERED")
        heartbeat_monitor._track(incident, "gateway", False, t0 + 60, "ALERT", "RECOVERED")
        heartbeat_monitor._track(incident, "gateway", False, t0 + 120, "ALERT", "RECOVERED")
        assert sent == ["ALERT", "RECOVERED"], sent
    finally:
        heartbeat_monitor.send_telegram = original
    print("PASS: a cleared incident sends exactly one RECOVERED, then stays quiet")


def test_gateway_is_still_checked_while_the_service_is_failed():
    """The other half of the 2026-09-08 fix: the gateway check used to sit
    behind bot_expected_to_be_producing_status(), which excludes 'failed'.
    So the moment the bot crashed ON a dead gateway, the check that would
    have named the cause stopped running. A clean 'inactive' is still
    skipped, preserving the anti-flip-flop design."""
    original = heartbeat_monitor._pairs_bot_service_state
    try:
        for state in ("active", "activating", "failed"):
            heartbeat_monitor._pairs_bot_service_state = lambda s=state: (s, "any")
            assert heartbeat_monitor.gateway_should_be_reachable() is True, state
        heartbeat_monitor._pairs_bot_service_state = lambda: ("inactive", "success")
        assert heartbeat_monitor.gateway_should_be_reachable() is False
    finally:
        heartbeat_monitor._pairs_bot_service_state = original
    print("PASS: the gateway is still checked while the service is failed, skipped when cleanly inactive")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
