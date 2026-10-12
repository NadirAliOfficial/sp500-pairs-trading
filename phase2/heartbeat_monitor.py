"""Independent heartbeat watchdog for the pairs bot.

Deliberately its own systemd service, separate from phase2/main.py -- if
the bot process or the whole VPS goes down, anything embedded inside that
same process would die with it and never alert. This runs standalone,
checking two independent signals every cycle:

  1. data/bot_status.json freshness -- the bot writes this every ~60s while
     its main loop is alive. If it goes stale or missing, the bot itself
     (or the VPS under it) is down.
  2. The IB Gateway's own port -- reachable even if bot_status.json happens
     to still look fresh, catches a gateway-specific outage separately.

Both are gated on pairs-bot.service's own systemd ActiveState rather than
a fixed clock window -- per the client 2026-08-08, the flip-flopping
ALERT/RECOVERED pairs were "confusing, not the solution." Root cause: the
bot exits cleanly and on purpose every day at end of session (flatten,
report, disconnect) and isn't started again until 8:15am ET the next
weekday via cron, plus it was manually restarted several times in one day
for deploys -- none of that is a real incident, but a naive staleness
timer alerted on every single one of those transitions, every day, since
it had no way to tell "expected to be off right now" apart from "crashed."
Checking systemd's own ActiveState fixes that at the source: 'inactive'
with a successful Result is the normal end-of-day/weekend/between-deploys
state and is silently skipped, 'active' or 'activating' is when the bot is
actually supposed to be producing a live status (so staleness there is a
real problem), and 'failed' or an unsuccessful Result is a genuine crash,
surfaced by its own separate alert so it isn't lost just because the
service has already stopped by the time this next runs.

Sends exactly one Telegram alert per incident (not one per check cycle),
and one recovery message when the signal clears, to both configured chat
IDs. Client's chat ID needs him to have started a conversation with the
bot first (t.me/YourPairsBot) -- Telegram rejects sends to a chat
that was never initiated from that side.
"""
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATUS_PATH = DATA_DIR / "bot_status.json"
CHECK_INTERVAL_SECONDS = 60
STALE_THRESHOLD_SECONDS = 240  # generous buffer over the bot's own 60s write cycle,
# wide enough to cover a normal startup sequence (pre-market news scan, IBKR handshake,
# contract resolution) without false-alerting while pairs-bot.service is still "activating"
ENDED_GRACE_SECONDS = 21_600  # 6 hours -- covers the longest realistic gap from an early
# exit (as early as 10:30am) to the 4:00pm report going out, with room to spare. Past this,
# an "ended" status this old means the report itself never went out -- worth a real alert,
# not silence forever (see status_is_stale's own comment on why "ended" is skipped at all).
REALERT_INTERVAL_SECONDS = 21_600  # 6 hours -- an UNRESOLVED incident re-alerts on this
# cadence instead of going quiet after the first message. Added 2026-09-08 after the live
# gateway sat stuck at an unauthenticated login screen from 2026-08-31 to 2026-09-07: the
# bot crashed on it every morning, this watchdog sent exactly one crash alert on the first
# day, latched its "already alerted" flag, and never spoke again -- so a week of no live
# connection looked identical to silence. One alert per incident is right for a blip; it is
# exactly wrong for an outage that persists across days.

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_IDS = [c.strip() for c in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if c.strip()]

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "4010"))


def send_telegram(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        print(f"[heartbeat] Telegram not configured, would have sent: {message}")
        return
    for chat_id in TELEGRAM_CHAT_IDS:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
        try:
            urllib.request.urlopen(url, data=data, timeout=10)
        except urllib.error.URLError as e:
            print(f"[heartbeat] failed to send Telegram alert to {chat_id}: {e}")


def status_is_stale(now: float) -> bool:
    try:
        data = json.loads(STATUS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return True
    # Per the client 2026-08-26: a session that ends early (no entry by
    # 10:30am, per the "Dashboard should shut down" instruction) leaves the
    # process alive -- ActiveState stays "active", it's still waiting to
    # send the 4:00pm report -- for hours after its main loop legitimately
    # stops writing fresh status. bot_expected_to_be_producing_status()
    # alone can't tell that apart from a real hang, since systemd still
    # reports "active" either way -- it flapped ALERT/RECOVERED every
    # STALE_THRESHOLD_SECONDS the whole quiet stretch. The status value
    # itself is the signal systemd can't see: "ended" is a real, current
    # read of the file (not stale data left over from before a hang), it's
    # just correctly reporting nothing more is expected until the report.
    # Bounded, not silenced forever: if "ended" itself is older than
    # ENDED_GRACE_SECONDS, the report never went out in any realistic
    # timeline and something is genuinely stuck -- fall through to the
    # normal staleness check below instead of staying quiet indefinitely.
    if data.get("status") == "ended" and now - data.get("ts", 0) <= ENDED_GRACE_SECONDS:
        return False
    return now - data.get("ts", 0) > STALE_THRESHOLD_SECONDS


def gateway_unreachable() -> bool:
    try:
        with socket.create_connection((IBKR_HOST, IBKR_PORT), timeout=5):
            return False
    except OSError:
        return True


def _pairs_bot_service_state() -> tuple[str, str]:
    """(ActiveState, Result) from systemd for pairs-bot.service itself --
    'active' while the main loop is running, 'activating' during its own
    startup sequence (ExecStartPre sleep + pre-market scan + IBKR handshake,
    before it's written its first status), 'inactive' after a stop (clean
    end-of-day exit or a manual restart), 'failed' if the process crashed.
    Falls back to ('unknown', 'unknown') if systemctl itself can't be
    reached rather than raising -- a watchdog failing because its OWN
    status check failed would be worse than one stale reading."""
    try:
        out = subprocess.run(
            ["systemctl", "show", "-p", "ActiveState", "-p", "Result", "pairs-bot.service"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return "unknown", "unknown"
    values = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    return values.get("ActiveState", "unknown"), values.get("Result", "unknown")


def gateway_should_be_reachable() -> bool:
    """True whenever an unreachable gateway is worth alerting on: the bot is
    running (or starting), OR it is in a FAILED state.

    The failed case is the point -- added 2026-09-08. The gateway check used
    to sit inside bot_expected_to_be_producing_status() below, which excludes
    'failed', so the moment the bot crashed ON an unreachable gateway the
    check that would have named the cause stopped running. That is precisely
    backwards: a crashed bot plus a dead gateway is the single most useful
    thing this watchdog can say. A clean 'inactive' (nights, weekends,
    between deploys) is still skipped, so the anti-flip-flop behaviour this
    module was built around is unchanged."""
    state, _ = _pairs_bot_service_state()
    return state in ("active", "activating", "failed")


def bot_expected_to_be_producing_status() -> bool:
    """True only while systemd itself expects pairs-bot.service to be alive
    right now -- 'active' (running its main loop) or 'activating' (still in
    its own startup sequence, hasn't written a first status yet). False for
    a clean scheduled/manual stop and for nights and weekends when the
    service was never started -- neither of those is an incident, so
    neither should trigger a staleness alert. Deliberately does NOT include
    'failed' -- a crash is real and still needs to surface, just via
    bot_crashed() below rather than this staleness check."""
    state, _ = _pairs_bot_service_state()
    return state in ("active", "activating")


def bot_crashed() -> bool:
    """True if the service's last run ended in a failure rather than a
    clean exit -- catches a crash even once systemd has already moved the
    service out of 'active', which bot_expected_to_be_producing_status()
    alone would miss (a crashed service isn't 'active' either, but neither
    is one that finished its day normally, and only one of those is real)."""
    state, result = _pairs_bot_service_state()
    return state == "failed" or (state == "inactive" and result not in ("success", "unknown"))


def _track(incident: dict, key: str, firing: bool, now: float,
           alert_message: str, recovered_message: str) -> None:
    """One incident's alert lifecycle: fire on the first cycle it's true,
    RE-fire every REALERT_INTERVAL_SECONDS for as long as it stays true, and
    send one RECOVERED when it clears.

    `incident[key]` holds the timestamp of the last alert sent for this
    incident, or None when it isn't currently firing. The re-fire is the
    2026-09-08 fix -- see REALERT_INTERVAL_SECONDS's own comment for the
    week-long silent outage that motivated it."""
    last_alert = incident.get(key)
    if firing:
        if last_alert is None:
            send_telegram(alert_message)
            incident[key] = now
        elif now - last_alert >= REALERT_INTERVAL_SECONDS:
            hours = int((now - last_alert) // 3600)
            send_telegram(f"{alert_message} (still unresolved, {hours}h since the last alert)")
            incident[key] = now
    elif last_alert is not None:
        send_telegram(recovered_message)
        incident[key] = None


def main() -> None:
    incident: dict = {"status": None, "gateway": None, "crash": None}
    print(f"[heartbeat] starting, checking every {CHECK_INTERVAL_SECONDS}s, "
          f"stale threshold {STALE_THRESHOLD_SECONDS}s, gateway {IBKR_HOST}:{IBKR_PORT}, "
          f"re-alerting every {REALERT_INTERVAL_SECONDS}s while unresolved, "
          f"gated on pairs-bot.service's own systemd state")
    while True:
        now = time.time()

        _track(incident, "crash", bot_crashed(), now,
               "pairs-bot ALERT: the bot process crashed (systemd reports a failed exit), "
               "not a normal end-of-day stop.",
               "pairs-bot RECOVERED: back to a clean state after the crash.")

        # Gateway reachability is checked independently of the staleness
        # block below, and INCLUDES the failed state -- see
        # gateway_should_be_reachable's own comment.
        if gateway_should_be_reachable():
            _track(incident, "gateway", gateway_unreachable(), now,
                   f"pairs-bot ALERT: IB Gateway ({IBKR_HOST}:{IBKR_PORT}) is not reachable. "
                   "If this persists, the gateway is likely sitting at an unauthenticated "
                   "login screen and needs a restart (and possibly a 2FA approval).",
                   "pairs-bot RECOVERED: IB Gateway is reachable again.")
        else:
            incident["gateway"] = None

        if bot_expected_to_be_producing_status():
            _track(incident, "status", status_is_stale(now), now,
                   "pairs-bot ALERT: bot_status.json has gone stale or is missing while the "
                   "service is supposed to be running. The bot process or the VPS itself may be hung.",
                   "pairs-bot RECOVERED: bot_status.json is updating again.")
        else:
            # Not expected to be running right now (clean end-of-day exit,
            # weekend, or between deploys) -- clear any in-flight alert
            # state quietly, without a RECOVERED message, since there was
            # never a real incident to recover from.
            incident["status"] = None

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
