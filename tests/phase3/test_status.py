"""Unit tests for phase3/status.py -- pure/cheap checks, no Flask needed.
is_gateway_reachable is tested against a real temporary local socket (both
open and closed) rather than mocked, since it's a thin wrapper around a
real TCP connect and that's exactly the behavior worth verifying."""
import socket
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase3.status import count_qualified_pairs, is_gateway_reachable, recent_log_lines


def test_is_gateway_reachable_true_when_something_is_listening():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert is_gateway_reachable(host="127.0.0.1", port=port) is True
    finally:
        server.close()
    print("PASS: is_gateway_reachable returns True when a socket is actually listening")


def test_is_gateway_reachable_false_when_nothing_is_listening():
    # bind and immediately close to get a port number almost certainly free
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    assert is_gateway_reachable(host="127.0.0.1", port=port, timeout=0.5) is False
    print("PASS: is_gateway_reachable returns False when nothing is listening")


def test_count_qualified_pairs_counts_data_rows_not_header():
    tmp = Path(tempfile.mkdtemp()) / "qualified_pairs.csv"
    tmp.write_text("symbol_a,symbol_b,hedge_ratio\nTYL,WDAY,2.13\nBX,KKR,0.95\n")
    assert count_qualified_pairs(path=tmp) == 2
    print("PASS: count_qualified_pairs counts data rows, excluding the header")


def test_count_qualified_pairs_none_when_file_missing():
    assert count_qualified_pairs(path=Path("/tmp/does-not-exist-pairs.csv")) is None
    print("PASS: count_qualified_pairs returns None when Phase 1 hasn't been run yet")


def test_recent_log_lines_returns_last_n_lines_in_order():
    tmp = Path(tempfile.mkdtemp()) / "pairs_bot.log"
    tmp.write_text("\n".join(f"line {i}" for i in range(1, 21)) + "\n")
    lines = recent_log_lines(n=5, path=tmp)
    assert lines == ["line 16", "line 17", "line 18", "line 19", "line 20"]
    print("PASS: recent_log_lines returns the last N lines in original order")


def test_recent_log_lines_empty_when_no_session_has_run():
    assert recent_log_lines(path=Path("/tmp/does-not-exist-pairs-bot.log")) == []
    print("PASS: recent_log_lines returns an empty list when no log file exists yet")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
