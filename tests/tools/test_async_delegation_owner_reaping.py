"""RED->GREEN tests for periodic owner-pid-liveness reaping (deploy#59).

``recover_abandoned_delegations()`` already classifies a durable ``running``
row whose owning process has exited as ``unknown`` — but before this change it
only ran ONCE at registry startup (via ``restore_undelivered_completions``).
On a long-lived gateway, a delegation whose owner_pid dies mid-flight (the turn
never starts) therefore sat at ``state=running`` forever, holding its slot and
never surfacing an outcome.

These tests pin two things:

1. ``recover_abandoned_delegations`` reaps a dead-owner ``running`` row even
   when there is NO in-memory record for it (the cross-process orphan: the CLI
   worker dispatched + persisted the row, then exited; the gateway is a
   different process with an empty ``_records``).
2. A live-owner ``running`` row is left untouched.
3. The periodic reaper thread invokes the recovery scan on its interval.
"""

import json
import time

from tools import async_delegation as ad


def _point_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")


def _insert_running_row(delegation_id, owner_pid, owner_started_at=None):
    now = time.time()
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, owner_pid,
                owner_started_at, task_json, origin_session_id)
               VALUES (?, '', '', NULL, 'running', ?, ?, 'pending', 0, ?, ?, ?, '')""",
            (delegation_id, now, now, owner_pid, owner_started_at,
             json.dumps({"goal": "orphan"})),
        )


def _row_state(delegation_id):
    with ad._DB_LOCK, ad._transaction() as conn:
        row = conn.execute(
            "SELECT state FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()
    return row[0] if row else None


def _dead_pid():
    """A pid that is (almost certainly) not alive."""
    import os
    pid = 999999
    while ad_pid_alive(pid) and pid > 2:
        pid -= 1
    return pid


def ad_pid_alive(pid):
    from gateway.status import _pid_exists
    return _pid_exists(pid)


def test_recover_reaps_dead_owner_running_row_without_in_memory_record(
    monkeypatch, tmp_path
):
    _point_ledger(monkeypatch, tmp_path)
    # No in-memory record — this is the cross-process orphan the bug describes.
    ad._records.clear()
    dead = _dead_pid()
    _insert_running_row("deleg_dead1", owner_pid=dead)

    assert _row_state("deleg_dead1") == "running"
    recovered = ad.recover_abandoned_delegations()
    assert recovered >= 1
    assert _row_state("deleg_dead1") == "unknown"


def test_recover_leaves_live_owner_running_row_untouched(monkeypatch, tmp_path):
    _point_ledger(monkeypatch, tmp_path)
    ad._records.clear()
    import os
    from gateway.status import get_process_start_time
    live_pid = os.getpid()
    _insert_running_row(
        "deleg_live1",
        owner_pid=live_pid,
        owner_started_at=get_process_start_time(live_pid),
    )

    ad.recover_abandoned_delegations()
    assert _row_state("deleg_live1") == "running"


def test_periodic_reaper_invokes_recovery_scan(monkeypatch):
    """The periodic durable reaper must call recover_abandoned_delegations
    on its interval, independent of any in-memory records."""
    calls = {"n": 0}

    def fake_recover():
        calls["n"] += 1
        return 0

    monkeypatch.setattr(ad, "recover_abandoned_delegations", fake_recover)
    # Drive one tick of the reaper body directly (no real sleep).
    ad._durable_reaper_tick()
    assert calls["n"] == 1


def test_durable_reaper_tick_swallows_errors(monkeypatch):
    """A DB error in a tick must not propagate (would kill the daemon)."""
    def boom():
        raise RuntimeError("db gone")

    monkeypatch.setattr(ad, "recover_abandoned_delegations", boom)
    # Must not raise.
    assert ad._durable_reaper_tick() == 0


def test_ensure_durable_reaper_is_idempotent_and_stoppable(monkeypatch):
    """Starting twice reuses one thread; stop signals it to exit.

    Also proves the loop TICKS on its interval (not just the immediate reap):
    with a tiny interval we wait for at least 2 recovery calls before stopping.
    """
    import threading as _t
    calls = {"n": 0}
    bumped = _t.Event()

    def counting_recover():
        calls["n"] += 1
        if calls["n"] >= 2:
            bumped.set()
        return 0

    monkeypatch.setattr(ad, "recover_abandoned_delegations", counting_recover)
    monkeypatch.setattr(ad, "_REAPER_INTERVAL_SECONDS", 0.02)
    ad.stop_durable_reaper()  # ensure clean slate
    try:
        ad.ensure_durable_reaper()
        first = ad._reaper_thread
        assert first is not None and first.is_alive()
        ad.ensure_durable_reaper()  # idempotent — same thread
        assert ad._reaper_thread is first
        # Immediate reap + at least one interval tick.
        assert bumped.wait(timeout=2.0), "reaper did not tick on its interval"
        assert calls["n"] >= 2
    finally:
        ad.stop_durable_reaper()
    assert not (first and first.is_alive())
