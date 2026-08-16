"""RED->GREEN tests for the ``hermes cron run`` CLI render of a background dispatch.

Regression guard for deploy#59: when ``cronjob(action='run')`` dispatches the
run to the background (``execution_mode == 'background'``), the CLI must NOT
force a succeeded/failed binary off ``execution_success`` (which the background
path never sets) — that made every background dispatch print "Ran now: failed."
even though the dispatch succeeded. It must instead report the dispatch and the
delegation id.
"""

from hermes_cli import cron as cron_cli
from hermes_cli.cron import _job_action


def _patch_api(monkeypatch, payload):
    monkeypatch.setattr(cron_cli, "_cron_api", lambda **kw: payload)


def test_run_background_dispatch_reports_dispatched_not_failed(monkeypatch, capsys):
    _patch_api(
        monkeypatch,
        {
            "success": True,
            "job": {
                "id": "job123",
                "name": "Nightly Audit",
                "executed": True,
                "execution_mode": "background",
                "delegation_id": "deleg_abcd1234",
            },
        },
    )
    rc = _job_action("run", "job123", "Triggered")
    out = capsys.readouterr().out
    assert rc == 0
    # The bug: prints "Ran now: failed." for a successful background dispatch.
    assert "failed" not in out.lower()
    assert "Dispatched to background" in out
    assert "deleg_abcd1234" in out


def test_run_inline_success_still_reports_succeeded(monkeypatch, capsys):
    _patch_api(
        monkeypatch,
        {
            "success": True,
            "job": {
                "id": "job123",
                "name": "Nightly Audit",
                "executed": True,
                "execution_success": True,
            },
        },
    )
    rc = _job_action("run", "job123", "Triggered")
    out = capsys.readouterr().out
    assert rc == 0
    assert "Ran now: succeeded." in out


def test_run_inline_failure_still_reports_failed(monkeypatch, capsys):
    _patch_api(
        monkeypatch,
        {
            "success": True,
            "job": {
                "id": "job123",
                "name": "Nightly Audit",
                "executed": True,
                "execution_success": False,
            },
        },
    )
    rc = _job_action("run", "job123", "Triggered")
    out = capsys.readouterr().out
    assert rc == 0
    assert "Ran now: failed." in out


def test_run_skipped_reports_skip_reason(monkeypatch, capsys):
    _patch_api(
        monkeypatch,
        {
            "success": True,
            "job": {
                "id": "job123",
                "name": "Nightly Audit",
                "executed": False,
                "execution_skipped": "Already being fired by the scheduler; not run again.",
            },
        },
    )
    rc = _job_action("run", "job123", "Triggered")
    out = capsys.readouterr().out
    assert rc == 0
    assert "Already being fired" in out
