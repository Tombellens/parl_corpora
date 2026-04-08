"""
dashboard.py
============
Single entry-point for the entire speaker enrichment pipeline.

Run with:
    python3 dashboard.py [--port 5050] [--debug]
    nohup python3 dashboard.py >> /home/tom/data/speaker_enrichment/logs/dashboard.log 2>&1 &

SSH tunnel:
    ssh -L 5050:localhost:5050 tom@workstation
    → open http://localhost:5050

Everything is launchable from the browser:
  - Import speakers from speaker_names.csv
  - Run any individual pipeline stage
  - Run the orchestrator (picks best stage automatically)
  - Run the full test pipeline with custom parameters
  - Load / unload LLM models
  - Activate failure batches for retry
  - Live log output from all running processes
  - Kill any running process
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, redirect, render_template_string, request, url_for

import config
from db import (
    activate_failure_batch, get_conn, get_stage_counts,
    init_db, now_iso, STAGES,
)
from llm_client import (
    get_loaded_models, load_model, read_llm_lock, unload_model,
)

HERE = Path(__file__).parent
app  = Flask(__name__)

# ---------------------------------------------------------------------------
# Process Manager
# ---------------------------------------------------------------------------

class ProcessManager:
    """
    Tracks background subprocesses launched from the dashboard.
    Each process gets a named slot and writes to its own log file.
    Thread-safe for concurrent Flask requests.
    """

    def __init__(self, log_dir: Path):
        self._lock  = threading.Lock()
        self._procs: dict[str, dict] = {}
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

    def start(self, name: str, cmd: list[str], env: dict | None = None,
              force: bool = False) -> dict:
        """Launch a subprocess. Replaces any finished process with the same name.
        If force=True, kills any running process with that name first."""
        with self._lock:
            existing = self._procs.get(name)
            if existing and existing["proc"].poll() is None:
                if force:
                    existing["proc"].terminate()
                    try:
                        existing["proc"].wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        existing["proc"].kill()
                else:
                    raise RuntimeError(f"Process '{name}' is already running (PID {existing['pid']})")

            log_path = self.log_dir / f"{name}.log"
            log_fh   = open(log_path, "w", buffering=1)

            # PYTHONUNBUFFERED ensures output appears in the log immediately
            full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
            proc = subprocess.Popen(
                cmd,
                cwd=str(HERE),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=full_env,
                text=True,
            )
            entry = {
                "name":       name,
                "cmd":        " ".join(cmd),
                "proc":       proc,
                "pid":        proc.pid,
                "log_path":   str(log_path),
                "started_at": now_iso(),
                "log_fh":     log_fh,
            }
            self._procs[name] = entry
            return entry

    def kill(self, name: str) -> bool:
        with self._lock:
            entry = self._procs.get(name)
            if not entry:
                return False
            proc = entry["proc"]
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            return True

    def status(self, name: str) -> dict | None:
        with self._lock:
            entry = self._procs.get(name)
            if not entry:
                return None
            return self._entry_info(entry)

    def list_all(self) -> list[dict]:
        with self._lock:
            return [self._entry_info(e) for e in self._procs.values()]

    def get_log(self, name: str, offset: int = 0) -> tuple[str, int]:
        """Return (new_text, new_offset) from the log file starting at byte offset."""
        with self._lock:
            entry = self._procs.get(name)
            if not entry:
                return "", 0
            log_path = entry["log_path"]
        try:
            with open(log_path, "r", errors="replace") as f:
                f.seek(offset)
                text = f.read()
                new_offset = f.tell()
            return text, new_offset
        except OSError:
            return "", offset

    def _entry_info(self, entry: dict) -> dict:
        rc = entry["proc"].poll()
        return {
            "name":       entry["name"],
            "cmd":        entry["cmd"],
            "pid":        entry["pid"],
            "log_path":   entry["log_path"],
            "started_at": entry["started_at"],
            "running":    rc is None,
            "returncode": rc,
            "status":     "running" if rc is None else ("ok" if rc == 0 else f"exit {rc}"),
        }


_pm = ProcessManager(Path(config.LOG_DIR))


def _launch(name: str, script: str, extra_args: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(HERE / script)] + (extra_args or [])
    return _pm.start(name, cmd)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speaker Enrichment</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Courier New',monospace;background:#0d1117;color:#e6edf3;padding:20px;font-size:13px}
h1{color:#58a6ff;font-size:20px;margin-bottom:2px}
.subtitle{color:#8b949e;font-size:11px;margin-bottom:20px}
h2{color:#79c0ff;font-size:14px;margin:24px 0 10px;border-bottom:1px solid #30363d;padding-bottom:4px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:14px}
.card h3{color:#79c0ff;font-size:13px;margin-bottom:10px;border-bottom:1px solid #21262d;padding-bottom:4px}
table{width:100%;border-collapse:collapse}
th{color:#8b949e;text-align:left;padding:5px 8px;border-bottom:1px solid #21262d;font-size:11px;text-transform:uppercase}
td{padding:4px 8px;border-bottom:1px solid #1c2128;vertical-align:middle}
tr:last-child td{border-bottom:none}
.badge{display:inline-block;padding:1px 7px;border-radius:12px;font-size:11px;font-weight:bold;white-space:nowrap}
.s-pending{background:#21262d;color:#8b949e}
.s-running{background:#1f4287;color:#58a6ff}
.s-success,.s-ok{background:#1a4731;color:#3fb950}
.s-failed{background:#4d1f1f;color:#f85149}
.s-skipped{background:#272d3a;color:#8b949e}
.bar-wrap{background:#21262d;border-radius:3px;height:10px;width:100%;display:flex;overflow:hidden}
.bar-success{background:#238636}.bar-failed{background:#da3633}
.bar-running{background:#1f6feb}.bar-skipped{background:#6e7681}.bar-pending{background:#30363d}
.lock-on{color:#f85149}.lock-off{color:#3fb950}
button,input[type=submit]{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:4px;
  padding:4px 12px;cursor:pointer;font-size:12px;font-family:inherit;margin:2px}
button:hover{background:#30363d}
.btn-green{border-color:#238636;color:#3fb950}
.btn-blue{border-color:#1f6feb;color:#58a6ff}
.btn-red{border-color:#da3633;color:#f85149}
.btn-orange{border-color:#d29922;color:#e3b341}
input[type=text],input[type=number],select{background:#0d1117;color:#e6edf3;border:1px solid #30363d;
  border-radius:4px;padding:4px 8px;font-size:12px;font-family:inherit}
label{font-size:12px;color:#8b949e;display:inline-block;margin-right:8px}
.flash{padding:8px 12px;border-radius:4px;margin-bottom:12px;font-size:12px}
.flash-ok{background:#1a4731;color:#3fb950}.flash-err{background:#4d1f1f;color:#f85149}
.terminal{background:#010409;border:1px solid #30363d;border-radius:4px;padding:10px;
  font-size:11px;line-height:1.5;height:280px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;
  color:#c9d1d9;font-family:'Courier New',monospace}
.proc-row{display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid #1c2128}
.proc-row:last-child{border-bottom:none}
.proc-name{color:#79c0ff;font-weight:bold;min-width:160px}
a{color:#58a6ff;text-decoration:none}
a:hover{text-decoration:underline}
.mono{font-family:'Courier New',monospace;font-size:11px;color:#8b949e}
.form-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:6px 0}
hr{border:none;border-top:1px solid #30363d;margin:12px 0}
.stage-btn{margin:2px;padding:3px 10px;font-size:11px}
</style>
</head>
<body>
<h1>Speaker Enrichment Pipeline</h1>
<div class="subtitle">{{ now }} &nbsp;·&nbsp; DB: {{ db_path }}
  &nbsp;·&nbsp; <a href="/">refresh</a></div>

{% if flash_msg %}
<div class="flash {{ 'flash-ok' if flash_ok else 'flash-err' }}">{{ flash_msg }}</div>
{% endif %}

<!-- ===================== PIPELINE OVERVIEW ===================== -->
<h2>Pipeline Overview</h2>
<table>
<tr><th>Stage</th><th>Pending</th><th>Running</th><th>Success</th><th>Failed</th><th>Skipped</th><th>Progress</th></tr>
{% for stage, counts in stage_counts.items() %}
{% set total=counts.values()|sum %}
{% set p=counts.get('pending',0) %}{% set r=counts.get('running',0) %}
{% set s=counts.get('success',0) %}{% set f=counts.get('failed',0) %}{% set k=counts.get('skipped',0) %}
<tr>
  <td><b>{{ stage }}</b></td>
  <td><span class="badge s-pending">{{ p }}</span></td>
  <td><span class="badge s-running">{{ r }}</span></td>
  <td><span class="badge s-success">{{ s }}</span></td>
  <td><span class="badge s-failed">{{ f }}</span></td>
  <td><span class="badge s-skipped">{{ k }}</span></td>
  <td style="width:180px">
    {% if total > 0 %}
    <div class="bar-wrap">
      <div class="bar-success" style="width:{{ (s/total*100)|int }}%"></div>
      <div class="bar-running" style="width:{{ (r/total*100)|int }}%"></div>
      <div class="bar-failed"  style="width:{{ (f/total*100)|int }}%"></div>
      <div class="bar-skipped" style="width:{{ (k/total*100)|int }}%"></div>
      <div class="bar-pending" style="flex:1"></div>
    </div>
    <span class="mono">{{ s }}/{{ total }}</span>
    {% endif %}
  </td>
</tr>
{% endfor %}
</table>

<!-- ===================== RUNNING PROCESSES ===================== -->
<h2>Running Processes</h2>
{% if processes %}
<div class="card" style="margin-bottom:12px">
{% for proc in processes %}
<div class="proc-row">
  <span class="proc-name">{{ proc.name }}</span>
  <span class="badge {{ 's-running' if proc.running else ('s-ok' if proc.returncode == 0 else 's-failed') }}">
    {{ proc.status }}</span>
  <span class="mono">PID {{ proc.pid }}</span>
  <span class="mono">{{ proc.started_at }}</span>
  <span class="mono" style="flex:1;overflow:hidden;text-overflow:ellipsis">{{ proc.cmd[:80] }}</span>
  <button type="button" class="stage-btn" onclick="showLog('{{ proc.name }}')">log</button>
  {% if proc.running %}
  <form method="post" action="/process/kill" style="display:inline;margin:0">
    <input type="hidden" name="name" value="{{ proc.name }}">
    <button class="stage-btn btn-red" type="submit">kill</button>
  </form>
  {% else %}
  <form method="post" action="/process/restart" style="display:inline;margin:0">
    <input type="hidden" name="name" value="{{ proc.name }}">
    <input type="hidden" name="force" value="1">
    <button class="stage-btn btn-orange" type="submit">restart</button>
  </form>
  {% endif %}
</div>
{% endfor %}
</div>
{% else %}
<p class="mono" style="margin-bottom:12px">No processes tracked this session.</p>
{% endif %}

<div class="card">
  <h3>Live Log</h3>
  <div class="form-row" style="margin-bottom:8px">
    <select id="log-select">
      <option value="">— select process —</option>
      {% for proc in processes %}
      <option value="{{ proc.name }}">{{ proc.name }}</option>
      {% endfor %}
    </select>
    <button type="button" onclick="startPolling(document.getElementById('log-select').value)">View</button>
    <button type="button" onclick="stopPolling()">Stop</button>
    <button type="button" onclick="document.getElementById('log-out').textContent=''">Clear</button>
    <span id="log-status" class="mono"></span>
  </div>
  <div class="terminal" id="log-out">Select a process above to view its output.</div>
</div>

<!-- ===================== LAUNCH CONTROLS ===================== -->
<h2>Launch Controls</h2>
<div class="grid3">

<!-- Main pipeline -->
<div class="card">
  <h3>Main Pipeline — Individual Stages</h3>
  <p class="mono" style="margin-bottom:8px">Each button launches the corresponding batch script in the background.</p>
  {% for stage, script in stage_scripts.items() %}
  <form method="post" action="/pipeline/launch" style="display:inline;margin:0">
    <input type="hidden" name="stage" value="{{ stage }}">
    <input type="hidden" name="script" value="{{ script }}">
    <button class="stage-btn btn-blue" type="submit">{{ stage }}</button>
  </form>
  {% endfor %}
  <hr>
  <form method="post" action="/pipeline/orchestrate">
    <button class="btn-green" type="submit">Auto-orchestrate</button>
    <span class="mono" style="font-size:10px">(picks best stage)</span>
  </form>
</div>

<!-- Import & Setup -->
<div class="card">
  <h3>Setup</h3>
  <p class="mono" style="margin-bottom:8px">Run once to initialise the DB and load speakers.</p>
  <form method="post" action="/setup/init" style="margin-bottom:8px">
    <button class="btn-blue" type="submit">Init database</button>
  </form>
  <form method="post" action="/setup/import">
    <button class="btn-blue" type="submit">Import speakers</button>
    <span class="mono" style="font-size:10px">from speaker_names.csv</span>
  </form>
</div>

<!-- LLM Controls -->
<div class="card">
  <h3>LLM — LM Studio</h3>
  <p>
    {% if lock and lock.alive %}
    <span class="lock-on">LOCKED</span> &nbsp;
    <span class="mono">{{ lock.task }} · {{ lock.model }}</span><br>
    <span class="mono" style="font-size:10px">PID {{ lock.pid }} since {{ lock.started_at }}</span>
    {% else %}
    <span class="lock-off">Free</span>
    {% endif %}
  </p>
  <hr>
  {% if loaded_models %}
  {% for m in loaded_models %}
  <div style="margin:4px 0;display:flex;align-items:center;gap:6px">
    <span class="badge s-success">loaded</span>
    <span class="mono">{{ (m.id or m.key or m.get('display_name','?'))[:40] }}</span>
    <form method="post" action="/llm/unload" style="display:inline;margin:0">
      <input type="hidden" name="instance_id" value="{{ m.id or m.key }}">
      <button class="stage-btn btn-red" type="submit">unload</button>
    </form>
  </div>
  {% endfor %}
  {% else %}
  <p class="mono" style="margin:4px 0">No models loaded</p>
  {% endif %}
  <hr>
  <form method="post" action="/llm/load">
    <div class="form-row">
      <input type="text" name="model_id" placeholder="model identifier" style="width:220px">
      <button class="btn-blue" type="submit">Load</button>
    </div>
  </form>
</div>

</div><!-- /grid3 -->

<!-- ===================== TEST PIPELINE ===================== -->
<h2>Test Pipeline</h2>
<div class="card">
  <h3>Run end-to-end test on a small sample (isolated DB, production untouched)</h3>
  <form method="post" action="/test/launch">
    <div class="form-row">
      <label>Speakers (n)</label>
      <input type="number" name="n" value="5" min="1" max="50" style="width:60px">

      <label>Country</label>
      <input type="text" name="country" placeholder="e.g. FR" style="width:60px" maxlength="2">

      <label>Stage only</label>
      <select name="stage">
        <option value="">— all stages —</option>
        {% for s in all_stages %}
        <option value="{{ s }}">{{ s }}</option>
        {% endfor %}
      </select>

      <label><input type="checkbox" name="verbose" value="1"> verbose</label>
      <label><input type="checkbox" name="keep" value="1"> keep files</label>

      <button class="btn-orange" type="submit">Run test</button>
    </div>
  </form>
</div>

<!-- ===================== BY COUNTRY ===================== -->
<h2>Progress by Country</h2>
<table>
<tr><th>Country</th><th>Total</th><th>Queried</th><th>Fetched</th>
    <th>URL synth</th><th>CV done</th><th>Ann A</th><th>Ann B</th><th>Ann C</th><th>Ann D</th></tr>
{% for row in country_stats %}
<tr>
  <td><b>{{ row.country }}</b></td>
  <td>{{ row.total }}</td>
  <td>{{ row.queried }}</td>
  <td>{{ row.fetched }}</td>
  <td>{{ row.url_synth }}</td>
  <td>{{ row.cv }}</td>
  <td>{{ row.ann_a }}</td>
  <td>{{ row.ann_b }}</td>
  <td>{{ row.ann_c }}</td>
  <td>{{ row.ann_d }}</td>
</tr>
{% endfor %}
</table>

<!-- ===================== FAILURE BATCHES ===================== -->
<h2>Failure Batches <span class="mono" style="font-size:11px">(manual retry only)</span></h2>
{% if failure_batches %}
<table>
<tr><th>ID</th><th>Stage</th><th>Name</th><th>N</th><th>Created</th><th>Activated</th><th></th></tr>
{% for fb in failure_batches %}
<tr>
  <td>{{ fb.id }}</td>
  <td><span class="badge s-failed">{{ fb.stage }}</span></td>
  <td>{{ fb.name }}</td>
  <td>{{ fb.n_speakers }}</td>
  <td class="mono">{{ fb.created_at }}</td>
  <td class="mono">{{ fb.activated_at or '—' }}</td>
  <td>
    {% if not fb.activated_at %}
    <form method="post" action="/failure/activate" style="display:inline;margin:0">
      <input type="hidden" name="batch_id" value="{{ fb.id }}">
      <button class="btn-green stage-btn" type="submit">Activate</button>
    </form>
    {% else %}<span class="badge s-skipped">activated</span>{% endif %}
  </td>
</tr>
{% endfor %}
</table>
{% else %}
<p class="mono">No failure batches.</p>
{% endif %}

<!-- ===================== RECENT RUNS ===================== -->
<h2>Recent Batch Runs</h2>
<table>
<tr><th>Run</th><th>Stage</th><th>Type</th><th>Started</th><th>Finished</th>
    <th>Tried</th><th>OK</th><th>Fail</th></tr>
{% for r in recent_runs %}
<tr>
  <td class="mono">{{ r.run_id[:8] }}</td>
  <td>{{ r.stage }}</td>
  <td>{{ r.batch_type }}</td>
  <td class="mono">{{ r.started_at }}</td>
  <td class="mono">{{ r.finished_at or '…' }}</td>
  <td>{{ r.n_attempted }}</td>
  <td><span class="badge s-success">{{ r.n_success }}</span></td>
  <td><span class="badge s-failed">{{ r.n_failed }}</span></td>
</tr>
{% endfor %}
</table>

<!-- ===================== SPEAKER SEARCH ===================== -->
<h2>Speaker Lookup</h2>
<form method="get" action="/search">
  <div class="form-row" style="margin-bottom:10px">
    <input type="text" name="q" value="{{ query or '' }}" placeholder="name or speaker_id" style="width:300px">
    <button type="submit">Search</button>
  </div>
</form>
{% if search_results is not none %}
  {% if search_results %}
  <table>
  <tr><th>Name</th><th>Country</th><th>Dataset</th>
      <th>Q</th><th>F</th><th>US</th><th>CV</th><th>A</th><th>B</th><th>C</th><th>D</th><th></th></tr>
  {% for r in search_results %}
  <tr>
    <td>{{ r.name_cleaned }}</td>
    <td>{{ r.country }}</td>
    <td class="mono" style="font-size:10px">{{ r.source_dataset }}</td>
    <td><span class="badge s-{{ r.query_status }}">{{ r.query_status[:3] }}</span></td>
    <td><span class="badge s-{{ r.fetch_status }}">{{ r.fetch_status[:3] }}</span></td>
    <td><span class="badge s-{{ r.url_synth_status }}">{{ r.url_synth_status[:3] }}</span></td>
    <td><span class="badge s-{{ r.cv_synth_status }}">{{ r.cv_synth_status[:3] }}</span></td>
    <td><span class="badge s-{{ r.annotate_a_status }}">{{ r.annotate_a_status[:3] }}</span></td>
    <td><span class="badge s-{{ r.annotate_b_status }}">{{ r.annotate_b_status[:3] }}</span></td>
    <td><span class="badge s-{{ r.annotate_c_status }}">{{ r.annotate_c_status[:3] }}</span></td>
    <td><span class="badge s-{{ r.annotate_d_status }}">{{ r.annotate_d_status[:3] }}</span></td>
    <td><a href="/speaker/{{ r.speaker_id }}">detail</a></td>
  </tr>
  {% endfor %}
  </table>
  {% else %}<p class="mono">No results.</p>{% endif %}
{% endif %}

<p style="margin-top:28px;color:#30363d;font-size:10px">
  Auto-refresh every 30s &nbsp;·&nbsp; <a href="/" style="color:#30363d">refresh now</a>
  &nbsp;·&nbsp; <a href="/api/status" style="color:#30363d">JSON status</a>
</p>
<script>setTimeout(()=>location.reload(),30000);</script>

<!-- ===================== LOG POLLING ===================== -->
<script>
let _pollTimer = null;
let _logOffset = 0;
let _logName   = null;

function showLog(name) {
  if (!name) return;
  document.getElementById('log-select').value = name;
  startPolling(name);
}

function startPolling(name) {
  if (!name) return;
  stopPolling();
  _logName   = name;
  _logOffset = 0;
  document.getElementById('log-out').textContent = '';
  document.getElementById('log-status').textContent = 'streaming ' + name + '…';
  _pollTimer = setInterval(pollLog, 1000);
  pollLog();
}

function stopPolling() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
  document.getElementById('log-status').textContent = '';
}

function pollLog() {
  if (!_logName) return;
  fetch('/api/log/' + encodeURIComponent(_logName) + '?offset=' + _logOffset)
    .then(r => r.json())
    .then(d => {
      if (d.text) {
        const box = document.getElementById('log-out');
        box.textContent += d.text;
        box.scrollTop = box.scrollHeight;
      }
      _logOffset = d.offset;
      if (!d.running) {
        stopPolling();
        const rc = d.returncode != null ? d.returncode : '?';
        document.getElementById('log-status').textContent =
          rc === 0 ? 'finished OK' : 'finished (exit ' + rc + ')';
      }
    })
    .catch(() => stopPolling());
}
</script>

</body>
</html>
"""

SPEAKER_DETAIL_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><title>{{ speaker.name_cleaned }}</title>
<style>
body{font-family:'Courier New',monospace;background:#0d1117;color:#e6edf3;padding:20px;font-size:13px}
h1{color:#58a6ff}h2{color:#79c0ff;margin:16px 0 8px;border-bottom:1px solid #30363d;padding-bottom:4px}
.badge{display:inline-block;padding:1px 7px;border-radius:12px;font-size:11px;font-weight:bold}
.s-pending{background:#21262d;color:#8b949e}.s-running{background:#1f4287;color:#58a6ff}
.s-success{background:#1a4731;color:#3fb950}.s-failed{background:#4d1f1f;color:#f85149}
.s-skipped{background:#272d3a;color:#8b949e}
pre{background:#1c2128;padding:10px;border-radius:4px;overflow-x:auto;font-size:11px;max-height:400px;white-space:pre-wrap}
table{width:100%;border-collapse:collapse}
th{color:#8b949e;text-align:left;padding:5px 8px;border-bottom:1px solid #21262d;font-size:11px}
td{padding:4px 8px;border-bottom:1px solid #1c2128;font-size:12px}
a{color:#58a6ff}
</style>
</head>
<body>
<p><a href="/">← Dashboard</a></p>
<h1>{{ speaker.name_cleaned }}</h1>
<p class="mono">{{ speaker.country }} · {{ speaker.source_dataset }}
   · {{ speaker.n_sentences }} sentences · {{ speaker.min_date }} – {{ speaker.max_date }}</p>

<h2>Stages</h2>
<table>
<tr><th>Stage</th><th>Status</th><th>At</th><th>Error</th></tr>
{% for stage, status, at, err in stages %}
<tr>
  <td>{{ stage }}</td>
  <td><span class="badge s-{{ status }}">{{ status }}</span></td>
  <td class="mono" style="font-size:10px">{{ at or '' }}</td>
  <td style="color:#f85149;font-size:10px">{{ err or '' }}</td>
</tr>
{% endfor %}
</table>

<h2>URLs ({{ urls|length }})</h2>
<table>
<tr><th>Rank</th><th>URL</th><th>Lang</th><th>Fetch</th><th>Chars</th><th>Synth</th></tr>
{% for u in urls %}
<tr>
  <td>{{ u.search_rank }}</td>
  <td style="font-size:10px;max-width:360px;word-break:break-all">
    <a href="{{ u.url }}" target="_blank">{{ u.url[:80] }}{% if u.url|length > 80 %}…{% endif %}</a>
  </td>
  <td>{{ u.query_language }}</td>
  <td><span class="badge s-{{ u.fetch_status }}">{{ u.fetch_status }}</span></td>
  <td>{{ u.cleaned_text_len or '' }}</td>
  <td><span class="badge s-{{ u.synthesis_status }}">{{ u.synthesis_status }}</span></td>
</tr>
{% endfor %}
</table>

{% if cv %}
<h2>CV</h2>
<pre>{{ cv.cv_text }}</pre>
<p class="mono" style="font-size:10px">Model: {{ cv.model }} · prompt v{{ cv.prompt_version }}
  · {{ cv.n_sources_used }} sources · {{ cv.created_at }}</p>
{% endif %}

{% for ann in annotations %}
<h2>Group {{ ann.group_name }} annotation</h2>
<pre>{{ ann.annotation_json }}</pre>
<p class="mono" style="font-size:10px">Model: {{ ann.model }} · prompt v{{ ann.prompt_version }}
  · {{ ann.annotated_at }}</p>
{% endfor %}
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

STAGE_SCRIPTS = {
    "query":      "batch_query.py",
    "fetch":      "batch_fetch.py",
    "url_synth":  "batch_synthesize_url.py",
    "cv_synth":   "batch_synthesize_cv.py",
    "annotate_a": "batch_annotate_a.py",
    "annotate_b": "batch_annotate_b.py",
    "annotate_c": "batch_annotate_c.py",
    "annotate_d": "batch_annotate_d.py",
}


def _base_ctx(**extra):
    with get_conn() as conn:
        stage_counts    = get_stage_counts(conn)
        country_stats   = conn.execute(
            """SELECT country,
                 COUNT(*) AS total,
                 SUM(query_status='success') AS queried,
                 SUM(fetch_status='success') AS fetched,
                 SUM(url_synth_status='success') AS url_synth,
                 SUM(cv_synth_status='success') AS cv,
                 SUM(annotate_a_status='success') AS ann_a,
                 SUM(annotate_b_status='success') AS ann_b,
                 SUM(annotate_c_status='success') AS ann_c,
                 SUM(annotate_d_status='success') AS ann_d
               FROM speakers GROUP BY country ORDER BY country"""
        ).fetchall()
        failure_batches = conn.execute(
            "SELECT * FROM failure_batches ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        recent_runs     = conn.execute(
            "SELECT * FROM batch_runs ORDER BY started_at DESC LIMIT 25"
        ).fetchall()

    try:
        loaded_models = get_loaded_models()
    except Exception:
        loaded_models = []

    ctx = dict(
        now=now_iso(),
        db_path=config.DB_PATH,
        stage_counts=stage_counts,
        country_stats=[dict(r) for r in country_stats],
        failure_batches=[dict(r) for r in failure_batches],
        recent_runs=[dict(r) for r in recent_runs],
        loaded_models=loaded_models,
        lock=read_llm_lock(),
        processes=_pm.list_all(),
        stage_scripts=STAGE_SCRIPTS,
        all_stages=list(STAGE_SCRIPTS.keys()),
        query=None,
        search_results=None,
        flash_msg=None,
        flash_ok=True,
    )
    ctx.update(extra)
    return ctx


def _flash(msg, ok=True):
    return {"flash_msg": msg, "flash_ok": ok}


def _render(**extra):
    return render_template_string(TEMPLATE, **_base_ctx(**extra))


# ---------------------------------------------------------------------------
# Routes — main
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return _render()


@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    results = []
    if q:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM speakers WHERE name_cleaned LIKE ? OR speaker_id LIKE ? "
                "ORDER BY name_cleaned LIMIT 100",
                (f"%{q}%", f"%{q}%"),
            ).fetchall()
        results = [dict(r) for r in rows]
    return _render(query=q, search_results=results)


@app.route("/speaker/<speaker_id>")
def speaker_detail(speaker_id):
    with get_conn() as conn:
        speaker     = conn.execute("SELECT * FROM speakers WHERE speaker_id=?", (speaker_id,)).fetchone()
        if not speaker:
            return "Not found", 404
        urls        = conn.execute(
            "SELECT * FROM speaker_urls WHERE speaker_id=? ORDER BY search_rank", (speaker_id,)
        ).fetchall()
        cv          = conn.execute("SELECT * FROM speaker_cvs WHERE speaker_id=?", (speaker_id,)).fetchone()
        annotations = conn.execute(
            "SELECT * FROM speaker_annotations WHERE speaker_id=? ORDER BY group_name", (speaker_id,)
        ).fetchall()

    stages = [
        ("query",      speaker["query_status"],      speaker["query_at"],      speaker["query_error"]),
        ("fetch",      speaker["fetch_status"],       speaker["fetch_at"],      speaker["fetch_error"]),
        ("url_synth",  speaker["url_synth_status"],   speaker["url_synth_at"],  speaker["url_synth_error"]),
        ("cv_synth",   speaker["cv_synth_status"],    speaker["cv_synth_at"],   speaker["cv_synth_error"]),
        ("annotate_a", speaker["annotate_a_status"],  speaker["annotate_a_at"], speaker["annotate_a_error"]),
        ("annotate_b", speaker["annotate_b_status"],  speaker["annotate_b_at"], speaker["annotate_b_error"]),
        ("annotate_c", speaker["annotate_c_status"],  speaker["annotate_c_at"], speaker["annotate_c_error"]),
        ("annotate_d", speaker["annotate_d_status"],  speaker["annotate_d_at"], speaker["annotate_d_error"]),
    ]
    ann_display = []
    for a in annotations:
        d = dict(a)
        try:
            d["annotation_json"] = json.dumps(json.loads(d["annotation_json"]), indent=2, ensure_ascii=False)
        except Exception:
            pass
        ann_display.append(d)

    return render_template_string(SPEAKER_DETAIL_TEMPLATE,
        speaker=dict(speaker), urls=[dict(u) for u in urls],
        cv=dict(cv) if cv else None, annotations=ann_display, stages=stages)


# ---------------------------------------------------------------------------
# Routes — setup
# ---------------------------------------------------------------------------

@app.route("/setup/init", methods=["POST"])
def setup_init():
    try:
        init_db()
        return _render(**_flash("Database initialised."))
    except Exception as e:
        return _render(**_flash(f"Error: {e}", ok=False))


@app.route("/setup/import", methods=["POST"])
def setup_import():
    try:
        proc = _launch("import_speakers", "import_speakers.py")
        return _render(**_flash(f"import_speakers started (PID {proc['pid']})."))
    except Exception as e:
        return _render(**_flash(f"Error: {e}", ok=False))


# ---------------------------------------------------------------------------
# Routes — pipeline launch
# ---------------------------------------------------------------------------

@app.route("/pipeline/launch", methods=["POST"])
def pipeline_launch():
    stage  = request.form.get("stage", "")
    script = request.form.get("script", "")
    if stage not in STAGE_SCRIPTS:
        return _render(**_flash(f"Unknown stage '{stage}'.", ok=False))
    try:
        proc = _launch(f"stage_{stage}", script)
        return _render(**_flash(f"Stage '{stage}' launched (PID {proc['pid']})."))
    except RuntimeError as e:
        return _render(**_flash(str(e), ok=False))
    except Exception as e:
        return _render(**_flash(f"Launch error: {e}", ok=False))


@app.route("/pipeline/orchestrate", methods=["POST"])
def pipeline_orchestrate():
    try:
        proc = _launch("orchestrator", "orchestrator.py")
        return _render(**_flash(f"Orchestrator launched (PID {proc['pid']})."))
    except RuntimeError as e:
        return _render(**_flash(str(e), ok=False))
    except Exception as e:
        return _render(**_flash(f"Error: {e}", ok=False))


# ---------------------------------------------------------------------------
# Routes — test pipeline
# ---------------------------------------------------------------------------

@app.route("/test/launch", methods=["POST"])
def test_launch():
    n       = request.form.get("n", "5").strip()
    country = request.form.get("country", "").strip().upper()
    stage   = request.form.get("stage", "").strip()
    verbose = request.form.get("verbose") == "1"
    keep    = request.form.get("keep") == "1"

    args = ["--n", n, "--keep"]          # always keep so dashboard can inspect
    if country:
        args += ["--country", country]
    if stage:
        args += ["--stage", stage]
    if verbose:
        args.append("--verbose")
    if not keep:
        args = [a for a in args if a != "--keep"]   # remove keep if user didn't tick it

    try:
        proc = _pm.start("test_pipeline",
                         [sys.executable, str(HERE / "test_pipeline.py")] + args,
                         force=True)
        return _render(**_flash(
            f"Test pipeline launched (PID {proc['pid']})  "
            f"n={n}" + (f" country={country}" if country else "") +
            (f" stage={stage}" if stage else "")
        ))
    except RuntimeError as e:
        return _render(**_flash(str(e), ok=False))
    except Exception as e:
        return _render(**_flash(f"Error: {e}", ok=False))


# ---------------------------------------------------------------------------
# Routes — process control
# ---------------------------------------------------------------------------

@app.route("/process/kill", methods=["POST"])
def process_kill():
    name = request.form.get("name", "")
    if _pm.kill(name):
        return _render(**_flash(f"Process '{name}' killed."))
    return _render(**_flash(f"Process '{name}' not found.", ok=False))


@app.route("/process/restart", methods=["POST"])
def process_restart():
    name = request.form.get("name", "")
    info = _pm.status(name)
    if not info:
        return _render(**_flash(f"Process '{name}' not found.", ok=False))
    try:
        # Re-launch with the same command, force-killing if still alive
        cmd = info["cmd"].split()
        proc = _pm.start(name, cmd, force=True)
        return _render(**_flash(f"Process '{name}' restarted (PID {proc['pid']})."))
    except Exception as e:
        return _render(**_flash(f"Restart error: {e}", ok=False))


# ---------------------------------------------------------------------------
# Routes — LLM management
# ---------------------------------------------------------------------------

@app.route("/llm/load", methods=["POST"])
def llm_load():
    model_id = request.form.get("model_id", "").strip()
    if not model_id:
        return _render(**_flash("No model ID provided.", ok=False))
    try:
        load_model(model_id)
        return _render(**_flash(f"Model '{model_id}' loaded."))
    except Exception as e:
        return _render(**_flash(f"Error loading model: {e}", ok=False))


@app.route("/llm/unload", methods=["POST"])
def llm_unload():
    instance_id = request.form.get("instance_id", "").strip()
    try:
        unload_model(instance_id)
        return _render(**_flash(f"Model '{instance_id}' unloaded."))
    except Exception as e:
        return _render(**_flash(f"Error unloading: {e}", ok=False))


# ---------------------------------------------------------------------------
# Routes — failure batches
# ---------------------------------------------------------------------------

@app.route("/failure/activate", methods=["POST"])
def failure_activate():
    batch_id = request.form.get("batch_id", "")
    try:
        with get_conn() as conn:
            ids = activate_failure_batch(conn, int(batch_id))
        return _render(**_flash(
            f"Failure batch {batch_id} activated — {len(ids)} speakers reset to pending."
        ))
    except Exception as e:
        return _render(**_flash(f"Error: {e}", ok=False))


# ---------------------------------------------------------------------------
# API routes (JSON)
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    with get_conn() as conn:
        counts = get_stage_counts(conn)
    return jsonify({
        "stage_counts": counts,
        "llm_lock":     read_llm_lock(),
        "processes":    _pm.list_all(),
    })


@app.route("/api/log/<name>")
def api_log(name):
    offset  = int(request.args.get("offset", 0))
    text, new_offset = _pm.get_log(name, offset)
    info = _pm.status(name)
    return jsonify({
        "text":       text,
        "offset":     new_offset,
        "running":    info["running"] if info else False,
        "returncode": info["returncode"] if info else None,
    })


@app.route("/api/processes")
def api_processes():
    return jsonify(_pm.list_all())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int,  default=config.DASHBOARD_PORT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    init_db()

    print(f"Dashboard:  http://0.0.0.0:{args.port}")
    print(f"SSH tunnel: ssh -L {args.port}:localhost:{args.port} tom@workstation")
    print(f"Database:   {config.DB_PATH}")
    app.run(host=config.DASHBOARD_HOST, port=args.port,
            debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
