"""
llm_client.py
=============
Wrapper around the LM Studio REST API for:
  - Listing loaded models            GET  /api/v1/models
  - Loading a model                  POST /api/v1/models/load
  - Unloading a model                POST /api/v1/models/unload
  - Chat completions (OpenAI-compat) POST /v1/chat/completions

Also manages the LLM lock file so only one script uses the GPU at a time.
The lock file is JSON:
    {"pid": 1234, "task": "synthesize_url", "model": "...", "started_at": "..."}
"""

import json
import os
import subprocess
import time
from pathlib import Path

import requests
from openai import OpenAI

from config import (
    LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, LLM_LOCK_FILE,
    LMS_BIN, LMS_SERVER_STARTUP_TIMEOUT,
)


# ---------------------------------------------------------------------------
# LM Studio model management  (uses /api/v1/ REST endpoints)
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {LM_STUDIO_API_KEY}",
        "Content-Type":  "application/json",
    }


# ---------------------------------------------------------------------------
# LM Studio server management  (uses the lms CLI)
# ---------------------------------------------------------------------------

def is_lm_studio_running() -> bool:
    """Return True if the LM Studio server is up and answering requests."""
    try:
        r = requests.get(
            f"{LM_STUDIO_BASE_URL}/api/v1/models",
            headers=_headers(), timeout=3,
        )
        return r.status_code < 500
    except Exception:
        return False


def _lms_run(lms: str, *args, timeout: int = 15) -> tuple[int, str]:
    """Run an lms subcommand, return (returncode, combined output)."""
    try:
        result = subprocess.run(
            [lms, *args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout}s"
    except FileNotFoundError:
        return -1, f"binary not found: {lms}"


def start_lm_studio_server() -> bool:
    """
    Attempt to start the LM Studio HTTP server via `lms server start`.

    NOTE: On headless Linux servers, LM Studio's daemon requires a running
    desktop environment and cannot be auto-started programmatically.
    If this fails, start LM Studio manually (open the GUI or configure it
    as a systemd service) and ensure the API server is enabled in settings.

    Returns True if the server comes up within LMS_SERVER_STARTUP_TIMEOUT,
    False otherwise.
    """
    lms = LMS_BIN if Path(LMS_BIN).exists() else "lms"
    print(f"  LM Studio not running — running `lms server start` in background…")

    # lms server start blocks for the lifetime of the server, so run it
    # as a background process and poll the API separately.
    try:
        subprocess.Popen(
            [lms, "server", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print(f"  ✗ lms binary not found at {lms}.")
        return False

    # Poll until the API responds
    deadline = time.time() + LMS_SERVER_STARTUP_TIMEOUT
    while time.time() < deadline:
        time.sleep(2)
        if is_lm_studio_running():
            print("  ✓ LM Studio server is up.")
            return True

    print(
        f"  ✗ LM Studio did not respond within {LMS_SERVER_STARTUP_TIMEOUT}s.\n"
        f"  On headless servers, start LM Studio manually and enable the API server\n"
        f"  in LM Studio Settings → Local Server, or run: {lms} server start"
    )
    return False


def stop_lm_studio_server() -> None:
    """Stop the LM Studio server via `lms server stop`."""
    lms = LMS_BIN if Path(LMS_BIN).exists() else "lms"
    try:
        subprocess.run([lms, "server", "stop"], timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  LM Studio server stopped.")
    except Exception as e:
        print(f"  Warning: could not stop LM Studio server: {e}")


def list_models() -> list[dict]:
    """Return the list of models known to LM Studio (loaded + available)."""
    r = requests.get(f"{LM_STUDIO_BASE_URL}/api/v1/models", headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json().get("data", [])


def get_loaded_models() -> list[dict]:
    """Return only models that currently have loaded instances."""
    models = list_models()
    return [m for m in models if m.get("loaded_instances")]


def _lms_bin() -> str:
    return LMS_BIN if Path(LMS_BIN).exists() else "lms"


# Remembers the most recent load so chat() can reload the model after a
# runtime crash (after which /v1 would otherwise JIT a 4096 default instance).
_LAST_LOAD: dict | None = None


def unload_all_instances() -> None:
    """
    Unload every loaded instance via the lms CLI. Guarantees a clean slate so
    chat requests cannot be misrouted to a stale or JIT-spawned default-context
    (4096) instance.
    """
    try:
        subprocess.run(
            [_lms_bin(), "unload", "--all"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=60,
        )
    except Exception as e:
        print(f"  Warning: 'lms unload --all' failed: {e}")


def load_model(model_id: str, context_length: int = 16384,
               flash_attention: bool = True, num_parallel: int = 1,
               **kwargs) -> dict:
    """
    Load a model into GPU memory via the `lms` CLI and return
    {"instance_id": ..., "load_time_seconds": ...}.

    IMPORTANT — why the CLI, not the REST API:
    Loading via POST /api/v1/models/load creates an instance that the
    OpenAI-compatible /v1/chat/completions endpoint does NOT use: with JIT
    loading, /v1 spins up its own default-context (4096) instance, producing
    'n_ctx: 4096' overflow errors even though the API instance was 65536.
    `lms load` registers the instance correctly so /v1 serves at the requested
    context. Verified: lms-loaded 65536 instance serves a ~25k-token prompt
    with HTTP 200, whereas an API-loaded one 400s at n_ctx 4096.

    num_parallel: concurrent sequence slots. LM Studio divides context_length
    across these slots (parallel=4 -> each request only gets context/4), so we
    default to 1 to give every sequential request the FULL context.
    """
    if not is_lm_studio_running():
        raise RuntimeError(
            "LM Studio server is not running. Start it with: lms server start"
        )

    # Clean slate so exactly one instance (ours) serves requests.
    unload_all_instances()

    cmd = [
        _lms_bin(), "load", model_id,
        "-c", str(context_length),
        "--parallel", str(num_parallel),
        "--gpu", "max",
        "-y",
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=300,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"lms load failed (exit {result.returncode}):\n{result.stdout}"
        )
    print(f"  Loaded {model_id} in {elapsed:.1f}s "
          f"(ctx={context_length}, parallel={num_parallel}) via lms CLI")

    global _LAST_LOAD
    _LAST_LOAD = {
        "model_id": model_id,
        "context_length": context_length,
        "num_parallel": num_parallel,
    }
    return {"instance_id": model_id, "load_time_seconds": elapsed}


def reload_last_model() -> bool:
    """Reload the most recently loaded model with the same config. Used to
    recover after a runtime crash. Returns True on success."""
    if not _LAST_LOAD:
        return False
    try:
        load_model(
            _LAST_LOAD["model_id"],
            context_length=_LAST_LOAD["context_length"],
            num_parallel=_LAST_LOAD["num_parallel"],
        )
        return True
    except Exception as e:
        print(f"  Reload failed: {e}")
        return False


def unload_model(instance_id: str) -> dict:
    """Unload a model instance from GPU memory via the lms CLI."""
    result = subprocess.run(
        [_lms_bin(), "unload", instance_id],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"  Warning: lms unload {instance_id} exited {result.returncode}: {result.stdout}")
    else:
        print(f"  Unloaded {instance_id}")
    return {"instance_id": instance_id, "returncode": result.returncode}


def unload_all_models() -> None:
    """Unload every currently loaded model."""
    for m in get_loaded_models():
        instance_id = m.get("id") or m.get("instance_id") or m.get("key")
        if instance_id:
            try:
                unload_model(instance_id)
            except Exception as e:
                print(f"  Warning: could not unload {instance_id}: {e}")


# ---------------------------------------------------------------------------
# Chat completions (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def _client() -> OpenAI:
    return OpenAI(base_url=f"{LM_STUDIO_BASE_URL}/v1", api_key=LM_STUDIO_API_KEY)


def _is_crash_or_fallback(err: str) -> bool:
    """
    True if the error indicates the runtime crashed, or that requests are being
    served by a JIT-spawned default (4096) instance instead of our loaded one.
    Both are recovered by reloading the model.
    """
    e = err.lower()
    return (
        "has crashed" in e
        or "exit code" in e
        or "n_ctx: 4096" in e
        or "no models loaded" in e
        or "model_not_found" in e
    )


def chat(messages: list[dict], model: str, temperature: float = 0,
         max_tokens: int = 4096, max_attempts: int = 4, **kwargs) -> str:
    """
    Send a chat completion request and return the raw response string.

    Retries transient errors. If the runtime crashes — or requests start
    hitting a JIT-spawned 4096-context instance — the model is reloaded with
    its last config before retrying, so one crash does not cascade into every
    subsequent request failing at n_ctx 4096.
    """
    client = _client()
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            err = str(e)
            if _is_crash_or_fallback(err):
                print(f"  LLM crash/fallback detected (attempt {attempt+1}/{max_attempts}) "
                      f"— reloading model: {err[:120]}")
                time.sleep(3)
                reload_last_model()
                time.sleep(2)
            else:
                print(f"  LLM call failed (attempt {attempt+1}/{max_attempts}): {e} — retrying in 5s")
                time.sleep(5)


def extract_json(response: str) -> dict | list:
    """
    Pull a JSON object or array out of an LLM response that may be wrapped
    in markdown code fences.
    """
    text = response.strip()
    if "```" in text:
        parts = text.split("```")
        # take the first fenced block
        for part in parts[1::2]:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return json.loads(text)


# ---------------------------------------------------------------------------
# LLM lock file management
# ---------------------------------------------------------------------------

def acquire_llm_lock(task: str, model: str) -> None:
    """
    Claim the LLM lock file for the current process.
    Raises RuntimeError if another live process holds the lock.
    """
    lock_path = Path(LLM_LOCK_FILE)
    if lock_path.exists():
        try:
            info = json.loads(lock_path.read_text())
            pid  = info.get("pid", -1)
            # Check whether that process is still alive
            try:
                os.kill(pid, 0)   # signal 0 = check existence only
                raise RuntimeError(
                    f"LLM already in use by PID {pid} "
                    f"(task={info.get('task')}, model={info.get('model')}, "
                    f"since {info.get('started_at')})"
                )
            except ProcessLookupError:
                print(f"  Stale lock (PID {pid} no longer running) — clearing.")
        except (json.JSONDecodeError, KeyError):
            pass   # corrupt lock file → overwrite

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({
        "pid":        os.getpid(),
        "task":       task,
        "model":      model,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))


def release_llm_lock() -> None:
    lock_path = Path(LLM_LOCK_FILE)
    if lock_path.exists():
        lock_path.unlink()


def read_llm_lock() -> dict | None:
    """Return the current lock info dict, or None if no lock exists."""
    lock_path = Path(LLM_LOCK_FILE)
    if not lock_path.exists():
        return None
    try:
        info = json.loads(lock_path.read_text())
        pid  = info.get("pid", -1)
        try:
            os.kill(pid, 0)
            info["alive"] = True
        except ProcessLookupError:
            info["alive"] = False
        return info
    except (json.JSONDecodeError, OSError):
        return None


def is_llm_locked() -> bool:
    info = read_llm_lock()
    return info is not None and info.get("alive", False)
