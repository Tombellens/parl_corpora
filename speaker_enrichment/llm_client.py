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


def load_model(model_id: str, context_length: int = 16384,
               flash_attention: bool = True, **kwargs) -> dict:
    """
    Ask LM Studio to load a model into GPU memory.
    Returns the response dict (includes instance_id and load_time_seconds).
    Raises RuntimeError if the LM Studio server is not running.
    """
    if not is_lm_studio_running():
        raise RuntimeError(
            "LM Studio server is not running. "
            "Start it with: lms server start"
        )

    payload = {
        "model": model_id,
        "context_length": context_length,
        "flash_attention": flash_attention,
        "echo_load_config": True,
        **kwargs,
    }
    r = requests.post(
        f"{LM_STUDIO_BASE_URL}/api/v1/models/load",
        headers=_headers(),
        json=payload,
        timeout=300,   # loading can take a while
    )
    r.raise_for_status()
    result = r.json()
    print(f"  Loaded {model_id} in {result.get('load_time_seconds', '?'):.1f}s  "
          f"(instance: {result.get('instance_id', '?')})")
    return result


def unload_model(instance_id: str) -> dict:
    """
    Unload a model instance from GPU memory.
    `instance_id` is the model identifier (same as model_id in most cases).
    """
    r = requests.post(
        f"{LM_STUDIO_BASE_URL}/api/v1/models/unload",
        headers=_headers(),
        json={"instance_id": instance_id},
        timeout=60,
    )
    r.raise_for_status()
    print(f"  Unloaded {instance_id}")
    return r.json()


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


def chat(messages: list[dict], model: str, temperature: float = 0,
         max_tokens: int = 4096, **kwargs) -> str:
    """
    Send a chat completion request and return the raw response string.
    Retries up to 3 times on transient errors.
    """
    client = _client()
    for attempt in range(3):
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
            if attempt == 2:
                raise
            print(f"  LLM call failed (attempt {attempt+1}/3): {e}  — retrying in 5s")
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
