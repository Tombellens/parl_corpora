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


def unload_all_instances() -> None:
    """
    Unload every loaded instance of every model. Used to guarantee a clean
    slate before loading, so chat requests cannot be misrouted to a stale or
    JIT-spawned default-context (4096) instance.
    """
    try:
        for m in list_models():
            for inst in (m.get("loaded_instances") or []):
                iid = inst.get("id")
                if not iid:
                    continue
                try:
                    requests.post(
                        f"{LM_STUDIO_BASE_URL}/api/v1/models/unload",
                        headers=_headers(),
                        json={"instance_id": iid},
                        timeout=60,
                    )
                    print(f"  Unloaded stale instance {iid}")
                except Exception as e:
                    print(f"  Warning: could not unload {iid}: {e}")
    except Exception as e:
        print(f"  Warning: could not enumerate instances to unload: {e}")


def load_model(model_id: str, context_length: int = 16384,
               flash_attention: bool = True, num_parallel: int = 1,
               **kwargs) -> dict:
    """
    Ask LM Studio to load a model into GPU memory.
    Returns the response dict (includes instance_id and load_time_seconds).
    Raises RuntimeError if the LM Studio server is not running.

    Unloads ALL existing instances first so there is exactly one instance
    serving requests — with multiple instances loaded, LM Studio can route a
    chat completion to a stale/default 4096-context instance, causing
    'n_ctx: 4096' overflow errors even though we asked for 65536.

    num_parallel: number of concurrent sequence slots. LM Studio divides the
    KV-cache context_length across these slots, so each in-flight request only
    gets context_length / num_parallel tokens. Our pipeline issues requests
    sequentially, so we default to 1 to give every request the FULL context
    (with parallel=4 a 65536 context yields only 16384 tokens per request,
    which is what caused the earlier 'n_ctx: 16384' overflow errors).
    """
    if not is_lm_studio_running():
        raise RuntimeError(
            "LM Studio server is not running. "
            "Start it with: lms server start"
        )

    # Clean slate: ensure no other (possibly default-context) instance can
    # serve requests alongside the one we are about to load.
    unload_all_instances()

    # NOTE: this LM Studio build expects FLAT snake_case keys at the top level
    # (a nested "config" object is rejected with "Unrecognized key(s): config").
    payload = {
        "model": model_id,
        "context_length": context_length,
        "flash_attention": flash_attention,
        "parallel": num_parallel,
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
