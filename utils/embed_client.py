import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from multiprocessing.connection import Client

DAEMON_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_daemon.py")
PID_FILE = "/tmp/embed_daemon.pid"
READY_FILE = "/tmp/embed_daemon.ready"
PROGRESS_DIR = "/tmp/embed_progress"


def _progress_path(job_id):
    return os.path.join(PROGRESS_DIR, f"{job_id}.json")


def _is_daemon_running():
    if not os.path.exists(PID_FILE):
        return False
    with open(PID_FILE) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        os.remove(PID_FILE)
        return False


def _start_daemon():
    subprocess.Popen(
        [sys.executable, DAEMON_SCRIPT],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    for _ in range(120):  # up to ~60s for model load
        if os.path.exists(READY_FILE) and _is_daemon_running():
            return
        time.sleep(0.5)
    raise RuntimeError("Embedding daemon failed to start")


def encode(sentences, chunk_size=32, show_progress=True):
    if not _is_daemon_running():
        _start_daemon()

    job_id = str(uuid.uuid4())
    result_holder = {}

    def _do_request():
        conn = Client(("localhost", 6000), authkey=b"secret")
        conn.send((job_id, sentences, chunk_size))
        status, payload = conn.recv()
        conn.close()
        result_holder["status"] = status
        result_holder["payload"] = payload

    t = threading.Thread(target=_do_request)
    t.start()

    if show_progress:
        path = _progress_path(job_id)
        while t.is_alive():
            try:
                with open(path) as f:
                    p = json.load(f)
                pct = round(100 * p["done"] / p["total"], 1) if p["total"] else 0
                print(f"\rJob {job_id[:8]}: {p['done']}/{p['total']} ({pct}%)", end="", flush=True)
            except (FileNotFoundError, json.JSONDecodeError, ZeroDivisionError):
                pass
            time.sleep(1)
        print()

    t.join()

    if result_holder["status"] == "error":
        raise RuntimeError(result_holder["payload"])
    return result_holder["payload"]