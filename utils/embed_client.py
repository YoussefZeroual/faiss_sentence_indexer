import os
import socket
import subprocess
import sys
import time
import uuid
from multiprocessing.connection import Client, Listener
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #filename='/home/miai_guest/zeroualy/module_faiss/app.log'
)
logger = logging.getLogger(__name__)
DAEMON_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_daemon.py")


def _try_connect():
    try:
        return Client(("localhost", 6000))
    except (ConnectionRefusedError, OSError):
        return None


def _start_daemon():
    logger.info("starting embedding daemon")
    subprocess.Popen(
        [sys.executable, DAEMON_SCRIPT],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    for _ in range(120):  # up to ~60s for model load
        conn = _try_connect()
        if conn is not None:
            return conn
        time.sleep(0.5)
    raise RuntimeError("Embedding daemon failed to start")


def encode(sentences, chunk_size=512, show_progress=True):
    conn = _try_connect()
    if conn is None:
        conn = _start_daemon()

    job_id = str(uuid.uuid4())
    conn.send((job_id, sentences, chunk_size))

    status, payload = None, None
    while True:
        status, *rest = conn.recv()

        if status == "progress":
            done, total = rest
            if show_progress:
                pct = round(100 * done / total, 1) if total else 0
                print(f"\rJob {job_id[:8]}: {done}/{total} ({pct}%)", end="", flush=True)
            continue

        payload = rest[0]
        break

    conn.close()
    if show_progress:
        print()

    if status == "error":
        raise RuntimeError(payload)
    return payload
