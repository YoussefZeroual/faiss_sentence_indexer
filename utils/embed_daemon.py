#!/usr/bin/env python3
import json
import logging
import os
import sys
import uuid
from multiprocessing.connection import Listener

import numpy as np
from sentence_transformers import SentenceTransformer

# --- paths ---
PID_FILE = "/tmp/embed_daemon.pid"
READY_FILE = "/tmp/embed_daemon.ready"
LOG_FILE = "/tmp/embed_daemon.log"
PROGRESS_DIR = "/tmp/embed_progress"

os.makedirs(PROGRESS_DIR, exist_ok=True)
model_mapping = {
    "camembert-base":"dangvantuan/sentence-camembert-base",
    "bge-m3":"BAAI/bge-m3"
    }

MODEL_NAME =  model_mapping["bge-m3"]

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("embed_daemon")


def _progress_path(job_id):
    return os.path.join(PROGRESS_DIR, f"{job_id}.json")


def _write_progress(job_id, done, total):
    with open(_progress_path(job_id), "w") as f:
        json.dump({"done": done, "total": total}, f)


def _cleanup_progress(job_id):
    try:
        os.remove(_progress_path(job_id))
    except FileNotFoundError:
        pass


def main():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    logger.info("Daemon starting, PID=%s", os.getpid())

    logger.info("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded successfully")

    listener = Listener(("localhost", 6000), authkey=b"secret")
    with open(READY_FILE, "w") as f:
        f.write("ready")
    logger.info("Listener ready on localhost:6000")

    while True:
        try:
            conn = listener.accept()
            logger.info("Accepted connection from %s", listener.last_accepted)
        except (EOFError, OSError) as e:
            logger.warning("Bad handshake attempt ignored: %s", e)
            continue

        job_id = None
        try:
            job_id, sentences, chunk_size = conn.recv()
            logger.info("Job %s: received %d sentences, chunk_size=%d",
                        job_id, len(sentences), chunk_size)

            all_vecs = []
            total = len(sentences)
            _write_progress(job_id, 0, total)

            for i in range(0, total, chunk_size):
                chunk = sentences[i:i + chunk_size]
                vecs = model.encode(chunk).astype(np.float32)
                all_vecs.append(vecs)
                done = min(i + chunk_size, total)
                _write_progress(job_id, done, total)
                logger.info("Job %s: encoded %d/%d", job_id, done, total)

            result = np.concatenate(all_vecs, axis=0)
            conn.send(("done", result))
            logger.info("Job %s: completed, sent %d embeddings", job_id, len(result))

        except Exception as e:
            logger.exception("Job %s: error", job_id)
            try:
                conn.send(("error", str(e)))
            except Exception:
                logger.warning("Job %s: could not send error, connection closed", job_id)
        finally:
            if job_id is not None:
                _cleanup_progress(job_id)
            conn.close()


if __name__ == "__main__":
    main()
