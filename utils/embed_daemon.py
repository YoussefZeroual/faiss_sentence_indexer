#!/usr/bin/env python3
import logging
import sys
from multiprocessing.connection import Listener

import numpy as np
from sentence_transformers import SentenceTransformer

model_mapping = {
    "camembert-base": "dangvantuan/sentence-camembert-base",
    "bge-m3": "BAAI/bge-m3",
}

MODEL_NAME = model_mapping["bge-m3"]

# --- logging (stdout only) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("embed_daemon")


def main():
    logger.info("Daemon starting")

    logger.info("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded successfully")

    # Binding successfully doubles as the "ready" signal - clients detect
    # readiness by successfully connecting, no separate ready flag needed.
    listener = Listener(("localhost", 6000))
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
            conn.send(("progress", 0, total))

            for i in range(0, total, chunk_size):
                chunk = sentences[i:i + chunk_size]
                vecs = model.encode(chunk).astype(np.float32)
                all_vecs.append(vecs)
                done = min(i + chunk_size, total)
                conn.send(("progress", done, total))
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
            conn.close()


if __name__ == "__main__":
    main()
