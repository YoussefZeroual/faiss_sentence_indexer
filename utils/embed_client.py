import os
import socket
import subprocess
import sys
import time
import uuid
from multiprocessing.connection import Client, Listener
import logging
from utils.embed_daemon import average_pool,average_pool_last_n_layers
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

def encode_no_daemon(sentences=None,token_mode=False):
    from torch import Tensor
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer
    MODEL_NAME = "BAAI/bge-m3"
    TOKEN_MODEL_NAME = 'intfloat/multilingual-e5-base'

    logger.info("Loading models...")
    model = SentenceTransformer(MODEL_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    token_tokenizer = AutoTokenizer.from_pretrained(TOKEN_MODEL_NAME)
    token_model = AutoModel.from_pretrained(TOKEN_MODEL_NAME)

    if token_mode:
        batch_dict = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}  # Move to GPU
        outputs = token_model(**batch_dict, output_hidden_states=True)
        vecs = average_pool_last_n_layers(outputs.hidden_states, batch_dict["attention_mask"], num_layers=4)
        vecs = vecs.cpu().detach().numpy().astype(np.float32)
    else:
        vecs = model.encode(sentences).astype(np.float32)

    return vecs


def encode(sentences, chunk_size=512, show_progress=True,token_mode=False,no_daemon=False):
    if no_daemon:
        logger.info("Using no daemon mode")
        embs = encode_no_daemon(sentences,token_mode=token_mode)
        return embs
    conn = _try_connect()
    if conn is None:
        conn = _start_daemon()

    job_id = str(uuid.uuid4())
    conn.send((job_id, sentences, chunk_size,token_mode))

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
