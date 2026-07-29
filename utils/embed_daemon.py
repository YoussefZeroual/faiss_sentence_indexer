#!/usr/bin/env python3
import logging
import queue
import sys
import threading
import time
from multiprocessing.connection import Listener
import numpy as np
torch_imported = False
MODEL_NAME = "BAAI/bge-m3"
TOKEN_MODEL_NAME ='intfloat/multilingual-e5-base'
HOST, PORT = "localhost", 6000
BACKLOG = 256
MAX_BATCH_SIZE = 256
BATCH_WINDOW_S = 0.015
KEEP_MODEL_LOADED_TIMEOUT = 10 #seconds
device = 'None'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("embed_daemon")

model = None
tokenizer = None
token_model = None
torch = None
F = None
Tensor = None
AutoTokenizer = None
AutoModel = None
SentenceTransformer = None
use_last_n_layers = False
last_connection_time = 0


def handle_timeout():
    global last_connection_time
    logger.info("Last connecttion time = %s",last_connection_time)
    if last_connection_time == 0:
        t0 = time.perf_counter()
        last_connection_time +=t0
        if last_connection_time>=KEEP_MODEL_LOADED_TIMEOUT:
            model = None
            token_model = None

def load_models(token_mode=False):
    global model
    global tokenizer
    global token_model
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device is ",device)
    if (token_mode) and (token_model is None):
        model = None
        logger.info("Loading model %s",TOKEN_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(TOKEN_MODEL_NAME)
        token_model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
        token_model.to(device)
    elif (not token_mode) and (model is None):
        token_model = None
        logger.info("Loading model %s",MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)
        model.to(device)
        logger.info("Models loaded successfully")
def _ensure_imports():
    global torch, F, Tensor, AutoTokenizer, AutoModel, SentenceTransformer,device
    if torch is None:
        import torch
        import torch.nn.functional as F
        from torch import Tensor
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import SentenceTransformer
batch_queue: "queue.Queue" = queue.Queue()  # items: (sentences_list, result_queue)
def all_but_the_top(X, n_components=3):
    logger.info("applying 'All but the top' to embeddings:%s",X.shape)
    import torch
    import numpy as np

    was_numpy = isinstance(X, np.ndarray)
    if was_numpy:
        X = torch.from_numpy(X)

    X = X - X.mean(dim=0, keepdim=True)
    n_samples, n_features = X.shape
    q = min(n_components, n_samples, n_features)
    if q > 0:
        U, S, V = torch.pca_lowrank(X, q=q)
        P = V[:, :q]
        X = X - X @ P @ P.T

    return X.numpy() if was_numpy else X


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def average_pool_last_n_layers(hidden_states: tuple[Tensor, ...],
                                attention_mask: Tensor,
                                num_layers: int = 4) -> Tensor:
    # hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    stacked = torch.stack(hidden_states[-num_layers:])   # (n, batch, seq, hidden)
    layer_avg = stacked.mean(dim=0)                       # (batch, seq, hidden)
    return average_pool(layer_avg, attention_mask)

def batching_worker():
    """Single thread owns the GPU. Pulls whatever chunks are waiting,
    concatenates them into one encode() call, splits results back out."""
    while True:
        sentences, result_q, token_mode = batch_queue.get()  # blocks for first item
        batch_items = [(sentences, result_q)]
        batch_sentences = list(sentences)
        load_models(token_mode=token_mode)
        deadline = time.monotonic() + BATCH_WINDOW_S
        while len(batch_sentences) < MAX_BATCH_SIZE:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                s, rq = batch_queue.get(timeout=timeout)
            except queue.Empty:
                break
            batch_items.append((s, rq))
            batch_sentences.extend(s)

        try:
            if token_mode:
                batch_dict = tokenizer(batch_sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
                batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
                if use_last_n_layers:
                    outputs = token_model(**batch_dict, output_hidden_states=True)
                    vecs = average_pool_last_n_layers(outputs.hidden_states, batch_dict["attention_mask"], num_layers=4)
                else:
                    outputs = token_model(**batch_dict, output_hidden_states=False)
                    vecs = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                vecs = vecs.cpu().detach().numpy().astype(np.float32)
            else:
                vecs = model.encode(batch_sentences,batch_size=256).astype(np.float32)
            if device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            logger.exception("Batch encode failed (%d sentences, %d jobs)",
                              len(batch_sentences), len(batch_items))
            for _, rq in batch_items:
                rq.put(("error", str(e)))
            continue

        offset = 0
        for sentences, rq in batch_items:
            n = len(sentences)
            rq.put(("ok", vecs[offset:offset + n]))
            offset += n

    #time.sleep(1)
def handle_client(conn):
    """One thread per connection. Pure I/O + queue handoff, no GPU work here."""
    job_id = None
    try:
        job_id, sentences, chunk_size, token_mode = conn.recv()
        logger.info("Job %s: %d sentences, chunk_size=%d", job_id, len(sentences), chunk_size)

        total = len(sentences)
        print(token_mode)
        all_vecs = []
        conn.send(("progress", 0, total))

        for i in range(0, total, chunk_size):
            chunk = sentences[i:i + chunk_size]
            result_q = queue.Queue()
            batch_queue.put((chunk, result_q, token_mode))
            status, payload = result_q.get()  # waits for the batching worker

            if status == "error":
                raise RuntimeError(payload)

            all_vecs.append(payload)
            done = min(i + chunk_size, total)
            conn.send(("progress", done, total))

        result = np.concatenate(all_vecs, axis=0)
        conn.send(("done", result))
        logger.info("Job %s: completed, sent %d embeddings", job_id, len(result))
        all_vecs = None
    except Exception as e:
        logger.exception("Job %s: error", job_id)
        try:
            conn.send(("error", str(e)))
        except Exception:
            logger.warning("Job %s: could not send error, connection closed", job_id)
    finally:
        conn.close()


def accept_loop(listener):
    while True:
        try:
            conn = listener.accept()
            logger.info("Accepted connection from %s", listener.last_accepted)
        except (EOFError, OSError) as e:
            logger.warning("Bad handshake attempt ignored: %s", e)
            continue
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()


def main():
    logger.info("Daemon starting")
    _ensure_imports()


    threading.Thread(target=batching_worker, daemon=True).start()

    listener = Listener((HOST, PORT), backlog=BACKLOG)
    logger.info("Listener ready on %s:%d (backlog=%d)", HOST, PORT, BACKLOG)

    accept_loop(listener)


if __name__ == "__main__":
    main()
