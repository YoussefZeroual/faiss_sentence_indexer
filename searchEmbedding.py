
import json
import logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
from utils.embed_client import encode

import faiss
def load_index(index_file=None):
    if index_file is not None:
        return faiss.read_index(index_file)
    else:
        raise ValueError("file name empty")
def load_metadata(matadata_file_path=None):
    metadata = None
    with open (matadata_file_path,"r") as f:
        metadata = json.load(f)
    if metadata is not None:
        return metadata
    else:
        logger.warning("Couldn't load metadata")
        raise ValueError("Couldnt load metadata")

def embedd_query(query_str=None):

    if query_str is not None:
        embeddings = encode([query_str])
        import numpy as np
        embeddings = np.array(embeddings,dtype=np.float32).reshape(1,-1)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        return embeddings
    else:
        logger.warning("Query is empty")
        raise ValueError("Query is empty")

def search(query_str, index=None, metric_type=None, top_k=10, metadata=None):
    metadata = metadata
    len_metadata = len(metadata["raw_text"])
    query_vector = embedd_query(query_str)
    faiss.normalize_L2(query_vector)
    if hasattr(index, "nprobe"):
        index.nprobe = 8
    distances, indices = index.search(query_vector, top_k)
    matches = [(metadata["sent_id"][idx],metadata["raw_text"][idx], float(distance))
               for idx, distance in zip(indices[0], distances[0])
               if 0 <= idx < len_metadata]
    return matches


if __name__ == "__main__":
    import sys
    metric_type = faiss.METRIC_INNER_PRODUCT
    top_k = 10
    query_str = sys.argv[1]
    index = faiss.read_index("index.faiss")
    logger.info("Loaded index from %s","index.faiss")
    print("ntotal:", index.ntotal)
    print("dimension:", index.d)
    matches = search(query_str=query_str,
        index=index,
        metric_type=faiss.METRIC_INNER_PRODUCT,
        top_k=top_k,
        metadata_file_path= "metadata.json"
        )

    print(matches)