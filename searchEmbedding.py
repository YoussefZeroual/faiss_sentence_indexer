import numpy as np
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
    logger.info("loading index from:%s",index_file)
    if index_file is not None:
        return faiss.read_index(index_file)
    else:
        raise ValueError("file name empty")
def load_metadata(matadata_file_path=None):
    logger.info("loading metadata from:%s",matadata_file_path)
    metadata = None
    with open (matadata_file_path,"r") as f:
        metadata = json.load(f)
    if metadata is not None:
        return metadata
    else:
        logger.warning("Couldn't load metadata")
        raise ValueError("Couldnt load metadata")

def embedd_query(query_str=None):
    import time
    t0 = time.perf_counter()
    if query_str is not None:
        logger.info("embedding query:%s",query_str)
        embeddings = encode([query_str],chunk_size=1)
        import numpy as np
        t1 = time.perf_counter()
        exec_time = t1-t0
        logger.info("Query encoded in %s seconds",np.round(exec_time,2))
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
    matches = [(metadata["sent_id"][idx],metadata["raw_text"][idx], np.round(float(distance),3))
               for idx, distance in zip(indices[0], distances[0])
               if 0 <= idx < len_metadata]
    return matches
def search_folder(input_folder=None,query_str=None,metric_type=faiss.METRIC_INNER_PRODUCT,top_k=10):
    logger.info("Folder embedding search")
    import glob
    import os

    file_list = glob.glob(input_folder+"/*"+"faiss")[:5]
    len_f = len(file_list)
    results = []
    logger.info("Found %s files in folder",len_f)
    for f in file_list:
        base, ext = os.path.splitext(f)
        logger.info("%s",f)
        index = load_index(f)
        metadata = load_metadata(base+".json")
        result = search(query_str, index=index, metric_type=metric_type, top_k=top_k, metadata=metadata)
        result = [(f,r[0],r[1],float(r[2]))
               for r  in result]
        results.extend(result)
    for r in results:
        print(f"{r[0]} | {r[1]} | {r[2]}")

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
