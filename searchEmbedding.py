import numpy as np
import json
import logging
MISSING_SENTENCE = "[phrase manquante]"
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

def embedd_query(query_str=None,token_mode=False):
    if token_mode:
        logger.info("Encoding query with token level mode")
    else:
              logger.info("Encoding query with sentence level mode")
    import time
    t0 = time.perf_counter()
    if query_str is not None:
        logger.info("embedding query: %s",query_str)
        if token_mode:
            embeddings = encode([query_str],chunk_size=1,token_mode=True)
        else:
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

def search(query_vector=None,query_str=None, index=None, metric_type=None, top_k=10, metadata=None,token_mode=False):
    metadata = metadata
    len_metadata = len(metadata["raw_text"])
    if query_vector is not None:
        query_vector =query_vector
    else:
        query_vector = embedd_query(query_str,token_mode)
    faiss.normalize_L2(query_vector)
    if hasattr(index, "nprobe"):
        index.nprobe = 8
    try:
        distances, indices = index.search(query_vector, top_k)
        matches = [(metadata["sent_id"][idx],metadata["raw_text"][idx], np.round(float(distance),3))
                for idx, distance in zip(indices[0], distances[0])
                if 0 <= idx < len_metadata]
    except AssertionError as e:
        logger.warning("Assert error: query was probably encoded using a different model than target embeddings, please reembed target texts")
        return []
    return matches
def search_folder(input_folder=None,query_str=None,query_vector=None,metric_type=faiss.METRIC_INNER_PRODUCT,top_k=10,verbose=True,token_mode=False):
    import time
    t0 = time.perf_counter()
    logger.info("Folder embedding search")
    import glob
    import os
    token_suffix = ""

    if token_mode:
        index_ext = "_token.faiss"
        token_suffix = "_token"
    else:
       index_ext = ".faiss"
    if '*' in input_folder:
        file_list = glob.glob(input_folder)
        file_list = [os.path.splitext(f)[0].replace("_token","")+index_ext  for f in file_list]
    else:
        file_list = glob.glob(input_folder+"/*"+index_ext)

    if token_mode:
        file_list = list(set([f for f in file_list if os.path.splitext(f)[1] ==".faiss" and '_token' in f]))
    else:
        file_list = list(set([f for f in file_list if os.path.splitext(f)[1] ==".faiss"]))
    print(file_list)
    len_f = len(file_list)
    results = []
    skipped = False
    logger.info("Found %s files in folder",len_f)
    if query_vector is not None:
        query_vector =query_vector
    else:
        query_vector = embedd_query(query_str,token_mode)
    faiss.normalize_L2(query_vector)
    for f in file_list:
        base, ext = os.path.splitext(f)
        try:
            index = load_index(f)
        except RuntimeError as e:
            logger.warning("%s index file not found in directory,skipping file",base+".faiss")
            skipped = True
            continue
        try:
            metadata = load_metadata(base.replace("_token","")+".json")
        except FileNotFoundError as e:
            logger.warning("%s metadata file not found in directory,skipping file",base+".json")
            skipped = True
            continue

        result = search(query_str=query_str,query_vector=query_vector, index=index, metric_type=metric_type, top_k=top_k, metadata=metadata,token_mode=token_mode)
        result = [(f,r[0],r[1],float(r[2]))
               for r  in result]
        results.extend(result)
    results.sort(key=lambda x: x[3],reverse=True)
    results = results[:top_k]
    t1 = time.perf_counter()
    exec_time = t1-t0
    logger.info("Folder search executed in %s seconds in %s files",np.round(exec_time,2),len_f)
    if skipped:
        logger.warning("Some index files were skipped because file or corresponding metadata files were not found")
    if results ==[]:
        logger.warning("Search query didn't return any results, input file list probaby empty")
    if verbose:
        print("index file                | Sent id               | Sentence    | similarity score")
        for r in results:
            print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]}")

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
