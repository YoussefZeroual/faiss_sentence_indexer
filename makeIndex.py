# **makeIndex.py** créer pour un répertoire de collection, des index FAISS adaptés à la recherche rapide d'embeddings

import faiss
import numpy as np
import json
import logging
import glob
import time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
def load_embeddings(embedding_file_path):
    embeddings = np.load(embedding_file_path)

    embeddings = np.ascontiguousarray(embeddings,dtype=np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings
def makeIndex(embeddings=None,embedding_file_path=None,metric_type=None,index_type=None,output_file_path=None,overwrite=False):
    import os
    if embedding_file_path is not None:
        base, ext = os.path.splitext(embedding_file_path)
        if not overwrite and (os.path.exists(base+".faiss")):
            logger.info("Index already exist, loading from %s",base+".faiss")
            from searchEmbedding import load_index
            index=load_index(base+".faiss")
            return index
    if embeddings is None:
       embeddings = load_embeddings(embedding_file_path)
    logger.info("making index for %s sentences, index type=%s",embeddings.shape[0],index_type)
    n,dim = embeddings.shape
    metric_type = metric_type
    t0 = time.perf_counter()
    if index_type == "flat":
        index = faiss.IndexFlat(dim,metric_type)
        index.add(embeddings)
    elif index_type == "hnsw":
        hnsw_m = 64
        index = faiss.IndexHNSWFlat(dim, hnsw_m, metric_type)
        index.hnsw.efConstruction = 40
        index.add(embeddings)
        index.hnsw.efSearch = 64
    elif index_type == "ivfpq":
        nlist = 4 * int(np.sqrt(n))
        m = 8
        nbits = 8
        quantizer = faiss.IndexFlat(dim, metric_type)
        index = faiss.IndexIVFPQ(quantizer,dim,nlist,m,nbits,metric_type)
        if n < (2 ** nbits) * 40:
            logger.warning("Not enough training points (%s) for IVFPQ (need ≥ {(2**nbits)*40} for nbits=%s); using 'flat' instead",n,nbits)
            index = faiss.IndexFlat(dim,metric_type)
        index.train(embeddings)
        index.add(embeddings)
    else:
        logger.warning("Index Type unrecognized or empty: %s",index_type)
        raise ValueError(f"Index type unkown or empty: {index_type}")

    faiss.write_index(index,output_file_path)
    t1 = time.perf_counter()
    exec_time = t1-t0
    logger.info("Index created successfully in %s seconds",np.round(exec_time,3))
    logger.info("index written successfully")


    return index
def makeIndex_folder(input_folder=None,metric_type=None,index_type=None,overwrite=False):
    file_list = glob.glob(input_folder+"/*"+"npy")
    len_f = len(file_list)
    logger.info("Found %s files in folder",len_f)
    for f in file_list:
        logger.info("%s",f)
    for f in file_list:
        logger.info("Making index for file %s",f)
        output_file_path = f.replace(".npy",".faiss")
        index = makeIndex(embedding_file_path=f,metric_type=metric_type,index_type=index_type,output_file_path=output_file_path,overwrite=overwrite)


if __name__ == "__main__":
    index = makeIndex_folder("test")
