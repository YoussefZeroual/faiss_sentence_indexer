# **makeIndex.py** créer pour un répertoire de collection, des index FAISS adaptés à la recherche rapide d'embeddings

import faiss
import numpy as np
import json
import logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def makeIndex(embeddings=None,embedding_file_path=None,metric_type=None,index_type=None,output_file_path=None):
    if embeddings is None:
        embeddings = np.load(embedding_file_path)
    import time
    t0 = time.perf_counter()
    logger.info("making index for %s sentences",embeddings.shape[0])
    embeddings = np.ascontiguousarray(embeddings,dtype=np.float32)
    faiss.normalize_L2(embeddings)
    n,dim = embeddings.shape
    metric_type = metric_type
    
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
            raise ValueError(f"Not enough training points ({n}) for IVFPQ (need ≥ {(2**nbits)*40} for nbits={nbits}); use 'flat' instead")
        index.train(embeddings)
        index.add(embeddings)
    else:
        logger.warning("Index Type unrecognized or empty: %s",index_type)
        raise ValueError(f"Index type unkown or empty: {index_type}")
    faiss.write_index(index,output_file_path)
    t1 = time.perf_counter()
    exec_time = t1-t0
    logger.info("Index created successfully in %s seconds",np.round(exec_time,2))
    logger.info("index written successfully")
    
    
    return index

if __name__ == "__main__":
    from calcEmbeddings import calcEMbeddings
    input_file ="HS36-6030v2tv8.conllu"
    embedding_file_path = "test_2.npy"
    index = makeIndex(embedding_file_path=embedding_file_path,metric_type=faiss.METRIC_INNER_PRODUCT,index_type="ivfpq",output_file_path="index.faiss")
