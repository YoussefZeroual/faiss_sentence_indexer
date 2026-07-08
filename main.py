from calcEmbeddings import calcEMbeddings,save_metadata,parse_sentences
from makeIndex import makeIndex
from utils.embed_client import encode
from searchEmbedding import search,load_metadata,load_index
import faiss
import sys
import numpy as np

def main():
    import re
    input_file = sys.argv[1]
    query_str = sys.argv[2]
    import os
    
   
    output_index = re.sub(r'\.(.*?)$','.faiss',input_file)
    output_metadata = re.sub(r'\.(.*?)$','.json',input_file)
    output_embeddings = re.sub(r'\.(.*?)$','.npy',input_file)
    
    
    if not (os.path.exists(output_embeddings)):
    
        embeddings,metadata = calcEMbeddings(input_file,output_embeddings,"conllu",reduce_precision=True)
        save_metadata(metadata,output_metadata)
    elif  not (os.path.exists(output_metadata)):
        __,metadata = parse_sentences(input_file,"conllu")
        save_metadata(metadata,output_metadata)
        index = load_index(output_index)
        embeddings = np.load(output_embeddings)
        
        result = search(query_str,index,faiss.METRIC_INNER_PRODUCT,10,metadata)
    elif not (os.path.exists(output_index)):
        embeddings = np.load(output_embeddings)
        metadata = load_metadata(output_metadata)
        index=makeIndex(embeddings=embeddings,embedding_file_path=None,metric_type=faiss.METRIC_INNER_PRODUCT,index_type="ivfpq",output_file_path=output_index)
        result = search(query_str,index,faiss.METRIC_INNER_PRODUCT,10,metadata)
    
    else:
        embeddings = np.load(output_embeddings)
        metadata = load_metadata(output_metadata)
        index = load_index(output_index)
        result = search(query_str,index,faiss.METRIC_INNER_PRODUCT,10,metadata)
    
    print(result)

if __name__ == "__main__":
    main()