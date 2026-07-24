#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import faiss
import numpy as np

from calcEmbeddings import calcEmbeddings, save_metadata, parse_sentences,encode_folder
from makeIndex import makeIndex,makeIndex_folder
from utils.embed_client import encode
from searchEmbedding import search, load_metadata, load_index, search_folder,embedd_query
import logging
MODE_BY_EXT = {".conllu": "conllu", ".xml": "xml",".trs":"trs",".faiss":"faiss"}
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #filename='/home/miai_guest/zeroualy/module_faiss/app.log'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Semantic search over a corpus using FAISS.")
    parser.add_argument("input_file", help="Path to the corpus file (.conllu or .xml)")
    parser.add_argument("query", help="Query string to search for")
    parser.add_argument("--index-type", choices=["flat", "hnsw", "ivfpq"], default="ivfpq",
                         help="FAISS index type (default: flat)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return (default: 10)")
    parser.add_argument("--reduce-precision", action="store_true",
                         help="Save embeddings as float16 to save disk space")
    parser.add_argument("--force", action="store_true",
                         help="Force recomputation of embeddings/metadata/index even if cached files exist")
    parser.add_argument("--folder", action="store_true",
                         help="process a folder")
    parser.add_argument("--log",action="store_true",help="enables logging info")
    parser.add_argument("--encode_only",action="store_true",help="encodes files and creates indexes without running queries (only in folder mode)")
    parser.add_argument("--search_only",action="store_true",help="searches presuming index files already exist to skip checking and improve speed (only in folder mode)")
    parser.add_argument("--regenerate_metadata",action="store_true",help="regenerate metadata of a folder or a file")
    parser.add_argument("--warn",action="store_true",help="enables log for warnings only")
    parser.add_argument("--token_emb",action="store_true",help="enables token level embedding mode instead of sentence embedding")
    return parser.parse_args()
def process(args,index, metric_type=faiss.METRIC_INNER_PRODUCT, metadata=None):
    query_str=args.query
    top_k=args.top_k
    import time
    t0 = time.perf_counter()
    result = search(query_str=args.query, index=index, metric_type=faiss.METRIC_INNER_PRODUCT, top_k=args.top_k, metadata=metadata,token_mode=args.token_emb)
    t1 = time.perf_counter()
    exec_time = t1 - t0
    print("Sent id               | Sentence    | similarity score")
    for r in result:
        print(f"{r[0]} | {r[1]} |  {r[2]}")
    print("temps d'exécution de la requête Faiss:",np.round(exec_time,2))
def process_folder(args,input_file):
        query_vector = embedd_query(args.query,args.token_emb)
        base, ext = os.path.splitext(input_file)
        if '*' in input_file:
            files = glob.glob(input_file)
        else:
            files = glob.glob(base+"/"+"*faiss")
        print("Processing folder: searching similarity in",len(files),"index files")
        if args.force:
            encode_folder(input_file,overwrite=True,token_mode=args.token_emb)
            makeIndex_folder(input_folder=input_file,metric_type=faiss.METRIC_INNER_PRODUCT,index_type="ivfpq",overwrite=True,token_mode= args.token_emb)
        print("------------")
        search_folder(input_file,query_vector=query_vector,metric_type=faiss.METRIC_INNER_PRODUCT,top_k=args.top_k,token_mode=args.token_emb)
def main():
    args = parse_args()
    input_file = args.input_file
    folder = args.folder
    base, ext = os.path.splitext(input_file)
    output_index = base + ".faiss"
    output_metadata = base + ".json"
    output_embeddings = base + ".npy"
    # Index must be rebuilt whenever embeddings were recomputed, not just when it's missing on disk.
    embeddings_missing = args.force or not os.path.exists(output_embeddings) or not os.path.exists(output_metadata)
    index_missing = args.force or (embeddings_missing) or (not os.path.exists(output_index))
    metadata_messing = not os.path.exists(output_metadata)
    if not args.log:
        logging.getLogger("searchEmbedding").setLevel(logging.ERROR)
        logging.getLogger("calcEmbeddings").setLevel(logging.ERROR)
        logging.getLogger("makeIndex").setLevel(logging.ERROR)
    if args.log:
        logging.getLogger("searchEmbedding").setLevel(logging.INFO)
        logging.getLogger("calcEmbeddings").setLevel(logging.INFO)
        logging.getLogger("makeIndex").setLevel(logging.INFO)
    elif args.warn:
        logging.getLogger("searchEmbedding").setLevel(logging.WARNING)
        logging.getLogger("calcEmbeddings").setLevel(logging.WARNING)
        logging.getLogger("makeIndex").setLevel(logging.WARNING)
    if '*' in input_file:
        files = glob.glob(input_file)
    elif not os.path.exists(input_file):
        sys.exit(f"Error: input file not found: {input_file}")

    if args.regenerate_metadata:
        files = glob.glob(base+"/"+"*conllu")
        files.extend (glob.glob(base+"/"+"*xml"))

        metadata = None
        len_f = len(files)
        if args.folder:
            logger.info("regenerating metadata for %s files",len(files))
            for i,f in enumerate(files):
                logger.info("regenerating metadata for file %s/%s filename=%s",i,len_f,f)
                _,metadata = parse_sentences(f)
                base, ext = os.path.splitext(f)
                save_metadata(metadata,base+".json")
            process_folder(args,base)
        elif not args.folder:
            logger.info("regenerating metadata for %s",input_file)
            _,metadata = parse_sentences(input_file)
            index = load_index(base+".faiss")
            save_metadata(metadata,base+".json")
            process(args,index=index,metadata=metadata)

        return True
    if ext == ".faiss" :
        metadata = load_metadata(output_metadata)
        index = load_index(output_index)
        process(args,index=index,metadata=metadata)
        return True
    if args.encode_only and folder:
        encode_folder(input_file,overwrite=args.force)
        makeIndex_folder(input_folder=input_file,metric_type=faiss.METRIC_INNER_PRODUCT,index_type="ivfpq",overwrite=args.force)
        return True
    if args.search_only and folder:
        query_vector = embedd_query(args.query)
        process_folder(args,base)
        return True
    if folder:
        process_folder(args,base)
        return True
    if'*' in input_file:
        process_folder(args,input_file)
        return True
    mode = MODE_BY_EXT.get(ext.lower())

    if embeddings_missing and not index_missing and not args.force:
        index = load_index(output_index)
        metadata = load_metadata(output_metadata)
    elif (mode is None) and (not folder) and (embeddings_missing) and (index_missing):
        sys.exit(f"Error: unsupported file extension '{ext}' (expected one of {list(MODE_BY_EXT)})")

    elif (embeddings_missing and index_missing) or args.force:
        embeddings, metadata = calcEmbeddings(input_file, output_embeddings, mode,
                                               reduce_precision=args.reduce_precision,overwrite=args.force,token_mode=args.token_emb)
        print(embeddings.shape)
        save_metadata(metadata, output_metadata)
        index=makeIndex(embeddings=embeddings, embedding_file_path=None,
                               metric_type=faiss.METRIC_INNER_PRODUCT,
                               index_type=args.index_type, output_file_path=output_index)
        embeddings_missing = False
        index_missing = False

    else:
        embeddings = np.load(output_embeddings)
        metadata = load_metadata(output_metadata)


    if index_missing:
        try:
            index = makeIndex(embeddings=embeddings, embedding_file_path=None,
                               metric_type=faiss.METRIC_INNER_PRODUCT,
                               index_type=args.index_type, output_file_path=output_index)
        except ValueError as e:
            if args.index_type == "ivfpq":
                print(f"Warning: {e}. Falling back to 'flat' index.", file=sys.stderr)
                index = makeIndex(embeddings=embeddings, embedding_file_path=None,
                                   metric_type=faiss.METRIC_INNER_PRODUCT,
                                   index_type="flat", output_file_path=output_index)
            else:
                raise

    if not index_missing:
        index = load_index(output_index)
        if index.ntotal != len(metadata["raw_text"]):
            sys.exit(f"Error: index/metadata mismatch (index has {index.ntotal} vectors, "
                    f"metadata has {len(metadata['raw_text'])} entries). "
                    f"Re-run with --force to rebuild.")
    process(args,index=index,metadata=metadata)


if __name__ == "__main__":
    main()
