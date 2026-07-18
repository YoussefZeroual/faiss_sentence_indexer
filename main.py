#!/usr/bin/env python3
import argparse
import os
import sys

import faiss
import numpy as np

from calcEmbeddings import calcEMbeddings, save_metadata, parse_sentences,encode_folder
from makeIndex import makeIndex,makeIndex_folder
from utils.embed_client import encode
from searchEmbedding import search, load_metadata, load_index, search_folder,embedd_query
import logging
MODE_BY_EXT = {".conllu": "conllu", ".xml": "xml",".trs":"trs",".faiss":"faiss"}


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic search over a corpus using FAISS.")
    parser.add_argument("input_file", help="Path to the corpus file (.conllu or .xml)")
    parser.add_argument("query", help="Query string to search for")
    parser.add_argument("--index-type", choices=["flat", "hnsw", "ivfpq"], default="flat",
                         help="FAISS index type (default: flat)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return (default: 10)")
    parser.add_argument("--reduce-precision", action="store_true",
                         help="Save embeddings as float16 to save disk space")
    parser.add_argument("--force", action="store_true",
                         help="Force recomputation of embeddings/metadata/index even if cached files exist")
    parser.add_argument("--folder", action="store_true",
                         help="process a folder")
    parser.add_argument("--log",action="store_true",help="disables logging info")
    parser.add_argument("--encode_only",action="store_true",help="encodes files and creates indexes without running queries (only in folder mode)")
    parser.add_argument("--search_only",action="store_true",help="searches presuming index files already exist to skip checking and improve speed (only in folder mode)")
    return parser.parse_args()


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
    index_missing = args.force or (embeddings_missing) and (not os.path.exists(output_index))
    if not args.log:
        logging.disable(logging.WARNING)
    if not os.path.exists(input_file):
        sys.exit(f"Error: input file not found: {input_file}")
    if args.encode_only and folder:
        encode_folder(input_file,overwrite=args.force)
        makeIndex_folder(input_folder=input_file,metric_type=faiss.METRIC_INNER_PRODUCT,index_type="ivfpq",overwrite=args.force)
        return True
    if args.search_only and folder:
        query_vector = embedd_query(args.query)
        print("Processing folder")
        search_folder(input_file,query_vector=query_vector,metric_type=faiss.METRIC_INNER_PRODUCT,top_k=args.top_k)
        return True
    if folder:
        query_vector = embedd_query(args.query)
        print("Processing folder")
        if args.force:
            encode_folder(input_file,overwrite=True)
            makeIndex_folder(input_folder=input_file,metric_type=faiss.METRIC_INNER_PRODUCT,index_type="ivfpq",overwrite=True)
        print("------------")
        search_folder(input_file,query_vector=query_vector,metric_type=faiss.METRIC_INNER_PRODUCT,top_k=args.top_k)
        return True

    mode = MODE_BY_EXT.get(ext.lower())
    if embeddings_missing and not index_missing and not args.force:
        index = load_index(output_index)
        metadata = load_metadata(output_metadata)
    elif (mode is None) and (not folder) and (embeddings_missing) and (index_missing):
        sys.exit(f"Error: unsupported file extension '{ext}' (expected one of {list(MODE_BY_EXT)})")

    elif embeddings_missing and index_missing:
        embeddings, metadata = calcEMbeddings(input_file, output_embeddings, mode,
                                               reduce_precision=args.reduce_precision,overwrite=args.force)
        save_metadata(metadata, output_metadata)
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
    import time
    t0 = time.perf_counter()
    result = search(query_str=args.query, index=index, metric_type=faiss.METRIC_INNER_PRODUCT, top_k=args.top_k, metadata=metadata)
    t1 = time.perf_counter()
    exec_time = t1 - t0
    for r in result:
        print(f"{r[0]} | {r[1]} |  {r[2]}")
    print("temps d'exécution de la requête Faiss:",np.round(exec_time,2))


if __name__ == "__main__":
    main()
