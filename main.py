#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import faiss
import time
import numpy as np
from calcEmbeddings import calcEmbeddings, save_metadata, parse_sentences,encode_folder
from makeIndex import makeIndex,makeIndex_folder,load_embeddings
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
def process_no_faiss(args):
    t0 = time.perf_counter()
    print("Using no faiss mode: calculating cosine sim directly on embeddings")
    print("-----------")
    results = []
    token_ext = ""
    if args.token_emb:
        token_ext = "_token"
    if not args.no_faiss:
        return False
    query_embs = embedd_query(query_str=args.query,token_mode=args.token_emb,no_daemon=args.no_daemon)
    if args.folder or '*' in args.input_file:
        files = glob.glob(args.input_file)
        files = [f for f in files if os.path.splitext(f)[1] in ['.conllu','.xml','.trs']]


        len_f = len(files)
        print(f"processing {len_f} files")
        for i,f in enumerate(files):
            #query_embs = embedd_query(query_str=args.query,token_mode=args.token_emb)
            base,ext = os.path.splitext(f)
            print(f"Processing file {i}/{len_f}: {base+token_ext+'.npy'}")
            try:
                embs = load_embeddings(base+token_ext+".npy")
                #combined = np.vstack([embs,query_embs])
                #from utils.embed_daemon import all_but_the_top
                #combined = all_but_the_top(combined,2)
                #embs = combined[len(embs):]
                #query_embs = combined[:len(embs)]
                faiss.normalize_L2(embs)
                faiss.normalize_L2(query_embs)
            except FileNotFoundError as e:
                print("File not found:",base+token_ext+".npy","skipping")
                continue
            try:
                result = query_embs@embs.T
            except ValueError as e:
                print("Embedding and query dimension mismatch, skipping file:",base+token_ext+".npy")
                print("query dim",query_embs.shape,"embs dim",embs.shape)
                continue
            result = result[0]
            result = [np.round(float(r),3) for r in result]
            metadata = load_metadata(base+".json")
            results.extend(zip([f]*len(result),metadata["sent_id"],metadata["raw_text"],result))
    else:
        base,ext = os.path.splitext(args.input_file)
        try:
            embs = load_embeddings(base+token_ext+".npy")
            #combined = np.vstack([embs,query_embs])
            #from utils.embed_daemon import all_but_the_top
            #combined = all_but_the_top(combined,5)
            #embs = combined[:len(embs)]
            #query_embs = combined[len(embs):]
            faiss.normalize_L2(embs)
            faiss.normalize_L2(query_embs)
        except FileNotFoundError as e:
            print(f"File not found {base+token_ext+'.npy'}")
            return False
        try:
            result = query_embs@embs.T
        except ValueError as e:
            print("Embedding and query dimension mismatch, skipping file")
            print("query dim",query_embs.shape,"embs dim",embs.shape)
            return False
        result = result[0]
        result = [np.round(float(r),3) for r in result]
        metadata = load_metadata(base+".json")
        results.extend(zip([args.input_file]*len(result),metadata["sent_id"],metadata["raw_text"],result))
    results.sort(key=lambda x:x[3],reverse=True)
    print("--------------")
    print("Query results")
    print("--------------")
    print("Collection file                             | sent_id         | Sentence     | SImilarity score")
    for r in results[:args.top_k]:
        print(r[0],"|   ",r[1],"   ",r[2],"|   ",r[3])
    t1 = time.perf_counter()
    exec_time = np.round(t1-t0,3)
    print(f"Execution time: {exec_time}s")
def parse_args():
    parser = argparse.ArgumentParser(description="Semantic search over a corpus using FAISS.")
    parser.add_argument("input_file", help="Path to the corpus file (.conllu, .xml or .trs)")
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
    parser.add_argument("--encode-only",action="store_true",help="encodes files and creates indexes without running queries (only in folder mode)")
    parser.add_argument("--search-only",action="store_true",help="searches presuming index files already exist to skip checking and improve speed (only in folder mode)")
    parser.add_argument("--regenerate-metadata",action="store_true",help="regenerate metadata of a folder or a file")
    parser.add_argument("--warn",action="store_true",help="enables log for warnings only")
    parser.add_argument("--token-emb",action="store_true",help="enables token level embedding mode instead of sentence embedding")
    parser.add_argument("--no-faiss",action="store_true",help="processes a query using models directly without FAISS indexation, for testing purpose")
    parser.add_argument("--no-daemon",action="store_true",help="loads embedding models locally and use them instead of calling the daemon")
    return parser.parse_args()
def process(args,index, metric_type=faiss.METRIC_INNER_PRODUCT, metadata=None):
    query_str=args.query
    top_k=args.top_k
    import time
    t0 = time.perf_counter()
    result = search(query_str=args.query, index=index, metric_type=faiss.METRIC_INNER_PRODUCT, top_k=args.top_k, metadata=metadata,token_mode=args.token_emb,no_daemon=args.no_daemon)
    t1 = time.perf_counter()
    exec_time = t1 - t0
    print("Sent id               | Sentence    | similarity score")
    for r in result:
        print(f"{r[0]} | {r[1]} |  {r[2]}")
    print("temps d'exécution de la requête Faiss:",np.round(exec_time,2))
def process_folder(args,input_file):
        query_vector = embedd_query(args.query,args.token_emb,no_daemon=args.no_daemon)
        base, ext = os.path.splitext(input_file)
        if '*' in input_file:
            files = glob.glob(input_file)
        else:
            files = glob.glob(base+"/"+"*faiss")
        print("Processing folder: searching similarity in",len(files),"index files")
        if args.force:
            encode_folder(input_file,overwrite=True,token_mode=args.token_emb,no_daemon=args.no_daemon)
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
        logging.getLogger("utils.embed_client").setLevel(logging.ERROR)

    if args.log:
        logging.getLogger("searchEmbedding").setLevel(logging.INFO)
        logging.getLogger("calcEmbeddings").setLevel(logging.INFO)
        logging.getLogger("makeIndex").setLevel(logging.INFO)
        logging.getLogger("utils.embed_client").setLevel(logging.INFO)
    elif args.warn:
        logging.getLogger("searchEmbedding").setLevel(logging.WARNING)
        logging.getLogger("calcEmbeddings").setLevel(logging.WARNING)
        logging.getLogger("makeIndex").setLevel(logging.WARNING)
        logging.getLogger("utils.embed_client").setLevel(logging.WARNING)
    if args.no_faiss:
        process_no_faiss(args)
        return True
    if '*' in input_file:
        files = glob.glob(input_file)
    elif not os.path.exists(input_file):
        sys.exit(f"Error: input file not found: {input_file}")

    if args.regenerate_metadata:
        files = []
        if '*' in input_file:
            files = glob.glob(input_file)
            files = [f for f in files if ".conllu" in f or '.xml' in f or '.trs' in f]
        if folder:
            files = glob.glob(input_file+"/*")
        metadata = None
        len_f = len(files)
        if args.folder or "*" in input_file:
            logger.info("regenerating metadata for %s files",len(files))
            for i,f in enumerate(files):
                logger.info("regenerating metadata for file %s/%s filename=%s",i,len_f,f)
                _,metadata = parse_sentences(f)
                base, ext = os.path.splitext(f)
                save_metadata(metadata,base+".json")
        elif not args.folder:
            logger.info("regenerating metadata for %s",input_file)
            _,metadata = parse_sentences(input_file)
            index = load_index(base+".faiss")
            save_metadata(metadata,base+".json")

        return True
    if ext == ".faiss" :
        metadata = load_metadata(output_metadata)
        index = load_index(output_index)
        process(args,index=index,metadata=metadata)
        return True
    if args.encode_only and (folder or "*" in input_file):
        print("Encoding files")
        files = glob.glob(input_file)
        files = [f for f in files if os.path.splitext(f)[1] in ['.conllu','.xml','.trs']]
        token_ext = ""
        if args.token_emb:
            token_ext = "_token"
        print("checking and encoding",len(files),"files")
        cnt = 1
        len_f = len(files)
        for f in files:
            print(f"processing file {cnt}/{len_f} {f}")
            print("------")
            base,ext = os.path.splitext(f)
            index_file = base+token_ext+'.faiss'
            metadata_file = base+'.json'
            embs_file = base+token_ext+'.npy'
            print(embs_file)
            if not os.path.exists(metadata_file) or args.force:
                if args.force:
                    print("force mode enabled")
                else:
                    print("metadata file not found",metadata_file)
                _,metadata = parse_sentences(f)
                save_metadata(metadata,base+".json")
            else:
                print("Found metadata file",metadata_file)
            if not os.path.exists(embs_file) or args.force:
                if args.force:
                    print("force mode enabled")
                else:
                    print("Embeddings file not found, regenerating",metadata_file)
                embeddings, metadata = calcEmbeddings(collection_file_path=f, output_file_path=embs_file, mode=ext.replace('.','').strip(),
                                               reduce_precision=args.reduce_precision,overwrite=args.force,token_mode=args.token_emb,no_daemon=args.no_daemon)
            else:
                print("Found embeddings file",embs_file)
            if not os.path.exists(index_file) or args.force:
                if args.force:
                    print("force mode enabled")
                else:
                    print("Index file not found, building",index_file)
                index=makeIndex(embeddings=None, embedding_file_path=embs_file,
                               metric_type=faiss.METRIC_INNER_PRODUCT,
                               index_type=args.index_type, output_file_path=index_file)
            else:
                print("Found index file",index_file)
            print("------")
            cnt+=1
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
                                               reduce_precision=args.reduce_precision,overwrite=args.force,token_mode=args.token_emb,no_daemon=args.no_daemon)
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
