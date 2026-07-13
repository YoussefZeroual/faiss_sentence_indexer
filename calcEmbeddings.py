# -*- coding: utf-8 -*-
#prend en argument le répertoire d'une collection du Lexiscoscope, et crée, pour chaque fichier XML, une liste d'embeddings (objet pickle).
import time
import logging 
import numpy as np
import json
from utils.embed_client import encode
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
import conllu
from lxml import etree
def parse_conllu_fast(file_path):
    metadata = {"sent_id": [], "raw_text": []}
    sent_list = []
    sent_id = None
    text_raw = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                # normalize: strip '#', split on first '=', strip whitespace on both sides
                key_value = line[1:].split("=", 1)
                if len(key_value) != 2:
                    continue
                key, value = key_value[0].strip(), key_value[1].strip()
                if key == "sent_id":
                    sent_id = value
                elif key == "text_raw":
                    text_raw = value
            elif line == "":  # blank line = end of sentence block
                if sent_id is not None:
                    metadata["sent_id"].append(sent_id)
                    metadata["raw_text"].append(text_raw)
                    sent_list.append(text_raw)
                sent_id, text_raw = None, None

    # catch the last sentence if file doesn't end with a blank line
    if sent_id is not None:
        metadata["sent_id"].append(sent_id)
        metadata["raw_text"].append(text_raw)
        sent_list.append(text_raw)

    return sent_list, metadata
from lxml import etree
import re
import json


def fix_punctuation_spaces(text):
    # 1. Fix apostrophes (remove spaces around them)
    text = re.sub(r"\s+'", "'", text)  # space before apostrophe
    text = re.sub(r"'\s+", "'", text)  # space after apostrophe

    # 2. Fix spaces before French punctuation (:, ;, ?, !, »)
    text = re.sub(r'\s+([:;?!])', r'\1', text)  # Remove space before
    text = re.sub(r'([:;?!])(\S)', r'\1 \2', text)  # Add space after if needed

    # 3. Fix French guillemets (« »)
    text = re.sub(r'\s+([»])', r'\1', text)  # Remove space before closing
    text = re.sub(r'([«])\s+', r'\1 ', text)  # Add space after opening

    # 4. Fix regular punctuation (.,)
    text = re.sub(r'\s+([.,])', r'\1', text)  # Remove space before
    text = re.sub(r'([.,])(\S)', r'\1 \2', text)  # Add space after

    # 5. Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def parse_sentences_xml_conllu(filepath):
    data = None
    with open(filepath,"r") as f:
        data = f.read()
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(data,parser=parser)
    sentences = tree.xpath("//s")
    len_s = len(sentences)
    metadata = {"sent_id":[],
                 "raw_text":[]}
    sent_list = []
    for s in sentences:
        sent_id = s.get("id")
        raw_text = re.findall(r'^\s*\S+\s+(\S+)', s.text, re.MULTILINE)
        raw_text = " ".join(raw_text)
        raw_text = fix_punctuation_spaces(raw_text)
        metadata["sent_id"].append(sent_id)
        metadata["raw_text"].append(raw_text)
        sent_list.append(raw_text)
    return sent_list,metadata

def parse_sentences(file_path= None,mode = "conllu"):
    if mode == "conllu":
        t0 = time.perf_counter()
        logger.info("Parsing CONLLU sentences")
        sentence_list,metadata = parse_conllu_fast(file_path)
        t1 = time.perf_counter()
        ex_time = t1-t0
        logger.info("Sentences parsed in %s seconds",np.round(ex_time,2))
        return sentence_list
    elif mode == "xml":
        t0 = time.perf_counter()
        logger.info("Parsing xml sentences")
        sentence_list,metadata = parse_sentences_xml_conllu(file_path)
        t1 = time.perf_counter()
        ex_time = t1-t0
        logger.info("Sentences parsed in %s seconds",np.round(ex_time,2))
        return sentence_list,metadata


def calcEMbeddings(collection_file_path=None, output_file_path=None, mode="conllu",reduce_precision=False):
    sentence_list,metadata = parse_sentences(collection_file_path,mode=mode)
    logger.info("Encoding sentences with model")
    import time
    t0 = time.perf_counter()
    embeddings = encode(sentence_list, chunk_size=10000)
    t1 = time.perf_counter()
    procession_time = t1-t0
    logger.info("Embeddings created in %s seconds",np.round(procession_time,2))
    logger.info("saving embeddings to %s", output_file_path)
    if reduce_precision:
        np.save(output_file_path,embeddings.astype(np.float16))
    else:
        np.save(output_file_path,embeddings)
    logger.info("saved successfully")
    
    return embeddings,metadata
def save_metadata(metadata,output_file=None):
    with open(output_file,"w",encoding="utf-8") as f:
        json.dump(metadata,f)
        
if __name__ == "__main__":
    input_file ="HS36-6030v2tv8.conllu"
    
    embeddings = calcEMbeddings(input_file,"test_2.npy")
