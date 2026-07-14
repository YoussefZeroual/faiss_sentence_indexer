# -*- coding: utf-8 -*-
#prend en argument le répertoire d'une collection du Lexiscoscope, et crée, pour chaque fichier XML, une liste d'embeddings (objet pickle).
import time
import logging
import numpy as np
import json
import glob
from makeIndex import load_embeddings
from searchEmbedding import load_metadata
from utils.embed_client import encode
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #filename='/home/miai_guest/zeroualy/module_faiss/app.log'
)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
import conllu
from lxml import etree

#---- helper functions -
def parse_conllu_raw_entries(file_content):
    return [s.strip() for s in file_content.split('\n\n') if s.strip()]

def concat_forms(text):
    raw_text = re.findall(r'^\d+\s+(\S+)', text, re.MULTILINE)
    raw_text = " ".join(raw_text)
    raw_text = fix_punctuation_spaces(raw_text)
    return raw_text
def get_sent_id(text):
    match = re.search(r'^#sent_id\s*=\s*(\S+)', text, re.MULTILINE)
    return match.group(1) if match else None

#--------parsing functions------
def parse_conllu_fast(file_path):
    metadata = {"sent_id": [], "raw_text": []}
    sent_list = []
    sent_id = None
    text_raw = None

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    for line in content.splitlines():
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

    #check if the raw text is empty to fallback to 2nd parsing method
    if (sent_list == []) or (sum(x is None for x in sent_list) >= 5):
        logger.warning("sent_list has 5 None entries, falling back to form concatenation method")
        raw_entries = parse_conllu_raw_entries(content)
        metadata = {"sent_id": [], "raw_text": []}
        sent_list = []
        sent_id = None
        text_raw = None
        for i,sent in enumerate(raw_entries):
            text_raw = concat_forms(sent)
            sent_id = get_sent_id(sent)
            if sent_id is None:
                sent_id = i
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
    with open(filepath,"rb") as f:
        data = f.read()
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(data,parser=parser)
    sentences = tree.xpath("//s")
    len_s = len(sentences)
    print(len_s)
    metadata = {"sent_id":[],
                 "raw_text":[]}
    sent_list = []
    for s in sentences:
        sent_id = s.get("id")
        if s.text is not None and "\n" in s.text and "\t" in s.text:
            logger.warning("detected multiline text inside <s>, using contained CONLLU mode, sentence_id=%s,sentence=%s",sent_id,s.text)
            raw_text = re.findall(r'^\s*\S+\s+(\S+)', s.text, re.MULTILINE)
            raw_text = " ".join(raw_text)
            raw_text = fix_punctuation_spaces(raw_text)
            logger.warning("Parsed sentence: %s",s.text)

        elif s.text is None:
            logger.warning("Sentid=%s:<s> text is empty, looking for children texts",sent_id)
            raw_text = "".join(s.itertext())
            logger.warning("using s.itertext() : sentence=%s",raw_text)

        else:
            raw_text = s.text
            logger.debug("Sentid %s, sent tex: %s",sent_id,raw_text)

        metadata["sent_id"].append(sent_id)
        metadata["raw_text"].append(raw_text)
        sent_list.append(raw_text)

    return sent_list,metadata
def parse_sentence_trs(file_path=None):
    logger.info("mode is trs")
    data = None
    with open(file_path,"rb") as f:
        data = f.read()
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(data,parser=parser)
    sentences = tree.xpath("//Turn")
    len_s = len(sentences)
    print(len_s)
    metadata = {"sent_id":[],
                 "raw_text":[]}
    sent_list = []
    for s in sentences:
        sent_id = s.get("startTime")
        raw_text = "".join(s.itertext())
        raw_text = fix_punctuation_spaces(raw_text).replace(' ',' ')
        logger.info("Sentid %s, sent tex: %s",sent_id,raw_text)

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
        len_s = len(sentence_list)
        ex_time = t1-t0
        logger.info("%s Sentences parsed in %s seconds",len_s,np.round(ex_time,2))
        return sentence_list,metadata
    elif mode == "xml":
        t0 = time.perf_counter()
        logger.info("Parsing xml sentences")
        sentences,metadata = parse_sentences_xml_conllu(file_path)
        len_s = len(sentences)
        t1 = time.perf_counter()
        ex_time = t1-t0
        logger.info("%s Sentences parsed in %s seconds",len_s,np.round(ex_time,2))
        return sentences,metadata
    elif mode == "trs":
        t0 = time.perf_counter()
        logger.info("Parsing trs sentences")
        sentences,metadata = parse_sentence_trs(file_path)
        len_s = len(sentences)
        t1 = time.perf_counter()
        ex_time = t1-t0
        logger.info("%s Sentences parsed in %s seconds",len_s,np.round(ex_time,2))
        return sentences,metadata


def calcEMbeddings(collection_file_path=None, output_file_path=None, mode="conllu",reduce_precision=False,overwrite=False):
    import os
    base, ext = os.path.splitext(collection_file_path)
    if (not overwrite) and os.path.exists(base+".npy") and os.path.exists(base+".json"):
        logger.warning("embedding file and metadata file already exist, loading from %s and %s",base+".npy",base+".json")
        embeddings = load_embeddings(base+".npy")
        metadata= load_metadata(base+".json")
        return embeddings,metadata
    logger.info("parsing sentences, file=%s mode=%s",collection_file_path,mode)
    sentence_list,metadata = parse_sentences(collection_file_path,mode=mode)
    logger.info("Encoding sentences with model")
    import time
    t0 = time.perf_counter()
    embeddings = encode(sentence_list, chunk_size=500)
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
    logger.info("saving metadata")
    with open(output_file,"w",encoding="utf-8") as f:
        json.dump(metadata,f)

def encode_folder(input_folder=None,overwrite=False):
    extensions = [".conllu",".xml",".trs"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(input_folder+"/*"+ext))
    len_f = len(file_list)
    logger.info("Found %s files in folder",len_f)
    found_extentions = [re.findall(r'\.([^.]+)$', f)[0] for f in file_list]
    logger.info("Found formats: %s",list(set(found_extentions)))
    cnt = 1
    for f in file_list:
        logger.info("%s",f)
    for f,ext in zip(file_list,found_extentions):
        logger.info("Encoding file %s/%s filename=%s",cnt,len_f,f)
        embeddings,metadata = calcEMbeddings(f,f.replace(ext,'npy'),ext,overwrite=overwrite)
        save_metadata(metadata,f.replace(ext,"json"))
        cnt +=1
if __name__ == "__main__":
    input_folder ="test"
    encode_folder(input_folder)
