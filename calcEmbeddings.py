# -*- coding: utf-8 -*-
#prend en argument le répertoire d'une collection du Lexiscoscope, et crée, pour chaque fichier XML, une liste d'embeddings (objet pickle).
import time
import logging
import numpy as np
import json
import glob
import re
from lxml import etree
import time
import json
import os
from makeIndex import load_embeddings
from searchEmbedding import load_metadata,load_index
from utils.embed_client import encode
MISSING_SENTENCE = "[phrase manquante]"
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
AMALGAM_REGEX = r'^\d+\-\d+'
#---- helper functions -
def parse_conllu_raw_entries(file_content):
    return [s.strip() for s in file_content.split('\n\n') if s.strip()]

def is_amalgame(line):
    return "".join(re.findall(AMALGAM_REGEX,line)) if re.findall(AMALGAM_REGEX,line) else None
def has_amalgams(text):
    lines_ = [l if re.findall(AMALGAM_REGEX,l) else None for l in text.split("\n") ]
    if set(lines_) == {None}:
        return False
    else:
        return True
def concat_forms(text):
    tokens = []
    if has_amalgams(text):
        lines = text.split('\n')
        for i,t in enumerate(lines):
            if (not is_amalgame(lines[i-2])) and (not is_amalgame(lines[i-1])):
                tokens.append(t)
        joined_tokens = "\n".join(tokens)
        # capture only amalgame lines or non amagame lines (skips amalgam child lines)
        raw_text = re.findall(r'(?:^\d+\-\d+\s+|^\d+\s+)(\S+)', joined_tokens, re.MULTILINE)
        raw_text = " ".join(raw_text)
        raw_text = fix_punctuation_spaces(raw_text)

    else:
        raw_text = re.findall(r'^\d+\s+(\S+)', text, re.MULTILINE)

        raw_text = " ".join(raw_text)
        raw_text = fix_punctuation_spaces(raw_text)
        tokens = get_tokens(raw_text)
    return raw_text,tokens

def get_sent_id(text):
    match = re.search(r'^#\s*sent_id\s*=\s*(\S+)', text, re.MULTILINE)
    return match.group(1) if match else None
def clean_sentence(sent,filename,sent_id):
    if sent is None or not sent.split():
        logger.warning("Empty sentence, filling with [phrase manquante]],sent_id=%s,filename=%s",sent_id,filename)
        return MISSING_SENTENCE
    return sent.replace("_","").replace("  "," ")
#--------parsing functions------
def get_tokens(text):
    if text is None:
        return text
    match = re.findall(r'\'',text)
    if match !=[]:
        text = text.replace("\'","\'\n").strip()
        text = text.replace(" ","\n").strip()
        return text.split("\n")
    else:
        return text.split(" ")

def parse_conllu_fast(file_path,text=None):
    metadata = {"sent_id": [],
                "raw_text": [],
                "tokens":[]}
    sent_list = []
    sent_id = None
    text_raw = None
    if text is not None:
        content = text
    else:
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
                metadata["tokens"].append(get_tokens(text_raw))
            sent_id, text_raw = None, None
    # catch the last sentence if file doesn't end with a blank line
    if sent_id is not None:
        metadata["sent_id"].append(sent_id)
        text_raw = clean_sentence(text_raw,file_path,sent_id)
        metadata["raw_text"].append(text_raw)
        sent_list.append(text_raw)

    #check if the raw text is empty to fallback to 2nd parsing method
    if (sent_list == []) or (sum(x is None for x in sent_list) >= 3):
        logger.info("sent_list has None entries, falling back to form concatenation method")
        raw_entries = parse_conllu_raw_entries(content)
        metadata = {"sent_id": [], "raw_text": [],"tokens":[]}
        sent_list = []
        sent_id = None
        text_raw = None
        for i,sent in enumerate(raw_entries):
            text_raw,tokens = concat_forms(sent)
            sent_id = get_sent_id(sent)
            if sent_id is None:
                sent_id = i
            text_raw = clean_sentence(text_raw,file_path,sent_id)
            metadata["sent_id"].append(sent_id)
            metadata["raw_text"].append(text_raw)
            metadata["tokens"].append(tokens)
            sent_list.append(text_raw)
    return sent_list, metadata




def fix_punctuation_spaces(text):
    if isinstance(text,list):
        text = ' '.join(text)
    # 1. Fix apostrophes (remove spaces around them)
    text = re.sub(r"\s+\'", "'", text)  # space before apostrophe
    text = re.sub(r"'\s+", "'", text)  # space after apostrophe

    # 2. Fix spaces before French punctuation (:, ;, ?, !, »)
    text = re.sub(r'\s+([.:;?!])', r'\1', text)  # Remove space before
    text = re.sub(r'([.:;?!])(\S)', r'\1 \2', text)  # Add space after if needed

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
    metadata = {"sent_id": [],
                "raw_text": [],
                "tokens":[]}
    sent_list = []
    xml_conllu = 0
    for s in sentences:
        sent_id = s.get("id")
        if s.text is not None and "\n" in s.text:# and "\t" in s.text:
            xml_conllu +=1
            #logger.info("detected multiline text inside <s>, using contained CONLLU mode, sentence_id=%s,",sent_id)
            raw_text,tokens = concat_forms(s.text)
            raw_text = fix_punctuation_spaces(raw_text)
        elif s.text is None:
            logger.warning("Sentid=%s:<s> text is empty, looking for children texts",sent_id)
            raw_text = "".join(s.itertext())
            logger.warning("using s.itertext() : sentence=%s",raw_text)

        else:
            raw_text = s.text
            logger.debug("Sentid %s, sent tex: %s",sent_id,raw_text)
            tokens = get_tokens(raw_text)
        metadata["sent_id"].append(sent_id)
        raw_text = clean_sentence(raw_text,filepath,sent_id)
        metadata["raw_text"].append(raw_text)
        metadata["tokens"].append(tokens)
        sent_list.append(raw_text)
    if xml_conllu > 0:
        logger.info("Detected and parsed %s XML-CoNLLU sentences,filename=%s",xml_conllu,filepath)
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
    metadata = {"sent_id": [],
                "raw_text": [],
                "tokens":[]}
    sent_list = []
    for s in sentences:
        sent_id = s.get("startTime")
        raw_text = "".join(s.itertext())
        raw_text = fix_punctuation_spaces(raw_text).replace(' ',' ')
        metadata["sent_id"].append(sent_id)
        text_raw = clean_sentence(raw_text,file_path,sent_id)
        metadata["raw_text"].append(raw_text)
        metadata["tokens"].append(get_tokens(raw_text))
        sent_list.append(raw_text)
    return sent_list,metadata
def parse_sentences(file_path= None,mode = None):
    base,ext = os.path.splitext(file_path)
    mode = ext.replace(".","")
    if mode == "conllu":
        t0 = time.perf_counter()
        logger.info("Parsing CONLLU sentences,filename=%s",file_path)
        sent_list,metadata = parse_conllu_fast(file_path)
        len_s = len(sent_list)
        n_tokens = 0
        try:
            n_tokens = sum(len(re.findall(r'\w+|[^\w\s]', sent)) for sent in sent_list if sent !=MISSING_SENTENCE)
        except TypeError as e:
            logger.warning("Could'nt calculate number of tokens")
        t1 = time.perf_counter()
        ex_time = t1-t0
        logger.info("Parsed %s sentences, %s tokens in %s seconds",len_s,n_tokens,np.round(ex_time,2))
        return sent_list,metadata
    elif mode == "xml":
        t0 = time.perf_counter()
        logger.info("Parsing xml sentences,filename=%s",file_path)
        sent_list,metadata = parse_sentences_xml_conllu(file_path)
        len_s = len(sent_list)
        n_tokens = sum(len(re.findall(r'\w+|[^\w\s]', sent)) for sent in sent_list if sent !=MISSING_SENTENCE)
        t1 = time.perf_counter()
        ex_time = t1-t0
        logger.info("Parsed %s sentences, %s tokens in %s seconds",len_s,n_tokens,np.round(ex_time,2))
        return sent_list,metadata
    elif mode == "trs":
        t0 = time.perf_counter()
        logger.info("Parsing trs sentences,filename=%s",file_path)
        sent_list,metadata = parse_sentence_trs(file_path)
        len_s = len(sent_list)
        n_tokens = sum(len(re.findall(r'\w+|[^\w\s]', sent)) for sent in sent_list if sent !=MISSING_SENTENCE)
        t1 = time.perf_counter()
        ex_time = t1-t0
        logger.info("Parsed %s sentences, %s tokens in %s seconds",len_s,n_tokens,np.round(ex_time,2))
    else:
        logger.warning("File format not recognized: %s,filename=%s",ext,file_path)
        return None,None
    return sent_list,metadata


def calcEmbeddings(collection_file_path=None, output_file_path=None, mode="conllu",reduce_precision=False,overwrite=False,token_mode=False):
    token_suffix=""
    if token_mode and "_token" not in output_file_path :
        token_suffix = "_token"

    base, ext = os.path.splitext(collection_file_path)
    if (not overwrite) and  (os.path.exists(output_file_path)) and os.path.exists(base+".json"):
        logger.warning("embedding file and metadata file already exist, loading from %s and %s",output_file_path,base+token_suffix+".json")
        embeddings = load_embeddings(base+".npy")
        metadata= load_metadata(base+".json")
        return embeddings,metadata

    logger.info("parsing sentences, file=%s mode=%s",collection_file_path,mode)
    sentence_list,metadata = parse_sentences(collection_file_path,mode=mode)
    logger.info("Encoding sentences with model")

    t0 = time.perf_counter()
    if token_mode:
        logger.info("using token level embedding mode")
        embeddings = encode(sentence_list, chunk_size=100,token_mode=True)
        output_file_path = output_file_path.replace(".npy",token_suffix+".npy")
    else:
        logger.info("using sentence level embedding mode")
        embeddings = encode(sentence_list, chunk_size=100)
    t1 = time.perf_counter()
    procession_time = t1-t0
    logger.info("Embeddings created in %s seconds",np.round(procession_time,2))
    logger.info("saving embeddings to %s", output_file_path.replace("_token_token","_token"))
    if reduce_precision:
        np.save(output_file_path,embeddings.astype(np.float16))
    else:
        np.save(output_file_path,embeddings)
    logger.info("saved successfully")

    return embeddings,metadata
def save_metadata(metadata,output_file=None,token_mode=False):
    logger.info("saving metadata tp %s",output_file)
    with open(output_file,"w",encoding="utf-8") as f:
        json.dump(metadata,f)

def encode_folder(input_folder=None,overwrite=False,token_mode=False):
    extensions = [".conllu",".xml",".trs"]
    if '*' in input_folder:
        file_list = glob.glob(input_folder)
        file_list = [f  for f in file_list if os.path.splitext(f)[1] in extensions]
    else:

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
        embeddings,metadata = calcEmbeddings(f,f.replace(ext,'npy'),ext,overwrite=overwrite,token_mode=token_mode)
        save_metadata(metadata,f.replace(ext,"json"),token_mode=token_mode)
        cnt +=1
if __name__ == "__main__":
    input_folder ="test"
    encode_folder(input_folder)
