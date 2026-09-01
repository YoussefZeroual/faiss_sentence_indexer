"""
Module d'extraction et d'encodage de corpus linguistiques pour la création des fichiers embeddings .npy.

Ce script analyse des fichiers de corpus (CoNLLU, XML, TRS transcripition les corpus oraux)
et les convertit en représentations vectorielles (embeddings). Il nettoie les textes bruts,
gère les spécificités linguistiques (comme les mots amalgames), et orchestre l'appel
aux modèles d'encodage (utilisation directe du modèle, via le daemon d'embeddings, ou via Ollama).

Fonctionnalités principales :
- Parsing multi-formats de données textuelles et de transcriptions.
- Nettoyage typographique (espaces, ponctuation française) et traitement des amalgames.
- Encodage massif par lots au niveau de la phrase ou du token.
- Sauvegarde synchronisée des métadonnées (.json) et des vecteurs (.npy).
"""
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
# étiquette utiliée pour remplacer les phrases manquantes quand
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
# regex utilisé pour détecter les lignes CoNLLU représentant des amalgames, ex. 'du' --> 'de' et 'le', afin d'ignorer les lignes correspondant aux deux sous-morphèmes et ne garder que celle de la forme amalgamée. Ces lignes sont reconnaissables par leur numérotation, ex. 1 du 1-2 de 1-3 le
AMALGAM_REGEX = r'^\d+\-\d+'
#---- helper functions -
def parse_conllu_raw_entries(file_content):
    """
    Découpe le contenu brut d'un fichier CoNLLU en blocs de phrases distinctes.

    Args:
        file_content (str): Le contenu textuel intégral du fichier CoNLLU.

    Valeur de retour:
        list: Une liste de chaînes de caractères, où chaque élément correspond à un bloc d'annotation de phrase.
    """
    # Sépare le texte sur les doubles retours à la ligne (standard CoNLLU pour délimiter les phrases)
    # et ignore les éventuels blocs vides grâce à la condition if s.strip()
    return [s.strip() for s in file_content.split('\n\n') if s.strip()]

def is_amalgame(line):
    """
    Vérifie si une ligne d'annotation correspond à un mot amalgame (multi-mots).
    En CoNLLU, ces lignes utilisent un intervalle d'identifiants (ex: dans les fichiers CoNLLU, 'du' est décomposé en de et le donc deux lignes correspondant aux deux morphèmes décomposées en plus d'une ligne de la forme amalgamée, le script ne conserve que la ligne de l'amalgame et ignore ses sous-composantes).
    """
    return "".join(re.findall(AMALGAM_REGEX,line)) if re.findall(AMALGAM_REGEX,line) else None
def has_amalgams(text):
    """
    Détermine si un bloc de texte CoNLLU contient au moins une ligne d'amalgame.
    """
    lines_ = [l if re.findall(AMALGAM_REGEX,l) else None for l in text.split("\n") ]
    if set(lines_) == {None}:
        return False
    else:
        return True
def concat_forms(text):
    """
    Reconstruit le texte brut à partir des formes d'un bloc CoNLLU.
    Gère spécifiquement les amalgames en ignorant leurs composants enfants pour éviter les doublons.
    """
    tokens = []
    if has_amalgams(text):
        lines = text.split('\n')
        for i,t in enumerate(lines):
            if (not is_amalgame(lines[i-2])) and (not is_amalgame(lines[i-1])):
                tokens.append(t)
        joined_tokens = "\n".join(tokens)
        # capture only amalgame lines or non amagame lines (skips amalgam child lines)
        # capture les formes de surface via une regexcapure uniquement les lignes des amalgames en ignorant leurs sous-composants
        raw_text = re.findall(r'(?:^\d+\-\d+\s+|^\d+\s+)(\S+)', joined_tokens, re.MULTILINE)
        raw_text = " ".join(raw_text)
        raw_text = fix_punctuation_spaces(raw_text)

    else:
        # Extraction normale si aucun amalgame n'est présent
        raw_text = re.findall(r'^\d+\s+(\S+)', text, re.MULTILINE)

        raw_text = " ".join(raw_text)
        raw_text = fix_punctuation_spaces(raw_text)
        tokens = get_tokens(raw_text)
    return raw_text,tokens

def get_sent_id(text):
    """
    Extrait l'identifiant unique de la phrase (sent_id) depuis les métadonnées du bloc CoNLLU.
    """
    match = re.search(r'^#\s*sent_id\s*=\s*(\S+)', text, re.MULTILINE)
    return match.group(1) if match else None
def clean_sentence(sent,filename,sent_id):
    """
    Nettoie une phrase reconstruite et gère les valeurs nulles en les marquant par l'étiquette [phrase manquante].
    """
    if sent is None or not sent.split():
        logger.warning("Empty sentence, filling with [phrase manquante]],sent_id=%s,filename=%s",sent_id,filename)
        # Remplace les phrases vides par un marqueur textuel pour maintenir l'alignement des vecteurs
        return MISSING_SENTENCE
    return sent.replace("_","").replace("  "," ")
#--------parsing functions------
def get_tokens(text):
    """
    Sépare un texte en tokens en gérant spécifiquement les apostrophes (fréquentes en français).
    Cette fonction prépare la récupération des tokens individuels des phrases traitées, afin de les enregistrer dans le fichier JSON des métadonnées et de les encoder en mode token
    """
    if text is None:
        return text
    match = re.findall(r'\'',text)
    if match !=[]:
        # Isole l'apostrophe pour forcer une césure de token à cet endroit
        text = text.replace("\'","\'\n").strip()
        text = text.replace(" ","\n").strip()
        return text.split("\n")
    else:
        return text.split(" ")

def parse_conllu_fast(file_path,text=None):
    """
    Analyse un fichier ou un texte au format CoNLLU pour extraire les phrases et leurs métadonnées.
    Possède une alternative (fallback) si le texte brut n'est pas explicitement annoté avec une balise #text_raw.

    Args:
        file_path (str): Le chemin d'accès au fichier CoNLLU.
        text (str): Contenu textuel direct (utile si le contenu du fichier a déjà été récupéré ou est intégré dans un XML).

    Valeurs de retour:
        tuple: (sent_list, metadata)
            - sent_list (list): Liste des phrases sous forme de texte brut.
            - metadata (dict): Dictionnaire contenant 'sent_id', 'raw_text', et 'tokens'.
    """
    metadata = {"sent_id": [],
                "raw_text": [],
                "tokens":[]}
    sent_list = []
    sent_id = None
    text_raw = None
    # Chargement du contenu : depuis la variable texte si fournie, sinon lecture du fichier
    if text is not None:
        content = text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    # Première tentative d'extraction rapide via les métadonnées de l'en-tête de chaque phrase (balises #text_raw)
    for line in content.splitlines():
        line = line.rstrip("\n")
        if line.startswith("#"):
            # Normalisation : supprime '#', sépare au premier '=', nettoie les espaces
            key_value = line[1:].split("=", 1)
            if len(key_value) != 2:
                continue
            key, value = key_value[0].strip(), key_value[1].strip()
            if key == "sent_id":
                sent_id = value
            elif key == "text_raw":
                text_raw = value
        elif line == "":  # Une ligne vide marque la fin du bloc d'une phrase en CoNLLU
            if sent_id is not None:
                metadata["sent_id"].append(sent_id)
                metadata["raw_text"].append(text_raw)
                sent_list.append(text_raw)
                metadata["tokens"].append(get_tokens(text_raw))
            sent_id, text_raw = None, None
    # Capture la toute dernière phrase si le fichier ne se termine pas proprement par une ligne vide
    if sent_id is not None:
        metadata["sent_id"].append(sent_id)
        text_raw = clean_sentence(text_raw,file_path,sent_id)
        metadata["raw_text"].append(text_raw)
        sent_list.append(text_raw)

    # VÉRIFICATION ET ALTERNATIVE (FALLBACK) BASÉE SUR LE PARSING ET CONCATÉNATION DES FORMES INDIVIDUELLES
    # Si la liste des phrases précédemment parsée est vide ou si au moins 3 phrases n'ont pas de 'text_raw', la méthode rapide est jugée échouée. On bascule vers la deuxième méthode (concaténation des formes)
    if (sent_list == []) or (sum(x is None for x in sent_list) >= 3):
        logger.info("sent_list has None entries, falling back to form concatenation method")
        # Découpage du fichier en blocs bruts
        raw_entries = parse_conllu_raw_entries(content)
        # Réinitialisation des structures de données
        metadata = {"sent_id": [], "raw_text": [],"tokens":[]}
        sent_list = []
        sent_id = None
        text_raw = None
        for i,sent in enumerate(raw_entries):
         # Reconstruction du texte à partir des formes individuelles (avec gestion des amalgames)
            text_raw,tokens = concat_forms(sent)
            sent_id = get_sent_id(sent)
            # utilisation de l'indice de la phrase dans le fichier si la balise #sent_id est absente
            if sent_id is None:
                sent_id = i
            text_raw = clean_sentence(text_raw,file_path,sent_id)
            metadata["sent_id"].append(sent_id)
            metadata["raw_text"].append(text_raw)
            metadata["tokens"].append(tokens)
            sent_list.append(text_raw)
    return sent_list, metadata




def fix_punctuation_spaces(text):
    """
    Normalise les espaces autour de la ponctuation selon les règles typographiques françaises.
    Nécessaire pour la reconstruction des phrases à partir des formes individuelles issues du fichier CoNLLU.
    """
    if isinstance(text,list):
        text = ' '.join(text)
    # 1. Corrige les apostrophes (supprime les espaces adjacents)
    text = re.sub(r"\s+\'", "'", text)  # espace avant l'appostrophe
    text = re.sub(r"'\s+", "'", text)  # espace après l'appostrophe

     # 2. Supprime l'espace avant et force l'espace après la ponctuation double ou forte (:, ;, ?, !)
    text = re.sub(r'\s+([.:;?!])', r'\1', text)  # enlève l'espace avant
    text = re.sub(r'([.:;?!])(\S)', r'\1 \2', text)  # ajoute l'espace après si nécessaire

    # 3. Gère les guillemets français (« »)
    text = re.sub(r'\s+([»])', r'\1', text)  # efface l'espace avant
    text = re.sub(r'([«])\s+', r'\1 ', text)  # ajoute l'espace après

    # # 4. Corrige la ponctuation simple (.,)
    text = re.sub(r'\s+([.,])', r'\1', text)  # efface l'espace avant
    text = re.sub(r'([.,])(\S)', r'\1 \2', text)  # ajoute l'espace après

    # 5. Nettoie les espaces multiples résiduel
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def parse_sentences_xml_conllu(filepath):
    """
    Analyse un fichier XML (simple ou avec du ConLLU contenu dans des balises <s>) pour extraire les phrases et générer les métadonnées associées.

    Cette fonction recherche toutes les balises <s> (sentences) et gère plusieurs formats
    de contenu : texte simple, éléments imbriqués, ou annotations CoNLLU multilignes.

    Args:
        filepath (str): Le chemin d'accès au fichier XML à analyser.

    Valeurs de retour:
        tuple: (sent_list, metadata)
            - sent_list (list): Liste des phrases sous forme de texte brut.
            - metadata (dict): Dictionnaire contenant 'sent_id', 'raw_text', et 'tokens'.
    """
    data = None
    # Lecture en mode binaire ("rb"), recommandé pour le parsing XML avec lxml
    with open(filepath,"rb") as f:
        data = f.read()
    # Initialisation d'un parseur tolérant aux erreurs de syntaxe XML (recover=True)
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(data,parser=parser)
    # Extraction de toutes les balises <s> via une requête XPath
    sentences = tree.xpath("//s")
    len_s = len(sentences)
    # Initialisation des structures de données
    metadata = {"sent_id": [],
                "raw_text": [],
                "tokens":[]}
    sent_list = []
    xml_conllu = 0
    # itération dans la liste des s
    for s in sentences:
        sent_id = s.get("id")
        #initialisation de la variable destinée à stoquer les tokens individuels pour le cas 3
        tokens = None
        # Cas 1 : Le texte de la balise <s> contient des sauts de ligne: signature probable d'un format CoNLLU contenu
        if s.text is not None and "\n" in s.text:# and "\t" in s.text:
            xml_conllu +=1
            #logger.info("detected multiline text inside <s>, using contained CONLLU mode, sentence_id=%s,",sent_id)
            raw_text,tokens = concat_forms(s.text)
            raw_text = fix_punctuation_spaces(raw_text)
        # Cas 2 : La balise <s> ne contient pas de texte direct (ex: texte framenté dans des sous-balises comme <w>)
        elif s.text is None:
            logger.warning("Sentid=%s:<s> text is empty, looking for children texts",sent_id)
            # Récupération de tout le texte contenu dans les nœuds enfants
            raw_text = "".join(s.itertext())
            logger.warning("using s.itertext() : sentence=%s",raw_text)
        # Cas 3 : La balise contient du texte simple et direct
        else:
            raw_text = s.text
            logger.debug("Sentid %s, sent tex: %s",sent_id,raw_text)
            tokens = get_tokens(raw_text)
        # Enregistrement et nettoyage final pour la phrase actuelle
        metadata["sent_id"].append(sent_id)
        raw_text = clean_sentence(raw_text,filepath,sent_id)
        metadata["raw_text"].append(raw_text)
        if tokens is not None:
            metadata["tokens"].append(tokens)
        else:
            logger.warning("token list is None")
            metadata["tokens"].append(None)
        sent_list.append(raw_text)
    # Journalisation récapitulative si du format hybride XML-CoNLLU a été détecté
    if xml_conllu > 0:
        logger.info("Detected and parsed %s XML-CoNLLU sentences,filename=%s",xml_conllu,filepath)
    return sent_list,metadata
def parse_sentence_trs(file_path=None):
    """
    Analyse un fichier de transcription audio au format TRS pour extraire les tours de parole.

    Cette fonction cible les balises <Turn> du fichier XML, qui représentent les interventions
    des locuteurs, et utilise l'attribut temporel 'startTime' comme identifiant unique.

    Args:
        file_path (str): Le chemin d'accès au fichier TRS à analyser.

    Valeurs de retour:
        tuple: (sent_list, metadata)
            - sent_list (list): Liste des tours de parole sous forme de texte brut.
            - metadata (dict): Dictionnaire contenant 'sent_id' (startTime), 'raw_text', et 'tokens'.
    """
    logger.info("mode is trs")
    data = None
    # Lecture en mode binaire ("rb") pour la compatibilité avec le parseur lxml
    with open(file_path,"rb") as f:
        data = f.read()
    # Initialisation d'un parseur tolérant aux erreurs de syntaxe XML (recover=True)
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(data,parser=parser)

    # Extraction de toutes les balises <Turn> via une requête XPath
    sentences = tree.xpath("//Turn")
    len_s = len(sentences)
    # Initialisation des structures de données
    metadata = {"sent_id": [],
                "raw_text": [],
                "tokens":[]}
    sent_list = []

    # Itération sur chaque tour de parole extrait

    for s in sentences:
        # Utilisation du marqueur temporel de début (en secondes) comme identifiant
        sent_id = s.get("startTime")
        # Concaténation de tout le texte contenu dans le nœud <Turn> et ses sous-nœuds éventuels
        raw_text = "".join(s.itertext())
        # Nettoyage de la ponctuation et standardisation des espaces
        raw_text = fix_punctuation_spaces(raw_text).replace(' ',' ')
        metadata["sent_id"].append(sent_id)
        # Nettoyage final et gestion des potentiels tours de parole vides
        text_raw = clean_sentence(raw_text,file_path,sent_id)
        metadata["raw_text"].append(raw_text)
        metadata["tokens"].append(get_tokens(raw_text))
        sent_list.append(raw_text)
    return sent_list,metadata
def parse_sentences(file_path=None,mode=None):
    """
    Sélectionne et exécute le mode de parsing approprié en fonction de l'extension du fichier.

    Elle calcule également des statistiques d'extraction (nombre total de phrases et de tokens)
    et mesure le temps d'exécution pour chaque format supporté.

    Args:
        file_path (str): Le chemin d'accès au fichier cible.
        mode (str, optional): Le format du fichier. Note : ce paramètre est écrasé
                              par l'extension réelle extraite de 'file_path'.

    Returns:
        tuple: (sent_list, metadata) ou (None, None) si le format n'est pas reconnu.
            - sent_list (list): Liste des phrases ou tours de parole extraits.
            - metadata (dict): Dictionnaire des métadonnées correspondantes.
    """
    # Extraction de l'extension du fichier pour déterminer automatiquement le mode de parsing s'il n'est pas déterminé dans l'appel de la fonction
    if mode is None:
        base,ext = os.path.splitext(file_path)
        mode = ext.replace(".","")
    # Routage pour le format CoNLLU
    if mode == "conllu":
        t0 = time.perf_counter()
        logger.info("Parsing CONLLU sentences,filename=%s",file_path)
        # Appel du parseur spécifique
        sent_list,metadata = parse_conllu_fast(file_path)
        len_s = len(sent_list)
        n_tokens = 0
        # Comptage approximatif des tokens via expression régulière (mots ou ponctuation)
        try:
            n_tokens = sum(len(re.findall(r'\w+|[^\w\s]', sent)) for sent in sent_list if sent !=MISSING_SENTENCE)
        except TypeError as e:
            logger.warning("Could'nt calculate number of tokens")
        t1 = time.perf_counter()
        ex_time = t1-t0
        logger.info("Parsed %s sentences, %s tokens in %s seconds",len_s,n_tokens,np.round(ex_time,2))
        return sent_list,metadata
    # Routage pour le format XML (Lexicoscope)
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
    # Routage pour le format TRS (Transcriber)
    elif mode == "trs":
        t0 = time.perf_counter()
        logger.info("Parsing trs sentences,filename=%s",file_path)
        sent_list,metadata = parse_sentence_trs(file_path)
        len_s = len(sent_list)
        n_tokens = sum(len(re.findall(r'\w+|[^\w\s]', sent)) for sent in sent_list if sent !=MISSING_SENTENCE)
        t1 = time.perf_counter()
        ex_time = t1-t0
        logger.info("Parsed %s sentences, %s tokens in %s seconds",len_s,n_tokens,np.round(ex_time,2))
    # Gestion des extensions non supportées
    else:
        logger.warning("File format not recognized: %s,filename=%s",ext,file_path)
        return None,None
    # Point de retour pour le bloc 'trs'
    return sent_list,metadata

#----encoding functions -----

def calcEmbeddings(collection_file_path=None, output_file_path=None, mode=None,reduce_precision=False,overwrite=False,token_mode=False,no_daemon=False,use_ollama=False,ollama_host='localhost:11434',ollama_model=None):
    """
    Fonction principale du script: elle permet d'extraire les phrases d'un fichier de corpus et génère leurs embeddings correspondants.
    Intègre un système de cache : si les fichiers de sortie existent déjà, ils sont chargés directement.

    Args:
        collection_file_path (str): Le chemin vers le fichier de corpus source (.conllu, .xml, .trs).
        output_file_path (str): Le chemin de destination pour l'enregistrement du fichier des embeddings (.npy).
        mode (str): Le format du corpus cible (par défaut 'conllu').
        reduce_precision (bool, optional): Si True, sauvegarde les embeddings en float16 pour économiser de l'espace disque.
        overwrite (bool): Si True, force le recalcul même si les fichiers de sortie existent déjà.
        token_mode (bool): Si True, utilise l'encodage par token et ajoute le suffixe '_token' au fichier de sortie.
        no_daemon (bool): Si True, exécute le modèle localement au lieu du processus démon.
        use_ollama (bool): Si True, délègue l'encodage à une API Ollama externe.
        ollama_host (str): L'adresse du serveur Ollama.
        ollama_model (str): Le nom du modèle Ollama à utiliser.

    Valeurs de retour:
        tuple: (embeddings, metadata)
            - embeddings (numpy.ndarray): Matrice des vecteurs générés ou chargés.
            - metadata (dict): Dictionnaire contenant les métadonnées (ID, texte, tokens).
    """
    token_suffix=""
    # Gestion du suffixe pour différencier les fichiers d'embeddings par token
    if token_mode and "_token" not in output_file_path :
        token_suffix = "_token"

    base, ext = os.path.splitext(collection_file_path)
    # Vérification de l'existence du fichier embeddings cible : si on ne force pas l'écrasement et que les fichiers existent, on charge depuis le disque
    if (not overwrite) and  (os.path.exists(output_file_path)) and os.path.exists(base+".json"):
        logger.warning("embedding file and metadata file already exist, loading from %s and %s",output_file_path,base+token_suffix+".json")
        embeddings = load_embeddings(base+".npy")
        metadata= load_metadata(base+".json")
        return embeddings,metadata
    # 1. Étape de parsing : extraction des données du fichier source
    logger.info("parsing sentences, file=%s mode=%s",collection_file_path,mode)
    sentence_list,metadata = parse_sentences(collection_file_path,mode=mode)
    # 2. Étape d'encodage : transformation des textes en vecteurs
    logger.info("Encoding sentences with model")

    t0 = time.perf_counter() # chronomètre pour mesurer le temps d'excécution
    if token_mode:
        logger.info("using token level embedding mode")
        embeddings = encode(sentence_list, chunk_size=64,token_mode=True,no_daemon=no_daemon,use_ollama=use_ollama,ollama_host=ollama_host,ollama_model=ollama_model)
        output_file_path = output_file_path.replace(".npy",token_suffix+".npy")
    else:
        logger.info("using sentence level embedding mode")
        embeddings = encode(sentence_list, chunk_size=64,no_daemon=no_daemon,use_ollama=use_ollama,ollama_host=ollama_host,ollama_model=ollama_model)
    t1 = time.perf_counter()
    procession_time = t1-t0
    logger.info("Embeddings created in %s seconds",np.round(procession_time,2))
    # 3. Étape de sauvegarde
    # Gestion du nom du fichier: le 'replace' nettoie préventivement le nom de fichier au cas où le suffixe s'accumulerait
    logger.info("saving embeddings to %s", output_file_path.replace("_token_token","_token"))
    # Option d'optimisation du stockage (réduit deux fois la talile du fichier embeddings .npy avec un coût sur la précision de l'information)
    if reduce_precision:
        np.save(output_file_path,embeddings.astype(np.float16))
    else:
        np.save(output_file_path,embeddings)
    logger.info("saved successfully")

    return embeddings,metadata
def save_metadata(metadata,output_file=None,token_mode=False):
    """
    Sauvegarde le dictionnaire de métadonnées dans un fichier JSON.

    Args:
        metadata (dict): Le dictionnaire contenant les métadonnées extraites ('sent_id', 'raw_text', 'tokens').
        output_file (str): Le chemin d'accès au fichier cible (généralement .json).
        token_mode (bool): Paramètre conservé pour des raisons de compatibilité de signature de la fonction.
    """
    # Journalisation de l'action de sauvegarde
    logger.info("saving metadata tp %s",output_file)
    # Ouverture du fichier en mode écriture ("w") avec l'encodage UTF-8,
    # indispensable pour préserver correctement les caractères spéciaux et accents du français.
    with open(output_file,"w",encoding="utf-8") as f:
        # Sérialisation et écriture du dictionnaire Python en format JSON
        json.dump(metadata,f)

def encode_folder(input_folder=None,overwrite=False,token_mode=False,no_daemon=False,use_ollama=False,ollama_host='localhost:11434',ollama_model=None):
    """
    Parcourt un répertoire ou un wildcard (ex. *Camus*) pour traiter en lot des fichiers de corpus,
    générer leurs embeddings et sauvegarder leurs métadonnées.

    Args:
        input_folder (str): Le chemin du répertoire cible ou un motif (ex: 'data/*').
        overwrite (bool, optional): Si True, force le recalcul des embeddings même s'ils existent déjà.
        token_mode (bool): Si True, génère les embeddings au niveau des tokens.
        no_daemon (bool): Si True, exécute le modèle d'encodage localement (sans démon).
        use_ollama (bool): Si True, utilise une instance Ollama pour l'encodage.
        ollama_host (str): L'adresse de l'hôte API Ollama (défaut: 'localhost:11434').
        ollama_model (str): Le modèle Ollama spécifique à interroger.

    Returns:
        None: Cette fonction opère par effets de bord (création de fichiers .npy et .json sur le disque).
    """
    # Liste des extensions de fichiers de corpus actuellement supportées par le pipeline
    extensions = [".conllu",".xml",".trs"]
    # récupération des chemins de fichiers
    if '*' in input_folder:
        file_list = glob.glob(input_folder)
        file_list = [f  for f in file_list if os.path.splitext(f)[1] in extensions]
    else:

        file_list = []
        for ext in extensions:
            file_list.extend(glob.glob(input_folder+"/*"+ext))
    len_f = len(file_list)
    logger.info("Found %s files in folder",len_f)
    # Extraction dynamique des extensions réellement trouvées parmi les fichiers du dossier
    found_extentions = [re.findall(r'\.([^.]+)$', f)[0] for f in file_list]
    logger.info("Found formats: %s",list(set(found_extentions)))

    cnt = 1
    # Affichage préalable de la liste complète des fichiers à traiter
    for f in file_list:
        logger.info("%s",f)
    # Boucle de traitement principale pour chaque fichier détecté
    for f,ext in zip(file_list,found_extentions):
        logger.info("Encoding file %s/%s filename=%s",cnt,len_f,f)
        # Appel de la fonction principale d'extraction et d'encodage
        # Le f.replace(ext, 'npy') remplace l'extension d'origine (ex: 'xml') par 'npy'
        # pour le fichier de sortie, mais présuppose que le nom de l'extension ne figure pas ailleurs dans le chemin.
        embeddings,metadata = calcEmbeddings(f,f.replace(ext,'npy'),ext,overwrite=overwrite,token_mode=token_mode,no_daemon=no_daemon,use_ollama=use_ollama,ollama_host=ollama_host,ollama_model=ollama_model)
        # Sauvegarde synchronisée des métadonnées associées en format JSON
        save_metadata(metadata,f.replace(ext,"json"),token_mode=token_mode)
        cnt +=1

# fonction main pour tester le script
if __name__ == "__main__":
    input_folder ="test"
    encode_folder(input_folder)
