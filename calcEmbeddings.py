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
import simplemma

CHUNK_SIZE = 64
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
    r"""
    Reconstruit le texte brut à partir des formes d'un bloc CoNLLU.
    Gère spécifiquement les amalgames en ignorant leurs composants enfants pour éviter les doublons.
    Utilise [^\t\n]+ plutôt que \S+ pour capturer la colonne FORM : certaines formes de
    surface contiennent un espace interne (ex. 'quand même', 'est-ce que', 'tout le long du'),
    et \S+ tronquait ces valeurs au premier espace, les colonnes CoNLLU étant séparées par
    des tabulations et non par n'importe quel espace.
    """
    tokens = []
    if has_amalgams(text):
            lines = text.split('\n')
            kept_lines = []
            for i,t in enumerate(lines):
                if (not is_amalgame(lines[i-2])) and (not is_amalgame(lines[i-1])):
                    kept_lines.append(t)
            joined_tokens = "\n".join(kept_lines)
            # capture only amalgame lines or non amagame lines (skips amalgam child lines)
            # capture les formes de surface via une regex, en capturant la colonne FORM jusqu'à la tabulation suivante
            forms = re.findall(r'(?:^\d+\-\d+\t|^\d+\t)([^\t\n]+)', joined_tokens, re.MULTILINE)
            tokens = forms  # 1 token par mot de surface (amalgames inclus), aligné avec les lemmes
            raw_text = " ".join(forms)
            raw_text = fix_punctuation_spaces(raw_text)

    else:
        # Extraction normale si aucun amalgame n'est présent
        forms = re.findall(r'^\d+\t([^\t\n]+)', text, re.MULTILINE)
        tokens = forms  # 1 token par ligne CoNLLU, aligné avec les lemmes

        raw_text = " ".join(forms)
        raw_text = fix_punctuation_spaces(raw_text)
    return raw_text, tokens

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
def get_lemmas(text):
    f"""
    Extrait la liste des lemmes à partir d'un bloc CoNLLU brut, alignée mot-de-surface
    par mot-de-surface avec concat_forms (un lemme par mot de surface, y compris les
    amalgames). Pour un amalgame, la colonne LEMMA vaut "_" (non renseignée) ; on utilise
    alors sa propre forme de surface (FORM) comme lemme de substitution, et on ignore
    ses sous-composants, exactement comme concat_forms le fait pour les tokens.

    Utilise [^\t\n]+ plutôt que \S+/\s+ pour capturer les colonnes FORM et LEMMA :
    certaines valeurs contiennent un espace interne (ex. 'quand même', 'en résumé'),
    et \S+ tronquait ces valeurs au premier espace, les colonnes CoNLLU étant séparées
    par des tabulations et non par n'importe quel espace.

    Args:
        text (str): Bloc CoNLLU brut d'une phrase (comme reçu par concat_forms).

    Valeur de retour:
        list: Liste des lemmes, une entrée par mot de surface, alignée avec tokens.
    """
    lines = text.split('\n')
    lemmas = []
    for i, line in enumerate(lines):
        # ignore les lignes correspondant aux sous-composants d'un amalgame déjà traité
        if (i >= 1 and is_amalgame(lines[i-1])) or (i >= 2 and is_amalgame(lines[i-2])):
            continue
        if is_amalgame(line):
            # ligne amalgame : LEMMA="_", on utilise sa forme de surface (colonne FORM) comme lemme de substitution
            match = re.match(r'^\d+\-\d+\t([^\t\n]+)', line)
        else:
            match = re.match(r'^\d+\t[^\t\n]+\t([^\t\n]+)', line)
        if match:
            lemmas.append(match.group(1))
    return lemmas

def get_lemmas_fast(text,lang='fr'):
    """
    Lemmatise un texte brut (sans annotation CoNLLU) à l'aide de simplemma.
    Utilisé comme solution de repli légère et rapide pour les formats sans colonne LEMMA
    (XML sans CoNLLU imbriqué, TRS). Moins précis qu'une vraie lemmatisation morphosyntaxique
    (pas de désambiguïsation par POS), mais suffisant comme approximation par lot.

    Args:
        text (str): Texte brut à lemmatiser.
        lang (str): Code langue simplemma (ex: 'fr' pour le français).

    Valeur de retour:
        list: Liste des lemmes, ou None si le texte est vide/manquant.
    """
    logger.warning("Lemmatisation absente (le fichier d'entrée n'est pas un ConLL-U): lemmatisation avec Simplemma")
    if text is None or not text.split() or text == MISSING_SENTENCE:
        return None
    return list(simplemma.text_lemmatizer(text,lang=lang))
def parse_conllu_fast(file_path,text=None):
    """
    Analyse un fichier ou un texte au format CoNLLU pour extraire les phrases et leurs métadonnées.
    Reconstruit systématiquement le texte brut à partir des formes (colonne FORM) et des
    lemmes (colonne LEMMA) de chaque ligne d'annotation, en gérant les amalgames.

    Args:
        file_path (str): Le chemin d'accès au fichier CoNLLU.
        text (str): Contenu textuel direct (utile si le contenu du fichier a déjà été récupéré ou est intégré dans un XML).

    Valeurs de retour:
        tuple: (sent_list, metadata)
            - sent_list (list): Liste des phrases sous forme de texte brut.
            - metadata (dict): Dictionnaire contenant 'sent_id', 'raw_text', 'tokens' et 'lemmas'.
    """
    metadata = {"sent_id": [],
                "raw_text": [],
                "tokens":[],
                "lemmas":[]}
    sent_list = []
    # Chargement du contenu : depuis la variable texte si fournie, sinon lecture du fichier
    if text is not None:
        content = text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

    # Découpage du fichier en blocs bruts (une entrée = une phrase, lignes séparées par \n)
    raw_entries = parse_conllu_raw_entries(content)
    for i,sent in enumerate(raw_entries):
        # Reconstruction du texte à partir des formes individuelles (avec gestion des amalgames)
        text_raw,tokens = concat_forms(sent)
        lemmas = get_lemmas(sent)
        sent_id = get_sent_id(sent)
        # utilisation de l'indice de la phrase dans le fichier si la balise #sent_id est absente
        if sent_id is None:
            sent_id = i
        text_raw = clean_sentence(text_raw,file_path,sent_id)
        metadata["sent_id"].append(sent_id)
        metadata["raw_text"].append(text_raw)
        metadata["tokens"].append(tokens)
        metadata["lemmas"].append(lemmas)
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
    with open(filepath,"rb") as f:
        data = f.read()
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(data,parser=parser)
    sentences = tree.xpath("//s")
    len_s = len(sentences)
    metadata = {"sent_id": [],
                "raw_text": [],
                "tokens":[],
                "lemmas":[]}
    sent_list = []
    xml_conllu = 0
    for s in sentences:
        sent_id = s.get("id")
        tokens = None
        lemmas = None
        # Cas 1 : CoNLLU imbriqué dans <s>
        if s.text is not None and "\n" in s.text:
            xml_conllu +=1
            raw_text,tokens = concat_forms(s.text)
            raw_text = fix_punctuation_spaces(raw_text)
            lemmas = get_lemmas(s.text)
        # Cas 2 : texte fragmenté dans des sous-balises, pas de colonnes CoNLLU -> lemmatisation rapide via simplemma
        elif s.text is None:
            logger.warning("Sentid=%s:<s> text is empty, looking for children texts",sent_id)
            raw_text = "".join(s.itertext())
            logger.warning("using s.itertext() : sentence=%s",raw_text)
            lemmas = get_lemmas_fast(raw_text)
        # Cas 3 : texte simple direct, pas de colonnes CoNLLU -> lemmatisation rapide via simplemma
        else:
            raw_text = s.text
            logger.debug("Sentid %s, sent tex: %s",sent_id,raw_text)
            tokens = get_tokens(raw_text)
            lemmas = get_lemmas_fast(raw_text)
        metadata["sent_id"].append(sent_id)
        raw_text = clean_sentence(raw_text,filepath,sent_id)
        metadata["raw_text"].append(raw_text)
        if tokens is not None:
            metadata["tokens"].append(tokens)
        else:
            logger.warning("token list is None")
            metadata["tokens"].append(None)
        metadata["lemmas"].append(lemmas)
        sent_list.append(raw_text)
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
            - metadata (dict): Dictionnaire contenant 'sent_id' (startTime), 'raw_text', 'tokens' et 'lemmas' (toujours None, non applicable pour une transcription orale non annotée).
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
                "tokens":[],
                "lemmas":[]}
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
        # pas de colonne LEMMA en TRS (transcription orale non annotée morphosyntaxiquement)
        metadata["lemmas"].append(get_lemmas_fast(raw_text))
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
        token_mode (bool): Si True, utilise l'encodage par token. Les embeddings de tokens
            individuels sont regroupés par lemme puis moyennés (voir build_lemma_embeddings) :
            le fichier '_token.npy' final contient une matrice uniforme (n_lemmes_uniques,
            hidden_dim), une ligne par lemme unique du fichier, et non les embeddings bruts
            par mot/par phrase.
        no_daemon (bool): Si True, exécute le modèle localement au lieu du processus démon.
        use_ollama (bool): Si True, délègue l'encodage à une API Ollama externe.
        ollama_host (str): L'adresse du serveur Ollama.
        ollama_model (str): Le nom du modèle Ollama à utiliser.

    Valeurs de retour:
        tuple: (embeddings, metadata)
            - embeddings (numpy.ndarray): Matrice des vecteurs générés ou chargés. En mode
              token, il s'agit de la matrice moyennée par lemme (n_lemmes_uniques, hidden_dim).
            - metadata (dict): Dictionnaire contenant les métadonnées (ID, texte, tokens, lemmes).
    """
    token_suffix=""
    if token_mode and "_token" not in output_file_path :
        token_suffix = "_token"
    # chemin réel du fichier d'embeddings (avec '_token' en mode token) et de l'index de lemmes
    effective_output_path = output_file_path.replace(".npy", token_suffix+".npy")
    lemma_index_file = effective_output_path.replace(".npy", "_lemma_index.json")

    base, ext = os.path.splitext(collection_file_path)
    cache_ok = os.path.exists(effective_output_path) and os.path.exists(base+".json")
    if token_mode:
        # en mode token, la matrice .npy est inutilisable sans son index de lemmes
        cache_ok = cache_ok and os.path.exists(lemma_index_file)
    if (not overwrite) and cache_ok:
        logger.warning("embedding file and metadata file already exist, loading from %s and %s",effective_output_path,base+".json")
        embeddings = load_embeddings(effective_output_path)
        metadata= load_metadata(base+".json")
        return embeddings,metadata
    logger.info("parsing sentences, file=%s mode=%s",collection_file_path,mode)
    sentence_list,metadata = parse_sentences(collection_file_path,mode=mode)
    logger.info("Encoding sentences with model")

    t0 = time.perf_counter()
    if token_mode:
        logger.info("using token level embedding mode")
        token_input_list = [
            " ".join(t.replace(" ", "_") for t in tok_list) if tok_list else ""
            for tok_list in metadata["tokens"]
        ]
        embeddings = encode(
            token_input_list,
            chunk_size=CHUNK_SIZE,
            token_mode=True,
            no_daemon=no_daemon,
            use_ollama=use_ollama,
            ollama_host=ollama_host,
            ollama_model=ollama_model
        )

        lemma_list, lemma_embeddings = build_lemma_embeddings(
            embeddings, metadata["lemmas"]
        )
        embeddings = lemma_embeddings

        if reduce_precision:
            embeddings = embeddings.astype(np.float16)

        np.save(effective_output_path, embeddings)

        lemma_index_path = effective_output_path.replace(".npy", "_lemma_index.json")
        with open(lemma_index_path, "w", encoding="utf-8") as f:
            json.dump(lemma_list, f, ensure_ascii=False)
        logger.info("lemma index saved to %s", lemma_index_path)
    else:
        logger.info("using sentence level embedding mode")
        embeddings = encode(
            sentence_list,
            chunk_size=CHUNK_SIZE,
            no_daemon=no_daemon,
            use_ollama=use_ollama,
            ollama_host=ollama_host,
            ollama_model=ollama_model
        )
        if reduce_precision:
            np.save(effective_output_path, embeddings.astype(np.float16))
        else:
            np.save(effective_output_path, embeddings)
    t1 = time.perf_counter()
    procession_time = t1-t0
    logger.info("Embeddings created in %s seconds",np.round(procession_time,2))
    logger.info("saving embeddings to %s", output_file_path.replace("_token_token","_token"))
    if token_mode:
        # embeddings est désormais une matrice uniforme (n_lemmes_uniques, hidden_dim)
        # -> sauvegarde directe, plus besoin de tableau 'object'/allow_pickle
        if reduce_precision:
            embeddings = embeddings.astype(np.float16)
        np.save(output_file_path, embeddings)
        # Sauvegarde de la liste ordonnée des lemmes (index ligne -> lemme)
        lemma_index_path = output_file_path.replace(".npy", "_lemma_index.json")
        with open(lemma_index_path, "w", encoding="utf-8") as f:
            json.dump(lemma_list, f, ensure_ascii=False)
        logger.info("lemma index saved to %s", lemma_index_path)
    else:
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
def build_lemma_embeddings(embeddings, lemmas_per_sentence):
    """
    Regroupe les embeddings de tokens par lemme à l'échelle d'un seul fichier de corpus,
    puis moyenne les vecteurs de chaque groupe (np.mean) pour produire un seul embedding
    par lemme unique.

    Args:
        embeddings (list): une entrée par phrase, chacune un tableau numpy
            (n_tokens_phrase, hidden_dim), tel que retourné par l'encodage en mode token.
        lemmas_per_sentence (list): metadata["lemmas"], une entrée par phrase, liste de
            lemmes alignée token-à-token avec embeddings (même longueur par phrase).

    Valeur de retour:
        tuple: (lemma_list, lemma_embeddings)
            - lemma_list (list): lemmes uniques triés (l'index dans cette liste = ligne dans lemma_embeddings)
            - lemma_embeddings (numpy.ndarray): matrice (n_lemmes_uniques, hidden_dim)
    """
    groups = {}
    for sent_embs, sent_lemmas in zip(embeddings, lemmas_per_sentence):
        if sent_embs is None or sent_lemmas is None:
            continue
        if len(sent_embs) != len(sent_lemmas):
            logger.warning("mismatch: %s embeddings vs %s lemmas, skipping sentence",
                            len(sent_embs), len(sent_lemmas))
            continue
        for vec, lemma in zip(sent_embs, sent_lemmas):
            if lemma is None or lemma == "_":
                continue
            groups.setdefault(lemma, []).append(vec)

    lemma_list = sorted(groups.keys())
    lemma_embeddings = np.stack([np.mean(groups[l], axis=0) for l in lemma_list]).astype(np.float32)
    return lemma_list, lemma_embeddings
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
            # splitext ne touche que l'extension finale (f.replace(ext,...) remplaçait toutes les occurrences dans le chemin)
            base = os.path.splitext(f)[0]
            embeddings,metadata = calcEmbeddings(f,base+".npy",ext,overwrite=overwrite,token_mode=token_mode,no_daemon=no_daemon,use_ollama=use_ollama,ollama_host=ollama_host,ollama_model=ollama_model)
            # Sauvegarde synchronisée des métadonnées associées en format JSON
            save_metadata(metadata,base+".json",token_mode=token_mode)
            cnt +=1

# fonction main pour tester le script
if __name__ == "__main__":
    input_folder ="test"
    encode_folder(input_folder)
