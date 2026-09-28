# ============================================================================
# searchEmbedding.py
#
# Charger les indices FAISS et les fichiers metadata (json), encoder une requête via le client d'embedding'
# et lancer la recherche de similarité entre la requête et les phrases indexées. Le script supporte la recherche dans un seul fichier index (.faiss) ou dans un dossier content plusieurs fichiers.
# les résultats d'une recherche en mode dossier sont rassemblés puis triés par score de similarité et affichés.
# ============================================================================


import numpy as np
import json
import logging
import os
# configuration du journal pour l'affichage des message et des avertissements
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
# le client d'embedding est utilisé pour communiquer avec le daemon d'embedding (un processus qui garde le modèle d'embeddings chargé en mémoire)
from utils.embed_client import encode

import faiss
def apply_abtt(vecs, abtt_file):
    with open(abtt_file) as f:
        params = json.load(f)
    mu = np.array(params["mean"], dtype=np.float32)
    P = np.array(params["components"], dtype=np.float32)  # (dim, 3)
    X = np.asarray(vecs, dtype=np.float32) - mu
    return X - (X @ P) @ P.T

def load_index(index_file=None):
    """
    Charge un index FAISS à partir d'un fichier spécifié sur le disque.

    Args:
        index_file (str): Le chemin d'accès vers le fichier d'index (extension .faiss).

    Valeur de retour:
        faiss.Index: L'objet index FAISS chargé en mémoire, prêt pour la recherche.

    Raises:
        ValueError: Si aucun nom de fichier n'est fourni (`index_file` est None).
    """
    # Journalisation de la tentative de chargement avec le chemin du fichier
    logger.info("loading index from:%s",index_file)
    if index_file is not None:
        # Lecture et renvoi de l'index via la bibliothèque FAISS
        return faiss.read_index(index_file)
    else:
        # Interruption de l'exécution si le chemin est invalide ou manquant
        raise ValueError("file name empty")

def load_metadata(matadata_file_path=None):
    """
    Charge les métadonnées d'un corpus à partir d'un fichier JSON.
    Les métadonnées sont les phrases elles-mêmes, avec leur identifieurs et leurs tokens individuels sous forme de dictionnaire, ex.
        {"sent_id": ["1", "2", "3", "4"], "raw_text": ["Bonjour!", "pas de souci!", "je vous en prie", "le roi de France n'est pas chauve"], "tokens": [["Bonjour","!"], ["pas","de","souci","!"], ["je", "vous","en","prie"], ["le", "roi", "de", "France", "n","est","pas","chauve"]]}
    Chaque entrée du dictionnaire est une liste qui a la même taille du fichier index correspondant. Grâce à l'identifiant de la phrase, celle-ci est associée à son embedding respectif et à son score de similarité après l'exécution de la recherche FAISS.
    Args:
        matadata_file_path (str): Le chemin d'accès vers le fichier JSON contenant les métadonnées.

    Returns:
        dict or None: Un dictionnaire contenant les métadonnées (ex: 'sent_id', 'raw_text') si le chargement réussit, sinon None.
    """
    # Journalisation de l'opération de chargement avec le chemin cible
    logger.info("loading metadata from:%s",matadata_file_path)
    metadata = None
    # Ouverture du fichier en mode lecture
    with open (matadata_file_path,"r") as f:
        # Conversion du contenu JSON du fichier en objet Python (un dictionnaire)
        metadata = json.load(f)
    # Vérification que les données ont bien été extraites et assignées
    if metadata is not None:
        return metadata
    else:
        # Enregistrement d'un avertissement dans les logs si la variable est toujours vide
        logger.warning("Couldn't load metadata")
        return None
def embedd_query(query_str=None,token_mode=False,no_daemon=False,use_ollama=False,ollama_host='localhost',ollama_model=None):
    """
    Permet de génèrer un vecteur d'embedding pour une requête textuelle donnée.

    Args:
        query_str (str,): Le texte de la requête à encoder.
        token_mode (bool): Si True, encode la requête au niveau des tokens puis moyenne
            les vecteurs de mots obtenus en un seul vecteur (la requête devient un point
            unique dans le même espace que l'index de lemmes moyennés).
        no_daemon (bool): Si True, exécute l'encodage localement sans utiliser le processus démon en arrière-plan.
        use_ollama (bool): Si True, délègue la création de l'embedding à un modèle d'embedding via Ollama (très lent).
        ollama_host (str): L'adresse de l'hôte Ollama (par défaut 'localhost').
        ollama_model (str): Le nom du modèle d'embedding Ollama cible.

    Valeurs de retour:
        numpy.ndarray: Un tableau NumPy 2D contigu de type float32 représentant l'embedding de la requête.

    Raises:
        ValueError: Si aucune chaîne de requête n'est fournie (`query_str` est None).
    """
    # Journalisation du mode d'encodage sélectionné
    if token_mode:
        logger.info("Encoding query with token level mode")
    else:
        logger.info("Encoding query with sentence level mode")
    import time
    t0 = time.perf_counter()
    if query_str is not None:
        logger.info("embedding query: %s",query_str)
        if token_mode:
            # encode() en mode token retourne une liste d'un seul élément (une phrase, la requête),
            # cet élément étant un tableau (n_mots_requete, hidden_dim) -> on moyenne pour obtenir
            # un seul vecteur, dans le même espace que l'index de lemmes moyennés
            word_vecs = encode([query_str],chunk_size=1,token_mode=True,no_daemon=no_daemon,use_ollama=use_ollama,ollama_host=ollama_host,ollama_model=ollama_model)
            embeddings = np.mean(word_vecs[0], axis=0, keepdims=True)
        else:
            embeddings = encode([query_str],chunk_size=1,no_daemon=no_daemon,use_ollama=use_ollama,ollama_host=ollama_host,ollama_model=ollama_model)
            embeddings = np.array(embeddings,dtype=np.float32).reshape(1,-1)
        t1 = time.perf_counter()
        exec_time = t1-t0
        logger.info("Query encoded in %s seconds",np.round(exec_time,2))
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        return embeddings
    else:
        logger.warning("Query is empty")
        raise ValueError("Query is empty")
def abtt_path_from_index(index_file):
    """corpus_token.faiss -> corpus_token_abtt.json"""
    return os.path.splitext(index_file)[0] + "_abtt.json"

def resolve_abtt_path(abtt_file=None, index_file=None):
    """
    Détermine le fichier de paramètres ABTT à utiliser.
    Priorité : abtt_file explicite > déduit de index_file.
    Lève ValueError si aucun des deux n'est fourni (ex. index passé directement en mémoire
    sans abtt_file).
    """
    if abtt_file is not None:
        return abtt_file
    if index_file is not None:
        return abtt_path_from_index(index_file)
    raise ValueError("Token mode requires abtt_file or index_file to locate the ABTT parameters")

def get_lemma_occurrences(lemmas, metadata):
    """
    Pour une liste de lemmes, renvoie {lemme: [(token, sent_id), ...]} en un seul
    passage sur les métadonnées.
    """
    if metadata is None:
        return {}
    wanted = {l: [] for l in lemmas}
    for token_list, lemma_list, sent_id in zip(metadata["tokens"], metadata["lemmas"], metadata["sent_id"]):
        for token, lemma in zip(token_list, lemma_list):
            if lemma in wanted:
                wanted[lemma].append((token, sent_id))
    return wanted

def search(query_vector=None, query_str=None, index=None, index_file=None, abtt_file=None,
           metric_type=None, top_k=10, metadata=None, token_mode=False,
           no_daemon=False, lemma_list=None, allow_no_abtt=False,metadata_file=None):
    """
    Exécute une recherche de similarité dans un index FAISS et renvoie les meilleures
    correspondances avec leurs métadonnées.

    Args:
        query_vector (numpy.ndarray): Vecteur d'embedding BRUT de la requête (ni normalisé,
            ni transformé par ABTT). Si None, il est calculé à partir de query_str.
        query_str (str): Texte de la requête (utilisé si query_vector est None).
        index (faiss.Index): Index FAISS dans lequel effectuer la recherche (peut venir
            directement de la mémoire).
        index_file (str): Chemin du fichier .faiss correspondant à `index` (optionnel).
            Sert à déduire le fichier '_abtt.json' si abtt_file n'est pas fourni.
        abtt_file (str): Chemin explicite du fichier '_abtt.json' (optionnel). Prioritaire
            sur index_file. Nécessaire en mode token si l'index est passé depuis la mémoire
            sans index_file.
        metric_type (int): Type de métrique FAISS (inner product sur vecteurs normalisés L2).
        top_k (int): Nombre maximal de résultats (défaut: 10).
        metadata (dict): Phrases et identifiants ('sent_id', 'raw_text'). Ignoré en mode token.
        token_mode (bool): Encodage/recherche au niveau des tokens (lemmes).
        no_daemon (bool): Encodage local sans passer par le démon.
        lemma_list (list): Requis en mode token, liste ordonnée des lemmes de l'index.
        allow_no_abtt (bool): Si True, en mode token, autorise la recherche sans ABTT quand
        aucun fichier ABTT n'est fourni ou trouvé (pour les tests, ex. index construit
        sans ABTT). Attention : si l'index a été construit AVEC ABTT, les scores seront
        faux (espaces différents). Défaut: False (erreur explicite).
    Valeurs de retour:
        list: En mode phrase, tuples (sent_id, raw_text, score). En mode token,
            tuples (lemme, score). Liste vide en cas d'erreur de compatibilité.
    """
    if query_vector is None:
        query_vector = embedd_query(query_str, token_mode, no_daemon=no_daemon)

    if token_mode:
        abtt_path = None
        try:
            abtt_path = resolve_abtt_path(abtt_file=abtt_file, index_file=index_file)
        except ValueError:
            if not allow_no_abtt:
                raise
        if abtt_path is not None and not os.path.exists(abtt_path):
            if not allow_no_abtt:
                raise FileNotFoundError(f"{abtt_path} not found, re-run calcEmbeddings")
            abtt_path = None

        if abtt_path is not None and (not allow_no_abtt):
            logger.info("Applying ABTT to query")
            query_vector = apply_abtt(query_vector, abtt_path)
        else:
            logger.warning("Searching in token mode WITHOUT ABTT (allow_no_abtt=True): "
                           "results are only valid if the index was built without ABTT")

    # copie : faiss.normalize_L2 travaille en place
    query_vector = np.array(query_vector, dtype=np.float32, order="C", copy=True)
    faiss.normalize_L2(query_vector)

    if hasattr(index, "nprobe"):
        index.nprobe = 64
    try:
        distances, indices = index.search(query_vector, top_k)
        if token_mode:
            len_lemmas = len(lemma_list)
            matches = [(lemma_list[idx], np.round(float(distance), 3))
                       for idx, distance in zip(indices[0], distances[0])
                       if 0 <= idx < len_lemmas]

            occurrences = {}
            if metadata is None and metadata_file is not None:
                try:
                    metadata = load_metadata(metadata_file)
                except FileNotFoundError:
                    logger.warning("%s not found, cannot display tokens and sent_ids", metadata_file)
            if metadata is not None:
                occurrences = get_lemma_occurrences([m[0] for m in matches], metadata)
            else:
                logger.warning("No metadata available, cannot display lemma tokens and sent_ids")
            # (lemme, score, [(token, sent_id), ...])
            matches = [(lemma, score, occurrences.get(lemma, [])) for lemma, score in matches]
        else:
            len_metadata = len(metadata["raw_text"])
            matches = [(metadata["sent_id"][idx], metadata["raw_text"][idx], np.round(float(distance), 3))
                       for idx, distance in zip(indices[0], distances[0])
                       if 0 <= idx < len_metadata]
    except AssertionError:
        logger.warning("Assert error: query was probably encoded using a different model than target embeddings, please reembed target texts")
        return []
    return matches

def load_lemma_index(lemma_index_path=None):
    """
    Charge la liste ordonnée des lemmes depuis un fichier '_lemma_index.json',
    où l'index i correspond à la ligne i de l'index FAISS token-mode.
    """
    logger.info("loading lemma index from:%s",lemma_index_path)
    with open(lemma_index_path,"r",encoding="utf-8") as f:
        return json.load(f)



def search_folder(input_folder=None, query_str=None, query_vector=None,
                  metric_type=faiss.METRIC_INNER_PRODUCT, top_k=10, verbose=True,
                  token_mode=False, no_daemon=False, allow_no_abtt=False,max_token_ids_occ=8):
    """
    Exécute une recherche de similarité sur un ensemble d'index FAISS contenus dans un dossier.

    Args:
        input_folder (str): Chemin du dossier ou pattern (wildcard, ex. *Camus*) contenant les '.faiss'.
        query_str (str): Texte brut de la requête.
        query_vector (numpy.ndarray): Vecteur BRUT pré-calculé (ni normalisé, ni transformé ABTT).
        metric_type (int, optional): Métrique FAISS (défaut: produit scalaire).
        top_k (int): Nombre global de meilleurs résultats à conserver et afficher.
        verbose (bool): Si True, affiche le tableau des résultats.
        token_mode (bool): Si True, cible les index '_token.faiss' (et leurs fichiers
            '_lemma_index.json' et '_abtt.json').
        no_daemon (bool): Si True, encodage local sans démon.
        allow_no_abtt (bool): Si True, les corpus sans fichier '_abtt.json' ne sont plus
            ignorés : la recherche se fait sans ABTT (tests uniquement).
        max_token_ids_occ: nombre max de tokens et sent_ids à afficher par lemme (mode token)
    Valeurs retournées:
        None: agrège et affiche les résultats.
    """

    import time
    import glob
    t0 = time.perf_counter()
    logger.info("Folder embedding search")

    index_ext = "_token.faiss" if token_mode else ".faiss"
    if '*' in input_folder:
        file_list = []
        for f in glob.glob(input_folder):
            stem = os.path.splitext(f)[0]
            # ignore les fichiers annexes (lemmes, ABTT)
            if stem.endswith("_lemma_index") or stem.endswith("_abtt"):
                continue
            file_list.append(stem.replace("_token", "") + index_ext)
    else:
        file_list = glob.glob(input_folder + "/*" + index_ext)
    if token_mode:
        file_list = list(set([f for f in file_list if os.path.splitext(f)[1] == ".faiss" and '_token' in f]))
    else:
        file_list = list(set([f for f in file_list if os.path.splitext(f)[1] == ".faiss"]))

    len_f = len(file_list)
    results = []
    skipped = False
    logger.info("Found %s files in folder", len_f)

    if query_vector is None:
        query_vector = embedd_query(query_str, token_mode, no_daemon=no_daemon)

    for f in file_list:
        base, ext = os.path.splitext(f)
        try:
            index = load_index(f)
        except RuntimeError:
            logger.warning("%s index file not found in directory,skipping file", base + ".faiss")
            skipped = True
            continue

        if token_mode:
            lemma_index_path = base + "_lemma_index.json"
            abtt_path = abtt_path_from_index(f)
            try:
                lemma_list = load_lemma_index(lemma_index_path)
            except FileNotFoundError:
                logger.warning("%s lemma index file not found in directory,skipping file", lemma_index_path)
                skipped = True
                continue
            if not os.path.exists(abtt_path):
                if allow_no_abtt:
                    abtt_path = None   # search() fera la recherche sans ABTT (avec avertissement)
                else:
                    logger.warning("%s ABTT file not found, skipping file (re-run calcEmbeddings)", abtt_path)
                    skipped = True
                    continue
            result = search(query_str=query_str, query_vector=query_vector, index=index,
                            index_file=f, abtt_file=abtt_path, metric_type=metric_type,
                            top_k=top_k, token_mode=token_mode, no_daemon=no_daemon,
                            lemma_list=lemma_list, allow_no_abtt=allow_no_abtt,metadata_file=base.replace("_token","")+".json")
            result = [(f, r[0], float(r[1]), r[2]) for r in result]
        else:
            try:
                metadata = load_metadata(base + ".json")
            except FileNotFoundError:
                logger.warning("%s metadata file not found in directory,skipping file", base + ".json")
                skipped = True
                continue
            result = search(query_str=query_str, query_vector=query_vector, index=index,
                            index_file=f, metric_type=metric_type, top_k=top_k,
                            metadata=metadata, token_mode=token_mode, no_daemon=no_daemon)
            result = [(f, r[0], r[1], float(r[2])) for r in result]
        results.extend(result)

    sort_idx = 2 if token_mode else 3
    results.sort(key=lambda x: x[sort_idx], reverse=True)
    results = results[:top_k]

    exec_time = time.perf_counter() - t0
    logger.info("Folder search executed in %s seconds in %s files", np.round(exec_time, 2), len_f)
    if skipped:
        logger.warning("Some index files were skipped because file or corresponding metadata files were not found")
    if results == []:
        logger.warning("Search query didn't return any results, input file list probaby empty")
    if verbose:
        if token_mode:
            max_occ = max_token_ids_occ  # nombre max d'occurrences affichées par lemme
            print("index file                | lemma       | similarity score | tokens (sent_id)")
            for r in results:
                occ = r[3]
                shown = ", ".join(f"{tok} ({sid})" for tok, sid in occ[:max_occ])
                if len(occ) > max_occ:
                    shown += f", ... (+{len(occ) - max_occ})"
                print(f"{r[0]} | {r[1]} | {r[2]} | {shown}")
        else:
            print("index file                | Sent id               | Sentence    | similarity score")
            for r in results:
                print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]}")
