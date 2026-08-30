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
        token_mode (bool): Si True, utilise l'encodage au niveau des tokens au lieu du niveau phrase.
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
    # Démarrage du chronomètre pour mesurer le temps d'encodage
    t0 = time.perf_counter()
    if query_str is not None:
        logger.info("embedding query: %s",query_str)
        # Encodage selon le mode choisi (token ou phrase), en envoyant une liste contenant un seul élément (la requête)
        if token_mode:
            #from utils.embed_daemon import all_but_the_top
            embeddings = encode([query_str],chunk_size=1,token_mode=True,no_daemon=no_daemon,use_ollama=use_ollama,ollama_host=ollama_host,ollama_model=ollama_model)
        else:
            embeddings = encode([query_str],chunk_size=1,no_daemon=no_daemon,use_ollama=use_ollama,ollama_host=ollama_host,ollama_model=ollama_model)
        import numpy as np
        # Arrêt du chronomètre et calcul du temps d'exécution
        t1 = time.perf_counter()
        exec_time = t1-t0
        logger.info("Query encoded in %s seconds",np.round(exec_time,2))
        # Transformation de la liste d'embeddings en tableau NumPy 2D de type float32 (requis par FAISS)
        embeddings = np.array(embeddings,dtype=np.float32).reshape(1,-1)
        # S'assure que le tableau est stocké de manière contiguë en mémoire pour des performances optimales lors de la recherche
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        return embeddings
    else:
        # Journalisation et déclenchement d'une erreur si la requête est absente
        logger.warning("Query is empty")
        logger.warning("Query is empty")
        raise ValueError("Query is empty")

def search(query_vector=None,query_str=None, index=None, metric_type=None, top_k=10, metadata=None,token_mode=False,no_daemon=False):
    """
    Fonction principale du script.
    Exécute une recherche de similarité dans un index FAISS et renvoie les meilleures correspondances avec leurs métadonnées (phrases, identifiants et score de similarité cosinus).

    Args:
        query_vector (numpy.ndarray): Le vecteur d'embedding pré-calculé de la requête.
        query_str (str): Le texte de la requête (utilisé pour générer l'embedding si query_vector est None).
        index (faiss.Index): L'objet index FAISS dans lequel effectuer la recherche.
        metric_type (int): Le type de métrique FAISS utilisé (par défaut faiss.METRIC_INNER_PRODUCT qui correspond à une similarité cosinus étant donné que les vecteurs sont déjà normalisés L2).
        top_k (int): Le nombre maximal de résultats similaires à retourner (défaut: 10).
        metadata (dict): Le dictionnaire contenant les phrases en texte brut avec leurs identifiants ('sent_id', 'raw_text').
        token_mode (bool): Détermine si l'encodage d'une nouvelle requête se fait au niveau des tokens.
        no_daemon (bool): Détermine si l'encodage local est utilisé sans le processus démon.

    Valeurs de retour:
        list: Une liste de tuples sous le format (identifiant_phrase, texte_brut, score_similarité). Retourne une liste vide en cas d'erreur de dimension qui est généralement un résultat d'incompatibilité entre les vecteurs de la requête et ceux du corpus cible (utilisation par erreur de deux modèles différents).
    """
    # Déterminer le nombre total d'entrées des métadonnées pour éviter les débordements d'indices lors du mapping
    len_metadata = len(metadata["raw_text"])
    # Si aucun vecteur pré-calculé n'est fourni, on encode la requête textuelle à la volée
    if query_vector is None:
        query_vector = embedd_query(query_str,token_mode,no_daemon=no_daemon)
    # Normalisation L2 du vecteur de requête, essentielle pour que le inner product soit équivalent à une similarité cosinus
    faiss.normalize_L2(query_vector)
    # Ajustement du paramètre nprobe pour certains types d'index (index partitionnés comme IVF qui divise l'espace vectoriel en régions) afin d'améliorer la précision de la recherche
    if hasattr(index, "nprobe"):
        index.nprobe = 8
    try:
        # Lancement de la recherche FAISS pour extraire les k plus proches voisins (distances et indices)
        distances, indices = index.search(query_vector, top_k)
        # Création de la liste finale en associant l'ID, le texte et la distance arrondie (seulement si l'index est valide)
        matches = [(metadata["sent_id"][idx],metadata["raw_text"][idx], np.round(float(distance),3))
                for idx, distance in zip(indices[0], distances[0])
                if 0 <= idx < len_metadata]
    # Capture des erreurs fréquentes (ex: discordance entre la dimensionnalité de la requête et celle de l'index)
    except AssertionError as e:
        logger.warning("Assert error: query was probably encoded using a different model than target embeddings, please reembed target texts")
        return []
    return matches
def search_folder(input_folder=None,query_str=None,query_vector=None,metric_type=faiss.METRIC_INNER_PRODUCT,top_k=10,verbose=True,token_mode=False,no_daemon=False):
    """
    Exécute une recherche de similarité textuelle sur un ensemble d'index FAISS contenus dans un dossier.

    Args:
        input_folder (str): Le chemin du dossier ou le pattern de recherche (wildcard, ex. *Camus* pour restreindre la recherche à tous les fichiers dont le nom contient 'Camus' au milieu) contenant les fichiers '.faiss'.
        query_str (str): Le texte brut de la requête à rechercher.
        query_vector (numpy.ndarray): Le vecteur d'embedding pré-calculé (évite de ré-encoder la requête).
        metric_type (int, optional): La métrique de distance utilisée par FAISS (défaut: produit scalaire).
        top_k (int): Le nombre global de meilleurs résultats à conserver et afficher.
        verbose (bool): Si True, affiche le tableau des résultats dans la console.
        token_mode (bool): Si True, cible spécifiquement les index encodés au niveau des tokens (fichies se terminant avec le suffixe '_token.faiss').
        no_daemon (bool): Si True, utilise l'encodage local sans passer par le processus démon.

    Valeurs retournées:
        None: La fonction agrège et affiche les résultats, mais ne retourne aucune valeur.
    """
    import time
    t0 = time.perf_counter()
    logger.info("Folder embedding search")
    import glob
    import os
    token_suffix = ""
    # Définition des extensions cibles selon le mode d'encodage choisi
    if token_mode:
        index_ext = "_token.faiss"
        token_suffix = "_token"
    else:
       index_ext = ".faiss"
    # Récupération des chemins des fichiers via le module glob (supporte les wildcards '*')
    if '*' in input_folder:
        file_list = glob.glob(input_folder)
        file_list = [os.path.splitext(f)[0].replace("_token","")+index_ext  for f in file_list]
    else:
        file_list = glob.glob(input_folder+"/*"+index_ext)
    # Filtrage strict et déduplication pour s'assurer de ne garder que les fichiers d'index pertinents
    if token_mode:
        file_list = list(set([f for f in file_list if os.path.splitext(f)[1] ==".faiss" and '_token' in f]))
    else:
        file_list = list(set([f for f in file_list if os.path.splitext(f)[1] ==".faiss"]))
    len_f = len(file_list)
    results = []
    skipped = False
    logger.info("Found %s files in folder",len_f)
    # Si aucun vecteur n'est fourni, on encode la requête
    if query_vector is None:
        query_vector = embedd_query(query_str,token_mode,no_daemon=no_daemon)
    faiss.normalize_L2(query_vector)
    # Itération sur chaque fichier d'index trouvé dans le répertoire
    for f in file_list:
        base, ext = os.path.splitext(f)
        # Chargement de l'index
        try:
            index = load_index(f)
        except RuntimeError as e:
            logger.warning("%s index file not found in directory,skipping file",base+".faiss")
            skipped = True
            continue
        # Chargement des métadonnées correspondantes
        try:
            metadata = load_metadata(base.replace("_token","")+".json")
        except FileNotFoundError as e:
            logger.warning("%s metadata file not found in directory,skipping file",base+".json")
            skipped = True
            continue
        # Recherche des correspondances locales dans ce fichier spécifique
        result = search(query_str=query_str,query_vector=query_vector, index=index, metric_type=metric_type, top_k=top_k, metadata=metadata,token_mode=token_mode,no_daemon=no_daemon)
        # Formatage des résultats pour inclure le nom du fichier source (f)
        result = [(f,r[0],r[1],float(r[2]))
               for r  in result]
        results.extend(result)
    # Tri global de tous les résultats agrégés par score de similarité (indice 3), du plus élevé au plus bas
    results.sort(key=lambda x: x[3],reverse=True)
    # Conservation des top_k meilleurs résultats sur l'ensemble du corpus
    results = results[:top_k]
    t1 = time.perf_counter()
    exec_time = t1-t0
    logger.info("Folder search executed in %s seconds in %s files",np.round(exec_time,2),len_f)
    if skipped:
        logger.warning("Some index files were skipped because file or corresponding metadata files were not found")
    if results ==[]:
        logger.warning("Search query didn't return any results, input file list probaby empty")
    # Affichage tabulaire des correspondances trouvées
    if verbose:
        print("index file                | Sent id               | Sentence    | similarity score")
        for r in results:
            print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]}")
# fonction main pouvant être exécutée pour tester le fonctionnement du script
if __name__ == "__main__":
    import sys
    metric_type = faiss.METRIC_INNER_PRODUCT
    top_k = 10
    query_str = sys.argv[1]
    index = faiss.read_index("index.faiss")
    logger.info("Loaded index from %s","index.faiss")
    print("ntotal:", index.ntotal)
    print("dimension:", index.d)
    matches = search(query_str=query_str,
        index=index,
        metric_type=faiss.METRIC_INNER_PRODUCT,
        top_k=top_k,
        metadata_file_path= "metadata.json"
        )

    print(matches)
