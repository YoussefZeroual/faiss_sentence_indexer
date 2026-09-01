"""
Module de création d'index FAISS pour la recherche sémantique.

Ce script est responsable de la structuration des vecteurs d'embeddings (fichiers .npy)
en index de similarité hautement optimisés avec le modue FAISS (fichiers .faiss). Il fait le pont entre
la phase d'extraction des caractéristiques (calcul des embeddings) et la phase de recherche.
Une fois les fichiers .faiss sont crées, les fichiers .npy des embeddings peuvent être effacés car la recherche sémantique se base désormais sur les index .faiss. Toutefois, si l'on veut recréer les index, il serait nécessaire de refaire les embeddings.'
Fonctionnalités principales :
- Chargement et normalisation (L2) des vecteurs d'embeddings.
- Support de multiples algorithmes d'indexation selon le compromis vitesse/mémoire souhaité :
    * 'flat'  : Recherche exhaustive (précision parfaite, adapté aux petits corpus, taille du fichier index considérable, égale à la taille du fichier embeddings .npy)
    * 'hnsw'  : Recherche approximative basée sur des graphes (très rapide, gourmand en mémoire).
    * 'ivfpq' : Recherche partitionnée (IVF) et quantifiée (PQ) (optimisé pour la RAM et les grands corpus).
- Traitement par lots : génération d'index pour un dossier complet ou via des wildcards (ex. *Camus*).
- Gestion des modes d'encodage (phrase entière ou niveau token).
"""

import faiss
import numpy as np
import json
import logging
import glob
import time
import os
# configuration du formatage du journal affiché
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_embeddings(embedding_file_path):
    """
    Charge les vecteurs d'embeddings depuis un fichier NumPy, les formate
    et les normalise pour l'indexation FAISS.

    Args:
        embedding_file_path (str): Le chemin vers le fichier d'embeddings (généralement .npy).

    Valeurs retournées:
        numpy.ndarray: Une matrice 2D contiguë de type float32, avec des vecteurs normalisés L2.
    """
    logger.info("Loading embeddings file %s",embedding_file_path)
    # Séparation du nom de base et de l'extension pour forcer le chargement du fichier .npy même si l'utilisateur fournit une extension différente.
    base,ext = os.path.splitext(embedding_file_path)
    embeddings = np.load(base+".npy")
    # FAISS (étant écrit en C++) exige un tableau de type float32 stocké de manière contiguë en mémoire.
    # Cette étape prévient de nombreuses erreurs de compatibilité ou de fuites mémoire.
    embeddings = np.ascontiguousarray(embeddings,dtype=np.float32)
    # Normalisation L2 des vecteurs.
    # Indispensable pour que la métrique de produit scalaire (Inner Product)
    # le produit scalaire appliqué à des vecteurs normalisés L2 est l'équivalent d'une similarité cosinus, c'est la métrique la plus utilisée en recherche sémantique.
    faiss.normalize_L2(embeddings)
    return embeddings
def makeIndex(embeddings=None,embedding_file_path=None,metric_type=None,index_type=None,m=512,output_file_path=None,overwrite=False,token_mode=False):
    """
    Construit, entraîne et sauvegarde un index FAISS à partir de vecteurs d'embeddings.

    Args:
        embeddings (numpy.ndarray): Matrice des vecteurs à indexer.
        embedding_file_path (str): Chemin vers le fichier d'embeddings à charger si l'objet embeddings n'est pas fourni.
        metric_type (int): Type de métrique de distance FAISS (ex: faiss.METRIC_INNER_PRODUCT).
        index_type (str): L'algorithme d'indexation cible ('flat', 'hnsw', ou 'ivfpq').
        m (int): # Nombre de sous-vecteurs pour la compression (PQ), ce paramètre a un impact direct sur la précision des résultats des requpetes: en testant avec m = 8, les résultats étaient modests, augmenté à 64,128 et 256 les résultats se sont approchés de ceux d'un index flat (sans compression). Il faut cependant noter que plus cette valeur est élevé plus la taille de l'index est grande (ex. pour un corpus de ~28M tokens et m=256, la taille de l'index ivfpq = 35MB, pour un index flat du même corpus, la taille est plus de 460 MB ). Meilleure valeur testée jusqu'à présent = 512
        output_file_path (str): Chemin de destination pour sauvegarder le fichier d'index.
        overwrite (bool): Si False, charge l'index existant s'il est déjà présent sur le disque.
        token_mode (bool): Si True, ajuste le nom du fichier de sortie pour refléter l'encodage par token.

    Valeurs de retour:
        faiss.Index: L'index FAISS prêt à être utilisé pour la recherche de similarité.

    Raises:
        ValueError: Si le type d'index fourni n'est pas reconnu par le script.
    """
    import os
    # Chargement ou préparation des embeddings en mémoire
    if embedding_file_path is not None:
        # Charge depuis le disque (applique également la normalisation L2 et la contiguïté)
        base, ext = os.path.splitext(embedding_file_path)
        if not overwrite and (os.path.exists(base+".faiss")):
            logger.info("Index already exist, loading from %s",base+".faiss")
            from searchEmbedding import load_index
            index=load_index(base+".faiss")
            return index
    if embeddings is None:
       embeddings = load_embeddings(embedding_file_path)
    else:
        # Assure la compatibilité mémoire C++ de FAISS et normalise les vecteurs passés en argument
        embeddings = np.ascontiguousarray(embeddings,dtype=np.float32)
        faiss.normalize_L2(embeddings)
    logger.info("making index for %s sentences, index type=%s",embeddings.shape[0],index_type)
    n,dim = embeddings.shape
    t0 = time.perf_counter()     #début du chronomètre pour calculer le temps d'exécution
    # Sélection de la stratégie d'indexation FAISS
    if index_type == "flat":
        # Index exact sans compressions ou approximation (recherche exhaustive) : Précision parfaite, idéal pour de petits corpus, génère des fichiers index de grande taille.
        index = faiss.IndexFlat(dim,metric_type)
        index.add(embeddings)
        # Index HNSW (graphe de proximité) : Recherche approximative très rapide, mais consomme beaucoup de RAM
    elif index_type == "hnsw":
        hnsw_m = 64
        index = faiss.IndexHNSWFlat(dim, hnsw_m, metric_type)
        index.hnsw.efConstruction = 40     # Profondeur de recherche lors de la construction
        index.add(embeddings)
        index.hnsw.efSearch = 64    # Profondeur de recherche lors des requêtes
        # Index IVFPQ (Partitionnement + Quantification) : Optimise l'utilisation de la RAM pour les très grands corpus, optimise très considérablement la taille du fichier index
    elif index_type == "ivfpq":
        nlist = 4 * int(np.sqrt(n))    # Nombre de clusters (cellules de Voronoï)
        m = m                          # Nombre de sous-vecteurs pour la compression (PQ), ce paramètre a un impact direct sur la précision des résultats des requpetes: en testant avec m = 8, les résultats étaient modests, augmenté à 64,128 et 256 les résultats se sont approchés de ceux d'un index flat (sans compression). Il faut cependant noter que plus cette valeur est élevé plus la taille de l'index est grande.
        nbits = 8                      # Bits alloués par sous-vecteur (2^8 = 256 centroïdes)


        # Restriction par mesure de sécurité : IVFPQ nécessite un grand nombre d'échantillons pour s'entraîner statistiquement.
        # Si le corpus est trop petit, le code bascule automatiquement sur un index 'flat'.

        min_points_ivf = nlist * 39
        min_points_pq = (2 ** nbits) * 39
        if n < min_points_ivf or n < min_points_pq:
            required_points = max(min_points_ivf, min_points_pq)
            logger.warning(
                "Not enough training points (%s) for IVFPQ (need ≥ %s); using 'flat' instead",
                n, required_points
            )
            index = faiss.IndexFlat(dim,metric_type)
            pass
        else:
            quantizer = faiss.IndexFlat(dim, metric_type)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, metric_type)
            index.train(embeddings)
        index.add(embeddings)
    else:
        logger.warning("Index Type unrecognized or empty: %s",index_type)
        raise ValueError(f"Index type unkown or empty: {index_type}")
    # Adaptation du nom de fichier de sortie selon le mode d'encodage (phrase ou token)
    if token_mode:
        output_file_path = output_file_path.replace(".faiss","_token.faiss")

    # Sauvegarde physique de l'index sur le disque
    faiss.write_index(index,output_file_path)
    t1 = time.perf_counter()  #fin du chronomètre pour calculer le temps d'exécution
    exec_time = t1-t0
    logger.info("Index created successfully in %s seconds",np.round(exec_time,3))
    logger.info("index written successfully to %s",output_file_path)


    return index
def makeIndex_folder(input_folder=None,metric_type=None,index_type=None,m=512,overwrite=False,token_mode=False):
    """
    Parcourt un dossier (ou un wildcard ex. *Camus*) pour trouver des fichiers d'embeddings
    et crée un index FAISS pour chacun d'eux.

    Args:
        input_folder (str): Le chemin du répertoire cible ou un wildcard (ex: 'dossier/*.npy', dossier/ESLO*).
        metric_type (int): Le type de métrique de distance pour FAISS (ex: faiss.METRIC_INNER_PRODUCT).
        index_type (str): L'algorithme d'indexation à utiliser ('flat', 'hnsw', ou 'ivfpq').
        overwrite (bool, optional): Si True, force le recalcul et l'écrasement des index existants.
        token_mode (bool, optional): Si True, cible spécifiquement les fichiers encodés au niveau des tokens ('_token.npy').

    Valeurs de retour:
        None: La fonction agit directement sur les fichiers ciblés en écrivant les fichiers .faiss sur le disque.
    """
    # Définition de l'extension cible en fonction du mode d'encodage (phrase ou token)
    if token_mode:
        emb_ext = "_token.npy"
    else:
        emb_ext = ".npy"
    # Récupération des chemins de fichiers
    # Si input_folder contient un wildcard ('*'), on utilise glob pour trouver les correspondances
    if '*' in input_folder:
        file_list = glob.glob(input_folder)
        file_list = [os.path.splitext(f)[0].replace("_token","")+emb_ext  for f in file_list]
    else:
        # Sinon, on recherche tous les fichiers avec l'extension appropriée dans le répertoire fourni
        file_list = glob.glob(input_folder+"/*"+emb_ext)
    # Suppression des éventuels doublons dans la liste des fichiers
    file_list = list(set(file_list))
    len_f = len(file_list)
    logger.info("Found %s files in folder",len_f)
    # Affichage de la liste complète des fichiers identifiés
    for f in file_list:
        logger.info("%s",f)
    # Boucle principale : création de l'index pour chaque fichier d'embedding trouvé
    for f in file_list:
        base,ext = os.path.splitext(f)
        logger.info("Making index for file %s",f)
        # Définition du chemin de sortie pour le fichier d'index
        output_file_path = base+".faiss"
        # Appel de la fonction makeIndex pour traiter ce fichier spécifique
        index = makeIndex(embedding_file_path=f,metric_type=metric_type,index_type=index_type,m=m,output_file_path=output_file_path,overwrite=overwrite)


if __name__ == "__main__":
    index = makeIndex_folder("test")
