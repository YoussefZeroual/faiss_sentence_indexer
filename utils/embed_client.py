"""
Client d'encodage vectoriel (Embedding Client).
Il constitue l'interface entre les différents scripts et le daemon d'embedding
Il implémente trois stratégies d'exécution distinctes :
1. Mode Ollama : Délègue l'encodage à un serveur Ollama (local ou distant) via requêtes HTTP.
2. Mode daemon (par défaut) : Communique via IPC avec le processus de 'embed_daemon.py' pour un traitement asynchrone par lots.
3. Mode No_daemon : Charge les modèles PyTorch directement dans le processus courant.
"""
import os
import socket
import subprocess
import sys
import time
import uuid
from multiprocessing.connection import Client, Listener
import logging
from utils.embed_daemon import average_pool,average_pool_last_n_layers
import numpy as np
import requests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #filename='/home/miai_guest/zeroualy/module_faiss/app.log'
)
MODEL_NAME = "BAAI/bge-m3"
TOKEN_MODEL_NAME = 'intfloat/multilingual-e5-base'
logger = logging.getLogger(__name__)
# Chemin absolu vers le script du démon, déterminé dynamiquement pour éviter toute incompatibilité avec le système d'exploitation
DAEMON_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_daemon.py")
# Variables globales pour le mode "no_daemon" (chargement direct en mémoire)
weights_loaded = False
model = None
token_model = None
token_tokenizer=None
device=None
def encode_ollama(host='localhost:11434',model=None,sentence_list=None):
    """
    Génère des embeddings en interrogeant une instance de l'API Ollama.
    Nécessite un serveur Ollama et un modèle d'embeddings préinstallés dans ce serveur.
    Args:
        host (str): L'adresse et le port du serveur Ollama.
        model (str): Le nom du modèle à utiliser sur le serveur Ollama.
        sentence_list (list): La liste des phrases à encoder.

    Returns:
        list ou None: La liste des vecteurs retournés par l'API, ou None en cas d'échec de connexion.
    """
    logger.info("Using ollama for embedding")
    try:
        # Envoi de la requête POST au point d'accès d'embedding de l'API Ollama
        response = requests.post(
            f"http://{host}/api/embed",
            json={"model":model, "input": sentence_list}
            )
    except requests.exceptions.ConnectionError as e:
        logger.warning("%s",e)
        return None
    data = response.json()
    embs = data.get('embeddings')
    return embs
def _try_connect():
    """
    Tente d'établir une connexion IPC avec le processus daemon sur le port 6000.

    valeur retournée:
        multiprocessing.connection.Client ou None: L'objet de connexion si réussi, sinon None.
    """
    try:
        return Client(("localhost", 6000))
    except (ConnectionRefusedError, OSError):
        return None


def _start_daemon():
    """
    Si le daemon d'embedding n'est pas déjà lancé, cette fonction le lance en tant que processus indépendant (détaché)
    et attend qu'il soit prêt à accepter des connexions.

    valeur de retour:
        multiprocessing.connection.Client: La connexion établie avec le démon fraîchement démarré.

    Raises:
        RuntimeError: Si le démon ne répond pas après environ 60 secondes d'attente.
    """
    logger.info("starting embedding daemon")
    # Lancement du processus en arrière-plan sans bloquer le script actuel
    subprocess.Popen(
        [sys.executable, DAEMON_SCRIPT],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True # Détache le démon du terminal courant
    )
    # Boucle d'attente active : tente de se connecter jusqu'à 120 fois (avec 0.5s d'intervalle)
    # Cela laisse le temps au daemon de démarrer et de charger potentiellement PyTorch en mémoire.
    for _ in range(120):
        conn = _try_connect()
        if conn is not None:
            return conn
        time.sleep(0.5)
    raise RuntimeError("Embedding daemon failed to start")

def encode_no_daemon(sentences=None,token_mode=False):
    """
    Exécute l'encodage vectoriel directement dans le processus courant (de manière synchrone),
    sans utiliser le système de daemon en arrière-plan.

    Cette fonction utilise un chargement paresseux (lazy loading) : les bibliothèques lourdes
    (PyTorch) et les poids des modèles ne sont chargés en mémoire qu'au premier appel.

    Args:
        sentences (list): La liste des phrases ou textes à encoder.
        token_mode (bool): Si True, utilise le modèle orienté tokens (multilingual-e5-base)
                                     avec un pooling sur les dernières couches.

    valeur retournée:
        numpy.ndarray: La matrice des vecteurs d'embeddings (float32).
    """
    global weights_loaded
    global model
    global token_model
    global token_tokenizer
    global device
    # Chargement conditionnel pour éviter de saturer inutilement la RAM si la fonction n'est jamais appelée
    if not weights_loaded:
        from torch import Tensor
        from transformers import AutoTokenizer, AutoModel
        import torch

        from sentence_transformers import SentenceTransformer
        logger.info("Loading models...")
        # Initialisation du modèle pour le mode 'phrases' (biliothèque SentenceTransformer)
        model = SentenceTransformer(MODEL_NAME)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        # Initialisation du modèle du mode 'tokens' (directement la bibliothèque HuggingFace)
        token_tokenizer = AutoTokenizer.from_pretrained(TOKEN_MODEL_NAME)
        token_model = AutoModel.from_pretrained(TOKEN_MODEL_NAME)
        token_model.to(device)
        weights_loaded = True

    if token_mode:
        # branche 1: mode token
        from tqdm import tqdm
        batch_size = 32
        vecs = []
        # Traitement par lots manuel avec affichage d'une barre de progression
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i+batch_size]
            # Préparation des tenseurs pour le modèle
            batch_dict = token_tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            # Traitement des tenseurs par le modèle avec récupération des états cachés
            outputs = token_model(**batch_dict, output_hidden_states=True)
            # Pooling mathématique spécifique (moyenne des 4 dernières couches)
            # N.B: le pooling sera probablement enlevé de la version à venir étant donné que le mode token concerne effectivement la création de vecteurs pour les tokens individuels et non un pooling de ces tokens
            vec = average_pool_last_n_layers(outputs.hidden_states, batch_dict["attention_mask"], num_layers=4)

            # Copie des données sur le CPU et conversion en NumPy
            vecs.append(vec.cpu().detach().numpy().astype(np.float32))
        # Concaténation de tous les lots en une seule matrice
        vecs = np.vstack(vecs)
    else:
        # branche 2: mode phrase
        # COntrairement au modèle de traitement token, le modèle SentenceTransformer gère lui-même son propre batching et sa barre de progression
        vecs = model.encode(sentences,show_progress_bar=True).astype(np.float32)

    return vecs


def encode(sentences, chunk_size=512, show_progress=True,token_mode=False,no_daemon=False,use_ollama=False,ollama_host='localhost',ollama_model=None):
    """
    Fonction routeur principale pour la génération d'embeddings.
    permet de choisir la stratégie d'exécution selon le besoin de l'utilisateur (Ollama, local, ou via Démon IPC).

    Args:
        sentences (list): Les textes à encoder.
        chunk_size (int): Taille des sous-lots envoyés au démon via le réseau, la valeur par défaut 512 est déterminé après avoir testé des taillés plus grandes qui ont causé des débordements de la mémoire du GPU.
        show_progress (bool): Affiche dynamiquement la progression dans la console.
        token_mode (bool): Bascule sur le modèle d'encodage par token.
        no_daemon (bool): Force l'exécution locale synchrone.
        use_ollama (bool): Tente d'utiliser un serveur Ollama externe: attention, l'utilisation d'Ollama est significativement plus lente que le daemon ou un modèle chargé directement. Il s'agit d'un problème connu dans Ollama qui n'a pas encore été corrigé.
        ollama_host (str): Adresse du serveur Ollama.
        ollama_model (str): Modèle Ollama cible.

    Valeur retournée:
        numpy.ndarray: La matrice finale des embeddings générés.

    Raises:
        RuntimeError: Si le processus d'encodage (démon ou réseau) échoue.
    """
    # 1. Stratégie Ollama (API distante)
    if use_ollama:
        embs = encode_ollama(host=ollama_host,model=ollama_model,sentence_list=sentences)
        if embs:
            return embs
        elif embs is None:
            logger.info("Ollama response is empty, falling back to normal mode")
    # 2. Stratégie basée sur le chargement direct des modèles sans passer par le daemon (utile pour le débogage ou l'exécution d'une seule requête)
    if no_daemon:
        logger.info("Using no daemon mode")
        embs = encode_no_daemon(sentences,token_mode=token_mode)
        return embs
    # 3. Stratégie par défaut : IPC avec le daemon asynchrone
    # Tente de se connecter, ou démarre le démon s'il n'est pas encore actif
    conn = _try_connect()
    if conn is None:
        conn = _start_daemon()
    # Création d'un identifiant unique pour suivre la tâche pour des fins de débogage et de journalisation
    job_id = str(uuid.uuid4())
    # Envoi des phrases à traiter au daemon en spécifiant les variables d'exécution
    conn.send((job_id, sentences, chunk_size,token_mode))

    status, payload = None, None
    # Boucle d'écoute pour les retours du daemon
    while True:
        status, *rest = conn.recv()
        # Mise à jour de l'affichage de la progression
        if status == "progress":
            done, total = rest
            if show_progress:
                pct = round(100 * done / total, 1) if total else 0
                # Utilise '\r' pour écraser la ligne précédente dans la console
                print(f"\rJob {job_id[:8]}: {done}/{total} ({pct}%)", end="", flush=True)
            continue
        # Si le statut n'est pas 'progress', c'est 'done' ou 'error'
        payload = rest[0]
        break
    # Nettoyage de la connexion réseau
    conn.close()
    if show_progress:
        print() # Saut de ligne final pour ne pas écraser la dernière ligne de progression
    # Levée de l'exception locale si le démon a crashé de son côté
    if status == "error":
        raise RuntimeError(payload)
    return payload
