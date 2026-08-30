#!/usr/bin/env python3
"""
Daemon d'encodage vectoriel par lots.

Ce script s'exécute en tâche de fond et reste à l'écoute des requêtes d'encodage via des sockets IPC (Inter-Process Communication).
Ce daemon a pour but d'optimiser l'encodage des corpus et des requêtes en évitant de recharger le modèle d'encodage chaque fois.
Son architecture sépare les entrées/sorties réseau du calcul pur : il regroupe les requêtes de plusieurs
clients dans une file d'attente et les traite par lots pour maximiser les performances du GPU
et éviter la saturation de la VRAM.

Fonctionnalités principales :
- Chargement fainéant (lazy loading) et déchargement automatique des modèles après inactivité: évite de charger les deux modèles à la fois en ne chargeant un modèle que s'il est demandé par l'utilisateur, une fois chargé, il reste dans la mémoire.
- Support du modèle par défaut (BAAI/bge-m3) et du modèle par tokens (intfloat/multilingual-e5-base).
Pour le mode token:
- Opérations de pooling avancées (moyenne des dernières couches cachées) (sera probablement supprimé dans la version à venir car le mode token nécessite de garder les embeddings des tokens individuels)
- Réduction de dimensionnalité algorithmique (All-but-the-top): permet d'optimiser les embeddings du mode token en éliminant certaines dimensions.
"""
import logging
import queue
import sys
import threading
import time
from multiprocessing.connection import Listener
import numpy as np
torch_imported = False
# modèle d'embeddings pour le mode phrase
MODEL_NAME = "BAAI/bge-m3"
# modèle d'embedding pour le mode token
TOKEN_MODEL_NAME = 'intfloat/multilingual-e5-base'

HOST, PORT = "localhost", 6000
BACKLOG = 256
MAX_BATCH_SIZE = 256
BATCH_WINDOW_S = 0.015
KEEP_MODEL_LOADED_TIMEOUT = 10 #seconds
device = 'None'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("embed_daemon")

# initiaisation des variables globales pour faciliter le chargement 'fainéant' afin d'optimiser au maximum la taille de la VRAM utilisée,
model = None
tokenizer = None
token_model = None
torch = None
F = None
Tensor = None
AutoTokenizer = None
AutoModel = None
SentenceTransformer = None
use_last_n_layers = False
last_connection_time = 0


def handle_timeout():
    """
    Gère le déchargement automatique des modèles de la mémoire vidéo (VRAM) après
    une période d'inactivité définie par KEEP_MODEL_LOADED_TIMEOUT.
    Cette fonction est encore en développement et elle n'est pas encore utilisée.
    """
    global last_connection_time
    logger.info("Last connecttion time = %s",last_connection_time)
    if last_connection_time == 0:
        t0 = time.perf_counter()
        last_connection_time +=t0
        # Si le temps écoulé dépasse la limite, on libère la mémoire en supprimant les références aux modèles
        if last_connection_time>=KEEP_MODEL_LOADED_TIMEOUT:
            model = None
            token_model = None

def load_models(token_mode=False):
    """
    Charge les modèles d'embedding en mémoire (VRAM si CUDA est disponible, sinon RAM)
    uniquement lorsqu'ils sont nécessaires (lazy loading).
    Chaque modèle est chargé après une première requête par l'utilisateur, puis gardé en mémoire en processus d'arrière plan (daemon)

    Args:
        token_mode (bool): Détermine quel modèle charger (E5 pour les tokens, BGE-M3 pour les phrases).
    """
    global model
    global tokenizer
    global token_model
    global device
    # Détection automatique de l'accélération matérielle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device is ",device)
    # Chargement du modèle orienté tokens
    if (token_mode) and (token_model is None):
        model = None
        logger.info("Loading model %s",TOKEN_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(TOKEN_MODEL_NAME)
        token_model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
        token_model.to(device)
    # Chargement du modèle orienté phrases (SentenceTransformer)
    elif (not token_mode) and (model is None):
        token_model = None
        logger.info("Loading model %s",MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)
        model.to(device)
        logger.info("Models loaded successfully")
def _ensure_imports():
    """
    Importe les bibliothèques lourdes (PyTorch, Transformers) uniquement au démarrage effectif
    du démon pour accélérer l'importation initiale du module parent.
    """
    global torch, F, Tensor, AutoTokenizer, AutoModel, SentenceTransformer,device
    if torch is None:
        import torch
        import torch.nn.functional as F
        from torch import Tensor
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import SentenceTransformer
batch_queue: "queue.Queue" = queue.Queue()  # items: (sentences_list, result_queue)
def all_but_the_top(X, n_components=3):
    logger.info("applying 'All but the top' to embeddings:%s",X.shape)
    import torch
    import numpy as np

    was_numpy = isinstance(X, np.ndarray)
    if was_numpy:
        X = torch.from_numpy(X)

    X = X - X.mean(dim=0, keepdim=True)
    n_samples, n_features = X.shape
    q = min(n_components, n_samples, n_features)
    if q > 0:
        U, S, V = torch.pca_lowrank(X, q=q)
        P = V[:, :q]
        X = X - X @ P @ P.T

    return X.numpy() if was_numpy else X


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def average_pool_last_n_layers(hidden_states: tuple[Tensor, ...],
                                attention_mask: Tensor,
                                num_layers: int = 4) -> Tensor:
    # hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    stacked = torch.stack(hidden_states[-num_layers:])   # (n, batch, seq, hidden)
    layer_avg = stacked.mean(dim=0)                       # (batch, seq, hidden)
    return average_pool(layer_avg, attention_mask)

def batching_worker():
    """
    Thread d'exécustion unique.
    Extrait les éléments en attente dans la file, les concatène en un seul lot
    pour l'encodage, puis redistribue les vecteurs résultants pour reconstruire les listes d'origine.
    """
    while True:
        # Bloque l'exécution jusqu'à ce qu'une première requête arrive dans la file
        sentences, result_q, token_mode = batch_queue.get()  # blocks for first item
        batch_items = [(sentences, result_q)]
        batch_sentences = list(sentences)
        # S'assure que le bon modèle est chargé en mémoire (lazy loading)
        load_models(token_mode=token_mode)
        # Définit une fenêtre de temps maximale pour accumuler d'autres requêtes (ex: 15ms)
        deadline = time.monotonic() + BATCH_WINDOW_S
        # Accumule des phrases jusqu'à atteindre la taille maximale du lot (MAX_BATCH_SIZE)
        while len(batch_sentences) < MAX_BATCH_SIZE:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break          # La fenêtre de temps est écoulée, on lance le calcul
            try:
                s, rq = batch_queue.get(timeout=timeout)
            except queue.Empty:
                break       # Aucune autre requête n'est arrivée à temps
            batch_items.append((s, rq))
            batch_sentences.extend(s)

        try:
            # Branche 1 : Encodage orienté tokens (multilingual-e5-base)
            if token_mode:
                # Tokenisation avec troncature et padding dynamique (max 512 tokens)
                batch_dict = tokenizer(batch_sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
                # Transfert des tenseurs sur le périphérique matériel (GPU ou CPU)
                batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
                # Pooling : combinaison des plongements des sous-mots: cette partie sera revue dans la version à venir étant donné que le mode token sera destiné à un encodage effectif des mots individuels sans pooling.
                if use_last_n_layers:
                    outputs = token_model(**batch_dict, output_hidden_states=True)
                    vecs = average_pool_last_n_layers(outputs.hidden_states, batch_dict["attention_mask"], num_layers=4)
                else:
                    outputs = token_model(**batch_dict, output_hidden_states=False)
                    vecs = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # Détachement de l'objet PyTorch et conversion en NumPy (CPU)
                vecs = vecs.cpu().detach().numpy().astype(np.float32)
            # Branche 2 : Encodage en mode phrases (SentenceTransformer)
            else:
                vecs = model.encode(batch_sentences,batch_size=256).astype(np.float32)
            if device == 'cuda':
                # Libération préventive du cache VRAM pour éviter les fuites mémoire (après avoir constaté une augmentation très rapide de la taille de la mémoire occupée lors de l'encodage de plusieurs corpus consécutifs)
                torch.cuda.empty_cache()

        except Exception as e:
            logger.exception("Batch encode failed (%d sentences, %d jobs)",
                              len(batch_sentences), len(batch_items))
            # En cas de crash du GPU (ex: Out of Memory), on avertit toutes les requêtes du lot
            for _, rq in batch_items:
                rq.put(("error", str(e)))
            continue

        # Redistribution du lot traité : découpe le gros tenseur de résultats
        # en petits segments correspondant à chaque requête d'origine
        offset = 0
        for sentences, rq in batch_items:
            n = len(sentences)
            # Renvoie le statut 'ok' et le sous-ensemble de vecteurs via la file de retour du client
            rq.put(("ok", vecs[offset:offset + n]))
            offset += n

    #time.sleep(1)
def handle_client(conn):
    """
    Gère la communication avec un client connecté.
    S'exécute dans un thread dédié (un thread par connexion).
    Ne fait aucun calcul GPU : gère uniquement les entrées/sorties réseau (I/O)
    et délègue le travail au worker GPU via le système de file d'attente (batch_queue).

    Args:
        conn (multiprocessing.connection.Connection): L'objet de connexion socket du client.
    """
    job_id = None
    try:
        # 1. Réception de la requête du client
        job_id, sentences, chunk_size, token_mode = conn.recv()
        logger.info("Job %s: %d sentences, chunk_size=%d", job_id, len(sentences), chunk_size)

        total = len(sentences)
        print(token_mode)
        all_vecs = []
        # Envoi d'un signal initial de progression pour confirmer la prise en charge
        conn.send(("progress", 0, total))
        # 2. Découpage en sous-lots (chunks) définis par le client
        for i in range(0, total, chunk_size):
            chunk = sentences[i:i + chunk_size]
            # Création d'une file d'attente privée et unique pour recevoir la réponse de ce sous-lot
            result_q = queue.Queue()
            # Envoi du sous-lot dans la file d'attente globale du worker GPU
            batch_queue.put((chunk, result_q, token_mode))
            # Le thread se met en pause ici (bloquant) jusqu'à ce que le GPU ait terminé d'encoder ce chunk
            status, payload = result_q.get()  # waits for the batching worker
            # Vérification si le GPU a rencontré une erreur (ex: Out of Memory)
            if status == "error":
                raise RuntimeError(payload)

            # ajout du lot en cours a la liste des vecteurs à retourner
            all_vecs.append(payload)
            # Calcul et envoi de l'avancement au client
            done = min(i + chunk_size, total)
            conn.send(("progress", done, total))
        # 3. Assemblage final
        # Concaténation de tous les sous-lots de vecteurs en une seule grande matrice NumPy
        result = np.concatenate(all_vecs, axis=0)
        # Envoi du résultat final au client
        conn.send(("done", result))
        logger.info("Job %s: completed, sent %d embeddings", job_id, len(result))
        # Libération de la mémoire locale
        all_vecs = None
    except Exception as e:
        # 4. Gestion globale des erreurs du job
        logger.exception("Job %s: error", job_id)
        try:
            # Tente d'avertir le client de l'échec
            conn.send(("error", str(e)))
        except Exception:
            # Si le réseau est coupé, on log l'échec d'envoi
            logger.warning("Job %s: could not send error, connection closed", job_id)
    finally:
        # 5. Nettoyage : fermeture systématique de la connexion (évite les connexions fantômes)
        conn.close()


def accept_loop(listener):
    while True:
        try:
            conn = listener.accept()
            logger.info("Accepted connection from %s", listener.last_accepted)
        except (EOFError, OSError) as e:
            logger.warning("Bad handshake attempt ignored: %s", e)
            continue
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

# function principale du démarrage du daemon
def main():
    """
    Boucle principale d'écoute du serveur (tourne indéfiniment).
    Accepte les nouvelles connexions entrantes et délègue leur traitement
    à des threads séparés pour ne jamais bloquer l'arrivée d'autres clients.
    """
    logger.info("Daemon starting")
    # Importe PyTorch et les Transformers uniquement au démarrage pour
    # éviter de ralentir d'autres scripts qui importeraient ce fichier
    _ensure_imports()

    # 1. Démarrage du "moteur" GPU en arrière-plan
    # Ce thread tournera en boucle infinie pour traiter la file d'attente
    threading.Thread(target=batching_worker, daemon=True).start()
    # 2. Création du socket d'écoute (Inter-Process Communication)
    # HOST, PORT et BACKLOG (nombre max de connexions en file d'attente non acceptées)
    # sont définis au début du fichier
    listener = Listener((HOST, PORT), backlog=BACKLOG)
    logger.info("Listener ready on %s:%d (backlog=%d)", HOST, PORT, BACKLOG)

    accept_loop(listener)


if __name__ == "__main__":
    main()
