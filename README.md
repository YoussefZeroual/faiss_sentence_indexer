# faiss_sentence_indexer

Outil de recherche sémantique de phrases pour des corpus littéraires (format CONLL-U / XML), basé sur des embeddings de phrases (modèle `sentence-camembert-base`) indexés avec **FAISS**.

Le pipeline permet de :
1. extraire les phrases d'un fichier de corpus (CONLL-U ou XML),
2. calculer leurs embeddings via un modèle SentenceTransformer,
3. construire un index FAISS optimisé pour la recherche par similarité cosinus,
4. interroger cet index avec une phrase de requête et récupérer les phrases les plus proches sémantiquement.

---

## Sommaire

- [Architecture générale](#architecture-générale)
- [Prérequis](#prérequis)
- [Structure des fichiers](#structure-des-fichiers)
- [Le serveur d'embeddings (`embed_daemon.py` / `embed_client.py`)](#le-serveur-dembeddings)
- [Calcul des embeddings (`calcEmbeddings.py`)](#calcul-des-embeddings)
- [Construction de l'index (`makeIndex.py`)](#construction-de-lindex)
- [Recherche (`searchEmbedding.py`)](#recherche)
- [Point d'entrée (`main.py`)](#point-dentrée)
- [Utilisation en ligne de commande](#utilisation-en-ligne-de-commande)
- [Choix du type d'index](#choix-du-type-dindex)
- [Limitations connues et points d'attention](#limitations-connues-et-points-dattention)

---

## Architecture générale

Le projet est organisé en briques indépendantes qui communiquent via des fichiers intermédiaires (`.npy`, `.json`, `.faiss`) :

```
fichier.conllu
      │
      ▼
calcEmbeddings.py ──► fichier.npy (embeddings)
      │                fichier.json (métadonnées : sent_id, raw_text)
      ▼
makeIndex.py ──► fichier.faiss (index FAISS normalisé)
      │
      ▼
searchEmbedding.py ──► résultats de recherche (top-k phrases similaires)
```

`main.py` orchestre l'ensemble de la chaîne de façon incrémentale : il ne recalcule que ce qui manque encore sur le disque (embeddings, métadonnées, ou index).

Le calcul des embeddings est délégué à un **daemon persistant** (`embed_daemon.py`), afin d'éviter de recharger le modèle CamemBERT à chaque appel. Le client (`embed_client.py`) démarre ce daemon si besoin et communique avec lui par socket local.

---

## Prérequis

- Python 3.9+
- Bibliothèques :
  ```bash
  pip install faiss-cpu numpy sentence-transformers conllu lxml
  ```
  (utiliser `faiss-gpu` à la place de `faiss-cpu` si un GPU CUDA est disponible)
- Le modèle `dangvantuan/sentence-camembert-base` sera téléchargé automatiquement depuis Hugging Face au premier lancement du daemon.

---

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `calcEmbeddings.py` | Parse un fichier CONLL-U (ou XML) et calcule les embeddings de chaque phrase |
| `makeIndex.py` | Construit un index FAISS à partir des embeddings (flat, HNSW ou IVFPQ) |
| `searchEmbedding.py` | Charge un index et effectue une recherche de similarité |
| `embed_client.py` | Interface cliente vers le daemon d'embedding (gestion du cycle de vie du process, envoi/réception des jobs) |
| `embed_daemon.py` | Processus serveur qui charge le modèle une seule fois et traite les requêtes d'encodage |
| `main.py` | Script d'orchestration : enchaîne automatiquement les étapes manquantes puis exécute une recherche |

> **Remarque d'organisation** : dans le code actuel, `calcEmbeddings.py` et `searchEmbedding.py` importent `from utils.embed_client import encode`, ce qui suppose un sous-dossier `utils/` contenant `embed_client.py`. Si ce dossier n'existe pas dans votre arborescence, il faut soit créer `utils/embed_client.py`, soit adapter les imports vers `from embed_client import encode`.

---

## Le serveur d'embeddings

### Pourquoi un daemon ?

Charger un modèle SentenceTransformer prend plusieurs secondes, voire dizaines de secondes. Pour éviter ce coût à chaque script (`calcEmbeddings.py`, `searchEmbedding.py`, etc.), le modèle est chargé **une seule fois** dans un processus séparé et persistant (`embed_daemon.py`), qui reste actif en arrière-plan.

### Fonctionnement (`embed_daemon.py`)

- Écrit son PID dans `/tmp/embed_daemon.pid`.
- Charge le modèle `dangvantuan/sentence-camembert-base`.
- Ouvre un `Listener` (`multiprocessing.connection`) sur `localhost:6000`, protégé par une clé d'authentification (`authkey=b"secret"`).
- Signale sa disponibilité en créant `/tmp/embed_daemon.ready`.
- Boucle indéfiniment : à chaque connexion, reçoit `(job_id, sentences, chunk_size)`, encode les phrases par lots (`chunk_size`), écrit la progression dans `/tmp/embed_progress/{job_id}.json`, puis renvoie le résultat sous forme `("done", vecteurs_numpy)` ou `("error", message)`.
- Les logs sont écrits dans `/tmp/embed_daemon.log` et sur la sortie standard.

### Fonctionnement (`embed_client.py`)

- `encode(sentences, chunk_size=32, show_progress=True)` :
  1. Vérifie si le daemon tourne (`_is_daemon_running`), sinon le démarre (`_start_daemon`, avec un délai d'attente de 60 s max pour le chargement du modèle).
  2. Ouvre une connexion cliente vers `localhost:6000` avec la même clé d'authentification.
  3. Envoie la liste de phrases et récupère le résultat dans un thread séparé, tout en affichant une barre de progression textuelle basée sur les fichiers de `/tmp/embed_progress/`.
  4. Lève une `RuntimeError` si le daemon renvoie une erreur.

> **Sécurité** : l'authentification par `authkey=b"secret"` protège uniquement contre des connexions accidentelles sur le port 6000 ; ce n'est pas un mécanisme de sécurité robuste. Le daemon écoute uniquement sur `localhost`, donc le risque est limité aux autres processus locaux de la même machine.

---

## Calcul des embeddings

Fichier : `calcEmbeddings.py`

### `parse_sentences(file_path, mode="conllu")`
Extrait la liste des phrases brutes d'un fichier :
- **mode `conllu`** : utilise `parse_conllu_fast()`, un parseur CONLL-U rapide et manuel qui lit ligne par ligne, extrait les métadonnées `# sent_id = ...` et `# text_raw = ...`, et découpe les phrases sur les lignes vides. Retourne `(sentence_list, metadata)` où `metadata = {"sent_id": [...], "raw_text": [...]}`.
- **mode `xml`** : parcourt un arbre XML (via `lxml`) et extrait le texte de toutes les balises `<s>`.

### `calcEMbeddings(collection_file_path, output_file_path, mode="conllu", reduce_precision=False)`
1. Parse les phrases et métadonnées.
2. Encode les phrases via `encode(sentence_list, chunk_size=10000)` (appel au daemon).
3. Sauvegarde les embeddings au format `.npy` :
   - en `float16` si `reduce_precision=True` (gain d'espace disque, au prix d'une légère perte de précision),
   - en `float32` (par défaut) sinon.
4. Retourne `(embeddings, metadata)`.

### `save_metadata(metadata, output_file)`
Sauvegarde le dictionnaire `metadata` en JSON (encodage UTF-8).

---

## Construction de l'index

Fichier : `makeIndex.py`

### `makeIndex(embeddings=None, embedding_file_path=None, metric_type=None, index_type=None, output_file_path=None)`

1. Charge les embeddings (`np.load`) si non fournis directement.
2. Force le type `float32` contigu en mémoire (`np.ascontiguousarray`), requis par FAISS.
3. **Normalise les vecteurs (`faiss.normalize_L2`)** afin que la métrique `METRIC_INNER_PRODUCT` équivaille à une similarité cosinus (voir section [normalisation](#pourquoi-normaliser)).
4. Construit l'index selon `index_type` :
   - **`"flat"`** : `IndexFlat(dim, metric_type)` — recherche exhaustive exacte, sans structure d'accélération. Idéal pour les petits corpus ou pour valider la qualité des résultats.
   - **`"hnsw"`** : `IndexHNSWFlat(dim, hnsw_m=64, metric_type)` — graphe de plus proches voisins hiérarchique. `efConstruction=40` (qualité de construction du graphe), `efSearch=64` (qualité/vitesse de recherche). Bon compromis vitesse/précision pour des corpus moyens à grands, sans entraînement préalable nécessaire.
   - **`"ivfpq"`** : `IndexIVFPQ` — index avec quantification par produit (compression) et listes inversées (`nlist = 4·√n`, `m=8` sous-quantifieurs, `nbits=8`). Nécessite un **entraînement** (`index.train`) sur les données, et un minimum de points d'entraînement (`(2**nbits) * 40`, soit 10 240 phrases minimum pour `nbits=8`). Recommandé pour de très grands corpus où la mémoire est une contrainte.
5. Écrit l'index sur disque (`faiss.write_index`).

### Pourquoi normaliser ?

FAISS ne propose pas nativement de métrique "cosinus" : seulement `METRIC_L2` et `METRIC_INNER_PRODUCT`. Le produit scalaire de deux vecteurs normalisés (norme = 1) est mathématiquement équivalent à leur similarité cosinus. C'est pourquoi `makeIndex.py` normalise systématiquement les embeddings avant indexation, et `searchEmbedding.py` fait de même sur le vecteur de requête — **la cohérence entre les deux côtés est essentielle**, sans quoi les scores de similarité seraient faussés.

---

## Recherche

Fichier : `searchEmbedding.py`

### `load_index(index_file)`
Charge un index FAISS depuis le disque (`faiss.read_index`).

### `load_metadata(matadata_file_path)`
Charge le fichier JSON de métadonnées (`sent_id`, `raw_text`).

### `embedd_query(query_str)`
Encode la phrase de requête via le daemon, et la met en forme (`float32`, tableau 2D contigu de forme `(1, dim)`) attendue par FAISS.

### `search(query_str, index, metric_type, top_k=10, metadata)`
1. Encode et normalise le vecteur de requête.
2. Configure `index.nprobe = 8` si l'index le permet (cas des index IVF, pour contrôler le nombre de listes inversées explorées lors de la recherche — plus `nprobe` est élevé, plus la recherche est précise mais lente).
3. Effectue la recherche (`index.search`), qui retourne les distances et indices des `top_k` plus proches voisins.
4. Reconstitue les résultats sous forme de tuples `(sent_id, raw_text, score)`, en filtrant les indices invalides (`-1` retourné par FAISS quand moins de `top_k` résultats existent).

---

## Point d'entrée

Fichier : `main.py`

Ce script orchestre le pipeline complet de façon **incrémentale** : à partir d'un fichier d'entrée, il déduit les noms des fichiers dérivés (`.npy`, `.json`, `.faiss`) et ne recalcule que ce qui manque encore.

```bash
python main.py mon_corpus.conllu "ma phrase de recherche"
```

Logique simplifiée :
1. Si `mon_corpus.npy` n'existe pas → calcule les embeddings **et** les métadonnées.
2. Sinon, si `mon_corpus.json` n'existe pas → reconstruit uniquement les métadonnées.
3. Sinon, si `mon_corpus.faiss` n'existe pas → construit l'index à partir des embeddings existants.
4. Sinon (tout existe déjà) → charge index + métadonnées et lance directement la recherche.

> ⚠️ **Point d'attention important** : cette logique utilise une chaîne de `if / elif`, ce qui signifie qu'**un seul cas peut s'exécuter par appel**. Par exemple, si seuls les embeddings existent (`.npy` présent, `.json` et `.faiss` absents), le script s'arrête après avoir recréé les métadonnées, **sans** construire l'index ni lancer de recherche — il faudra relancer `main.py` une seconde fois pour que l'étape suivante (construction de l'index) s'exécute, puis une troisième fois pour obtenir une recherche complète une fois tout généré. Une fois les trois fichiers (`.npy`, `.json`, `.faiss`) présents, un seul appel suffit à chaque recherche suivante.

---

## Utilisation en ligne de commande

**Étape par étape (manuel) :**

```bash
# 1. Calcul des embeddings + métadonnées
python calcEmbeddings.py   # (adapter input_file / output_file_path dans le bloc __main__)

# 2. Construction de l'index
python makeIndex.py        # (adapter embedding_file_path / output_file_path)

# 3. Recherche
python searchEmbedding.py "ma phrase de recherche"
```

**Pipeline automatique (recommandé) :**

```bash
python main.py mon_corpus.conllu "ma phrase de recherche"
```
(à relancer si besoin comme précisé plus haut, tant que tous les fichiers dérivés n'existent pas encore)

---

## Choix du type d'index

| Type | Cas d'usage | Avantages | Inconvénients |
|---|---|---|---|
| `flat` | Petits corpus (< quelques dizaines de milliers de phrases), ou besoin de résultats exacts | Résultats exacts, aucun entraînement | Recherche en O(n), lente sur gros volumes |
| `hnsw` | Corpus moyens à grands, recherche rapide sans entraînement | Très bon compromis vitesse/précision, pas d'entraînement requis | Consommation mémoire plus élevée que IVFPQ, construction plus lente |
| `ivfpq` | Très grands corpus, contrainte mémoire forte | Compression forte, recherche rapide | Résultats approximatifs, nécessite un entraînement et un nombre minimum de vecteurs |

---

## Limitations connues et points d'attention

- **Dépendance `utils/embed_client`** : `calcEmbeddings.py` et `searchEmbedding.py` importent `from utils.embed_client import encode`, alors que `embed_client.py` est fourni à la racine du projet. Il faut soit créer un dossier `utils/` avec ce module, soit corriger les imports.
- **`searchEmbedding.py` (bloc `__main__`)** : appelle `search(...)` avec un argument `metadata_file_path="metadata.json"`, alors que la fonction attend un paramètre `metadata` contenant déjà les métadonnées chargées (dictionnaire), pas un chemin de fichier. Il faut appeler `load_metadata("metadata.json")` au préalable et passer le résultat via `metadata=...`.
- **IVFPQ et petits corpus** : si le corpus est trop petit (moins de `(2**nbits) * 40`, soit 10 240 phrases pour `nbits=8`), `makeIndex.py` lève volontairement une erreur explicite invitant à utiliser `"flat"` à la place.
- **Cohérence de la normalisation** : toute modification du pipeline doit impérativement conserver la normalisation L2 identique côté indexation (`makeIndex.py`) et côté requête (`searchEmbedding.py`), sous peine de résultats de similarité incohérents.
- **Persistance du daemon** : le daemon d'embedding reste actif en arrière-plan après utilisation (fichier `/tmp/embed_daemon.pid`). Pour l'arrêter manuellement :
  ```bash
  kill $(cat /tmp/embed_daemon.pid)
  ```
- **Textes anciens ou dialectaux** : le modèle `sentence-camembert-base` est entraîné principalement sur du français contemporain ; les résultats sur des textes littéraires très anciens ou fortement dialectaux peuvent être moins fiables et méritent une vérification manuelle.
