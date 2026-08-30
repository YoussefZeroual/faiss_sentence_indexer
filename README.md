# faiss_sentence_indexer

Outil de recherche sémantique pour corpus linguistiques et littéraires (formats **CoNLL-U**, **XML** et **TRS**), basé sur des embeddings de phrases (ou de tokens) indexés avec **FAISS**.

Le pipeline permet de :

1. **extraire** les phrases (ou tours de parole) d'un fichier de corpus (CoNLL-U, XML ou TRS) ;
2. **calculer leurs embeddings** via un modèle d'encodage (local, via un daemon persistant, ou via Ollama) ;
3. **construire un index FAISS** optimisé pour la recherche par similarité cosinus ;
4. **interroger** cet index avec une phrase de requête et récupérer les résultats les plus proches sémantiquement.

---

## Sommaire

- [Architecture générale](#architecture-générale)
- [Prérequis et installation](#prérequis-et-installation)
- [Structure du projet](#structure-du-projet)
- [Formats de corpus supportés](#formats-de-corpus-supportés)
- [Les trois stratégies d'encodage](#les-trois-stratégies-dencodage)
- [Le serveur d'embeddings (daemon)](#le-serveur-dembeddings-daemon)
- [Calcul des embeddings (`calcEmbeddings.py`)](#calcul-des-embeddings-calcembeddingspy)
- [Construction de l'index (`makeIndex.py`)](#construction-de-lindex-makeindexpy)
- [Tests empiriques : impact de `nprobe` et de `m`](#tests-empiriques--impact-de-nprobe-et-de-m)
- [Recherche (`searchEmbedding.py`)](#recherche-searchembeddingpy)
- [Point d'entrée et CLI (`main.py`)](#point-dentrée-et-cli-mainpy)
- [Système de cache incrémental](#système-de-cache-incrémental)
- [Choix de conception et garde-fous](#choix-de-conception-et-garde-fous)
- [Limitations connues et points d'attention](#limitations-connues-et-points-dattention)

---

## Architecture générale

Le projet est organisé en briques indépendantes qui communiquent via des fichiers intermédiaires (`.npy`, `.json`, `.faiss`) et via un processus daemon en arrière-plan pour l'encodage :

```
corpus.conllu / corpus.xml / corpus.trs
              │
              ▼
    calcEmbeddings.py ──► corpus.npy (embeddings, ou corpus_token.npy en mode token)
              │             corpus.json (métadonnées : sent_id, raw_text, tokens)
              │
              │   (appelle utils/embed_client.py, qui délègue
              │    l'encodage au daemon, à Ollama, ou au modèle local)
              ▼
      makeIndex.py ──► corpus.faiss (index FAISS normalisé, ou corpus_token.faiss)
              │
              ▼
   searchEmbedding.py ──► résultats de recherche (top-k phrases similaires)
```

`main.py` orchestre l'ensemble de la chaîne de façon **incrémentale** : il ne recalcule que ce qui manque encore sur le disque (embeddings, métadonnées ou index), et propose de nombreux modes d'exécution (fichier unique, dossier complet, encodage seul, recherche seule, mode sans FAISS pour du débogage, régénération des métadonnées, etc.).

Le calcul des embeddings peut être délégué à un **daemon persistant** (`utils/embed_daemon.py`), afin d'éviter de recharger un modèle lourd (PyTorch / Transformers) à chaque appel de script. Le client (`utils/embed_client.py`) démarre ce daemon si besoin et communique avec lui par socket local ; il peut aussi, selon les besoins, s'en passer complètement (mode local synchrone) ou déléguer l'encodage à un serveur **Ollama** externe.

---

## Prérequis et installation

- Python 3.9+ (le code utilise des annotations de type modernes, ex. `tuple[Tensor, ...]`)
- Bibliothèques principales :
  ```bash
  pip install faiss-cpu numpy torch transformers sentence-transformers conllu lxml requests
  ```
  (utiliser `faiss-gpu` à la place de `faiss-cpu`, et une version de `torch` compilée pour CUDA, si un GPU est disponible)
- Les modèles Hugging Face sont téléchargés automatiquement au premier appel qui en a besoin (voir [modèles utilisés](#les-modèles-dencodage)).
- Pour la stratégie Ollama : une instance [Ollama](https://ollama.com) locale ou distante, avec un modèle d'embedding installé (ex. `nomic-embed-text-v2-moe`).

### ⚠️ Organisation obligatoire des fichiers

L'organisation en sous-dossier `utils/` est **requise pour que le code fonctionne**, car `embed_client.py` importe lui-même `average_pool` et `average_pool_last_n_layers` depuis `utils.embed_daemon` :

```python
from utils.embed_daemon import average_pool, average_pool_last_n_layers
```

L'arborescence attendue est donc :

```
projet/
├── main.py
├── calcEmbeddings.py
├── makeIndex.py
├── searchEmbedding.py
└── utils/
    ├── __init__.py
    ├── embed_client.py
    └── embed_daemon.py
```

---

## Structure du projet

| Fichier | Rôle |
|---|---|
| `main.py` | Point d'entrée CLI : enchaîne automatiquement les étapes manquantes du pipeline, gère de nombreux modes d'exécution spéciaux, et lance la recherche finale. |
| `calcEmbeddings.py` | Parse un fichier de corpus (CoNLL-U, XML ou TRS), nettoie le texte, et calcule les embeddings de chaque phrase ou tour de parole. |
| `makeIndex.py` | Construit un index FAISS (`flat`, `hnsw` ou `ivfpq`) à partir des embeddings. |
| `searchEmbedding.py` | Charge un index et ses métadonnées, encode une requête, et effectue la recherche de similarité (fichier unique ou dossier entier). |
| `utils/embed_client.py` | Interface cliente vers l'encodage : trois stratégies possibles (daemon IPC, exécution locale directe, ou API Ollama). |
| `utils/embed_daemon.py` | Processus serveur qui charge les modèles d'encodage une seule fois, regroupe les requêtes en lots dynamiques, et les traite au fil de l'eau. |

---

## Formats de corpus supportés

Le format cible est déterminé automatiquement à partir de l'**extension du fichier** (le paramètre `mode` passé en argument est écrasé par cette détection).

| Extension | Format | Fonction de parsing | Description |
|---|---|---|---|
| `.conllu` | CoNLL-U | `parse_conllu_fast()` | Corpus annotés en dépendances syntaxiques (texte littéraire, corpus écrit annoté). |
| `.xml` | XML / XML-CoNLLU | `parse_sentences_xml_conllu()` | Corpus au format XML (ex. Lexicoscope), avec gestion des cas où du CoNLL-U est imbriqué dans les balises `<s>`. |
| `.trs` | Transcriber | `parse_sentence_trs()` | Transcriptions de corpus oraux ; chaque `<Turn>` (tour de parole) est traité comme une unité de recherche, identifiée par son `startTime`. |

### Parsing CoNLL-U (`parse_conllu_fast`)

Deux stratégies successives sont utilisées :

1. **Extraction rapide par en-têtes** : le parseur lit ligne par ligne et récupère directement les métadonnées `# sent_id = ...` et `# text_raw = ...` lorsqu'elles sont présentes, ce qui est le chemin le plus rapide.
2. **Repli automatique (fallback) par concaténation des formes** : si la liste de phrases obtenue est vide, ou si au moins 3 phrases n'ont pas de `text_raw`, le script bascule automatiquement sur une reconstruction du texte à partir des formes de surface (`concat_forms`), y compris la gestion des **mots amalgames** français (ex. `du` → `de` + `le`) qui ne doivent pas être comptés deux fois.

Dans les deux cas, la ponctuation est renormalisée (`fix_punctuation_spaces`) selon les conventions typographiques françaises (espace avant `:` `;` `?` `!`, pas d'espace avant `.` `,`, gestion des guillemets français « » et des apostrophes).

### Parsing XML (`parse_sentences_xml_conllu`)

Le parseur (basé sur `lxml`, en mode tolérant aux erreurs `recover=True`) cible toutes les balises `<s>` et gère trois cas de figure :

- texte direct simple à l'intérieur de `<s>` ;
- texte multi-lignes ressemblant à du CoNLL-U imbriqué (reconstruction via `concat_forms`) ;
- absence de texte direct : le contenu est reconstitué à partir des balises enfants (`s.itertext()`).

### Parsing TRS (`parse_sentence_trs`)

Chaque `<Turn>` du fichier de transcription est extrait avec son texte concaténé et son `startTime` comme identifiant. La ponctuation est nettoyée de la même façon que pour le CoNLL-U.

### Gestion des phrases vides

Toute phrase reconstruite comme vide ou nulle est remplacée par l'étiquette `[phrase manquante]` (constante `MISSING_SENTENCE`), afin de préserver l'alignement entre les phrases, leurs métadonnées et leurs vecteurs d'embeddings — un décalage d'index à ce niveau invaliderait silencieusement toute la recherche en aval.

---

## Les trois stratégies d'encodage

`utils/embed_client.py` expose une unique fonction routeur, `encode()`, qui choisit entre trois stratégies d'exécution selon les arguments fournis :

| Stratégie | Argument | Comportement |
|---|---|---|
| **Daemon IPC** (par défaut) | *(aucun flag)* | Envoie les phrases à `embed_daemon.py` via une socket locale ; démarre le daemon automatiquement s'il n'est pas déjà lancé. |
| **Sans daemon** | `no_daemon=True` | Charge les modèles directement dans le processus courant (utile pour le débogage, l'exécution ponctuelle, ou les environnements où un processus persistant n'est pas souhaitable). |
| **Ollama** | `use_ollama=True` | Délègue l'encodage à un serveur Ollama externe via une requête HTTP POST sur `/api/embed`. |

Ces trois stratégies sont mutuellement exclusives dans l'ordre de priorité suivant : Ollama est tenté en premier s'il est demandé, puis, en cas d'échec de connexion, le script bascule automatiquement sur le mode daemon/local. Le mode `no_daemon` est prioritaire sur le mode daemon s'il est explicitement demandé.

> ⚠️ **Ollama est explicitement documenté comme significativement plus lent** que le daemon ou un modèle chargé directement — problème connu côté Ollama, non résolu à ce jour. Cette stratégie est donc surtout pertinente pour des besoins ponctuels ou d'intégration avec une infrastructure existante, pas pour l'indexation de gros corpus.

### Les modèles d'encodage

Deux modèles distincts sont utilisés selon le mode choisi :

| Mode | Modèle | Usage |
|---|---|---|
| **Phrase** (par défaut) | `BAAI/bge-m3` (via `SentenceTransformer`) | Encode chaque phrase entière en un seul vecteur — c'est le mode utilisé pour l'indexation FAISS et la recherche sémantique classique. |
| **Token** (`token_mode=True`) | `intfloat/multilingual-e5-base` (via `transformers.AutoModel`) | Encode au niveau des tokens, avec un pooling (moyenne des 4 dernières couches cachées, `average_pool_last_n_layers`, ou pooling simple sur la dernière couche selon le contexte). Ce mode est expérimental et documenté dans le code comme susceptible d'être révisé, car il est actuellement utilisé pour produire un vecteur agrégé par phrase plutôt que des vecteurs par token individuel. |

> Le modèle par défaut pour l'encodage au niveau phrase est `BAAI/bge-m3`, un modèle d'embedding multilingue.

Une fonctionnalité de **réduction de dimensionnalité "All-but-the-top"** (`all_but_the_top`, dans `embed_daemon.py`) est également disponible : elle retire les composantes principales dominantes des embeddings (via `torch.pca_lowrank`), une technique connue pour améliorer la qualité de la similarité cosinus sur certains modèles. Elle n'est pas branchée par défaut dans le pipeline (le code d'appel est présent mais commenté dans `main.py`), et reste donc une fonctionnalité exploratoire à activer manuellement si besoin.

---

## Le serveur d'embeddings (daemon)

### Pourquoi un daemon ?

Charger un modèle d'encodage (PyTorch + poids Hugging Face) prend de quelques secondes à quelques dizaines de secondes. Pour éviter ce coût à chaque script, le modèle est chargé **une seule fois** dans un processus séparé et persistant (`utils/embed_daemon.py`), qui reste actif en arrière-plan et peut servir plusieurs scripts et plusieurs jobs successifs.

### Fonctionnement (`embed_daemon.py`)

- Ouvre un `Listener` (`multiprocessing.connection`) sur `localhost:6000`.
- Effectue un **chargement fainéant (lazy loading)** des modèles : rien n'est chargé en mémoire tant qu'aucune requête ne l'exige, et seul le modèle réellement demandé (phrase ou token) est chargé.
- Traite les requêtes via un **thread de traitement par lots dynamique** (`batching_worker`) :
  - une file d'attente (`batch_queue`) reçoit les sous-lots de phrases envoyés par tous les clients connectés ;
  - le worker regroupe plusieurs requêtes arrivées dans une **fenêtre de temps de 15 ms** (`BATCH_WINDOW_S`), jusqu'à une taille maximale de lot de 256 phrases (`MAX_BATCH_SIZE`), avant de lancer l'encodage ;
  - ce mécanisme permet de mutualiser le coût GPU entre plusieurs requêtes concurrentes sans bloquer un seul gros job.
- Chaque connexion cliente est gérée dans son propre thread (`handle_client`), dédié exclusivement aux entrées/sorties réseau : le calcul lourd est toujours délégué au thread unique du worker, ce qui évite toute concurrence d'accès au GPU.
- Envoie la progression (`("progress", fait, total)`) au fur et à mesure du traitement des sous-lots (`chunk_size` défini côté client), puis le résultat final (`("done", vecteurs_numpy)`) ou une erreur (`("error", message)`).
- Libère explicitement le cache CUDA (`torch.cuda.empty_cache()`) après chaque lot lorsqu'un GPU est utilisé, en réaction à des fuites de VRAM observées lors de l'encodage de plusieurs corpus consécutifs.
- Un mécanisme de **déchargement automatique des modèles après inactivité** (`handle_timeout`, `KEEP_MODEL_LOADED_TIMEOUT`) existe dans le code mais est **encore en développement et n'est pas branché** dans la boucle principale : les modèles chargés restent donc en mémoire jusqu'à l'arrêt du processus.

### Fonctionnement (`embed_client.py` → fonction `encode()`)

1. Vérifie si le daemon tourne (`_try_connect`), sinon le démarre en tâche de fond détachée (`_start_daemon`), avec une boucle d'attente active de 120 tentatives × 0,5 s (≈ 60 s max) pour laisser le temps au modèle de se charger.
2. Ouvre une connexion cliente vers `localhost:6000`.
3. Envoie `(job_id, sentences, chunk_size, token_mode)` et récupère les résultats progressivement, tout en affichant une barre de progression textuelle dans la console si `show_progress=True`.
4. Lève une `RuntimeError` si le daemon renvoie une erreur (par exemple un `OutOfMemoryError` GPU sur le lot en cours).

> **Sécurité** : le daemon écoute uniquement sur `localhost` et n'impose pas d'authentification par clé. Cela signifie que **toute application locale sur la même machine peut s'y connecter**. Ce n'est pas un problème sur une machine mono-utilisateur dédiée, mais cela mérite d'être noté si le daemon est exécuté sur un serveur partagé.

---

## Calcul des embeddings (`calcEmbeddings.py`)

### `parse_sentences(file_path, mode=...)`

Point d'entrée unique de parsing, qui redirige automatiquement (selon l'extension du fichier) vers `parse_conllu_fast`, `parse_sentences_xml_conllu` ou `parse_sentence_trs`, et journalise le nombre de phrases et de tokens extraits ainsi que le temps d'exécution. Retourne `(sent_list, metadata)`, où `metadata` contient `sent_id`, `raw_text`, et `tokens` (liste de tokens individuels par phrase, utile pour le mode d'encodage par token et pour d'éventuelles analyses linguistiques ultérieures).

### `calcEmbeddings(collection_file_path, output_file_path, mode, reduce_precision=False, overwrite=False, token_mode=False, no_daemon=False, use_ollama=False, ollama_host=..., ollama_model=None)`

1. **Vérification du cache** : si `overwrite=False` et que les fichiers `.npy` et `.json` existent déjà, ils sont simplement chargés depuis le disque (pas de recalcul).
2. Sinon, parse les phrases et métadonnées via `parse_sentences`.
3. Encode les phrases (`chunk_size=64` par défaut) via `encode()`, en respectant le mode et la stratégie demandés.
4. Sauvegarde les embeddings au format `.npy` :
   - en `float16` si `reduce_precision=True` (gain d'espace disque, au prix d'une légère perte de précision) ;
   - en `float32` par défaut sinon.
5. En mode token, le suffixe `_token` est automatiquement ajouté au nom du fichier de sortie pour éviter d'écraser les embeddings "phrase".
6. Retourne `(embeddings, metadata)`.

### `save_metadata(metadata, output_file)`

Sauvegarde le dictionnaire `metadata` en JSON, encodage UTF-8 (indispensable pour préserver correctement les accents et caractères spéciaux français).

### `encode_folder(input_folder, ...)`

Traite un dossier entier (ou un motif avec wildcard, ex. `corpus/*Camus*`) : détecte tous les fichiers `.conllu`, `.xml` et `.trs`, puis appelle `calcEmbeddings` et `save_metadata` pour chacun.

---

## Construction de l'index (`makeIndex.py`)

### `load_embeddings(embedding_file_path)`

Charge les vecteurs `.npy`, les convertit en tableau contigu `float32` (requis par FAISS), puis les **normalise en norme L2** (`faiss.normalize_L2`).

### `makeIndex(embeddings=None, embedding_file_path=None, metric_type=None, index_type=None, m=256, output_file_path=None, overwrite=False, token_mode=False)`

1. Si `overwrite=False` et qu'un index `.faiss` existe déjà pour ce fichier, il est chargé directement depuis le disque plutôt que reconstruit.
2. Sélectionne la stratégie d'indexation selon `index_type` :

| Type | Description | Garde-fou |
|---|---|---|
| `flat` | `IndexFlat` — recherche exhaustive, exacte, sans structure d'accélération. Idéal pour petits corpus ou pour valider la qualité des autres index. | Aucun — toujours disponible, quel que soit le nombre de vecteurs. |
| `hnsw` | `IndexHNSWFlat` (`hnsw_m=64`, `efConstruction=40`, `efSearch=64`) — graphe de plus proches voisins. Bon compromis vitesse/précision, sans entraînement nécessaire. | Aucun garde-fou spécifique — mais consomme davantage de mémoire que `ivfpq`. |
| `ivfpq` | `IndexIVFPQ` — listes inversées (`nlist = 4·√n`) + quantification produit (`m`, `nbits=8`). Nécessite un entraînement (`index.train`). Optimisé pour les très grands corpus contraints en mémoire. | **Garde-fou automatique** : si le nombre de vecteurs `n` est inférieur au minimum requis pour un entraînement statistiquement fiable (`nlist·39` et `(2**nbits)·39`, soit ~10 000+ points selon la taille du corpus), la fonction bascule **automatiquement** sur un index `flat` et journalise un avertissement, plutôt que d'entraîner un index IVFPQ de mauvaise qualité. |

3. Normalise les vecteurs fournis directement en argument (si `embeddings` est passé plutôt que chargé depuis un fichier), pour garantir la même cohérence de normalisation.
4. Écrit l'index sur disque (`faiss.write_index`) ; en mode token, le suffixe `_token` est ajouté au nom du fichier.

#### Le paramètre `m` (nombre de sous-quantifieurs PQ)

`m` détermine en combien de sous-vecteurs chaque embedding est découpé pour la quantification produit (`IndexIVFPQ`) : chaque sous-vecteur est ensuite compressé en un code de `nbits=8` (soit 256 centroïdes possibles). C'est le paramètre qui a le plus d'impact direct sur la qualité des résultats en mode `ivfpq`, et il est désormais exposé comme argument de `makeIndex()` (valeur par défaut : `m=256`, contre `m=8` figé auparavant).

- `m` doit être un **diviseur de la dimension des embeddings** (1024 pour `BAAI/bge-m3`) : valeurs valides `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}`.
- Plus `m` est élevé, plus chaque sous-vecteur couvre peu de dimensions d'origine, donc moins d'information est perdue à la compression — mais plus l'index occupe d'espace sur le disque.
- Voir la section [Tests empiriques : impact de `nprobe` et de `m`](#tests-empiriques--impact-de-nprobe-et-de-m) pour des mesures concrètes de ce compromis sur un corpus réel.

> ⚠️ **`makeIndex_folder()` n'expose pas encore `m`** : la fonction de traitement par dossier appelle `makeIndex()` sans passer ce paramètre, qui retombe donc systématiquement sur sa valeur par défaut (`256`) pour tout traitement par lot. Si une valeur différente est souhaitée en mode dossier, il faut actuellement soit modifier `makeIndex_folder()` pour relayer un paramètre `m`, soit appeler `makeIndex()` fichier par fichier.

### Pourquoi normaliser ?

FAISS ne propose pas nativement de métrique "cosinus" : seulement `METRIC_L2` et `METRIC_INNER_PRODUCT`. Le produit scalaire de deux vecteurs normalisés (norme = 1) est mathématiquement équivalent à leur similarité cosinus. C'est pourquoi `makeIndex.py` normalise systématiquement les embeddings avant indexation, et `searchEmbedding.py` fait de même sur le vecteur de requête — **la cohérence entre les deux côtés est essentielle**, sans quoi les scores de similarité seraient faussés.

### `makeIndex_folder(input_folder, ...)`

Traite un dossier entier (ou un motif wildcard) : détecte les fichiers d'embeddings (`.npy` ou `_token.npy` selon `token_mode`) et construit un index pour chacun via `makeIndex`.

---

## Tests empiriques : impact de `nprobe` et de `m`

Cette section rassemble les résultats de tests comparatifs menés sur un corpus réel (`HS36-6030v2tv8`, **114 064 phrases**, embeddings `BAAI/bge-m3` en 1024 dimensions, index `ivfpq` avec `nlist=1348`), en comparant systématiquement les résultats d'une recherche `ivfpq` à une **référence exacte** (recherche `flat` / calcul direct du produit scalaire sur les `.npy` via `--no-faiss`), sur la même requête (`"pas de souci"`, top-30).

### `nprobe` (paramètre de recherche, sans impact mesuré dans ce test)

`nprobe` contrôle le nombre de listes inversées (clusters) explorées lors d'une recherche `ivfpq` — c'est un paramètre appliqué **au moment de la requête**, sur un index déjà construit, sans nécessiter de reconstruction.

| `nprobe` testé | Résultat |
|---|---|
| `8` (valeur d'origine codée en dur dans `search()`) | Référence de départ |
| `64` | **Résultats strictement identiques** à `nprobe=8` (mêmes phrases, même ordre, mêmes scores à 3 décimales) |
| `1348` (= `nlist`, recherche exhaustive sur toutes les listes inversées) | **Résultats toujours strictement identiques** |

**Conclusion** : sur ce corpus et cette requête, faire varier `nprobe` de 8 jusqu'à une couverture exhaustive de tous les clusters n'a produit **aucun changement mesurable**. La couverture des clusters n'était donc pas le facteur limitant de la qualité des résultats — l'écart observé avec la recherche exacte provient d'ailleurs.

### `m` (paramètre de construction, impact majeur mesuré)

`m` (nombre de sous-quantifieurs PQ) doit en revanche être fixé **à la construction de l'index** — le faire varier impose de reconstruire l'index (`--force`). Le test a comparé le recouvrement du top-30 `ivfpq` avec le top-30 exact, ainsi que la taille du fichier `.faiss` produit, pour plusieurs valeurs de `m` :

| `m` | Recouvrement avec le top-30 exact | Écart de score sur les meilleurs résultats | Taille approximative de l'index |
|---|---|---|---|
| `8` (valeur d'origine) | 17 / 30 | jusqu'à ~0.05 (ex. 0.788 vs 0.835) | ~8 MB |
| `64` | 21 / 30 | ~0.01–0.02 | ~14 MB |
| `256` | 25 / 30 | ~0.005–0.01 | intermédiaire (non mesurée précisément) |
| `512` | **29 / 30** | ~0.001–0.002 (quasi exact) | ~60–70 MB (estimation à partir du calcul `n × m` octets) |

À titre de comparaison, un index `flat` du même corpus (aucune compression, vecteurs `float32` complets) pèse environ **467 MB**.

**Conclusion** : l'écart entre `ivfpq` et une recherche exacte observé dans ce projet provient **entièrement de l'erreur de reconstruction de la quantification produit (PQ)**, et non d'une couverture insuffisante des clusters (`nprobe`). Augmenter `m` réduit directement et significativement cette erreur, avec des rendements encore nettement positifs jusqu'à `m=256`, et des rendements plus marginaux (mais un score de similarité presque parfaitement exact) à `m=512`. C'est ce constat empirique qui justifie le changement de valeur par défaut de `m` (désormais `256` dans le code, contre `8` auparavant), et sa documentation comme paramètre explicite plutôt que constante figée.

### Recommandations pratiques issues de ces tests

- **Ne pas chercher à ajuster `nprobe`** pour améliorer la qualité des résultats `ivfpq` sur ce type de corpus — ce paramètre agit sur la vitesse de recherche en présence de nombreux clusters non pertinents, pas sur l'erreur de reconstruction vectorielle.
- **Pour un corpus dont la taille reste raisonnable en mémoire** (de l'ordre de quelques centaines de milliers de phrases, comme dans ce test), `flat` reste l'option la plus simple si l'exactitude prime : le surcoût de chargement mesuré (~750 ms pour ce corpus) est négligeable pour un usage interactif ou ponctuel.
- **Si la compression disque/RAM est nécessaire** (très grands corpus, contrainte matérielle), `m=256` à `m=512` offre un compromis solide : recouvrement de 25 à 29 sur 30 avec la recherche exacte, pour une taille d'index restant très inférieure à celle d'un index `flat` équivalent.
- **`hnsw` n'a pas encore été mesuré dans cette série de tests** et reste une alternative à évaluer : en théorie, il offre un meilleur recouvrement que `ivfpq` à taille de corpus comparable, sans nécessiter de réglage de `m`/`nbits`, au prix d'une consommation mémoire plus élevée qu'`ivfpq`.

---

## Recherche (`searchEmbedding.py`)

### `load_index(index_file)` / `load_metadata(metadata_file_path)`

Chargent respectivement un index FAISS (`faiss.read_index`) et un dictionnaire de métadonnées JSON (`sent_id`, `raw_text`, `tokens`).

### `embedd_query(query_str, token_mode=False, no_daemon=False, use_ollama=False, ollama_host=..., ollama_model=None)`

Encode la phrase de requête via `encode()` (avec les mêmes options de stratégie que pour le corpus), puis met en forme le résultat en tableau `float32` 2D contigu de forme `(1, dim)`, tel qu'attendu par FAISS.

### `search(query_vector=None, query_str=None, index=None, metric_type=None, top_k=10, metadata=None, token_mode=False, no_daemon=False)`

1. Encode la requête si un vecteur pré-calculé n'est pas déjà fourni (`query_vector`), ce qui permet, en mode dossier, de **n'encoder la requête qu'une seule fois** puis de réutiliser le même vecteur sur tous les index du dossier — optimisation notable pour les recherches multi-fichiers.
2. Normalise le vecteur de requête (`faiss.normalize_L2`), pour rester cohérent avec la normalisation appliquée côté indexation.
3. Configure `index.nprobe = 8` si l'index le permet (cas des index IVF), afin de contrôler le compromis vitesse/précision de la recherche partitionnée.
4. Effectue la recherche (`index.search`) et reconstitue les résultats sous forme de tuples `(sent_id, raw_text, score)`, en filtrant les indices invalides (`-1`, retourné par FAISS lorsqu'il existe moins de `top_k` résultats que demandé).
5. **Garde-fou de dimension** : si un `AssertionError` est levé par FAISS (typiquement parce que la requête a été encodée avec un modèle différent de celui utilisé pour indexer le corpus cible — par exemple mode token vs mode phrase), la fonction journalise un avertissement explicite et retourne une liste vide plutôt que de faire planter tout le script.

### `search_folder(input_folder, query_str=None, query_vector=None, ..., verbose=True)`

Exécute la recherche sur tous les index `.faiss` d'un dossier (ou motif wildcard) :

- encode la requête une seule fois si nécessaire ;
- ignore et journalise en avertissement les fichiers index sans métadonnées correspondantes (ou l'inverse), plutôt que d'interrompre toute la recherche ;
- agrège les résultats de tous les fichiers, les trie globalement par score décroissant, et n'en conserve que les `top_k` meilleurs ;
- affiche un tableau récapitulatif si `verbose=True`.

---

## Point d'entrée et CLI (`main.py`)

### Utilisation de base

```bash
python main.py mon_corpus.conllu "ma phrase de recherche"
```

### Arguments principaux

| Argument | Description |
|---|---|
| `input_file` | Fichier de corpus (`.conllu`, `.xml`, `.trs`), fichier `.faiss` existant, dossier, ou motif wildcard. |
| `query` | La phrase de requête. |
| `--index-type {flat,hnsw,ivfpq}` | Type d'index FAISS à construire (par défaut : `ivfpq` — voir note ci-dessous). |
| `--top-k N` | Nombre de résultats à retourner (défaut : 10). |
| `--reduce-precision` | Sauvegarde les embeddings en `float16` pour économiser de l'espace disque. |
| `--force` | Force le recalcul complet (embeddings, métadonnées, index), même si les fichiers en cache existent déjà. |
| `--folder` | Traite un dossier entier plutôt qu'un fichier unique. |
| `--encode-only` | Encode et indexe les fichiers d'un dossier sans lancer de recherche (utile pour une phase de pré-calcul). |
| `--search-only` | Recherche directement sur des index supposés déjà construits (optimisation de vitesse, saute les vérifications de cache). |
| `--regenerate-metadata` | Régénère uniquement les métadonnées JSON, sans toucher aux embeddings ni à l'index (utile après une mise à jour du parseur). |
| `--token-emb` | Active le mode d'encodage par token (`intfloat/multilingual-e5-base`) au lieu du mode phrase. |
| `--no-faiss` | Effectue une recherche par produit scalaire direct sur les `.npy`, sans passer par un index FAISS — sert à valider la fiabilité d'un index FAISS en comparant ses résultats à un calcul exact. |
| `--no-daemon` | Charge les modèles localement dans le processus courant plutôt que de passer par le daemon. |
| `--use-ollama` | Utilise un serveur Ollama pour l'encodage. |
| `--log` / `--warn` | Ajustent le niveau de verbosité des logs de tous les modules du pipeline. |

> ⚠️ Il existe une incohérence mineure dans le code actuel : la valeur par défaut de `--index-type` est `ivfpq`, alors que le texte d'aide affiché (`help=`) indique encore `"default: flat"`. Vérifiez systématiquement la valeur réellement appliquée avec `--log` si vous ne spécifiez pas explicitement `--index-type`.

### Les 7 modes d'exécution de `main.py`

`main()` route l'exécution vers l'un des modes suivants, dans cet ordre de priorité :

1. **`--no-faiss`** : recherche directe par produit scalaire sur les fichiers `.npy`, sans FAISS (mode de validation/débogage).
2. **`--regenerate-metadata`** : reparse le(s) fichier(s) source(s) et réécrit uniquement le(s) fichier(s) `.json`, sans toucher aux embeddings ni à l'index.
3. **Fichier `.faiss` fourni directement** : charge l'index et les métadonnées correspondantes, puis lance directement la recherche.
4. **`--encode-only`** (sur dossier ou wildcard) : encode et indexe tous les fichiers manquants du dossier, étape par étape (métadonnées → embeddings → index), sans lancer de recherche.
5. **`--search-only`** (sur dossier) : suppose que tous les index existent déjà et lance directement la recherche, pour un gain de vitesse en évitant les vérifications de cache.
6. **Mode dossier ou wildcard standard** (`--folder` ou motif `*` dans `input_file`) : encode si nécessaire (en mode `--force`), puis effectue une recherche agrégée sur tous les index du dossier.
7. **Mode fichier unique standard** : applique la logique de cache incrémental (voir section suivante), puis lance la recherche sur ce fichier.

### Contexte autour d'une phrase (`get_sent_context`)

`main.py` fournit également une fonction utilitaire `get_sent_context(file, sent_id, context_size=10)`, qui permet de récupérer le texte environnant une phrase trouvée par la recherche (les `context_size` phrases précédentes et suivantes), utile pour examiner un résultat dans son contexte discursif d'origine plutôt qu'isolément.

---

## Système de cache incrémental

Pour un fichier corpus unique, `main.py` déduit les chemins des fichiers dérivés attendus (`.npy`, `.json`, `.faiss`) et applique la logique suivante :

1. Si les **embeddings et métadonnées existent déjà**, et pas `--force` → ils sont chargés directement, on ne reconstruit que l'index s'il manque.
2. Si le fichier fourni a une extension **non reconnue** et qu'aucun fichier dérivé n'existe → le script s'arrête avec une erreur explicite.
3. Si **embeddings et index sont absents**, ou `--force` est utilisé → tout est reconstruit depuis le fichier source (parsing → embeddings → métadonnées → index).
4. Sinon → les embeddings bruts (`.npy`) sont rechargés depuis le disque pour ne reconstruire que l'index manquant.

**Garde-fou d'intégrité** : une fois l'index chargé (ou reconstruit), `main.py` vérifie que le nombre de vecteurs de l'index (`index.ntotal`) correspond bien au nombre d'entrées des métadonnées (`len(metadata["raw_text"])`). En cas de désaccord — signe probable d'une incohérence entre un ancien index et de nouvelles métadonnées régénérées séparément — le script s'arrête avec un message d'erreur explicite invitant à relancer avec `--force`, plutôt que de retourner silencieusement des résultats erronés ou décalés.

**Repli automatique IVFPQ → flat** : si `makeIndex` lève une `ValueError` parce que le corpus est trop petit pour un entraînement IVFPQ fiable, `main.py` intercepte l'erreur et reconstruit automatiquement un index `flat` à la place, en avertissant l'utilisateur.

> Un seul appel à `main.py` sur un fichier unique suffit à traverser toutes les étapes manquantes du pipeline en une seule exécution.

---

## Choix de conception et garde-fous

Cette section récapitule les décisions structurantes du projet et les raisons qui les motivent :

- **Daemon persistant plutôt que rechargement systématique du modèle** : le coût de chargement d'un modèle Transformer (plusieurs secondes à dizaines de secondes) est trop élevé pour être payé à chaque script ; le daemon amortit ce coût sur toute une session de travail.
- **Traitement par lots dynamique côté daemon** (fenêtre de 15 ms, taille max 256) : permet de mutualiser efficacement le GPU entre plusieurs requêtes concurrentes sans imposer à un client isolé d'attendre qu'un très gros lot d'un autre client soit terminé.
- **Chargement fainéant des modèles** (phrase vs token) : évite de saturer la VRAM en gardant les deux modèles en mémoire simultanément si un seul est réellement utilisé.
- **Libération explicite du cache CUDA après chaque lot** : réponse directe à des fuites de mémoire GPU constatées en production lors de l'encodage de plusieurs corpus consécutifs.
- **Normalisation L2 systématique et symétrique** (côté indexation *et* côté requête) : condition nécessaire pour que la métrique `METRIC_INNER_PRODUCT` de FAISS soit mathématiquement équivalente à une similarité cosinus.
- **Valeur par défaut de `m=256` pour `ivfpq`** (paramètre désormais exposé dans `makeIndex()`, contre `m=8` figé auparavant) : choix directement issu de tests comparatifs contre une recherche exacte, qui ont montré que `nprobe` n'avait aucun impact mesurable sur la qualité des résultats, alors que `m` en avait un très net — voir [Tests empiriques](#tests-empiriques--impact-de-nprobe-et-de-m).
- **Bascule automatique IVFPQ → flat** lorsque le corpus est trop petit pour un entraînement statistiquement fiable : évite de construire silencieusement un index de mauvaise qualité, au prix d'un avertissement explicite dans les logs.
- **Vérification d'intégrité index/métadonnées** (`index.ntotal == len(metadata)`) avant toute recherche : détecte les désynchronisations entre fichiers dérivés générés à des moments différents.
- **Cache incrémental à trois niveaux** (embeddings → métadonnées → index) : évite de recalculer des embeddings coûteux en GPU/CPU si seule l'étape d'indexation ou de reconstruction des métadonnées doit être rejouée.
- **Suffixe `_token` systématique** sur les fichiers dérivés du mode token : évite d'écraser accidentellement les embeddings/index "phrase" d'un même corpus.
- **Mode `--no-faiss` de validation** : permet de comparer, sur un même corpus, les résultats d'une recherche FAISS (potentiellement approximative selon le type d'index) à un calcul de similarité cosinus exact, pour détecter une éventuelle perte de qualité liée à l'indexation.
- **Repli progressif du parseur CoNLL-U** (en-têtes rapides → reconstruction par concaténation des formes) : maximise la robustesse face à des fichiers de corpus hétérogènes ou partiellement annotés.
- **Étiquetage explicite des phrases manquantes** (`[phrase manquante]`) plutôt que leur suppression : préserve l'alignement entre indices FAISS, métadonnées et phrases d'origine.

---

## Limitations connues et points d'attention

- **`searchEmbedding.py` (bloc `__main__`)** : le bloc de test en bas du fichier appelle encore `search(..., metadata_file_path="metadata.json")`, alors que la fonction `search()` attend un paramètre nommé `metadata` contenant un dictionnaire déjà chargé, et non un chemin de fichier. Ce bloc de test plantera donc (`TypeError`) tel quel ; utilisez plutôt `main.py` ou `search_folder`, ou corrigez l'appel en `load_metadata("metadata.json")` suivi de `metadata=...`.
- **Incohérence de l'aide CLI** : `--index-type` a pour valeur par défaut `"ivfpq"` dans le code, mais le texte d'aide affiché mentionne encore `"default: flat"`.
- **Mode `--search-only`** : la fonction `process_folder` ré-encode systématiquement la requête en interne ; le vecteur encodé séparément en amont dans ce mode n'est pas réutilisé, ce qui annule partiellement le gain de vitesse recherché par cette option.
- **IVFPQ sur petits corpus** : si le corpus ne dépasse pas le seuil minimal de points d'entraînement, `makeIndex.py` bascule automatiquement sur `flat` — ceci est un comportement voulu, mais peut surprendre si l'on s'attend explicitement à un index compressé.
- **Mode token expérimental** : le mode d'encodage par token, et son pooling associé (moyenne des dernières couches), sont documentés dans le code même comme susceptibles d'être révisés ou supprimés dans une version future, car le mode ne produit actuellement qu'un vecteur agrégé par phrase plutôt que des vecteurs par token individuels.
- **`all_but_the_top`** (réduction de dimensionnalité) : présent dans le code mais non branché par défaut dans le pipeline (appels commentés dans `main.py`) ; à activer manuellement si l'on souhaite l'expérimenter.
- **Déchargement automatique des modèles après inactivité** : implémenté partiellement (`handle_timeout`) mais non actif dans la boucle principale du daemon — les modèles restent chargés en mémoire tant que le processus daemon n'est pas arrêté manuellement.
- **Textes anciens ou dialectaux** : les modèles d'encodage sont entraînés principalement sur du texte contemporain ; les résultats sur des textes littéraires très anciens ou fortement dialectaux peuvent être moins fiables et méritent une vérification manuelle.
- **Persistance du daemon** : le daemon reste actif en arrière-plan après utilisation. Pour l'arrêter manuellement, il faut identifier son PID (par exemple via `ps aux | grep embed_daemon`) et l'arrêter avec `kill <PID>`.
- **Cohérence de la normalisation** : toute modification future du pipeline doit impérativement conserver une normalisation L2 identique côté indexation et côté requête, sous peine de scores de similarité incohérents.
- **Doublons en recherche multi-fichiers (`search_folder`)** : si deux fichiers indexés du même dossier contiennent le même texte source (ex. un corpus encodé une fois via le daemon et une fois en mode `no_daemon` pour comparaison), chaque phrase correspondante occupe **deux emplacements distincts** dans le classement agrégé top-k, ce qui peut évincer d'autres résultats légitimes du top-k final. Ce n'est pas un défaut de l'algorithme de recherche en tant que tel, mais un effet mécanique de l'agrégation sur des corpus dupliqués — à garder en tête lors de l'interprétation de résultats en mode dossier.
