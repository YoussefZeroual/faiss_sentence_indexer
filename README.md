# faiss_sentence_indexer

Outil de recherche sémantique pour corpus linguistiques et littéraires (formats **CoNLL-U**, **XML** et **TRS**), basé sur des embeddings de phrases (ou de tokens) indexés avec **FAISS**.
Le mode token est en cours de développement et n'est pas encore utilisable dans la version actuelle.

Le pipeline permet de :

1. **parser** les phrases (ou tours de parole) d'un fichier de corpus (CoNLL-U, XML ou TRS) ;
2. **calculer leurs embeddings** via un modèle d'encodage (local, via un daemon persistant, ou via Ollama) ;
3. **construire un index FAISS** optimisé pour la recherche par similarité cosinus ;
4. **interroger** cet index avec une phrase de requête et récupérer les résultats les plus proches sémantiquement.

---

## Sommaire

- [Architecture générale](#architecture-générale)
- [Prérequis et installation](#prérequis-et-installation)
- [Structure du projet](#structure-du-projet)
- [Module `calcEmbeddings.py`](#module-calcembeddingspy)
- [Module `makeIndex.py`](#module-makeindexpy)
- [Tests : impact de `nprobe` et de `m`](#tests--impact-de-nprobe-et-de-m)
- [Module `searchEmbedding.py`](#module-searchembeddingpy)
- [Module `utils/embed_client.py`](#module-utilsembed_clientpy)
- [Module `utils/embed_daemon.py`](#module-utilsembed_daemonpy)
- [Module `main.py`](#module-mainpy)
- [Choix de conception et garde-fous](#choix-de-conception-et-garde-fous)
- [Limitations connues et points d'attention](#limitations-connues-et-points-dattention)

---

## Architecture générale

Le projet est organisé en briques indépendantes qui communiquent via des fichiers intermédiaires (`.npy`, `.json`, `.faiss`) et via un processus daemon en arrière-plan pour l'encodage optimisé des phrases :

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
      makeIndex.py ──► corpus.faiss (index FAISS pour le mode phrase, ou corpus_token.faiss pour le mode token)
              │
              ▼
   searchEmbedding.py ──► résultats de la recherche sémantique (top-k phrases similaires)
```

`main.py` orchestre l'ensemble de la chaîne de façon **incrémentale** : il ne recalcule que ce qui manque encore sur le disque (embeddings, métadonnées ou index), et propose de nombreux modes d'exécution (fichier unique, dossier complet, encodage seul, recherche seule, mode sans FAISS pour le débogage, régénération des métadonnées, etc.).

Le calcul des embeddings peut être délégué à un **daemon persistant en arrière-plan** (`utils/embed_daemon.py`), afin d'éviter de recharger un modèle lourd (PyTorch / Transformers) à chaque appel de script. Le client (`utils/embed_client.py`) démarre ce daemon si besoin et communique avec lui par socket local ; il peut aussi, selon les besoins, s'en passer complètement (mode local synchrone) ou déléguer l'encodage à un serveur **Ollama** externe.

---

## Prérequis et installation

- Python 3.9+ (le code utilise des annotations de type modernes, ex. `tuple[Tensor, ...]`)
- Bibliothèques principales :
  ```bash
  pip install faiss-cpu numpy torch transformers sentence-transformers conllu lxml requests
  ```
  (utiliser `faiss-gpu` à la place de `faiss-cpu`, et une version de `torch` compilée pour CUDA, si un GPU est disponible)
- Les modèles Hugging Face sont téléchargés automatiquement au premier appel qui en a besoin (voir le module [`utils/embed_daemon.py`](#module-utilsembed_daemonpy)).
- Pour la stratégie Ollama : une instance [Ollama](https://ollama.com) locale ou distante, avec un modèle d'embedding installé (ex. `nomic-embed-text-v2-moe`).

### ⚠️ Organisation obligatoire des fichiers

L'organisation en sous-dossier `utils/` est **requise pour que le code fonctionne**

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
| `main.py` | Point d'entrée CLI : enchaîne automatiquement les étapes du pipeline, gère de nombreux modes d'exécution spéciaux, et lance la recherche finale. |
| `calcEmbeddings.py` | Parse un fichier de corpus (CoNLL-U, XML ou TRS), nettoie le texte, et calcule les embeddings de chaque phrase ou tour de parole. |
| `makeIndex.py` | Construit un index FAISS (`flat`, `hnsw` ou `ivfpq`) à partir des embeddings. |
| `searchEmbedding.py` | Charge un index et ses métadonnées, encode une requête, et effectue la recherche de similarité (fichier unique ou dossier entier). |
| `utils/embed_client.py` | Interface cliente vers l'encodage : trois stratégies possibles (daemon IPC, exécution locale directe, ou API Ollama). |
| `utils/embed_daemon.py` | Processus serveur qui charge les modèles d'encodage une seule fois, regroupe les requêtes en lots dynamiques, et les traite au fil de l'eau. |

Les sections suivantes documentent chaque module fonction par fonction: signature, paramètres, valeur de retour, comportement détaillé et un exemple d'appel.

---

## Module `calcEmbeddings.py`

Module d'extraction et d'encodage de corpus linguistiques pour la création des fichiers embeddings `.npy`. Il analyse des fichiers de corpus (CoNLL-U, XML, TRS pour les transcriptions de corpus oraux) et les convertit en représentations vectorielles normalisées. Il nettoie les textes bruts, gère certaines spécificités linguistiques (les mots amalgames ex. du --> de+le), et orchestre l'appel aux modèles d'encodage.

Le format cible est déterminé automatiquement à partir de l'**extension du fichier** (le paramètre `mode` permet de déterminer et de forcer un mode spécifique).

| Extension | Format | Fonction de parsing | Description |
|---|---|---|---|
| `.conllu` | CoNLL-U | `parse_conllu_fast()` | Corpus annotés en dépendances syntaxiques (souvent créés avec Stanza). |
| `.xml` | XML / XML-CoNLLU | `parse_sentences_xml_conllu()` | Corpus au format XML, avec gestion des cas où du CoNLL-U est imbriqué dans les balises `<s>`. |
| `.trs` | Transcriber | `parse_sentence_trs()` | Transcriptions de corpus oraux ; chaque `<Turn>` (tour de parole) est traité comme une phrase, identifiée par son `startTime`. |

**Pourquoi ne pas utiliser un module CoNLL-U dédié ?** En testant le module Python spécialisé dans le parsing du format CoNLL-U, d'importants ralentissements ont été constatés à cause de la complexité du traitement qui tente de parser toutes les informations. Étant donné que l'objectif se limite à extraire la totalité des phrases et de leurs identifiants, une simple fonction de parsing reposant sur des regex est jugée comme un choix optimal. De plus, le même code est réutilisé dans le traitement du format hybride où du CoNLL-U est imbriqué à l'intérieur de balises `<s>` de certains fichiers XML.

### Fonctions internes de parsing CoNLL-U

#### `parse_conllu_raw_entries(file_content)`

**Description** — Découpe le contenu brut d'un fichier CoNLL-U en blocs de phrases distinctes.

| Paramètre | Type | Description |
|---|---|---|
| `file_content` | `str` | Le contenu textuel intégral du fichier CoNLL-U. |

**Valeur de retour** — `list[str]` : une liste de chaînes de caractères, où chaque élément correspond à un bloc d'annotation de phrase.

**Comportement** — Sépare le texte sur les doubles retours à la ligne (standard CoNLL-U pour délimiter les phrases) et ignore les éventuels blocs vides.

**Exemple**
```python
with open("corpus.conllu", encoding="utf-8") as f:
    content = f.read()
blocks = parse_conllu_raw_entries(content)
# blocks == ["# sent_id = 1\n1\tBonjour\t...", "# sent_id = 2\n1\tAu\t...", ...]
```

#### `is_amalgame(line)`

**Description** — Vérifie si une ligne d'annotation correspond à un mot amalgame (multi-mots). En CoNLL-U, ces lignes utilisent un intervalle d'identifiants (ex. `du` est décomposé en `de` et `le` : deux lignes correspondent aux deux morphèmes, en plus d'une ligne pour la forme amalgamée ; le script ne conserve que la ligne de l'amalgame et ignore ses sous-composantes).

| Paramètre | Type | Description |
|---|---|---|
| `line` | `str` | Une ligne d'annotation CoNLL-U. |

**Valeur de retour** — `str | None` : la portion de la ligne correspondant au pattern d'intervalle (ex. `"1-2"`) si c'est une ligne d'amalgame, sinon `None`.

**Exemple**
```python
is_amalgame("1-2\tdu\t_\t_")   # -> "1-2"
is_amalgame("1\tde\tDE\t_")    # -> None
```

#### `has_amalgams(text)`

**Description** — Détermine si un bloc de texte CoNLL-U contient au moins une ligne d'amalgame.

| Paramètre | Type | Description |
|---|---|---|
| `text` | `str` | Un bloc CoNLL-U (une phrase, lignes séparées par `\n`). |

**Valeur de retour** — `bool` : `True` si au moins une ligne du bloc est un amalgame, `False` sinon.

**Exemple**
```python
has_amalgams("1-2\tdu\t_\n2\tde\t_\n3\tle\t_")  # -> True
```

#### `concat_forms(text)`

**Description** — Reconstruit le texte brut à partir des formes de surface d'un bloc CoNLL-U. Gère spécifiquement les amalgames en ignorant leurs composants enfants pour éviter les doublons (cette fonction est utilisée par  `parse_conllu_fast` et par `parse_sentences_xml_conllu` pour le CoNLL-U imbriqué en XML).

| Paramètre | Type | Description |
|---|---|---|
| `text` | `str` | Un bloc CoNLL-U correspondant à une phrase. |

**Valeur de retour** — `tuple[str, list]` : `(raw_text, tokens)`, le texte reconstruit et la ponctuation nettoyée (`fix_punctuation_spaces`), ainsi que la liste des tokens (uniquement calculée dans le cas sans amalgame, via `get_tokens`).

**Comportement**
- Si le bloc contient des amalgames (`has_amalgams`), les lignes correspondant aux sous-composantes des amalgames sont ignorées (comparaison des deux lignes précédentes via `is_amalgame`), puis les formes de surface sont extraites par regex sur les lignes restantes.
- Sinon, extraction directe des formes de surface via `re.findall(r'^\d+\s+(\S+)', text, re.MULTILINE)`.
- Dans les deux cas, le texte assemblé est passé à `fix_punctuation_spaces` avant d'être renvoyé.

**Exemple**
```python
raw_text, tokens = concat_forms("1-2\tdu\t_\n2\tde\tDE\t_\n3\tle\tLE\t_\n4\tchat\tNOUN\t_")
# raw_text == "du le chat"  (les sous-composantes 'de'/'le' de la ligne 1-2 sont ignorées)
```

#### `get_sent_id(text)`

**Description** — Extrait l'identifiant unique de la phrase (`sent_id`) depuis les métadonnées du bloc CoNLL-U.

| Paramètre | Type | Description |
|---|---|---|
| `text` | `str` | Un bloc CoNLL-U. |

**Valeur de retour** — `str | None` : la valeur du champ `# sent_id = ...` si présente, sinon `None`.

**Exemple**
```python
get_sent_id("# sent_id = s42\n1\tBonjour\t_")   # -> "s42"
```

#### `clean_sentence(sent, filename, sent_id)`

**Description** — Nettoie une phrase reconstruite et gère les valeurs nulles en les marquant par l'étiquette `[phrase manquante]` (constante `MISSING_SENTENCE`).

| Paramètre | Type | Description |
|---|---|---|
| `sent` | `str \| None` | La phrase reconstruite à nettoyer. |
| `filename` | `str` | Le nom du fichier source, utilisé uniquement pour le message de log en cas de phrase vide. |
| `sent_id` | `str \| int` | L'identifiant de la phrase, utilisé uniquement pour le message de log. |

**Valeur de retour** — `str` : la phrase nettoyée (underscores et doubles espaces supprimés), ou `MISSING_SENTENCE` si `sent` est `None` ou vide.

**Comportement** — Remplace les phrases vides par l'étiquette `[phrase manquante]` pour maintenir l'alignement entre phrases, métadonnées et vecteurs d'embeddings — un décalage d'index à ce niveau invaliderait silencieusement toute la recherche en aval.

**Exemple**
```python
clean_sentence("Il fait  beau_", "corpus.conllu", "s3")   # -> "Il fait beau"
clean_sentence(None, "corpus.conllu", "s4")                # -> "[phrase manquante]"
```

#### `get_tokens(text)`

**Description** — Sépare un texte en tokens en gérant les apostrophes. Prépare la récupération des tokens individuels des phrases traitées, afin de les enregistrer dans le fichier JSON des métadonnées et de les encoder en mode token.

| Paramètre | Type | Description |
|---|---|---|
| `text` | `str \| None` | Le texte à tokeniser. |

**Valeur de retour** — `list[str] | None` : la liste des tokens, ou `None` si `text` est `None`.

**Comportement** — Si le texte contient au moins une apostrophe, celle-ci est isolée pour forcer une césure de token à cet endroit (`d'accord` → `["d'", "accord"]`) ; sinon, découpage simple sur les espaces.

**Exemple**
```python
get_tokens("Il n'y a pas de souci")
# -> ["Il", "n'", "y", "a", "pas", "de", "souci"]
```

#### `fix_punctuation_spaces(text)`

**Description** — Normalise les espaces autour de la ponctuation selon les règles typographiques françaises. Nécessaire pour la reconstruction des phrases à partir des formes individuelles issues du fichier CoNLL-U.

| Paramètre | Type | Description |
|---|---|---|
| `text` | `str \| list` | Le texte à normaliser (une liste est automatiquement jointe par des espaces). |

**Valeur de retour** — `str` : le texte avec une ponctuation typographiquement correcte.

**Comportement** (dans l'ordre d'application) :
1. Corrige les apostrophes (supprime les espaces adjacents).
2. Supprime l'espace avant, et force l'espace après, la ponctuation double/forte (`:` `;` `?` `!`).
3. Gère les guillemets français « » (pas d'espace avant `»`, espace après `«`).
4. Corrige la ponctuation simple (`.` `,`) : pas d'espace avant, espace après.
5. Nettoie les espaces multiples résiduels.

**Exemple**
```python
fix_punctuation_spaces("Bonjour , comment ça va ?")
# -> "Bonjour, comment ça va ?"
```

### Fonctions de parsing par format

#### `parse_conllu_fast(file_path, text=None)`

**Description** — Analyse un fichier ou un texte au format CoNLL-U pour extraire les phrases et leurs métadonnées. Possède une alternative (fallback) si le texte brut n'est pas explicitement annoté avec une balise `#text_raw`.

| Paramètre | Type | Description |
|---|---|---|
| `file_path` | `str` | Le chemin d'accès au fichier CoNLL-U. |
| `text` | `str`, optionnel | Contenu textuel direct (utile si le contenu du fichier a déjà été récupéré, ou est imbriqué dans un XML). |

**Valeur de retour** — `tuple[list, dict]` : `(sent_list, metadata)`, où `sent_list` est la liste des phrases en texte brut, et `metadata` est un dictionnaire `{"sent_id": [...], "raw_text": [...], "tokens": [...]}`.

**Comportement**
1. **Extraction rapide par en-têtes** : lecture ligne par ligne, récupération directe des métadonnées `# sent_id = ...` et `# text_raw = ...` lorsqu'elles sont présentes — c'est le chemin le plus rapide.
2. **Alternative automatique (fallback) par concaténation des formes** : si la liste de phrases obtenue est vide, ou si au moins 3 phrases n'ont pas de `text_raw`, le script bascule sur une reconstruction du texte à partir des formes de surface (`concat_forms`), y compris la gestion des mots amalgames.

**Exemple**
```python
sent_list, metadata = parse_conllu_fast("corpus.conllu")
# metadata["sent_id"][0]   -> "s1"
# metadata["raw_text"][0]  -> "Bonjour, comment allez-vous ?"
```

#### `parse_sentences_xml_conllu(filepath)`

**Description** — Analyse un fichier XML (simple, ou avec du CoNLL-U imbriqué dans des balises `<s>`) pour extraire les phrases et générer les métadonnées associées.

| Paramètre | Type | Description |
|---|---|---|
| `filepath` | `str` | Le chemin d'accès au fichier XML à analyser. |

**Valeur de retour** — `tuple[list, dict]` : `(sent_list, metadata)`, identique en structure à `parse_conllu_fast`.

**Comportement** — Le parseur (basé sur `lxml`, en mode tolérant aux erreurs `recover=True`) cible toutes les balises `<s>` via XPath et gère trois cas de figure, par ordre de détection :
1. **Texte multi-lignes** dans `<s>` (signature probable de CoNLL-U imbriqué) → reconstruction via `concat_forms` puis `fix_punctuation_spaces`.
2. **Absence de texte direct** dans `<s>` (contenu fragmenté dans des sous-balises, ex. `<w>`) → récupération via `s.itertext()`.
3. **Texte simple et direct** dans `<s>` → utilisation telle quelle, tokenisation via `get_tokens`.

**Exemple**
```python
sent_list, metadata = parse_sentences_xml_conllu("corpus.xml")
```

#### `parse_sentence_trs(file_path=None)`

**Description** — Analyse un fichier de transcription audio au format TRS (Transcriber) pour extraire les tours de parole.

| Paramètre | Type | Description |
|---|---|---|
| `file_path` | `str` | Le chemin d'accès au fichier TRS à analyser. |

**Valeur de retour** — `tuple[list, dict]` : `(sent_list, metadata)`, où `metadata["sent_id"]` contient les valeurs de l'attribut `startTime` de chaque `<Turn>`.

**Comportement** — Cible les balises `<Turn>` du fichier XML (parseur `lxml` tolérant aux erreurs) et utilise l'attribut temporel `startTime` comme identifiant unique ; concatène tout le texte contenu dans le nœud `<Turn>` et ses sous-nœuds via `s.itertext()`, puis nettoie la ponctuation via `fix_punctuation_spaces`.

**Exemple**
```python
sent_list, metadata = parse_sentence_trs("interview.trs")
# metadata["sent_id"][0] -> "12.34"  (startTime en secondes)
```

### Point d'entrée de parsing

#### `parse_sentences(file_path=None, mode=None)`

**Description** — Sélectionne et exécute le mode de parsing approprié en fonction de l'extension du fichier. Calcule également des statistiques d'extraction (nombre total de phrases et de tokens) et mesure le temps d'exécution.

| Paramètre | Type | Description |
|---|---|---|
| `file_path` | `str` | Le chemin d'accès au fichier cible. |
| `mode` | `str`, optionnel | Le format du fichier. **Ce paramètre est écrasé** par l'extension réelle extraite de `file_path`. |

**Valeur de retour** — `tuple[list, dict]` : `(sent_list, metadata)`, ou `(None, None)` si le format du fichier n'est pas reconnu.

**Comportement** — Redirige, selon l'extension détectée (`.conllu`, `.xml`, `.trs`), vers `parse_conllu_fast`, `parse_sentences_xml_conllu` ou `parse_sentence_trs`, puis journalise le nombre de phrases et de tokens extraits (comptés par regex `\w+|[^\w\s]`, en ignorant les phrases marquées `[phrase manquante]`) ainsi que le temps d'exécution.

**Exemple**
```python
sent_list, metadata = parse_sentences("corpus.conllu")
if sent_list is None:
    print("Format de fichier non reconnu")
```

### Fonctions d'encodage et de sauvegarde

#### `calcEmbeddings(collection_file_path=None, output_file_path=None, mode="conllu", reduce_precision=False, overwrite=False, token_mode=False, no_daemon=False, use_ollama=False, ollama_host='localhost:11434', ollama_model=None)`

**Description** — Fonction principale du module : extrait les phrases d'un fichier de corpus et génère leurs embeddings correspondants. Intègre un système de cache : si les fichiers de sortie existent déjà, ils sont chargés directement.

| Paramètre | Type | Description |
|---|---|---|
| `collection_file_path` | `str` | Le chemin vers le fichier de corpus source (`.conllu`, `.xml`, `.trs`). |
| `output_file_path` | `str` | Le chemin de destination pour l'enregistrement du fichier des embeddings (`.npy`). |
| `mode` | `str` | Le format du corpus cible (par défaut `"conllu"`). |
| `reduce_precision` | `bool` | Si `True`, sauvegarde les embeddings en `float16` pour économiser de l'espace disque. |
| `overwrite` | `bool` | Si `True`, force le recalcul même si les fichiers de sortie existent déjà. |
| `token_mode` | `bool` | Si `True`, utilise l'encodage par token et ajoute le suffixe `_token` au fichier de sortie. |
| `no_daemon` | `bool` | Si `True`, exécute le modèle localement au lieu du processus daemon. |
| `use_ollama` | `bool` | Si `True`, délègue l'encodage à une API Ollama externe. |
| `ollama_host` | `str` | L'adresse du serveur Ollama. |
| `ollama_model` | `str` | Le nom du modèle Ollama à utiliser. |

**Valeur de retour** — `tuple[numpy.ndarray, dict]` : `(embeddings, metadata)`, la matrice des vecteurs générés (ou chargés depuis le cache) et le dictionnaire de métadonnées associé.

**Comportement**
1. **Vérification du cache** : si `overwrite=False` et que les fichiers `.npy` et `.json` existent déjà, ils sont simplement chargés depuis le disque (pas de recalcul).
2. Sinon, parse les phrases et métadonnées via `parse_sentences`.
3. Encode les phrases (`chunk_size=64`) via `encode()` du client d'embedding, en respectant le mode et la stratégie demandés.
4. Sauvegarde les embeddings au format `.npy`, en `float16` si `reduce_precision=True`, en `float32` sinon.
5. En mode token, le suffixe `_token` est automatiquement ajouté au nom du fichier de sortie pour éviter d'écraser les embeddings « phrase ».

**Exemple**
```python
embeddings, metadata = calcEmbeddings(
    collection_file_path="corpus.conllu",
    output_file_path="corpus.npy",
    mode="conllu",
)
```

#### `save_metadata(metadata, output_file=None, token_mode=False)`

**Description** — Sauvegarde le dictionnaire de métadonnées dans un fichier JSON.

| Paramètre | Type | Description |
|---|---|---|
| `metadata` | `dict` | Le dictionnaire contenant les métadonnées extraites (`sent_id`, `raw_text`, `tokens`). |
| `output_file` | `str` | Le chemin d'accès au fichier cible (généralement `.json`). |
| `token_mode` | `bool` | Paramètre conservé pour des raisons de compatibilité de signature de la fonction (non utilisé dans le corps actuel). |

**Valeur de retour** — `None`.

**Comportement** — Ouverture du fichier en écriture avec encodage UTF-8 (indispensable pour préserver correctement les accents et caractères spéciaux français), puis sérialisation JSON du dictionnaire.

**Exemple**
```python
save_metadata(metadata, "corpus.json")
```

#### `encode_folder(input_folder=None, overwrite=False, token_mode=False, no_daemon=False, use_ollama=False, ollama_host='localhost:11434', ollama_model=None)`

**Description** — Parcourt un répertoire ou un wildcard (ex. `test/*Camus*`) pour traiter en lot des fichiers de corpus, générer leurs embeddings et sauvegarder leurs métadonnées.

| Paramètre | Type | Description |
|---|---|---|
| `input_folder` | `str` | Le chemin du répertoire cible ou un motif (ex. `data/*`). |
| `overwrite` | `bool` | Si `True`, force le recalcul des embeddings même s'ils existent déjà. |
| `token_mode` | `bool` | Si `True`, génère les embeddings au niveau des tokens. |
| `no_daemon` | `bool` | Si `True`, exécute le modèle d'encodage localement (sans daemon). |
| `use_ollama` | `bool` | Si `True`, utilise une instance Ollama pour l'encodage. |
| `ollama_host` | `str` | L'adresse de l'hôte API Ollama (défaut : `'localhost:11434'`). |
| `ollama_model` | `str` | Le modèle Ollama spécifique à interroger. |

**Valeur de retour** — `None` : cette fonction opère par effets de bord (création de fichiers `.npy` et `.json` sur le disque).

**Comportement** — Détecte tous les fichiers `.conllu`, `.xml` et `.trs` du dossier (ou du motif wildcard), puis appelle `calcEmbeddings` et `save_metadata` pour chacun, en journalisant la progression (`fichier i/N`).

> ⚠️ Le nom de fichier de sortie est calculé par `f.replace(ext, 'npy')` (remplacement de l'extension d'origine), ce qui suppose que le nom de l'extension ne figure pas ailleurs dans le chemin du fichier.

**Exemple**
```python
encode_folder("corpus/*Camus*", overwrite=False)
```

---

## Module `makeIndex.py`

Module de création d'index FAISS pour la recherche sémantique. Il fait le pont entre la phase d'extraction des caractéristiques (calcul des embeddings) et la phase de recherche. Une fois les fichiers `.faiss` créés, les fichiers `.npy` des embeddings peuvent être effacés car la recherche sémantique se base désormais sur les index `.faiss` — pour recréer un index, il faudrait néanmoins refaire les embeddings au préalable.

#### `load_embeddings(embedding_file_path)`

**Description** — Charge les vecteurs d'embeddings depuis un fichier NumPy, les formate et les normalise pour l'indexation FAISS.

| Paramètre | Type | Description |
|---|---|---|
| `embedding_file_path` | `str` | Le chemin vers le fichier d'embeddings (généralement `.npy`). |

**Valeur de retour** — `numpy.ndarray` : une matrice 2D contiguë de type `float32`, avec des vecteurs normalisés L2.

**Comportement**
1. Sépare le nom de base et l'extension pour forcer le chargement du fichier `.npy` même si une autre extension est fournie.
2. Convertit le tableau en `float32` contigu en mémoire (`np.ascontiguousarray`), requis par FAISS (écrit en C++).
3. Normalise en norme L2 (`faiss.normalize_L2`) — nécessaire pour que la métrique de produit scalaire (Inner Product) soit l'équivalent d'une similarité cosinus entre les vecteurs.

**Exemple**
```python
embeddings = load_embeddings("corpus.npy")
```

#### `makeIndex(embeddings=None, embedding_file_path=None, metric_type=None, index_type=None, m=256, output_file_path=None, overwrite=False, token_mode=False)`

**Description** — Construit, entraîne et sauvegarde un index FAISS à partir de vecteurs d'embeddings.

| Paramètre | Type | Description |
|---|---|---|
| `embeddings` | `numpy.ndarray` | Matrice des vecteurs à indexer (alternative à `embedding_file_path`). |
| `embedding_file_path` | `str` | Chemin vers le fichier d'embeddings à charger si `embeddings` n'est pas fourni. |
| `metric_type` | `int` | Type de métrique de distance FAISS (ex. `faiss.METRIC_INNER_PRODUCT`). |
| `index_type` | `str` | L'algorithme d'indexation cible : `"flat"`, `"hnsw"` ou `"ivfpq"`. |
| `m` | `int` | Nombre de sous-vecteurs pour la compression PQ (mode `ivfpq` uniquement, valeur par défaut : `512`). Voir la sous-section dédiée ci-dessous. |
| `output_file_path` | `str` | Chemin de destination pour sauvegarder le fichier d'index. |
| `overwrite` | `bool` | Si `False`, charge l'index existant s'il est déjà présent sur le disque. |
| `token_mode` | `bool` | Si `True`, ajuste le nom du fichier de sortie pour refléter l'encodage par token. |

**Valeur de retour** — `faiss.Index` : l'index FAISS prêt à être utilisé pour la recherche de similarité.

**Exceptions** — `ValueError` si `index_type` fourni n'est pas reconnu (`"flat"`, `"hnsw"`, `"ivfpq"`).

**Comportement**
1. Si `overwrite=False` et qu'un index `.faiss` existe déjà pour ce fichier, il est chargé directement depuis le disque plutôt que reconstruit.
2. Sélectionne la stratégie d'indexation selon `index_type` :

| Type | Description | Garde-fou |
|---|---|---|
| `flat` | `IndexFlat` : recherche exhaustive, exacte, sans structure d'accélération. Idéal pour petits corpus ou pour valider la qualité des autres index. Produit des index non compressés : la taille est donc équivalente à celle des embeddings bruts. | Aucun, toujours disponible, quel que soit le nombre de vecteurs. |
| `hnsw` | `IndexHNSWFlat` (`hnsw_m=64`, `efConstruction=40`, `efSearch=64`) : graphe de plus proches voisins. Bon compromis vitesse/précision, sans entraînement nécessaire. | Aucun garde-fou spécifique — mais consomme davantage de mémoire que `ivfpq`, avec une taille sur disque proche ou équivalente au mode `flat`. |
| `ivfpq` | `IndexIVFPQ` : listes inversées (`nlist = 4·√n`) + quantification produit (`m`, `nbits=8`). Nécessite un entraînement (`index.train`). Optimisé pour les très grands corpus contraints en mémoire ; permet un gain considérable en taille du fichier (ex. pour `m=512`, un embedding de ~460 MB produit un index de ~60 MB). | **Garde-fou automatique** : si le nombre de vecteurs `n` est inférieur au minimum requis pour un entraînement statistiquement fiable (`nlist·39` et `(2**nbits)·39`), la fonction bascule **automatiquement** sur un index `flat` et journalise un avertissement, plutôt que d'entraîner un index IVFPQ de mauvaise qualité. |

3. Normalise les vecteurs fournis directement en argument (si `embeddings` est passé plutôt que chargé depuis un fichier), pour garantir la même cohérence de normalisation qu'avec `load_embeddings`.
4. Écrit l'index sur disque (`faiss.write_index`) ; en mode token, le suffixe `_token` est ajouté au nom du fichier.

**Le paramètre `m` (nombre de sous-quantifieurs PQ)** — `m` détermine en combien de sous-vecteurs chaque embedding est découpé pour la quantification produit (`IndexIVFPQ`) : chaque sous-vecteur est ensuite compressé en un code de `nbits=8` (soit 256 centroïdes possibles). C'est le paramètre qui a le plus d'impact direct sur la qualité des résultats en mode `ivfpq`.

- `m` doit être un **diviseur de la dimension des embeddings** (1024 pour `BAAI/bge-m3`) : valeurs valides `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}`.
- Plus `m` est élevé, plus chaque sous-vecteur couvre peu de dimensions d'origine, donc moins d'information est perdue à la compression — mais plus l'index occupe d'espace sur le disque.
- Voir la section [Tests : impact de `nprobe` et de `m`](#tests--impact-de-nprobe-et-de-m) pour des mesures concrètes de ce compromis sur un corpus réel.


**Pourquoi normaliser ?** FAISS ne propose pas nativement de métrique « cosinus » : seulement `METRIC_L2` et `METRIC_INNER_PRODUCT`. Le produit scalaire de deux vecteurs normalisés (norme = 1) est mathématiquement équivalent à leur similarité cosinus. C'est pourquoi `makeIndex.py` normalise systématiquement les embeddings avant indexation, et `searchEmbedding.py` fait de même sur le vecteur de requête — **la cohérence entre les deux côtés est essentielle**, sans quoi les scores de similarité seraient faussés.

**Exemple**
```python
index = makeIndex(
    embedding_file_path="corpus.npy",
    metric_type=faiss.METRIC_INNER_PRODUCT,
    index_type="ivfpq",
    m=512,
    output_file_path="corpus.faiss",
)
```

#### `makeIndex_folder(input_folder=None, metric_type=None, index_type=None, overwrite=False, token_mode=False)`

**Description** — Parcourt un dossier (ou un motif wildcard, ex. `test/ESLO*`) pour trouver des fichiers d'embeddings et crée un index FAISS pour chacun d'eux.

| Paramètre | Type | Description |
|---|---|---|
| `input_folder` | `str` | Le chemin du répertoire cible ou un motif wildcard (ex. `dossier/*.npy`, `dossier/ESLO*`). |
| `metric_type` | `int` | Le type de métrique de distance pour FAISS (ex. `faiss.METRIC_INNER_PRODUCT`). |
| `index_type` | `str` | L'algorithme d'indexation à utiliser (`"flat"`, `"hnsw"` ou `"ivfpq"`). |
| `m` | `int` | Nombre de sous-vecteurs pour la compression PQ (mode `ivfpq` uniquement, valeur par défaut : `512`). |
| `overwrite` | `bool` | Si `True`, force le recalcul et l'écrasement des index existants. |
| `token_mode` | `bool` | Si `True`, cible spécifiquement les fichiers encodés au niveau des tokens (`_token.npy`). |

**Valeur de retour** — `None` : la fonction agit directement sur les fichiers ciblés en écrivant les fichiers `.faiss` sur le disque.

**Comportement** — Détecte les fichiers d'embeddings (`.npy` ou `_token.npy` selon `token_mode`) via `glob`, déduplique la liste, puis construit un index pour chacun via `makeIndex` (avec `m` implicitement fixé à sa valeur par défaut, voir avertissement ci-dessus).

**Exemple**
```python
makeIndex_folder(
    input_folder="test",
    metric_type=faiss.METRIC_INNER_PRODUCT,
    index_type="ivfpq",
)
```

---

## Tests : impact de `nprobe` et de `m`

Cette section rassemble les résultats de tests comparatifs menés sur un corpus réel (`HS36-6030v2tv8.conllu`, **114 064 phrases**, embeddings `BAAI/bge-m3` en 1024 dimensions, index `ivfpq` avec `nlist=1348`), en comparant systématiquement les résultats d'une recherche `ivfpq` à une **référence exacte** (recherche `flat` / calcul direct du produit scalaire sur les `.npy` via `--no-faiss`), sur la même requête (`"pas de souci"`, top-30).

### `nprobe` (paramètre de recherche, sans impact mesuré dans ce test)

`nprobe` contrôle le nombre de listes inversées (clusters) explorées lors d'une recherche `ivfpq` — c'est un paramètre appliqué **au moment de la requête**, sur un index déjà construit, sans nécessiter de reconstruction.

| `nprobe` testé | Résultat |
|---|---|
| `8` (valeur d'origine codée en dur dans `search()`) | Référence de départ |
| `64` | **Résultats strictement identiques** à `nprobe=8` (mêmes phrases, même ordre, mêmes scores à 3 décimales) |
| `1348` (= `nlist`, recherche exhaustive sur toutes les listes inversées) | **Résultats toujours strictement identiques** |

**Conclusion** : sur ce corpus et cette requête, faire varier `nprobe` de 8 jusqu'à une couverture exhaustive de tous les clusters n'a produit **aucun changement mesurable**. La couverture des clusters n'était donc pas le facteur limitant de la qualité des résultats — l'écart observé avec la recherche exacte provient d'ailleurs.

### `m` (paramètre de construction, impact majeur mesuré)

`m` (nombre de sous-quantifieurs PQ) doit en revanche être fixé **à la construction de l'index** — le faire varier impose de reconstruire l'index (`--force`, ou en effaçant le fichier `.faiss` et en ajoutant l'option `--encode-only`). Le test a comparé le recouvrement du top-30 `ivfpq` avec le top-30 exact, ainsi que la taille du fichier `.faiss` produit, pour plusieurs valeurs de `m` :

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
- **`hnsw` n'a pas encore été mesuré dans cette série de tests** et reste une alternative à évaluer : en théorie, il offre un meilleur recouvrement que `ivfpq` à taille de corpus comparable, sans nécessiter de réglage de `m`/`nbits`, au prix d'une consommation mémoire et disque plus élevée qu'`ivfpq`.

---

## Module `searchEmbedding.py`

Module de chargement des indices FAISS et des fichiers de métadonnées (JSON), d'encodage d'une requête via le client d'embedding, et d'exécution de la recherche de similarité entre la requête et les phrases indexées. Le module supporte la recherche dans un seul fichier index (`.faiss`) ou dans un dossier contenant plusieurs fichiers — les résultats d'une recherche en mode dossier sont rassemblés puis triés par score de similarité.

#### `load_index(index_file=None)`

**Description** — Charge un index FAISS à partir d'un fichier spécifié sur le disque.

| Paramètre | Type | Description |
|---|---|---|
| `index_file` | `str` | Le chemin d'accès vers le fichier d'index (extension `.faiss`). |

**Valeur de retour** — `faiss.Index` : l'objet index FAISS chargé en mémoire, prêt pour la recherche.

**Exceptions** — `ValueError` si `index_file` est `None`.

**Exemple**
```python
index = load_index("corpus.faiss")
```

#### `load_metadata(matadata_file_path=None)`

**Description** — Charge les métadonnées d'un corpus à partir d'un fichier JSON. Les métadonnées sont les phrases elles-mêmes, avec leurs identifiants et leurs tokens individuels sous forme de dictionnaire, ex. `{"sent_id": ["1", "2"], "raw_text": ["Bonjour !", "pas de souci !"], "tokens": [["Bonjour","!"], ["pas","de","souci","!"]]}`. Chaque entrée du dictionnaire est une liste de la même taille que le fichier index correspondant.

| Paramètre | Type | Description |
|---|---|---|
| `matadata_file_path` | `str` | Le chemin d'accès vers le fichier JSON contenant les métadonnées. |

**Valeur de retour** — `dict | None` : le dictionnaire de métadonnées (`sent_id`, `raw_text`, `tokens`) si le chargement réussit, sinon `None` (avec un avertissement journalisé).

**Exemple**
```python
metadata = load_metadata("corpus.json")
```

#### `embedd_query(query_str=None, token_mode=False, no_daemon=False, use_ollama=False, ollama_host='localhost', ollama_model=None)`

**Description** — Génère un vecteur d'embedding pour une requête textuelle donnée.

| Paramètre | Type | Description |
|---|---|---|
| `query_str` | `str` | Le texte de la requête à encoder. |
| `token_mode` | `bool` | Si `True`, utilise l'encodage au niveau des tokens au lieu du niveau phrase. |
| `no_daemon` | `bool` | Si `True`, exécute l'encodage localement sans passer par le daemon. |
| `use_ollama` | `bool` | Si `True`, délègue la création de l'embedding à un modèle Ollama (plus lent). |
| `ollama_host` | `str` | L'adresse de l'hôte Ollama (par défaut `'localhost'`). |
| `ollama_model` | `str` | Le nom du modèle d'embedding Ollama cible. |

**Valeur de retour** — `numpy.ndarray` : un tableau 2D contigu de type `float32` représentant l'embedding de la requête, de forme `(1, dim)`.

**Exceptions** — `ValueError` si `query_str` est `None`.

**Comportement** — Encode la requête via `encode()` (avec les mêmes options de stratégie que pour le corpus, en envoyant une liste à un seul élément), mesure et journalise le temps d'encodage, puis met en forme le résultat en tableau `float32` 2D contigu tel qu'attendu par FAISS.

**Exemple**
```python
query_vector = embedd_query("pas de souci")
```

#### `search(query_vector=None, query_str=None, index=None, metric_type=None, top_k=10, metadata=None, token_mode=False, no_daemon=False)`

**Description** — Fonction principale du module : exécute une recherche de similarité dans un index FAISS et renvoie les meilleures correspondances avec leurs métadonnées.

| Paramètre | Type | Description |
|---|---|---|
| `query_vector` | `numpy.ndarray` | Le vecteur d'embedding pré-calculé de la requête (évite un ré-encodage si déjà disponible). |
| `query_str` | `str` | Le texte de la requête (utilisé pour générer l'embedding si `query_vector` est `None`). |
| `index` | `faiss.Index` | L'objet index FAISS dans lequel effectuer la recherche. |
| `metric_type` | `int` | Le type de métrique FAISS utilisé (par défaut `faiss.METRIC_INNER_PRODUCT`, équivalent à une similarité cosinus puisque les vecteurs sont normalisés L2). |
| `top_k` | `int` | Le nombre maximal de résultats similaires à retourner (défaut : 10). |
| `metadata` | `dict` | Le dictionnaire contenant les phrases en texte brut avec leurs identifiants (`sent_id`, `raw_text`). |
| `token_mode` | `bool` | Détermine si l'encodage d'une nouvelle requête se fait au niveau des tokens. |
| `no_daemon` | `bool` | Détermine si l'encodage local est utilisé sans le processus daemon. |

**Valeur de retour** — `list[tuple]` : une liste de tuples `(sent_id, raw_text, score)`. Retourne une liste vide en cas d'erreur de dimension (généralement due à une incompatibilité entre les vecteurs de la requête et ceux du corpus cible — utilisation par erreur de deux modèles différents).

**Comportement**
1. Encode la requête si un vecteur pré-calculé n'est pas déjà fourni (`query_vector`), ce qui permet, en mode dossier, de **n'encoder la requête qu'une seule fois** puis de réutiliser le même vecteur sur tous les index du dossier — optimisation notable pour les recherches multi-fichiers.
2. Normalise le vecteur de requête (`faiss.normalize_L2`), pour rester cohérent avec la normalisation appliquée côté indexation.
3. Configure `index.nprobe = 8` si l'index le permet (cas des index IVF), afin de contrôler le compromis vitesse/précision de la recherche partitionnée.
4. Effectue la recherche (`index.search`) et reconstitue les résultats sous forme de tuples `(sent_id, raw_text, score)`, en filtrant les indices invalides (`-1`, retourné par FAISS lorsqu'il existe moins de `top_k` résultats que demandé).
5. **Garde-fou de dimension** : si un `AssertionError` est levé par FAISS (typiquement parce que la requête a été encodée avec un modèle différent de celui utilisé pour indexer le corpus cible — par exemple mode token vs mode phrase), la fonction journalise un avertissement explicite et retourne une liste vide plutôt que de faire planter tout le script.

**Exemple**
```python
results = search(
    query_str="pas de souci",
    index=index,
    metric_type=faiss.METRIC_INNER_PRODUCT,
    top_k=10,
    metadata=metadata,
)
for sent_id, raw_text, score in results:
    print(sent_id, raw_text, score)
```

#### `search_folder(input_folder=None, query_str=None, query_vector=None, metric_type=faiss.METRIC_INNER_PRODUCT, top_k=10, verbose=True, token_mode=False, no_daemon=False)`

**Description** — Exécute une recherche de similarité textuelle sur un ensemble d'index FAISS contenus dans un dossier.

| Paramètre | Type | Description |
|---|---|---|
| `input_folder` | `str` | Le chemin du dossier ou le motif wildcard (ex. `*Camus*` pour restreindre la recherche aux fichiers dont le nom contient « Camus ») contenant les fichiers `.faiss`. |
| `query_str` | `str` | Le texte brut de la requête à rechercher. |
| `query_vector` | `numpy.ndarray` | Le vecteur d'embedding pré-calculé (évite de ré-encoder la requête). |
| `metric_type` | `int` | La métrique de distance utilisée par FAISS (défaut : produit scalaire). |
| `top_k` | `int` | Le nombre global de meilleurs résultats à conserver et afficher. |
| `verbose` | `bool` | Si `True`, affiche le tableau des résultats dans la console. |
| `token_mode` | `bool` | Si `True`, cible spécifiquement les index encodés au niveau des tokens (fichiers `_token.faiss`). |
| `no_daemon` | `bool` | Si `True`, utilise l'encodage local sans passer par le daemon. |

**Valeur de retour** — `None` : la fonction agrège et affiche les résultats, mais ne retourne aucune valeur.

**Comportement**
- Encode la requête une seule fois si nécessaire (`embedd_query`), puis normalise le vecteur.
- Récupère les fichiers `.faiss` du dossier (ou du motif wildcard) via `glob`, en filtrant selon `token_mode`, et déduplique la liste.
- Ignore et journalise en avertissement les fichiers index sans métadonnées correspondantes (ou l'inverse), plutôt que d'interrompre toute la recherche.
- Agrège les résultats de tous les fichiers (chacun annoté du nom du fichier source), les trie globalement par score décroissant, et n'en conserve que les `top_k` meilleurs.
- Affiche un tableau récapitulatif si `verbose=True`.

> ⚠️ Si deux fichiers du dossier contiennent le même texte source (ex. un corpus encodé une fois via le daemon et une fois en mode `no_daemon` pour comparaison), chaque phrase correspondante occupe deux emplacements distincts dans le classement agrégé — ce qui peut évincer d'autres résultats légitimes du top-k final.

**Exemple**
```python
search_folder(
    input_folder="test/*",
    query_str="pas de souci",
    top_k=30,
    verbose=True,
)
```

---

## Module `utils/embed_client.py`

Client d'encodage vectoriel (« Embedding Client »). Il constitue l'interface entre les différents scripts et le daemon d'embedding, et implémente trois stratégies d'exécution distinctes :

| Stratégie | Argument | Comportement |
|---|---|---|
| **Daemon IPC** (par défaut) | *(aucun flag)* | Envoie les phrases à `embed_daemon.py` via une socket locale ; démarre le daemon automatiquement s'il n'est pas déjà lancé. |
| **Sans daemon** | `no_daemon=True` | Charge les modèles directement dans le processus courant (utile pour le débogage, l'exécution ponctuelle, ou les environnements où un processus persistant n'est pas souhaitable). |
| **Ollama** | `use_ollama=True` | Délègue l'encodage à un serveur Ollama externe via une requête HTTP POST sur `/api/embed`. |

Ces trois stratégies sont mutuellement exclusives dans l'ordre de priorité suivant : Ollama est tenté en premier s'il est demandé, puis, en cas d'échec de connexion, le script bascule automatiquement sur le mode daemon/local. Le mode `no_daemon` est prioritaire sur le mode daemon s'il est explicitement demandé.

> ⚠️ **Ollama est explicitement documenté comme significativement plus lent** que le daemon ou un modèle chargé directement, problème connu côté Ollama, non résolu à ce jour. Cette stratégie reste donc expérimentale et peut être utile pour effectuer des tests.

**Les modèles d'encodage** — Deux modèles distincts sont utilisés selon le mode choisi :

| Mode | Modèle | Usage |
|---|---|---|
| **Phrase** (par défaut) | `BAAI/bge-m3` (via `SentenceTransformer`) | Encode chaque phrase entière en un seul vecteur ; c'est le mode utilisé pour l'indexation FAISS et la recherche sémantique de phrases. |
| **Token** (`token_mode=True`) | `intfloat/multilingual-e5-base` (via `transformers.AutoModel`) | Encode au niveau des tokens. Ce mode est en développement et servira dans les versions futures à créer des embeddings de tokens. |

Une fonctionnalité de **réduction de dimensionnalité « All-but-the-top »** (`all_but_the_top`, dans `embed_daemon.py`) est également disponible : elle retire les composantes principales dominantes des embeddings, une technique connue pour améliorer la qualité de la similarité cosinus sur certains modèles. Cette fonction est destinée à être utilisée dans le mode token.

#### `encode_ollama(host='localhost:11434', model=None, sentence_list=None)`

**Description** — Génère des embeddings en interrogeant une instance de l'API Ollama. Nécessite un serveur Ollama et un modèle d'embeddings préinstallés sur ce serveur.

| Paramètre | Type | Description |
|---|---|---|
| `host` | `str` | L'adresse et le port du serveur Ollama. |
| `model` | `str` | Le nom du modèle à utiliser sur le serveur Ollama. |
| `sentence_list` | `list` | La liste des phrases à encoder. |

**Valeur de retour** — `list | None` : la liste des vecteurs retournés par l'API, ou `None` en cas d'échec de connexion (`requests.exceptions.ConnectionError`).

**Exemple**
```python
embs = encode_ollama(host="localhost:11434", model="nomic-embed-text-v2-moe:latest",
                      sentence_list=["pas de souci"])
```

#### `_try_connect()`

**Description** — Tente d'établir une connexion IPC avec le processus daemon sur le port 6000.

**Paramètres** — Aucun.

**Valeur de retour** — `multiprocessing.connection.Client | None` : l'objet de connexion si réussi, sinon `None` (`ConnectionRefusedError` ou `OSError` interceptées).

**Exemple**
```python
conn = _try_connect()
if conn is None:
    print("Daemon non démarré")
```

#### `_start_daemon()`

**Description** — Si le daemon d'embedding n'est pas déjà lancé, cette fonction le lance en tant que processus indépendant (détaché) et attend qu'il soit prêt à accepter des connexions.

**Paramètres** — Aucun.

**Valeur de retour** — `multiprocessing.connection.Client` : la connexion établie avec le daemon fraîchement démarré.

**Exceptions** — `RuntimeError` si le daemon ne répond pas après environ 60 secondes d'attente.

**Comportement** — Lance le processus en arrière-plan (`subprocess.Popen`, `start_new_session=True` pour le détacher du terminal courant), puis tente de s'y connecter jusqu'à 120 fois avec un intervalle de 0,5 s, ce qui laisse le temps au daemon de démarrer et de charger potentiellement PyTorch en mémoire.

**Exemple**
```python
conn = _try_connect()
if conn is None:
    conn = _start_daemon()
```

#### `encode_no_daemon(sentences=None, token_mode=False)`

**Description** — Exécute l'encodage vectoriel directement dans le processus courant (de manière synchrone), sans utiliser le système de daemon en arrière-plan. Utilise un chargement paresseux (lazy loading) : les bibliothèques lourdes (PyTorch) et les poids des modèles ne sont chargés en mémoire qu'au premier appel.

| Paramètre | Type | Description |
|---|---|---|
| `sentences` | `list` | La liste des phrases ou textes à encoder. |
| `token_mode` | `bool` | Si `True`, utilise le modèle orienté tokens (`multilingual-e5-base`) avec un pooling sur les dernières couches. |

**Valeur de retour** — `numpy.ndarray` : la matrice des vecteurs d'embeddings (`float32`).

**Comportement**
- **Mode token** : tokenisation par lots de 32 (`tqdm` pour la barre de progression), pooling par moyenne des 4 dernières couches cachées (`average_pool_last_n_layers`).
- **Mode phrase** : délègue le batching et la barre de progression au `SentenceTransformer` lui-même (`model.encode(sentences, show_progress_bar=True)`).

**Exemple**
```python
embeddings = encode_no_daemon(["pas de souci", "aucun problème"])
```

#### `encode(sentences, chunk_size=512, show_progress=True, token_mode=False, no_daemon=False, use_ollama=False, ollama_host='localhost', ollama_model=None)`

**Description** — Fonction routeur principale pour la génération d'embeddings. Permet de choisir la stratégie d'exécution selon le besoin de l'utilisateur (Ollama, local, ou via daemon IPC).

| Paramètre | Type | Description |
|---|---|---|
| `sentences` | `list` | Les textes à encoder. |
| `chunk_size` | `int` | Taille des sous-lots envoyés au daemon via le réseau ; la valeur par défaut (512) a été déterminée après avoir testé des tailles plus grandes qui ont causé des débordements de mémoire GPU. |
| `show_progress` | `bool` | Affiche dynamiquement la progression dans la console. |
| `token_mode` | `bool` | Bascule sur le modèle d'encodage par token. |
| `no_daemon` | `bool` | Force l'exécution locale synchrone. |
| `use_ollama` | `bool` | Tente d'utiliser un serveur Ollama externe (plus lent que le daemon ou un modèle chargé directement — problème connu côté Ollama). |
| `ollama_host` | `str` | Adresse du serveur Ollama. |
| `ollama_model` | `str` | Modèle Ollama cible. |

**Valeur de retour** — `numpy.ndarray` : la matrice finale des embeddings générés.

**Exceptions** — `RuntimeError` si le processus d'encodage (daemon ou réseau) échoue.

**Comportement**
1. **Stratégie Ollama** (si `use_ollama=True`) : tentée en premier ; en cas de réponse vide, bascule automatiquement sur le mode suivant.
2. **Stratégie sans daemon** (si `no_daemon=True`) : chargement direct des modèles dans le processus courant.
3. **Stratégie par défaut (daemon)** : tente une connexion au daemon (`_try_connect`), le démarre si besoin (`_start_daemon`), envoie `(job_id, sentences, chunk_size)` avec un identifiant unique (`uuid.uuid4()`) pour le suivi/débogage, puis écoute les messages de progression et le résultat final, en affichant une barre de progression textuelle si `show_progress=True`.

**Exemple**
```python
embeddings = encode(["pas de souci", "aucun problème"], chunk_size=64)
```

---

## Module `utils/embed_daemon.py`

Daemon d'encodage vectoriel par lots. Ce script s'exécute en tâche de fond et reste à l'écoute des requêtes d'encodage via des sockets IPC (Inter-Process Communication), afin d'optimiser l'encodage des corpus et des requêtes en évitant de recharger le modèle d'encodage à chaque fois. Son architecture sépare les entrées/sorties réseau du calcul pur : il regroupe les requêtes de plusieurs clients dans une file d'attente et les traite par lots pour maximiser les performances du GPU et éviter la saturation de la VRAM.

Pour le moment, ce daemon n'a pas été développé en tant que service en arrière-plan géré par le système ; par conséquent, la seule manière de le terminer est de le tuer manuellement (ex. sous Linux, `kill [PID du processus]`).

#### `handle_timeout()`

**Description** — Gère le déchargement automatique des modèles de la mémoire vidéo (VRAM) après une période d'inactivité définie par `KEEP_MODEL_LOADED_TIMEOUT`. Cette fonction est encore en développement et **n'est pas encore utilisée** dans la boucle principale du daemon.

**Paramètres** — Aucun.

**Valeur de retour** — `None`.

**Comportement** — Journalise le temps écoulé depuis la dernière connexion ; si celui-ci dépasse `KEEP_MODEL_LOADED_TIMEOUT`, met les références de modèles à `None` (mais ce déchargement n'est actuellement jamais déclenché par le reste du code).

#### `load_models(token_mode=False)`

**Description** — Charge les modèles d'embedding en mémoire (VRAM si CUDA est disponible, sinon RAM), uniquement lorsqu'ils sont nécessaires (lazy loading). Chaque modèle est chargé après une première requête, puis gardé en mémoire pour traiter les requêtes futures.

| Paramètre | Type | Description |
|---|---|---|
| `token_mode` | `bool` | Détermine quel modèle charger (E5 pour les tokens, BGE-M3 pour les phrases). |

**Valeur de retour** — `None` (modifie les variables globales `model`, `tokenizer`, `token_model`, `device`).

**Comportement** — Détecte automatiquement l'accélération matérielle disponible (`cuda` ou `cpu`). Si `token_mode=True` et que le modèle token n'est pas déjà chargé, charge `intfloat/multilingual-e5-base` et libère la référence au modèle phrase (et inversement). Un seul des deux modèles reste donc chargé en mémoire à la fois.

**Exemple**
```python
load_models(token_mode=False)   # charge BAAI/bge-m3 si pas déjà chargé
```

#### `_ensure_imports()`

**Description** — Importe les bibliothèques lourdes (PyTorch, Transformers) uniquement au démarrage effectif du daemon, pour accélérer l'importation initiale du module parent.

**Paramètres** — Aucun.

**Valeur de retour** — `None` (peuple les variables globales `torch`, `F`, `Tensor`, `AutoTokenizer`, `AutoModel`, `SentenceTransformer`).

#### `all_but_the_top(X, n_components=3)`

**Description** — Applique la technique de réduction de dimensionnalité « All-but-the-top » sur une matrice d'embeddings : retire les composantes principales dominantes, une technique connue pour améliorer la qualité de la similarité cosinus sur certains modèles.

| Paramètre | Type | Description |
|---|---|---|
| `X` | `numpy.ndarray \| torch.Tensor` | La matrice d'embeddings à traiter. |
| `n_components` | `int` | Le nombre de composantes principales à retirer (défaut : 3). |

**Valeur de retour** — Même type que l'entrée (`numpy.ndarray` ou `torch.Tensor`) : la matrice après centrage et projection hors des `n_components` premières composantes principales.

**Comportement** — Centre la matrice (soustraction de la moyenne), calcule une PCA tronquée (`torch.pca_lowrank`) avec `q = min(n_components, n_samples, n_features)`, puis projette les vecteurs hors du sous-espace des `q` premières composantes.

**Exemple**
```python
reduced = all_but_the_top(embeddings, n_components=2)
```

#### `average_pool(last_hidden_states, attention_mask)`

**Description** — Calcule un pooling par moyenne sur les états cachés d'un modèle Transformer, en ignorant les tokens de padding.

| Paramètre | Type | Description |
|---|---|---|
| `last_hidden_states` | `torch.Tensor` | Les états cachés de la dernière couche du modèle. |
| `attention_mask` | `torch.Tensor` | Le masque d'attention associé (1 pour un token réel, 0 pour du padding). |

**Valeur de retour** — `torch.Tensor` : le vecteur moyenné par phrase, masqué des tokens de padding.

**Comportement** — Met à zéro les positions de padding (`masked_fill`), puis divise la somme des états cachés par le nombre de tokens réels (`attention_mask.sum`).

#### `average_pool_last_n_layers(hidden_states, attention_mask, num_layers=4)`

**Description** — Calcule un pooling par moyenne sur les `num_layers` dernières couches cachées d'un modèle Transformer (plutôt que sur la seule dernière couche).

| Paramètre | Type | Description |
|---|---|---|
| `hidden_states` | `tuple[torch.Tensor, ...]` | L'ensemble des couches cachées (`output_hidden_states=True` du modèle). |
| `attention_mask` | `torch.Tensor` | Le masque d'attention associé. |
| `num_layers` | `int` | Le nombre de dernières couches à moyenner (défaut : 4). |

**Valeur de retour** — `torch.Tensor` : le vecteur pondéré par phrase, calculé à partir de la moyenne inter-couches puis du pooling intra-phrase (`average_pool`).

> Cette fonction, et le pooling multi-couches associé, sont documentés dans le code même comme probablement amenés à être supprimés dans une version future, car le mode token est destiné à créer des embeddings de tokens individuels plutôt qu'un pooling agrégé par phrase.

**Exemple**
```python
vec = average_pool_last_n_layers(outputs.hidden_states, attention_mask, num_layers=4)
```

#### `batching_worker()`

**Description** — Thread d'exécution unique. Extrait les éléments en attente dans la file (`batch_queue`), les concatène en un seul lot pour l'encodage, puis redistribue les vecteurs résultants pour reconstruire les listes d'origine.

**Paramètres** — Aucun (boucle infinie, destinée à être lancée comme thread daemon).

**Valeur de retour** — Ne retourne jamais (boucle `while True`).

**Comportement**
1. Bloque en attente du premier élément de la file (`batch_queue.get()`).
2. S'assure que le bon modèle est chargé en mémoire (`load_models`).
3. Accumule d'autres requêtes arrivées dans une fenêtre de temps de `BATCH_WINDOW_S` (15 ms), jusqu'à une taille maximale `MAX_BATCH_SIZE` (256 phrases).
4. Encode le lot complet (branche token ou branche phrase selon `token_mode`).
5. Libère explicitement le cache CUDA (`torch.cuda.empty_cache()`) si un GPU est utilisé, en réponse à des fuites de VRAM constatées lors de l'encodage de plusieurs corpus consécutifs (observables via `nvidia-smi` pendant l'exécution).
6. En cas d'erreur d'encodage (ex. `OutOfMemoryError`), avertit toutes les requêtes du lot via leur file de retour respective plutôt que de faire planter le thread.
7. Redistribue le tenseur de résultats en sous-segments correspondant à chaque requête d'origine.

#### `handle_client(conn)`

**Description** — Gère la communication avec un client connecté. S'exécute dans un thread dédié (un thread par connexion). Ne fait aucun calcul GPU : gère uniquement les entrées/sorties réseau et délègue le travail au worker GPU via `batch_queue`.

| Paramètre | Type | Description |
|---|---|---|
| `conn` | `multiprocessing.connection.Connection` | L'objet de connexion socket du client. |

**Valeur de retour** — `None`.

**Comportement**
1. Reçoit `(job_id, sentences, chunk_size, token_mode)`.
2. Envoie un signal initial de progression (`("progress", 0, total)`).
3. Découpe les phrases en sous-lots de taille `chunk_size` (défini côté client), en soumettant chacun à `batch_queue` et en attendant la réponse du worker (bloquant), puis envoie la progression après chaque sous-lot.
4. Concatène tous les sous-lots en une seule matrice NumPy et envoie le résultat final (`("done", résultat)`).
5. En cas d'erreur, tente d'avertir le client (`("error", message)`) ; journalise un avertissement si l'envoi échoue (connexion coupée).
6. Ferme systématiquement la connexion (`finally`), pour éviter les connexions fantômes.

#### `accept_loop(listener)`

**Description** — Boucle d'acceptation des connexions entrantes sur le `Listener` du daemon.

| Paramètre | Type | Description |
|---|---|---|
| `listener` | `multiprocessing.connection.Listener` | L'objet listener en écoute sur `localhost:6000`. |

**Valeur de retour** — Ne retourne jamais (boucle `while True`).

**Comportement** — Accepte chaque nouvelle connexion et délègue son traitement à un thread dédié (`handle_client`), pour ne jamais bloquer l'arrivée d'autres clients ; ignore et journalise les tentatives de connexion mal formées (`EOFError`, `OSError`).

#### `main()`

**Description** — Boucle principale de démarrage du daemon.

**Paramètres** — Aucun.

**Valeur de retour** — Ne retourne jamais.

**Comportement**
1. Importe PyTorch et Transformers (`_ensure_imports`), différé au démarrage effectif pour ne pas ralentir d'autres scripts qui importeraient ce fichier.
2. Démarre le thread du worker GPU en arrière-plan (`batching_worker`, `daemon=True`).
3. Crée le socket d'écoute IPC (`Listener((HOST, PORT), backlog=BACKLOG)`) sur `localhost:6000`.
4. Lance la boucle d'acceptation des connexions (`accept_loop`).

**Sécurité** — Le daemon écoute uniquement sur `localhost` et n'impose pas d'authentification par clé. Cela signifie que toute application locale sur la même machine peut s'y connecter. Ce n'est pas un problème sur une machine mono-utilisateur dédiée, mais cela mérite d'être noté si le daemon est exécuté sur un serveur partagé.

**Exemple**
```bash
python -m utils.embed_daemon
```

---

## Module `main.py`

Interface en ligne de commande (CLI) pour la recherche sémantique sur corpus linguistiques. Elle orchestre l'utilisation des autres modules (`makeIndex.py`, `calcEmbeddings.py`, `searchEmbedding.py`), centralise toutes les opérations du pipeline (extraction de textes, génération d'embeddings, création d'index FAISS, exécution de requêtes de similarité), et gère un système de cache pour ne recalculer que ce qui est manquant.

### Arguments de la ligne de commande

| Argument | Description |
|---|---|
| `input_file` | Fichier de corpus (`.conllu`, `.xml`, `.trs`), fichier `.faiss` existant, dossier, ou motif wildcard. |
| `query` | La phrase de requête. |
| `--index-type {flat,hnsw,ivfpq}` | Type d'index FAISS à construire (par défaut : `ivfpq`). |
| `--top-k N` | Nombre de résultats à retourner (défaut : 10). |
| `--reduce-precision` | Sauvegarde les embeddings en `float16` pour économiser de l'espace disque. |
| `--force` | Force le recalcul complet (embeddings, métadonnées, index), même si les fichiers en cache existent déjà. |
| `--folder` | Traite un dossier entier plutôt qu'un fichier unique : la requête est effectuée sur tous les fichiers `.faiss` trouvés dans le dossier spécifié. |
| `--encode-only` | Encode et indexe les fichiers d'un dossier sans lancer de recherche (utile pour une phase de pré-calcul). Opère de manière incrémentale : cherche les fichiers corpus et leurs fichiers embeddings/index respectifs, et force le recalcul si ces fichiers sont absents. |
| `--search-only` | Recherche directement sur des index supposés déjà construits (optimisation de vitesse, saute les vérifications de cache). |
| `--regenerate-metadata` | Régénère uniquement les métadonnées JSON, sans toucher aux embeddings ni à l'index (utile après une mise à jour du parseur). |
| `--token-emb` | Active le mode d'encodage par token (`intfloat/multilingual-e5-base`) au lieu du mode phrase. |
| `--no-faiss` | Effectue une recherche par produit scalaire direct sur les `.npy`, sans passer par un index FAISS : sert à valider la fiabilité d'un index FAISS en comparant ses résultats à un calcul exact. |
| `--no-daemon` | Charge les modèles localement dans le processus courant plutôt que de passer par le daemon. |
| `--use-ollama` | Utilise un serveur Ollama pour l'encodage. |
| `--log` / `--warn` | Ajustent le niveau de verbosité des logs de tous les modules du pipeline. `--log` affiche un journal détaillé, `--warn` se limite aux avertissements. |

#### `parse_args()`

**Description** — Définit et analyse les arguments de l'interface en ligne de commande via `argparse`.

**Paramètres** — Aucun (lit `sys.argv`).

**Valeur de retour** — `argparse.Namespace` : un objet contenant tous les arguments parsés (voir tableau ci-dessus).

**Exemple**
```bash
python main.py mon_corpus.conllu "ma phrase de recherche" --top-k 30 --index-type ivfpq --log
```

#### `get_sent_context(f, sent_id, context_size=10)`

**Description** — Récupère le contexte textuel autour d'une phrase spécifique (phrases précédentes et suivantes). Utile pour examiner les correspondances de recherche dans leur environnement discursif d'origine.

| Paramètre | Type | Description |
|---|---|---|
| `f` | `str` | Chemin vers le fichier de métadonnées (`.json`) correspondant au corpus. |
| `sent_id` | `int \| str` | L'identifiant ou l'index de la phrase cible. |
| `context_size` | `int` | Le nombre de phrases à inclure avant et après la cible (fenêtre de contexte, défaut : 10). |

**Valeur de retour** — `str | None` : le bloc de texte concaténé contenant le contexte, ou `None` si introuvable ou hors limites.

**Comportement** — Charge les métadonnées via `load_metadata`, recherche l'index exact correspondant à `sent_id`, concatène les phrases de la fenêtre `[i-context_size : i+context_size]`, puis applique un nettoyage typographique final (`fix_punctuation_spaces`).

**Exemple**
```python
context = get_sent_context("corpus.json", sent_id=134358, context_size=5)
```

#### `process_no_faiss(args)`

**Description** — Effectue une recherche de similarité par produit matriciel direct (dot product) sur les vecteurs normalisés L2 (équivalent de la similarité cosinus), sans utiliser d'index FAISS. Sert à vérifier la fiabilité des recherches basées sur FAISS en comparant leurs résultats à une recherche directe dans les embeddings.

| Paramètre | Type | Description |
|---|---|---|
| `args` | `argparse.Namespace` | Les arguments de la ligne de commande (voir `parse_args`). |

**Valeur de retour** — `bool` : `False` si `args.no_faiss` n'est pas activé (fonction non applicable), sinon complète l'exécution par un affichage dans la console et ne renvoie pas explicitement de valeur en fin de fonction.

**Comportement**
1. Encode la requête (`embedd_query`).
2. **Mode dossier/wildcard** : parcourt tous les fichiers `.conllu`/`.xml`/`.trs` correspondants, charge leurs embeddings bruts (`load_embeddings`), normalise en L2, calcule le produit matriciel `query_embs @ embs.T`, ignore les fichiers introuvables ou en incompatibilité de dimension.
3. **Mode fichier unique** : même logique sur le seul fichier fourni.
4. Trie tous les résultats agrégés par score décroissant et affiche un tableau récapitulatif, avec le temps d'exécution total.

**Exemple**
```bash
python main.py "test/*" "pas de souci" --top-k 30 --no-faiss --log
```

#### `process(args, index, metric_type=faiss.METRIC_INNER_PRODUCT, metadata=None)`

**Description** — Exécute une requête de recherche sémantique sur un index FAISS unique et affiche les résultats.

| Paramètre | Type | Description |
|---|---|---|
| `args` | `argparse.Namespace` | Objet arguments de la ligne de commande contenant la requête et les options. |
| `index` | `faiss.Index` | L'index FAISS chargé en mémoire. |
| `metric_type` | `int` | Type de métrique de distance (défaut : produit scalaire). |
| `metadata` | `dict` | Métadonnées associées aux vecteurs (ID, phrases, tokens). |

**Valeur de retour** — `None` : affiche les résultats dans la console.

**Comportement** — Appelle `search()` du module `searchEmbedding.py` avec les options extraites de `args` (requête, `top_k`, `token_mode`, `no_daemon`), mesure le temps d'exécution, puis affiche chaque résultat.

**Exemple**
```python
process(args, index=index, metadata=metadata)
```

#### `process_folder(args, input_file)`

**Description** — Exécute une recherche sémantique sur un répertoire entier ou un lot de fichiers (via un wildcard).

| Paramètre | Type | Description |
|---|---|---|
| `args` | `argparse.Namespace` | Arguments de la ligne de commande. |
| `input_file` | `str` | Chemin du dossier cible ou motif de recherche (ex. `corpus/*.conllu`). |

**Valeur de retour** — `None`.

**Comportement**
1. **Encodage unique de la requête** pour gagner du temps lors du parcours des multiples fichiers (`embedd_query`) — optimisation majeure du mode dossier.
2. Récupère les fichiers `.faiss` du dossier (ou du motif wildcard).
3. **Mode `--force`** : recalcule tout le dossier (embeddings et index, via `encode_folder` et `makeIndex_folder`) avant la recherche.
4. Lance la recherche globale sur le répertoire avec le vecteur pré-calculé (`search_folder`).

**Exemple**
```python
process_folder(args, "test/*")
```

#### `main()`

**Description** — Point d'entrée principal du script CLI. Orchestre la logique conditionnelle du pipeline : analyse des arguments, détermination de l'état du cache, routage vers un mode d'exécution spécifique, gestion du traitement par lots ou par fichier unique, reconstruction des éléments manquants, puis lancement de la requête finale.

**Paramètres** — Aucun (lit `sys.argv` via `parse_args()`).

**Valeur de retour** — Ne retourne pas de valeur exploitée (utilisée en `if __name__ == "__main__"`).

**Comportement — les 7 modes d'exécution, dans cet ordre de priorité :**

1. **`--no-faiss`** : recherche directe par produit scalaire sur les fichiers `.npy`, sans FAISS (mode de validation/débogage) → `process_no_faiss`.
2. **`--regenerate-metadata`** : reparse le(s) fichier(s) source(s) et réécrit uniquement le(s) fichier(s) `.json`, sans toucher aux embeddings ni à l'index.
3. **Fichier `.faiss` fourni directement** : charge l'index et les métadonnées correspondantes, puis lance directement la recherche (`process`).
4. **`--encode-only`** (sur dossier ou wildcard) : encode et indexe tous les fichiers manquants du dossier, étape par étape (métadonnées → embeddings → index), sans lancer de recherche.
5. **`--search-only`** (sur dossier) : suppose que tous les index existent déjà et lance directement la recherche, pour un gain de vitesse en évitant les vérifications de cache.
6. **Mode dossier ou wildcard standard** (`--folder` ou motif `*` dans `input_file`) : encode si nécessaire (en mode `--force`), puis effectue une recherche agrégée sur tous les index du dossier (`process_folder`).
7. **Mode fichier unique standard** : applique la logique de cache incrémental ci-dessous, puis lance la recherche sur ce fichier (`process`).

**Système de cache incrémental (mode 7)** — Pour un fichier corpus unique, `main()` déduit les chemins des fichiers dérivés attendus (`.npy`, `.json`, `.faiss`) et applique la logique suivante :

1. Si les **embeddings et métadonnées existent déjà**, et pas `--force` → ils sont chargés directement, on ne reconstruit que l'index s'il manque.
2. Si le fichier fourni a une extension **non reconnue** et qu'aucun fichier dérivé n'existe → le script s'arrête avec une erreur explicite.
3. Si **embeddings et index sont absents**, ou `--force` est utilisé → tout est reconstruit depuis le fichier source (parsing → embeddings → métadonnées → index).
4. Sinon → les embeddings bruts (`.npy`) sont rechargés depuis le disque pour ne reconstruire que l'index manquant.

**Garde-fou d'intégrité** : une fois l'index chargé (ou reconstruit), `main()` vérifie que le nombre de vecteurs de l'index (`index.ntotal`) correspond bien au nombre d'entrées des métadonnées (`len(metadata["raw_text"])`). En cas de désaccord, signe probable d'une incohérence entre un ancien index et de nouvelles métadonnées régénérées séparément, le script s'arrête avec un message d'erreur explicite invitant à relancer avec `--force`, plutôt que de retourner silencieusement des résultats erronés ou décalés.

**Alternative automatique IVFPQ → flat** : si `makeIndex` signale une `ValueError` parce que le corpus est trop petit pour un entraînement IVFPQ fiable, `main()` intercepte l'erreur et reconstruit automatiquement un index `flat` à la place, en avertissant l'utilisateur.

> Un seul appel à `main.py` sur un fichier unique suffit à traverser toutes les étapes manquantes du pipeline en une seule exécution.

**Exemple**
```bash
python main.py mon_corpus.conllu "ma phrase de recherche"
```

---

## Choix de conception et garde-fous

Cette section récapitule les décisions structurantes du projet et les raisons qui les motivent :

- **Daemon persistant plutôt que rechargement systématique du modèle** : le coût de chargement d'un modèle Transformer (plusieurs secondes à dizaines de secondes) est trop élevé pour être payé à chaque script et à chaque requête ; le daemon amortit ce coût sur toute une session de travail.
- **Traitement par lots dynamique côté daemon** (fenêtre de 15 ms, taille max 256) : permet de mutualiser efficacement le GPU entre plusieurs requêtes concurrentes sans imposer à un client isolé d'attendre qu'un très gros lot d'un autre client soit terminé.
- **Chargement fainéant des modèles** (phrase vs token) : chaque modèle n'est chargé que lorsqu'il est demandé par le client ; une fois chargé, il est gardé en mémoire pour traiter les requêtes futures. Cela permet d'économiser de la VRAM.
- **Libération explicite du cache CUDA après chaque lot** : réponse directe à des fuites de mémoire GPU constatées lors de l'encodage de plusieurs corpus consécutifs.
- **Normalisation L2 systématique et symétrique** (côté indexation *et* côté requête) : condition nécessaire pour que la métrique `METRIC_INNER_PRODUCT` de FAISS soit mathématiquement équivalente à une similarité cosinus.
- **Valeur par défaut de `m=256` pour `ivfpq`** (paramètre exposé dans `makeIndex()`) : choix directement issu de tests comparatifs contre une recherche exacte, qui ont montré que `nprobe` n'avait aucun impact mesurable sur la qualité des résultats, alors que `m` en avait un très net — voir [Tests](#tests--impact-de-nprobe-et-de-m).
- **Bascule automatique IVFPQ → flat** lorsque le corpus est trop petit pour un entraînement statistiquement fiable : évite de construire silencieusement un index de mauvaise qualité, au prix d'un avertissement explicite dans les logs.
- **Vérification d'intégrité index/métadonnées** (`index.ntotal == len(metadata)`) avant toute recherche : détecte les désynchronisations entre fichiers dérivés générés à des moments différents.
- **Cache incrémental à trois niveaux** (embeddings → métadonnées → index) : évite de recalculer des embeddings coûteux en GPU/CPU si seule l'étape d'indexation ou de reconstruction des métadonnées doit être rejouée.
- **Suffixe `_token` systématique** sur les fichiers dérivés du mode token : évite d'écraser accidentellement les embeddings/index « phrase » d'un même corpus.
- **Mode `--no-faiss` de validation** : permet de comparer, sur un même corpus, les résultats d'une recherche FAISS (potentiellement approximative selon le type d'index) à un calcul de similarité cosinus exact, pour détecter une éventuelle perte de qualité liée à l'indexation.
- **Mécanisme progressif du parseur CoNLL-U** (en-têtes rapides → reconstruction par concaténation des formes) : maximise la robustesse face à des fichiers de corpus hétérogènes ou partiellement annotés.
- **Étiquetage explicite des phrases manquantes** (`[phrase manquante]`) plutôt que leur suppression : préserve l'alignement entre indices FAISS, métadonnées et phrases d'origine.

---
