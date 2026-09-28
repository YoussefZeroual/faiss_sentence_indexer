# faiss_sentence_indexer

Outil de recherche sémantique pour corpus linguistiques et littéraires (formats **CoNLL-U**, **XML** et **TRS**), basé sur des embeddings de phrases (ou de tokens) indexés avec **FAISS**.
Deux modes d'encodage sont disponibles : le mode **phrase** (un vecteur par phrase) et le mode **token** (un vecteur par **lemme**, obtenu en moyennant les embeddings contextuels de tous les tokens qui partagent ce lemme dans un fichier). Voir [Modes d'encodage](#modes-dencodage--phrase-et-token).

Le pipeline permet de :

1. **parser** les phrases (ou tours de parole) d'un fichier de corpus (CoNLL-U, XML ou TRS) ;
2. **calculer leurs embeddings** via un modèle d'encodage (local, via un daemon persistant, ou via Ollama) ;
3. **construire un index FAISS** optimisé pour la recherche par similarité cosinus ;
4. **interroger** cet index avec une phrase de requête et récupérer les résultats les plus proches sémantiquement.

---

## Sommaire

- [Architecture générale](#architecture-générale)
- [Modes d'encodage : phrase et token](#modes-dencodage--phrase-et-token)
- [Prérequis et installation](#prérequis-et-installation)
- [Structure du projet](#structure-du-projet)
- [Module `calcEmbeddings.py`](#module-calcembeddingspy)
- [Module `makeIndex.py`](#module-makeindexpy)
- [Tests : impact de `nprobe` et de `m`](#tests--impact-de-nprobe-et-de-m)
- [Module `searchEmbedding.py`](#module-searchembeddingpy)
- [Module `utils/embed_client.py`](#module-utilsembed_clientpy)
- [Module `utils/embed_daemon.py`](#module-utilsembed_daemonpy)
- [Module `main.py`](#module-mainpy)
- [Exemples de commandes (cas d'usage)](#exemples-de-commandes-cas-dusage)
- [Choix de conception et garde-fous](#choix-de-conception-et-garde-fous)
- [Limitations connues et points d'attention](#limitations-connues-et-points-dattention)

---

## Architecture générale

Le projet est organisé en briques indépendantes qui communiquent via des fichiers intermédiaires (`.npy`, `.json`, `.faiss`) et via un processus daemon en arrière-plan pour l'encodage optimisé des phrases :

```
corpus.conllu / corpus.xml / corpus.trs
              │
              ▼
    calcEmbeddings.py ──► corpus.npy (embeddings de phrases)
              │             corpus.json (métadonnées : sent_id, raw_text, tokens, lemmas)
              │             mode token : corpus_token.npy (un vecteur par lemme),
              │                          corpus_token_lemma_index.json (lemme de chaque ligne),
              │                          corpus_token_abtt.json (paramètres ABTT, sauf --no-abtt)
              │
              │   (appelle utils/embed_client.py, qui délègue
              │    l'encodage au daemon, à Ollama, ou au modèle local)
              ▼
      makeIndex.py ──► corpus.faiss (index FAISS pour le mode phrase, ou corpus_token.faiss pour le mode token)
              │
              ▼
   searchEmbedding.py ──► résultats de la recherche sémantique (top-k phrases similaires, ou top-k lemmes en mode token)
```

`main.py` orchestre l'ensemble de la chaîne de façon **incrémentale** : il ne recalcule que ce qui manque encore sur le disque (embeddings, métadonnées ou index), et propose de nombreux modes d'exécution (fichier unique, dossier complet, encodage seul, recherche seule, mode sans FAISS pour le débogage, régénération des métadonnées, etc.).

Le calcul des embeddings peut être délégué à un **daemon persistant en arrière-plan** (`utils/embed_daemon.py`), afin d'éviter de recharger un modèle lourd (PyTorch / Transformers) à chaque appel de script. Le client (`utils/embed_client.py`) démarre ce daemon si besoin et communique avec lui par socket local ; il peut aussi, selon les besoins, s'en passer complètement (mode local synchrone) ou déléguer l'encodage à un serveur **Ollama** externe.

---

## Modes d'encodage : phrase et token

| | Mode phrase (par défaut) | Mode token (`--token-emb` / `token_mode=True`) |
|---|---|---|
| **Unité indexée** | une phrase (ou un tour de parole) = un vecteur | un **lemme unique** du fichier = un vecteur |
| **Modèle** | `BAAI/bge-m3` (`SentenceTransformer`, 1024 dimensions) | `intfloat/multilingual-e5-base` (`transformers`, 768 dimensions) |
| **Fichiers dérivés** | `corpus.npy`, `corpus.json`, `corpus.faiss` | `corpus_token.npy`, `corpus_token_lemma_index.json`, `corpus_token_abtt.json` (sauf `--no-abtt`), `corpus_token.faiss` ; `corpus.json` est **partagé** avec le mode phrase |
| **Requête** | phrase encodée en un vecteur | requête encodée mot par mot puis **moyennée** en un seul vecteur, puis transformée par ABTT avec les paramètres du corpus |
| **Résultat** | `(sent_id, phrase, score)` | `(lemme, score, [(token, sent_id), ...])` : le lemme trouvé, et où il apparaît dans le corpus |

### Chaîne de traitement du mode token

1. **Parsing** : en plus de `sent_id`, `raw_text` et `tokens`, le parseur extrait la liste des **lemmes** de chaque phrase (colonne `LEMMA` du CoNLL-U ; `simplemma` en repli pour XML sans CoNLL-U imbriqué et pour TRS). Les listes `tokens` et `lemmas` d'une phrase sont alignées position par position.
2. **Encodage contextuel** : les tokens de chaque phrase sont joints par des espaces (les espaces internes à un token, ex. `quand même`, sont remplacés par `_` pour qu'il reste un seul mot pour le tokenizer) et envoyés au modèle E5. On moyenne les 4 dernières couches cachées, puis les sous-mots d'un même mot sont moyennés (`merge_subwords_to_words`). On obtient un tableau `(n_mots, 768)` par phrase.
3. **Regroupement par lemme** (`build_lemma_embeddings`) : dans un fichier de corpus, tous les vecteurs de tokens partageant le même lemme sont moyennés en un seul vecteur. La liste triée des lemmes uniques est sauvegardée dans `_lemma_index.json` (la position dans cette liste = la ligne dans la matrice = l'identifiant dans l'index FAISS).
4. **ABTT** (« All-but-the-top ») : par défaut, on centre la matrice de lemmes et on retire ses 3 premières composantes principales. La moyenne et les composantes sont sauvegardées dans `_abtt.json`, car la **requête doit subir exactement la même transformation** pour se trouver dans le même espace que le corpus. `--no-abtt` désactive cette étape.
5. **Indexation** : `makeIndex` construit `corpus_token.faiss` à partir de `corpus_token.npy` (avec normalisation L2 comme en mode phrase).
6. **Recherche** : la requête est encodée en mode token, moyennée, transformée par ABTT, normalisée, puis cherchée dans l'index. Les lemmes trouvés sont ensuite mis en correspondance avec leurs occurrences (`token`, `sent_id`) grâce à `corpus.json`.

> **Migration** : si vos fichiers `.json` ont été générés avant l'ajout du mode token, ils ne contiennent pas la clé `lemmas`. Régénérez-les avec `--regenerate-metadata` (ou `--force`) avant toute recherche en mode token.

---

## Prérequis et installation

- Python 3.9+ (le code utilise des annotations de type modernes, ex. `tuple[Tensor, ...]`)
- Bibliothèques principales :
  ```bash
  pip install faiss-cpu numpy torch transformers sentence-transformers conllu lxml requests simplemma pandas tqdm
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
| `calcEmbeddings.py` | Parse un fichier de corpus (CoNLL-U, XML ou TRS), nettoie le texte, et calcule les embeddings de chaque phrase ou tour de parole (en mode token : de chaque lemme, avec transformation ABTT). |
| `makeIndex.py` | Construit un index FAISS (`flat`, `hnsw` ou `ivfpq`) à partir des embeddings. |
| `searchEmbedding.py` | Charge un index et ses métadonnées, encode une requête, et effectue la recherche de similarité (fichier unique ou dossier entier), en mode phrase ou en mode token (lemmes). |
| `utils/embed_client.py` | Interface cliente vers l'encodage : trois stratégies possibles (daemon IPC, exécution locale directe, ou API Ollama). |
| `utils/embed_daemon.py` | Processus serveur qui charge les modèles d'encodage une seule fois, regroupe les requêtes en lots dynamiques, et les traite au fil de l'eau. |

Les sections suivantes documentent chaque module fonction par fonction: signature, paramètres, valeur de retour, comportement détaillé et un exemple d'appel.

---

## Module `calcEmbeddings.py`

Module d'extraction et d'encodage de corpus linguistiques pour la création des fichiers embeddings `.npy`. Il analyse des fichiers de corpus (CoNLL-U, XML, TRS pour les transcriptions de corpus oraux) et les convertit en représentations vectorielles normalisées. Il nettoie les textes bruts, gère certaines spécificités linguistiques (les mots amalgames ex. du --> de+le), et orchestre l'appel aux modèles d'encodage. Il extrait aussi les **lemmes** de chaque phrase (colonne `LEMMA` du CoNLL-U, ou `simplemma` en repli) et sait construire des embeddings **par lemme** pour le mode token.

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

**Description** — Reconstruit le texte brut à partir des formes de surface d'un bloc CoNLL-U. Gère spécifiquement les amalgames en ignorant leurs composants enfants pour éviter les doublons (cette fonction est utilisée par `parse_conllu_fast` et par `parse_sentences_xml_conllu` pour le CoNLL-U imbriqué en XML).

| Paramètre | Type | Description |
|---|---|---|
| `text` | `str` | Un bloc CoNLL-U correspondant à une phrase. |

**Valeur de retour** — `tuple[str, list]` : `(raw_text, tokens)`. `raw_text` est le texte reconstruit puis nettoyé par `fix_punctuation_spaces` ; `tokens` est la liste des formes de surface, **toujours calculée** (avec ou sans amalgame), à raison d'un token par mot de surface. Cette liste est alignée position par position avec celle de `get_lemmas`, ce qui est indispensable en mode token.

**Comportement**
- Si le bloc contient des amalgames (`has_amalgams`), les lignes correspondant aux sous-composantes des amalgames sont ignorées (comparaison des deux lignes précédentes via `is_amalgame`), puis les formes de surface sont extraites par regex sur les lignes restantes.
- Sinon, extraction directe des formes de surface sur chaque ligne d'annotation.
- La colonne `FORM` est capturée avec `[^\t\n]+` et non `\S+` : certaines formes contiennent un espace interne (`quand même`, `est-ce que`), et les colonnes CoNLL-U sont séparées par des tabulations, pas par n'importe quel espace. `\S+` tronquait ces formes au premier espace, ce qui décalait l'alignement avec les lemmes.

**Exemple**
```python
raw_text, tokens = concat_forms("1-2\tdu\t_\t_\n2\tde\tDE\t_\n3\tle\tLE\t_\n4\tchat\tNOUN\t_")
# raw_text == "du chat"  (les sous-composantes 'de'/'le' de la ligne 1-2 sont ignorées)
# tokens   == ["du", "chat"]
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

**Description** — Sépare un texte en tokens en gérant les apostrophes. Utilisée pour les formats **sans colonne `FORM`** (XML à texte simple, TRS) afin de remplir la clé `tokens` du fichier JSON de métadonnées. Pour le CoNLL-U, les tokens viennent directement de `concat_forms`.

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

#### `get_lemmas(text)`

**Description** — Extrait la liste des lemmes d'un bloc CoNLL-U brut (colonne `LEMMA`), alignée mot de surface par mot de surface avec `concat_forms`.

| Paramètre | Type | Description |
|---|---|---|
| `text` | `str` | Bloc CoNLL-U brut d'une phrase (comme reçu par `concat_forms`). |

**Valeur de retour** — `list[str]` : un lemme par mot de surface, dans le même ordre que `tokens`.

**Comportement**
- Ignore les sous-composantes des amalgames, exactement comme `concat_forms`.
- Pour une ligne d'amalgame, la colonne `LEMMA` vaut `_` (non renseignée) : c'est alors sa propre forme de surface (`FORM`) qui sert de lemme de substitution.
- Les colonnes sont capturées avec `[^\t\n]+` (voir `concat_forms`) pour supporter les valeurs contenant un espace interne.

**Exemple**
```python
get_lemmas("1-2\tdu\t_\t_\n2\tde\tde\t_\n3\tle\tle\t_\n4\tchats\tchat\t_")
# -> ["du", "chat"]
```

#### `get_lemmas_fast(text, lang='fr')`

**Description** — Lemmatise un texte brut (sans annotation CoNLL-U) avec `simplemma`. Solution de repli rapide pour les formats sans colonne `LEMMA` (XML sans CoNLL-U imbriqué, TRS). Moins précise qu'une vraie lemmatisation morphosyntaxique (pas de désambiguïsation par catégorie grammaticale), mais suffisante comme approximation.

| Paramètre | Type | Description |
|---|---|---|
| `text` | `str` | Texte brut à lemmatiser. |
| `lang` | `str` | Code langue `simplemma` (défaut : `'fr'`). |

**Valeur de retour** — `list[str] | None` : la liste des lemmes, ou `None` si le texte est vide ou vaut `[phrase manquante]`.

**Comportement** — Journalise un avertissement à chaque appel (lemmatisation approximative) puis appelle `simplemma.text_lemmatizer`.

> ⚠️ `simplemma.text_lemmatizer` segmente le texte à sa manière (la ponctuation devient un élément à part), alors que `tokens` provient de `get_tokens`. Les deux listes peuvent donc avoir des longueurs différentes ; en mode token, `build_lemma_embeddings` ignore alors la phrase concernée avec un avertissement (voir [Limitations](#limitations-connues-et-points-dattention)).

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

**Description** — Analyse un fichier ou un texte au format CoNLL-U pour extraire les phrases et leurs métadonnées.

| Paramètre | Type | Description |
|---|---|---|
| `file_path` | `str` | Le chemin d'accès au fichier CoNLL-U. |
| `text` | `str`, optionnel | Contenu textuel direct (utile si le contenu du fichier a déjà été récupéré, ou est imbriqué dans un XML). |

**Valeur de retour** — `tuple[list, dict]` : `(sent_list, metadata)`, où `sent_list` est la liste des phrases en texte brut, et `metadata` est un dictionnaire `{"sent_id": [...], "raw_text": [...], "tokens": [...], "lemmas": [...]}`.

**Comportement** — Pour chaque bloc de phrase (`parse_conllu_raw_entries`) :
1. Le texte et les tokens sont **toujours reconstruits à partir des formes de surface** (`concat_forms`, amalgames inclus). L'ancien chemin rapide par l'en-tête `# text_raw` et son alternative automatique ont été retirés : les tokens et les lemmes doivent venir de la même source pour rester alignés.
2. Les lemmes sont extraits par `get_lemmas`.
3. L'identifiant vient de `# sent_id = ...` ; s'il est absent, l'indice de la phrase dans le fichier est utilisé.
4. Le texte est nettoyé par `clean_sentence` (les phrases vides deviennent `[phrase manquante]`).

**Exemple**
```python
sent_list, metadata = parse_conllu_fast("corpus.conllu")
# metadata["sent_id"][0]  -> "s1"
# metadata["raw_text"][0] -> "Bonjour, comment allez-vous ?"
# metadata["tokens"][0]   -> ["Bonjour", ",", "comment", "allez", "-vous", "?"]
# metadata["lemmas"][0]   -> ["bonjour", ",", "comment", "aller", "vous", "?"]
```

#### `parse_sentences_xml_conllu(filepath)`

**Description** — Analyse un fichier XML (simple, ou avec du CoNLL-U imbriqué dans des balises `<s>`) pour extraire les phrases et générer les métadonnées associées.

| Paramètre | Type | Description |
|---|---|---|
| `filepath` | `str` | Le chemin d'accès au fichier XML à analyser. |

**Valeur de retour** — `tuple[list, dict]` : `(sent_list, metadata)`, identique en structure à `parse_conllu_fast` (`sent_id`, `raw_text`, `tokens`, `lemmas`). `sent_id` vient de l'attribut `id` de `<s>`.

**Comportement** — Le parseur (basé sur `lxml`, en mode tolérant aux erreurs `recover=True`) cible toutes les balises `<s>` via XPath et gère trois cas de figure, par ordre de détection :

| Cas | Détection | Texte | Tokens | Lemmes |
|---|---|---|---|---|
| 1. CoNLL-U imbriqué | texte multi-lignes dans `<s>` | `concat_forms` puis `fix_punctuation_spaces` | `concat_forms` | `get_lemmas` (colonne `LEMMA`) |
| 2. Contenu fragmenté | pas de texte direct dans `<s>` (sous-balises, ex. `<w>`) | `s.itertext()` | **`None`** | `get_lemmas_fast` (`simplemma`) |
| 3. Texte simple | texte direct dans `<s>` | tel quel | `get_tokens` | `get_lemmas_fast` (`simplemma`) |

Dans le cas 2, l'absence de tokens est journalisée ; en mode token, une phrase sans tokens est encodée comme une chaîne vide et n'apporte donc aucun vecteur.

**Exemple**
```python
sent_list, metadata = parse_sentences_xml_conllu("corpus.xml")
```

#### `parse_sentence_trs(file_path=None)`

**Description** — Analyse un fichier de transcription audio au format TRS (Transcriber) pour extraire les tours de parole.

| Paramètre | Type | Description |
|---|---|---|
| `file_path` | `str` | Le chemin d'accès au fichier TRS à analyser. |

**Valeur de retour** — `tuple[list, dict]` : `(sent_list, metadata)`, où `metadata["sent_id"]` contient les valeurs de l'attribut `startTime` de chaque `<Turn>`, et où `metadata` contient aussi `tokens` (via `get_tokens`) et `lemmas` (via `get_lemmas_fast`, faute de colonne `LEMMA` dans une transcription non annotée).

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
| `mode` | `str`, optionnel | Le format du fichier (`"conllu"`, `"xml"`, `"trs"`). S'il vaut `None`, il est déduit de l'extension de `file_path` ; s'il est fourni, il est utilisé tel quel. |

**Valeur de retour** — `tuple[list, dict]` : `(sent_list, metadata)`, ou `(None, None)` si le format du fichier n'est pas reconnu.

**Comportement** — Redirige, selon le format (fourni ou déduit de l'extension : `conllu`, `xml`, `trs`), vers `parse_conllu_fast`, `parse_sentences_xml_conllu` ou `parse_sentence_trs`, puis journalise le nombre de phrases et de tokens extraits (comptés par regex `\w+|[^\w\s]`, en ignorant les phrases marquées `[phrase manquante]`) ainsi que le temps d'exécution.

**Exemple**
```python
sent_list, metadata = parse_sentences("corpus.conllu")
if sent_list is None:
    print("Format de fichier non reconnu")
```

### Fonctions d'encodage et de sauvegarde

#### `all_but_the_top(X, n_components=3)`

**Description** — Applique la technique « All-but-the-top » à une matrice d'embeddings : centre la matrice puis retire ses `n_components` premières composantes principales (correction de l'anisotropie des espaces d'embeddings, alternative au *whitening*). **Utilisée uniquement en mode token**, une fois par fichier de corpus, sur la matrice des lemmes.

| Paramètre | Type | Description |
|---|---|---|
| `X` | `numpy.ndarray \| torch.Tensor` | La matrice d'embeddings à traiter (une ligne par lemme). |
| `n_components` | `int` | Le nombre de composantes principales à retirer (défaut : 3). |

**Valeur de retour** — `tuple` : `(X_transformée, mu, P)`.
- `X_transformée` : même type que l'entrée ;
- `mu` : `numpy.ndarray`, la moyenne soustraite (dimension `hidden_dim`) ;
- `P` : `numpy.ndarray` `(hidden_dim, q)`, les composantes retirées (ou `None` si `q = 0`).

**Comportement** — Soustrait la moyenne, calcule une PCA tronquée (`torch.pca_lowrank`, avec `q = min(n_components, n_samples, n_features)`), puis retire de chaque vecteur sa projection sur le sous-espace des `q` premières composantes. Contrairement à la version du daemon (`utils/embed_daemon.py`), celle-ci **renvoie `mu` et `P`** : ils sont sauvegardés dans `_abtt.json` pour être réappliqués à la requête (voir `apply_abtt` dans `searchEmbedding.py`).

> ⚠️ Cette fonction est volontairement appliquée à la matrice des **lemmes** et non aux embeddings individuels de tokens : la matrice complète des tokens d'un grand corpus demanderait une quantité de mémoire prohibitive.

#### `calcEmbeddings(collection_file_path=None, output_file_path=None, mode=None, reduce_precision=False, overwrite=False, token_mode=False, no_daemon=False, use_ollama=False, ollama_host='localhost:11434', ollama_model=None, apply_abtt=True)`

**Description** — Fonction principale du module : extrait les phrases d'un fichier de corpus et génère leurs embeddings correspondants (un vecteur par phrase en mode phrase, un vecteur par lemme en mode token). Intègre un système de cache : si les fichiers de sortie existent déjà, ils sont chargés directement.

| Paramètre | Type | Description |
|---|---|---|
| `collection_file_path` | `str` | Le chemin vers le fichier de corpus source (`.conllu`, `.xml`, `.trs`). |
| `output_file_path` | `str` | Le chemin de destination du fichier d'embeddings (`.npy`). En mode token, le suffixe `_token` est ajouté automatiquement s'il est absent. |
| `mode` | `str` | Le format du corpus (`"conllu"`, `"xml"`, `"trs"`). Par défaut `None` : déduit de l'extension. |
| `reduce_precision` | `bool` | Si `True`, sauvegarde les embeddings en `float16` pour économiser de l'espace disque. |
| `overwrite` | `bool` | Si `True`, force le recalcul même si les fichiers de sortie existent déjà. |
| `token_mode` | `bool` | Si `True`, encode au niveau des tokens puis regroupe par lemme (voir [Modes d'encodage](#modes-dencodage--phrase-et-token)). |
| `no_daemon` | `bool` | Si `True`, exécute le modèle localement au lieu du processus daemon. |
| `use_ollama` | `bool` | Si `True`, délègue l'encodage à une API Ollama externe (**incompatible avec le mode token**, Ollama ne renvoie pas de vecteurs par mot). |
| `ollama_host` | `str` | L'adresse du serveur Ollama. |
| `ollama_model` | `str` | Le nom du modèle Ollama à utiliser. |
| `apply_abtt` | `bool` | Mode token uniquement : si `True` (défaut), applique `all_but_the_top` à la matrice de lemmes et écrit `_abtt.json`. Si `False`, aucun `_abtt.json` n'est écrit (et un fichier existant est supprimé). |

**Valeur de retour** — `tuple[numpy.ndarray, dict]` : `(embeddings, metadata)`.
- Mode phrase : matrice `(n_phrases, 1024)`.
- Mode token : matrice `(n_lemmes_uniques, 768)`, une ligne par lemme unique du fichier (et non par phrase ou par mot).
- `metadata` : dictionnaire `sent_id`, `raw_text`, `tokens`, `lemmas`.

**Comportement**
1. **Chemins** : en mode token, calcule le chemin `..._token.npy`, ainsi que `..._token_lemma_index.json` et `..._token_abtt.json`.
2. **Vérification du cache** : si `overwrite=False` et que le `.npy` et le `.json` existent, ils sont chargés depuis le disque. En mode token, le cache n'est valide que si `_lemma_index.json` existe **et** si la présence de `_abtt.json` correspond à `apply_abtt` (changer `--no-abtt` invalide donc le cache).
3. Sinon, parse les phrases et métadonnées via `parse_sentences`.
4. **Mode token** :
   1. chaque phrase devient la chaîne de ses tokens séparés par des espaces (les espaces internes à un token sont remplacés par `_`) ;
   2. `encode(..., token_mode=True, chunk_size=64)` renvoie une liste de tableaux `(n_mots, 768)`, un par phrase ;
   3. `build_lemma_embeddings` regroupe et moyenne par lemme ;
   4. si `apply_abtt`, `all_but_the_top` est appliquée et `_abtt.json` (`mean`, `components`) est écrit ;
   5. `float16` si `reduce_precision`, puis sauvegarde de `_token.npy` et de `_token_lemma_index.json`.
5. **Mode phrase** : `encode(..., chunk_size=64)` puis sauvegarde en `float16` si `reduce_precision`, `float32` sinon.

**Exemple**
```python
# mode phrase
embeddings, metadata = calcEmbeddings(
    collection_file_path="corpus.conllu",
    output_file_path="corpus.npy",
    mode="conllu",
)

# mode token : produit corpus_token.npy, corpus_token_lemma_index.json, corpus_token_abtt.json
embeddings, metadata = calcEmbeddings(
    collection_file_path="corpus.conllu",
    output_file_path="corpus.npy",
    token_mode=True,
)
```

#### `build_lemma_embeddings(embeddings, lemmas_per_sentence)`

**Description** — Regroupe les embeddings de tokens par lemme, à l'échelle d'un fichier de corpus, puis moyenne les vecteurs de chaque groupe pour produire un seul embedding par lemme unique.

| Paramètre | Type | Description |
|---|---|---|
| `embeddings` | `list` | Une entrée par phrase, chacune un tableau `(n_tokens_phrase, hidden_dim)`, tel que renvoyé par l'encodage en mode token. |
| `lemmas_per_sentence` | `list` | `metadata["lemmas"]`, une liste de lemmes par phrase, alignée token à token avec `embeddings`. |

**Valeur de retour** — `tuple[list, numpy.ndarray]` : `(lemma_list, lemma_embeddings)`. `lemma_list` contient les lemmes uniques **triés** (l'indice dans cette liste correspond à la ligne dans `lemma_embeddings`) ; `lemma_embeddings` est une matrice `float32` `(n_lemmes_uniques, hidden_dim)`.

**Comportement** — Ignore (avec un avertissement `mismatch`) les phrases dont le nombre de vecteurs diffère du nombre de lemmes, les phrases sans vecteurs ou sans lemmes, et les lemmes valant `None` ou `_`.

**Exemple**
```python
lemma_list, lemma_embs = build_lemma_embeddings(token_embeddings, metadata["lemmas"])
# lemma_list[0] -> "a" ; lemma_embs[0] = moyenne de tous les vecteurs de tokens dont le lemme est "a"
```

#### `save_metadata(metadata, output_file=None, token_mode=False)`

**Description** — Sauvegarde le dictionnaire de métadonnées dans un fichier JSON.

| Paramètre | Type | Description |
|---|---|---|
| `metadata` | `dict` | Le dictionnaire contenant les métadonnées extraites (`sent_id`, `raw_text`, `tokens`, `lemmas`). |
| `output_file` | `str` | Le chemin d'accès au fichier cible (généralement `.json`). |
| `token_mode` | `bool` | Paramètre conservé pour des raisons de compatibilité de signature de la fonction (non utilisé dans le corps actuel). |

**Valeur de retour** — `None`.

**Comportement** — Ouverture du fichier en écriture avec encodage UTF-8 (indispensable pour préserver correctement les accents et caractères spéciaux français), puis sérialisation JSON du dictionnaire. Le fichier est **commun aux modes phrase et token**.

**Exemple**
```python
save_metadata(metadata, "corpus.json")
```

#### `encode_folder(input_folder=None, overwrite=False, token_mode=False, no_daemon=False, use_ollama=False, ollama_host='localhost:11434', ollama_model=None, apply_abtt=True)`

**Description** — Parcourt un répertoire ou un wildcard (ex. `test/*Camus*`) pour traiter en lot des fichiers de corpus, générer leurs embeddings et sauvegarder leurs métadonnées.

| Paramètre | Type | Description |
|---|---|---|
| `input_folder` | `str` | Le chemin du répertoire cible ou un motif (ex. `data/*`). |
| `overwrite` | `bool` | Si `True`, force le recalcul des embeddings même s'ils existent déjà. |
| `token_mode` | `bool` | Si `True`, génère les embeddings de lemmes (mode token). |
| `no_daemon` | `bool` | Si `True`, exécute le modèle d'encodage localement (sans daemon). |
| `use_ollama` | `bool` | Si `True`, utilise une instance Ollama pour l'encodage (incompatible avec le mode token). |
| `ollama_host` | `str` | L'adresse de l'hôte API Ollama (défaut : `'localhost:11434'`). |
| `ollama_model` | `str` | Le modèle Ollama spécifique à interroger. |
| `apply_abtt` | `bool` | Mode token : applique ABTT (défaut : `True`). |

**Valeur de retour** — `None` : cette fonction opère par effets de bord (création de fichiers `.npy`, `.json`, et en mode token `_lemma_index.json` / `_abtt.json`).

**Comportement** — Détecte tous les fichiers `.conllu`, `.xml` et `.trs` du dossier (ou du motif wildcard), puis appelle `calcEmbeddings` et `save_metadata` pour chacun, en journalisant la progression (`fichier i/N`). Le nom de sortie est calculé avec `os.path.splitext` : seule l'extension finale est remplacée, même si son nom apparaît ailleurs dans le chemin.

**Exemple**
```python
encode_folder("corpus/*Camus*", overwrite=False)
encode_folder("corpus/*Camus*", token_mode=True)
```

---

## Module `makeIndex.py`

Module de création d'index FAISS pour la recherche sémantique. Il fait le pont entre la phase d'extraction des caractéristiques (calcul des embeddings) et la phase de recherche. Une fois les fichiers `.faiss` créés, les fichiers `.npy` des embeddings peuvent être effacés car la recherche sémantique se base désormais sur les index `.faiss` — pour recréer un index, il faudrait néanmoins refaire les embeddings au préalable. Le module traite indifféremment les embeddings de phrases (`corpus.npy`) et de lemmes (`corpus_token.npy`) : ces derniers produisent `corpus_token.faiss`.

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

#### `makeIndex(embeddings=None, embedding_file_path=None, metric_type=None, index_type=None, m=512, output_file_path=None, overwrite=False, token_mode=False)`

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

**Mode token** — Une matrice de lemmes contient bien moins de lignes qu'un corpus de phrases (un vecteur par lemme unique). Avec `ivfpq`, le garde-fou décrit ci-dessus (`n ≥ 156·√n`, soit environ 24 000 vecteurs) fait donc très souvent basculer automatiquement sur un index `flat`, ce qui est sans inconvénient à cette taille.

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

#### `makeIndex_folder(input_folder=None, metric_type=None, index_type=None, m=512, overwrite=False, token_mode=False)`

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

**Comportement** — Détecte les fichiers d'embeddings (`.npy` ou `_token.npy` selon `token_mode`) via `glob`, déduplique la liste, puis construit un index pour chacun via `makeIndex` (en lui transmettant `m`).

Avec un wildcard, chaque fichier trouvé est d'abord ramené au nom de base du corpus (les suffixes `_lemma_index` et `_token` en fin de nom sont retirés), puis l'extension d'embeddings voulue est rajoutée ; les chemins qui ne correspondent à aucun fichier réel sont ignorés. Ainsi, `dossier/*Camus*` cible correctement `Camus_token.npy` en mode token, même si le wildcard capte aussi `_lemma_index.json` ou `_abtt.json`.

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
| `8` (valeur d'origine codée en dur dans `search()` ; le code utilise désormais `64`) | Référence de départ |
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

**Conclusion** : l'écart entre `ivfpq` et une recherche exacte observé dans ce projet provient **entièrement de l'erreur de reconstruction de la quantification produit (PQ)**, et non d'une couverture insuffisante des clusters (`nprobe`). Augmenter `m` réduit directement et significativement cette erreur, avec des rendements encore nettement positifs jusqu'à `m=256`, et des rendements plus marginaux (mais un score de similarité presque parfaitement exact) à `m=512`. C'est ce constat empirique qui justifie le changement de valeur par défaut de `m` (désormais `512` dans le code, contre `8` auparavant), et sa documentation comme paramètre explicite plutôt que constante figée.

### Recommandations pratiques issues de ces tests

- **Ne pas chercher à ajuster `nprobe`** pour améliorer la qualité des résultats `ivfpq` sur ce type de corpus — ce paramètre agit sur la vitesse de recherche en présence de nombreux clusters non pertinents, pas sur l'erreur de reconstruction vectorielle.
- **Pour un corpus dont la taille reste raisonnable en mémoire** (de l'ordre de quelques centaines de milliers de phrases, comme dans ce test), `flat` reste l'option la plus simple si l'exactitude prime : le surcoût de chargement mesuré (~750 ms pour ce corpus) est négligeable pour un usage interactif ou ponctuel.
- **Si la compression disque/RAM est nécessaire** (très grands corpus, contrainte matérielle), `m=256` à `m=512` offre un compromis solide : recouvrement de 25 à 29 sur 30 avec la recherche exacte, pour une taille d'index restant très inférieure à celle d'un index `flat` équivalent.
- **`hnsw` n'a pas encore été mesuré dans cette série de tests** et reste une alternative à évaluer : en théorie, il offre un meilleur recouvrement que `ivfpq` à taille de corpus comparable, sans nécessiter de réglage de `m`/`nbits`, au prix d'une consommation mémoire et disque plus élevée qu'`ivfpq`.

---

## Module `searchEmbedding.py`

Module de chargement des indices FAISS et des fichiers de métadonnées (JSON), d'encodage d'une requête via le client d'embedding, et d'exécution de la recherche de similarité entre la requête et les phrases indexées. Le module supporte la recherche dans un seul fichier index (`.faiss`) ou dans un dossier contenant plusieurs fichiers — les résultats d'une recherche en mode dossier sont rassemblés puis triés par score de similarité. Il gère les deux modes d'encodage : en mode phrase les résultats sont des phrases ; en mode token ce sont des **lemmes**, accompagnés de leurs occurrences dans le corpus.

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

**Description** — Charge les métadonnées d'un corpus à partir d'un fichier JSON. Les métadonnées sont les phrases elles-mêmes, avec leurs identifiants, leurs tokens et leurs lemmes, sous forme de dictionnaire, ex. `{"sent_id": ["1", "2"], "raw_text": ["Bonjour !", "pas de souci !"], "tokens": [["Bonjour","!"], ["pas","de","souci","!"]], "lemmas": [["bonjour","!"], ["pas","de","souci","!"]]}`. Chaque entrée du dictionnaire est une liste de la même taille que le nombre de phrases du corpus. En mode phrase, cette taille est celle de l'index FAISS ; en mode token, l'index a une entrée par lemme unique et c'est `_lemma_index.json` qui joue le rôle de table de correspondance.

| Paramètre | Type | Description |
|---|---|---|
| `matadata_file_path` | `str` | Le chemin d'accès vers le fichier JSON contenant les métadonnées. |

**Valeur de retour** — `dict | None` : le dictionnaire de métadonnées (`sent_id`, `raw_text`, `tokens`, `lemmas`) si le chargement réussit, sinon `None` (avec un avertissement journalisé).

**Exemple**
```python
metadata = load_metadata("corpus.json")
```

#### `load_lemma_index(lemma_index_path=None)`

**Description** — Charge la liste ordonnée des lemmes depuis un fichier `_lemma_index.json`. L'indice `i` de cette liste correspond à la ligne `i` de l'index FAISS du mode token.

| Paramètre | Type | Description |
|---|---|---|
| `lemma_index_path` | `str` | Chemin du fichier `..._token_lemma_index.json`. |

**Valeur de retour** — `list[str]` : les lemmes uniques triés.

**Exemple**
```python
lemma_list = load_lemma_index("corpus_token_lemma_index.json")
```

#### `apply_abtt(vecs, abtt_file)`

**Description** — Applique à des vecteurs (typiquement la requête) la transformation « All-but-the-top » dont les paramètres ont été sauvegardés lors du calcul des embeddings du corpus.

| Paramètre | Type | Description |
|---|---|---|
| `vecs` | `numpy.ndarray` | Vecteurs bruts, de forme `(n, dim)`. |
| `abtt_file` | `str` | Chemin du fichier `_abtt.json` (clés `mean` et `components`). |

**Valeur de retour** — `numpy.ndarray` : les vecteurs centrés (moyenne du corpus) puis projetés hors des composantes principales du corpus.

**Comportement** — La requête doit se trouver **dans le même espace que les lemmes indexés** : on réutilise donc `mu` et `P` du corpus, on ne les recalcule jamais sur la requête.

#### `abtt_path_from_index(index_file)`

**Description** — Déduit le chemin du fichier ABTT à partir de celui de l'index (`corpus_token.faiss` → `corpus_token_abtt.json`).

**Valeur de retour** — `str`.

#### `resolve_abtt_path(abtt_file=None, index_file=None)`

**Description** — Détermine le fichier ABTT à utiliser. Priorité : `abtt_file` explicite, puis chemin déduit de `index_file`.

**Exceptions** — `ValueError` si ni l'un ni l'autre n'est fourni (ex. index passé directement en mémoire sans `abtt_file`).

#### `get_lemma_occurrences(lemmas, metadata)`

**Description** — Pour une liste de lemmes, renvoie leurs occurrences dans le corpus en un seul passage sur les métadonnées.

| Paramètre | Type | Description |
|---|---|---|
| `lemmas` | `list[str]` | Les lemmes recherchés. |
| `metadata` | `dict \| None` | Métadonnées du corpus (`tokens`, `lemmas`, `sent_id`). |

**Valeur de retour** — `dict` : `{lemme: [(token, sent_id), ...]}` ; dictionnaire vide si `metadata` est `None`.

**Exemple**
```python
get_lemma_occurrences(["aller"], metadata)
# -> {"aller": [("allez", "s12"), ("vais", "s40"), ...]}
```

#### `embedd_query(query_str=None, token_mode=False, no_daemon=False, use_ollama=False, ollama_host='localhost', ollama_model=None)`

**Description** — Génère un vecteur d'embedding pour une requête textuelle donnée.

| Paramètre | Type | Description |
|---|---|---|
| `query_str` | `str` | Le texte de la requête à encoder. |
| `token_mode` | `bool` | Si `True`, encode la requête au niveau des tokens (mots) puis **moyenne** les vecteurs obtenus en un seul vecteur : la requête devient un point unique, dans le même espace que l'index de lemmes. |
| `no_daemon` | `bool` | Si `True`, exécute l'encodage localement sans passer par le daemon. |
| `use_ollama` | `bool` | Si `True`, délègue la création de l'embedding à un modèle Ollama (plus lent ; non compatible avec le mode token). |
| `ollama_host` | `str` | L'adresse de l'hôte Ollama (par défaut `'localhost'`). |
| `ollama_model` | `str` | Le nom du modèle d'embedding Ollama cible. |

**Valeur de retour** — `numpy.ndarray` : un tableau 2D contigu de type `float32`, de forme `(1, dim)` (`dim` = 1024 en mode phrase, 768 en mode token).

**Exceptions** — `ValueError` si `query_str` est `None`.

**Comportement** — Encode la requête via `encode()` (liste à un seul élément), mesure et journalise le temps d'encodage, puis met en forme le résultat. Le vecteur renvoyé est **brut** : ni normalisé, ni transformé par ABTT (c'est `search()` qui s'en charge).

**Exemple**
```python
query_vector = embedd_query("pas de souci")
query_vector_token = embedd_query("aller", token_mode=True)
```

#### `search(query_vector=None, query_str=None, index=None, index_file=None, abtt_file=None, metric_type=None, top_k=10, metadata=None, token_mode=False, no_daemon=False, lemma_list=None, allow_no_abtt=False, metadata_file=None)`

**Description** — Fonction principale du module : exécute une recherche de similarité dans un index FAISS et renvoie les meilleures correspondances avec leurs métadonnées.

| Paramètre | Type | Description |
|---|---|---|
| `query_vector` | `numpy.ndarray` | Le vecteur d'embedding **brut** de la requête (ni normalisé ni transformé par ABTT). Calculé à partir de `query_str` s'il est `None`. |
| `query_str` | `str` | Le texte de la requête (utilisé si `query_vector` est `None`). |
| `index` | `faiss.Index` | L'objet index FAISS dans lequel effectuer la recherche. |
| `index_file` | `str` | Chemin du `.faiss` correspondant à `index` (optionnel). Sert à déduire `_abtt.json`. |
| `abtt_file` | `str` | Chemin explicite du `_abtt.json` (optionnel, prioritaire sur `index_file`). Nécessaire en mode token si l'index est passé depuis la mémoire sans `index_file`. |
| `metric_type` | `int` | Le type de métrique FAISS (produit scalaire sur vecteurs normalisés L2 = similarité cosinus). |
| `top_k` | `int` | Le nombre maximal de résultats à retourner (défaut : 10). |
| `metadata` | `dict` | Phrases et identifiants (`sent_id`, `raw_text`). Mode phrase : obligatoire. Mode token : optionnel, sert à retrouver les occurrences des lemmes. |
| `token_mode` | `bool` | Recherche au niveau des lemmes. |
| `no_daemon` | `bool` | Encodage local sans passer par le daemon. |
| `lemma_list` | `list` | **Requis en mode token** : lemmes de l'index, dans l'ordre (`load_lemma_index`). |
| `allow_no_abtt` | `bool` | Mode token : si `True`, la recherche se fait **sans** ABTT, même si un fichier ABTT existe (tests uniquement). Les scores sont faux si l'index a été construit avec ABTT. Défaut : `False`. |
| `metadata_file` | `str` | Mode token : chemin de `corpus.json`, chargé si `metadata` n'est pas fourni, pour afficher les occurrences. |

**Valeur de retour** — `list[tuple]`.
- Mode phrase : `(sent_id, raw_text, score)`.
- Mode token : `(lemme, score, [(token, sent_id), ...])`.
- Liste vide en cas d'erreur de dimension (généralement une requête encodée avec un autre modèle que celui du corpus).

**Comportement**
1. Encode la requête si `query_vector` n'est pas fourni. En mode dossier, la requête n'est encodée qu'**une seule fois** puis réutilisée sur tous les index.
2. **Mode token — ABTT** : détermine le fichier `_abtt.json` (`resolve_abtt_path`) et transforme la requête avec `apply_abtt`. Lève `ValueError` si aucun chemin ne peut être déterminé et `FileNotFoundError` si le fichier est absent (« re-run calcEmbeddings »), sauf si `allow_no_abtt=True`.
3. Copie le vecteur puis le normalise en L2 (`faiss.normalize_L2` travaille en place ; la requête brute fournie n'est donc jamais modifiée), pour rester cohérent avec l'indexation.
4. Configure `index.nprobe = 64` si l'index le permet (index IVF).
5. Effectue la recherche (`index.search`) et filtre les indices invalides (`-1`, renvoyé par FAISS s'il y a moins de `top_k` résultats).
6. **Mode phrase** : associe chaque indice à `sent_id` et `raw_text`. **Mode token** : associe chaque indice à un lemme (`lemma_list`), puis ajoute ses occurrences via `get_lemma_occurrences`. Sans métadonnées, un avertissement est émis et les occurrences sont vides.
7. **Garde-fou de dimension** : si FAISS lève une `AssertionError` (requête encodée avec un modèle différent, par exemple mode token contre index de phrases), un avertissement explicite est journalisé et une liste vide est retournée.

**Exemple**
```python
# mode phrase
results = search(query_str="pas de souci", index=index,
                 metric_type=faiss.METRIC_INNER_PRODUCT, top_k=10, metadata=metadata)
for sent_id, raw_text, score in results:
    print(sent_id, raw_text, score)

# mode token
index = load_index("corpus_token.faiss")
lemma_list = load_lemma_index("corpus_token_lemma_index.json")
results = search(query_str="aller", index=index, index_file="corpus_token.faiss",
                 metric_type=faiss.METRIC_INNER_PRODUCT, top_k=10, token_mode=True,
                 lemma_list=lemma_list, metadata_file="corpus.json")
for lemma, score, occurrences in results:
    print(lemma, score, occurrences[:3])
```

#### `save_results_csv(results, output_csv, token_mode=False)`

**Description** — Enregistre des résultats de recherche dans un fichier CSV (encodage `utf-8-sig`, pour qu'Excel reconnaisse correctement l'UTF-8).

| Paramètre | Type | Description |
|---|---|---|
| `results` | `list[tuple]` | Résultats déjà annotés du fichier d'index : `(index_file, sent_id, phrase, score)` en mode phrase, `(index_file, lemme, score, occurrences)` en mode token. |
| `output_csv` | `str` | Chemin du fichier CSV à écrire. |
| `token_mode` | `bool` | Choisit le format des colonnes. |

**Comportement**
- Mode phrase : une ligne par résultat (`index_file`, `sent_id`, `sentence`, `score`).
- Mode token : une ligne **par occurrence** (`index_file`, `lemma`, `score`, `token`, `sent_id`) ; un lemme sans occurrence produit une ligne aux colonnes `token` et `sent_id` vides.

#### `search_folder(input_folder=None, query_str=None, query_vector=None, metric_type=faiss.METRIC_INNER_PRODUCT, top_k=10, verbose=True, token_mode=False, no_daemon=False, allow_no_abtt=False, max_token_ids_occ=8, output_csv=None)`

**Description** — Exécute une recherche de similarité textuelle sur un ensemble d'index FAISS contenus dans un dossier.

| Paramètre | Type | Description |
|---|---|---|
| `input_folder` | `str` | Le chemin du dossier ou le motif wildcard (ex. `*Camus*`) contenant les fichiers `.faiss`. |
| `query_str` | `str` | Le texte brut de la requête à rechercher. |
| `query_vector` | `numpy.ndarray` | Le vecteur d'embedding brut pré-calculé (évite de ré-encoder la requête). |
| `metric_type` | `int` | La métrique de distance utilisée par FAISS (défaut : produit scalaire). |
| `top_k` | `int` | Le nombre global de meilleurs résultats à conserver et afficher. |
| `verbose` | `bool` | Si `True`, affiche le tableau des résultats dans la console. |
| `token_mode` | `bool` | Si `True`, cible les index `_token.faiss` (avec leurs `_lemma_index.json` et `_abtt.json`). |
| `no_daemon` | `bool` | Si `True`, utilise l'encodage local sans passer par le daemon. |
| `allow_no_abtt` | `bool` | Mode token : les corpus sans `_abtt.json` ne sont plus ignorés, la recherche se fait sans ABTT (tests uniquement). |
| `max_token_ids_occ` | `int` | Mode token : nombre maximal d'occurrences `token (sent_id)` affichées par lemme (défaut : 8 ; `0` = toutes). |
| `output_csv` | `str` | Si fourni, enregistre les résultats dans ce fichier CSV (`save_results_csv`). |

**Valeur de retour** — `list[tuple]` : les `top_k` meilleurs résultats, triés par score décroissant.
- Mode phrase : `(index_file, sent_id, raw_text, score)`.
- Mode token : `(index_file, lemme, score, [(token, sent_id), ...])`.

**Comportement**
- Encode la requête une seule fois si nécessaire (`embedd_query`).
- Récupère les fichiers `.faiss` du dossier (ou du motif wildcard) via `glob`. En mode token, seuls les `_token.faiss` sont retenus ; en mode phrase, les fichiers `_lemma_index.json` et `_abtt.json` que capte un wildcard sont ignorés. La liste est dédupliquée.
- Ignore et journalise en avertissement les fichiers dont il manque un élément (métadonnées en mode phrase ; `_lemma_index.json` ou `_abtt.json` en mode token), plutôt que d'interrompre toute la recherche.
- Agrège les résultats de tous les fichiers (chacun annoté du fichier source), les trie globalement par score décroissant, et ne conserve que les `top_k` meilleurs.
- Écrit le CSV si `output_csv` est fourni, puis affiche un tableau récapitulatif si `verbose=True`.

> ⚠️ Si deux fichiers du dossier contiennent le même texte source (ex. un corpus encodé une fois via le daemon et une fois en mode `no_daemon` pour comparaison), chaque phrase correspondante occupe deux emplacements distincts dans le classement agrégé — ce qui peut évincer d'autres résultats légitimes du top-k final.

**Exemple**
```python
search_folder(input_folder="test/*", query_str="pas de souci", top_k=30, verbose=True)

# mode token, avec export CSV
search_folder(input_folder="test/*", query_str="aller", token_mode=True,
              top_k=30, output_csv="resultats.csv")
```

---

## Module `utils/embed_client.py`

Client d'encodage vectoriel (« Embedding Client »). Il constitue l'interface entre les différents scripts et le daemon d'embedding, et implémente trois stratégies d'exécution distinctes :

| Stratégie | Argument | Comportement |
|---|---|---|
| **Daemon IPC** (par défaut) | *(aucun flag)* | Envoie les phrases à `embed_daemon.py` via une socket locale ; démarre le daemon automatiquement s'il n'est pas déjà lancé. |
| **Sans daemon** | `no_daemon=True` | Charge les modèles directement dans le processus courant (utile pour le débogage, l'exécution ponctuelle, ou les environnements où un processus persistant n'est pas souhaitable). |
| **Ollama** | `use_ollama=True` | Délègue l'encodage à un serveur Ollama externe via une requête HTTP POST sur `/api/embed`. Renvoie un vecteur par texte : **non compatible avec le mode token**, qui a besoin d'un vecteur par mot. |

Ces trois stratégies sont mutuellement exclusives dans l'ordre de priorité suivant : Ollama est tenté en premier s'il est demandé, puis, en cas d'échec de connexion, le script bascule automatiquement sur le mode daemon/local. Le mode `no_daemon` est prioritaire sur le mode daemon s'il est explicitement demandé.

> ⚠️ **Ollama est explicitement documenté comme significativement plus lent** que le daemon ou un modèle chargé directement, problème connu côté Ollama, non résolu à ce jour. Cette stratégie reste donc expérimentale et peut être utile pour effectuer des tests.

**Les modèles d'encodage** — Deux modèles distincts sont utilisés selon le mode choisi :

| Mode | Modèle | Usage |
|---|---|---|
| **Phrase** (par défaut) | `BAAI/bge-m3` (via `SentenceTransformer`) | Encode chaque phrase entière en un seul vecteur (1024 dimensions) ; c'est le mode utilisé pour l'indexation FAISS et la recherche sémantique de phrases. |
| **Token** (`token_mode=True`) | `intfloat/multilingual-e5-base` (via `transformers.AutoModel`) | Encode chaque phrase en **un vecteur par mot** (768 dimensions) : moyenne des 4 dernières couches cachées, puis moyenne des sous-mots d'un même mot (`merge_subwords_to_words`). Ces vecteurs servent ensuite à construire les embeddings de lemmes (voir [Modes d'encodage](#modes-dencodage--phrase-et-token)). |

**Format de retour selon le mode** — En mode phrase, `encode()` renvoie une matrice `numpy.ndarray` `(n_phrases, 1024)`. En mode token, il renvoie une **liste** de tableaux `numpy`, un par phrase, de forme `(n_mots_phrase, 768)` : les phrases ayant des longueurs différentes, elles ne peuvent pas former une matrice uniforme.

**Réduction « All-but-the-top » (ABTT)** — Elle n'est plus appliquée dans le client ni dans le daemon en cours d'utilisation : `calcEmbeddings.all_but_the_top` l'applique à la matrice de lemmes d'un fichier, après le regroupement par lemme (voir [`calcEmbeddings.py`](#module-calcembeddingspy)). Le daemon conserve sa propre version de la fonction (sans effet sur le pipeline actuel).

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

**Description** — Exécute l'encodage vectoriel directement dans le processus courant (de manière synchrone), sans utiliser le système de daemon en arrière-plan. Utilise un chargement paresseux (lazy loading) : les bibliothèques lourdes (PyTorch) et les poids des **deux** modèles sont chargés en mémoire au premier appel.

| Paramètre | Type | Description |
|---|---|---|
| `sentences` | `list` | La liste des phrases ou textes à encoder. |
| `token_mode` | `bool` | Si `True`, utilise le modèle orienté tokens (`multilingual-e5-base`) et renvoie un vecteur par mot. |

**Valeur de retour** — Mode phrase : `numpy.ndarray` `(n_phrases, 1024)` en `float32`. Mode token : liste de tableaux `(n_mots_phrase, 768)`, un par phrase.

**Comportement**
- **Mode token** : tokenisation par lots de 32 (`tqdm` pour la barre de progression). Le tokenizer rapide est conservé tel quel (méthode `word_ids()`), le modèle est appelé avec `output_hidden_states=True`, on moyenne les 4 dernières couches, puis `merge_subwords_to_words` (importée de `utils.embed_daemon`) regroupe les sous-mots par mot.
- **Mode phrase** : délègue le batching et la barre de progression au `SentenceTransformer` lui-même (`model.encode(sentences, show_progress_bar=True)`).

**Exemple**
```python
embeddings = encode_no_daemon(["pas de souci", "aucun problème"])
word_vecs  = encode_no_daemon(["pas de souci"], token_mode=True)   # [array (3, 768)]
```

#### `encode(sentences, chunk_size=512, show_progress=True, token_mode=False, no_daemon=False, use_ollama=False, ollama_host='localhost', ollama_model=None)`

**Description** — Fonction routeur principale pour la génération d'embeddings. Permet de choisir la stratégie d'exécution selon le besoin de l'utilisateur (Ollama, local, ou via daemon IPC).

| Paramètre | Type | Description |
|---|---|---|
| `sentences` | `list` | Les textes à encoder. |
| `chunk_size` | `int` | Taille des sous-lots envoyés au daemon via le réseau ; la valeur par défaut (512) a été déterminée après avoir testé des tailles plus grandes qui ont causé des débordements de mémoire GPU (`calcEmbeddings` utilise `64`). |
| `show_progress` | `bool` | Affiche dynamiquement la progression dans la console. |
| `token_mode` | `bool` | Bascule sur le modèle d'encodage par token (un vecteur par mot). |
| `no_daemon` | `bool` | Force l'exécution locale synchrone. |
| `use_ollama` | `bool` | Tente d'utiliser un serveur Ollama externe (plus lent que le daemon ou un modèle chargé directement — problème connu côté Ollama ; non compatible avec le mode token). |
| `ollama_host` | `str` | Adresse du serveur Ollama. |
| `ollama_model` | `str` | Modèle Ollama cible. |

**Valeur de retour** — Mode phrase : `numpy.ndarray` `(n_phrases, dim)`. Mode token : liste plate de tableaux `numpy` `(n_mots_phrase, 768)`, un par phrase.

**Exceptions** — `RuntimeError` si le processus d'encodage (daemon ou réseau) échoue.

**Comportement**
1. **Stratégie Ollama** (si `use_ollama=True`) : tentée en premier ; en cas de réponse vide, bascule automatiquement sur le mode suivant.
2. **Stratégie sans daemon** (si `no_daemon=True`) : chargement direct des modèles dans le processus courant.
3. **Stratégie par défaut (daemon)** : tente une connexion au daemon (`_try_connect`), le démarre si besoin (`_start_daemon`), envoie `(job_id, sentences, chunk_size, token_mode)` avec un identifiant unique (`uuid.uuid4()`) pour le suivi/débogage, puis écoute les messages de progression et le résultat final, en affichant une barre de progression textuelle si `show_progress=True`.

**Exemple**
```python
embeddings = encode(["pas de souci", "aucun problème"], chunk_size=64)
word_vecs  = encode(["pas de souci"], chunk_size=1, token_mode=True)
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

**Comportement** — Détecte automatiquement l'accélération matérielle disponible (`cuda` ou `cpu`). Si `token_mode=True` et que le modèle token n'est pas déjà chargé, charge `intfloat/multilingual-e5-base` (modèle et tokenizer) et libère la référence au modèle phrase (et inversement). Un seul des deux modèles reste donc chargé en mémoire à la fois : alterner les modes phrase et token recharge donc un modèle à chaque changement.

**Exemple**
```python
load_models(token_mode=False)   # charge BAAI/bge-m3 si pas déjà chargé
```

#### `_ensure_imports()`

**Description** — Importe les bibliothèques lourdes (PyTorch, Transformers) uniquement au démarrage effectif du daemon, pour accélérer l'importation initiale du module parent.

**Paramètres** — Aucun.

**Valeur de retour** — `None` (peuple les variables globales `torch`, `F`, `Tensor`, `AutoTokenizer`, `AutoModel`, `SentenceTransformer`).

#### `all_but_the_top(X, n_components=3)`

**Description** — Version « daemon » de la technique « All-but-the-top » : centre la matrice et retire ses `n_components` premières composantes principales.

| Paramètre | Type | Description |
|---|---|---|
| `X` | `numpy.ndarray \| torch.Tensor` | La matrice d'embeddings à traiter. |
| `n_components` | `int` | Le nombre de composantes principales à retirer (défaut : 3). |

**Valeur de retour** — Même type que l'entrée : la matrice transformée.

> Cette fonction n'est **pas appelée** par le pipeline actuel : c'est la version de `calcEmbeddings.py`, qui renvoie aussi `mu` et `P`, qui est utilisée (les paramètres doivent être sauvegardés pour être réappliqués à la requête).

#### `merge_subwords_to_words(hidden_states, encoded_batch)`

**Description** — Regroupe les vecteurs de sous-mots (*subwords*) en un vecteur par mot, en moyennant les sous-tokens appartenant au même mot selon la segmentation du tokenizer. Les tokens spéciaux (CLS, SEP, PAD) n'appartiennent à aucun mot et sont ignorés. C'est cette fonction qui produit le « un vecteur par mot » du mode token.

| Paramètre | Type | Description |
|---|---|---|
| `hidden_states` | `torch.Tensor` | `(batch, seq_len, hidden_dim)` : sortie du modèle (moyenne des 4 dernières couches, ou dernière couche seule si `use_last_n_layers` est `False`). |
| `encoded_batch` | `BatchEncoding` | Sortie brute du tokenizer, **avant** conversion en dictionnaire de tenseurs sur le device : il faut un tokenizer *rapide* pour disposer de `word_ids()`. |

**Valeur de retour** — `list[numpy.ndarray]` : une entrée par phrase du lot, chacune de forme `(n_mots_phrase, hidden_dim)` en `float32` (un tableau `(0, hidden_dim)` si la phrase est vide).

**Comportement** — Pour chaque phrase, regroupe les positions de sous-tokens par identifiant de mot (`word_ids`), moyenne chaque groupe, puis empile les vecteurs dans l'ordre naturel des mots. Un « mot » est défini par le tokenizer à partir des espaces de la chaîne d'entrée, d'où la construction des entrées de `calcEmbeddings` (tokens joints par des espaces, espaces internes remplacés par `_`).

> Les fonctions `average_pool` et `average_pool_last_n_layers`, qui produisaient un vecteur agrégé par phrase, ont été supprimées : le mode token repose désormais sur des vecteurs par mot.

#### `batching_worker()`

**Description** — Thread d'exécution unique. Extrait les éléments en attente dans la file (`batch_queue`), les concatène en un seul lot pour l'encodage, puis redistribue les vecteurs résultants pour reconstruire les listes d'origine.

**Paramètres** — Aucun (boucle infinie, destinée à être lancée comme thread daemon).

**Valeur de retour** — Ne retourne jamais (boucle `while True`).

**Comportement**
1. Bloque en attente du premier élément de la file (`batch_queue.get()`).
2. S'assure que le bon modèle est chargé en mémoire (`load_models`).
3. Accumule d'autres requêtes arrivées dans une fenêtre de temps de `BATCH_WINDOW_S` (15 ms), jusqu'à une taille maximale `MAX_BATCH_SIZE` (64 phrases).
4. Encode le lot complet :
   - **branche token** : tokenisation (`max_length=512`, avec troncature), passage dans le modèle sous `torch.no_grad()` avec `output_hidden_states=True`, moyenne des 4 dernières couches (`use_last_n_layers=True`, valeur par défaut) ou dernière couche seule, puis `merge_subwords_to_words` : le résultat est une liste de tableaux `(n_mots, 768)` ;
   - **branche phrase** : `SentenceTransformer.encode(...)`, résultat en matrice `float32`.
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
4. Assemble les sous-lots et envoie le résultat final (`("done", résultat)`) : en mode phrase, une seule matrice NumPy (`np.concatenate`) ; en mode token, une **liste plate** de tableaux, un par phrase (leurs tailles diffèrent, ils ne peuvent pas former une matrice).
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
| `--token-emb` | Active le mode d'encodage par token (`intfloat/multilingual-e5-base`) : un vecteur par **lemme**, avec ABTT. Les fichiers dérivés portent le suffixe `_token`. Voir [Modes d'encodage](#modes-dencodage--phrase-et-token). |
| `--no-faiss` | Effectue une recherche par produit scalaire direct sur les `.npy`, sans passer par un index FAISS : sert à valider la fiabilité d'un index FAISS en comparant ses résultats à un calcul exact. |
| `--no-daemon` | Charge les modèles localement dans le processus courant plutôt que de passer par le daemon. |
| `--use-ollama` | Utilise un serveur Ollama pour l'encodage (non compatible avec `--token-emb`). |
| `--no-abtt` | Mode token : construit les embeddings **sans** All-but-the-top (aucun `_abtt.json` n'est écrit). À combiner avec `--force` pour reconstruire un corpus existant, et avec `--allow-no-abtt` pour la recherche en fichier unique (sans fichier ABTT, la recherche s'arrête sinon avec une erreur). |
| `--allow-no-abtt` | Mode token : autorise la recherche sans le fichier `_abtt.json` du corpus (tests / comparaison A-B). Les résultats sont faux si l'index a été construit **avec** ABTT. |
| `--output-csv FICHIER` | Enregistre les résultats de recherche dans un fichier CSV (une ligne par occurrence en mode token). |
| `--max-token-ids-occ N` | Mode token : nombre maximal de `token (sent_id)` affichés par lemme (défaut : 8, `0` = toutes les occurrences). |
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

#### `prepare_query_no_faiss(args, raw_query, abtt_path)`

**Description** — Renvoie une **copie** de la requête brute, transformée par ABTT en mode token (mêmes `mu` et `P` que le corpus). La requête brute n'est jamais modifiée, ce qui permet de la réutiliser d'un fichier à l'autre (`faiss.normalize_L2` travaille en place).

| Paramètre | Type | Description |
|---|---|---|
| `args` | `argparse.Namespace` | Arguments de la ligne de commande (`token_emb`, `allow_no_abtt`). |
| `raw_query` | `numpy.ndarray` | Vecteur brut de la requête. |
| `abtt_path` | `str` | Chemin du `_abtt.json` du corpus. |

**Valeur de retour** — `numpy.ndarray` : la requête prête à l'emploi.

**Exceptions** — `FileNotFoundError` si le fichier ABTT est absent en mode token, sauf avec `--allow-no-abtt` (la requête reste alors brute, avec un avertissement).

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

En mode token, les embeddings chargés sont ceux de `_token.npy` et la requête reçoit la transformation ABTT du corpus (`prepare_query_no_faiss`).

> ⚠️ L'affichage de `--no-faiss` associe les scores aux phrases de `corpus.json` : cela n'a de sens qu'en mode phrase (voir [Limitations](#limitations-connues-et-points-dattention)).

**Exemple**
```bash
python main.py "test/*" "pas de souci" --top-k 30 --no-faiss --log
```

#### `process(args, index, metric_type=faiss.METRIC_INNER_PRODUCT, metadata=None, lemma_list=None, abtt_file=None)`

**Description** — Exécute une requête de recherche sémantique sur un index FAISS unique et affiche les résultats.

| Paramètre | Type | Description |
|---|---|---|
| `args` | `argparse.Namespace` | Objet arguments de la ligne de commande contenant la requête et les options. |
| `index` | `faiss.Index` | L'index FAISS chargé en mémoire. |
| `metric_type` | `int` | Type de métrique de distance (défaut : produit scalaire). |
| `metadata` | `dict` | Métadonnées associées aux vecteurs (ID, phrases, tokens, lemmes). En mode token, sert à afficher les occurrences des lemmes. |
| `lemma_list` | `list` | **Requis en mode token** : lemmes de l'index, chargés depuis `_lemma_index.json`. |
| `abtt_file` | `str` | Chemin du `_abtt.json` du corpus (mode token). Sans lui, et sans `--allow-no-abtt`, la recherche s'arrête avec un message d'erreur. |

**Valeur de retour** — `None` : affiche les résultats dans la console (et écrit un CSV avec `--output-csv`).

**Comportement** — Appelle `search()` du module `searchEmbedding.py` avec les options extraites de `args`, mesure le temps d'exécution, puis affiche les résultats :
- mode phrase : `Sent id | Sentence | similarity score` ;
- mode token : `lemma | similarity score | tokens (sent_id)`, limité à `--max-token-ids-occ` occurrences par lemme (`... (+N)` indique les occurrences non affichées).

Si le fichier ABTT est manquant, `process` quitte proprement avec un message invitant à relancer le calcul des embeddings (`--force`) ou à utiliser `--allow-no-abtt`.

**Exemple**
```python
process(args, index=index, metadata=metadata)                                # mode phrase
process(args, index=index, metadata=metadata, lemma_list=lemma_list,
        abtt_file="corpus_token_abtt.json")                                  # mode token
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
3. **Mode `--force`** : recalcule tout le dossier (embeddings et index, via `encode_folder` et `makeIndex_folder`, avec `apply_abtt = not --no-abtt`) avant la recherche.
4. Lance la recherche globale sur le répertoire avec le vecteur pré-calculé (`search_folder`), en transmettant `--token-emb`, `--max-token-ids-occ` et `--output-csv`. Les corpus sans fichier ABTT ne sont tolérés que si `--allow-no-abtt` ou `--no-abtt` est actif.

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
2. **`--regenerate-metadata`** : reparse le(s) fichier(s) source(s) et réécrit uniquement le(s) fichier(s) `.json`, sans toucher aux embeddings ni à l'index. À lancer après une mise à jour du parseur ; nécessaire notamment pour ajouter la clé `lemmas` à d'anciens `.json` avant d'utiliser le mode token.
3. **Fichier `.faiss` fourni directement** : charge l'index et les métadonnées correspondantes (et, en mode token, la liste des lemmes et le chemin ABTT), puis lance directement la recherche (`process`).
4. **`--encode-only`** (sur dossier ou wildcard) : encode et indexe tous les fichiers manquants du dossier, étape par étape (métadonnées → embeddings → index), sans lancer de recherche. En mode token, les fichiers ciblés portent le suffixe `_token`.
5. **`--search-only`** (sur dossier) : suppose que tous les index existent déjà et lance directement la recherche, pour un gain de vitesse en évitant les vérifications de cache.
6. **Mode dossier ou wildcard standard** (`--folder` ou motif `*` dans `input_file`) : encode si nécessaire (en mode `--force`), puis effectue une recherche agrégée sur tous les index du dossier (`process_folder`).
7. **Mode fichier unique standard** : applique la logique de cache incrémental ci-dessous, puis lance la recherche sur ce fichier (`process`).

**Système de cache incrémental (mode 7)** — Pour un fichier corpus unique, `main()` déduit les chemins des fichiers dérivés attendus (`.npy`, `.json`, `.faiss`) et applique la logique suivante :

1. Si les **embeddings et métadonnées existent déjà**, et pas `--force` → ils sont chargés directement, on ne reconstruit que l'index s'il manque.
2. Si le fichier fourni a une extension **non reconnue** et qu'aucun fichier dérivé n'existe → le script s'arrête avec une erreur explicite.
3. Si **embeddings et index sont absents**, ou `--force` est utilisé → tout est reconstruit depuis le fichier source (parsing → embeddings → métadonnées → index).
4. Sinon → les embeddings bruts (`.npy`) sont rechargés depuis le disque pour ne reconstruire que l'index manquant.

**Fichiers dérivés en mode token** — `main()` attend `corpus_token.npy`, `corpus_token.faiss` et `corpus_token_lemma_index.json` ; `corpus.json` reste commun. L'index de lemmes est indissociable de la matrice : son absence compte comme « embeddings manquants » et déclenche un recalcul complet (cas 7.3), pas une simple reconstruction d'index.

**Garde-fou d'intégrité** : une fois l'index chargé (ou reconstruit), `main()` vérifie que le nombre de vecteurs de l'index (`index.ntotal`) correspond bien au nombre d'entrées du fichier de correspondance : les métadonnées (`len(metadata["raw_text"])`) en mode phrase, la liste de lemmes (`_lemma_index.json`) en mode token. En cas de désaccord, signe probable d'une incohérence entre un ancien index et de nouvelles métadonnées régénérées séparément, le script s'arrête avec un message d'erreur explicite invitant à relancer avec `--force`, plutôt que de retourner silencieusement des résultats erronés ou décalés.

**Alternative automatique IVFPQ → flat** : si `makeIndex` signale une `ValueError` parce que le corpus est trop petit pour un entraînement IVFPQ fiable, `main()` intercepte l'erreur et reconstruit automatiquement un index `flat` à la place, en avertissant l'utilisateur.

> Un seul appel à `main.py` sur un fichier unique suffit à traverser toutes les étapes manquantes du pipeline en une seule exécution.

**Exemple**
```bash
python main.py mon_corpus.conllu "ma phrase de recherche"

# mode token : recherche de lemmes proches, avec export CSV
python main.py mon_corpus.conllu "aller" --token-emb --top-k 20 --output-csv resultats.csv

# mode token sans ABTT (reconstruit les fichiers du corpus ; --allow-no-abtt est nécessaire pour la recherche finale)
python main.py mon_corpus.conllu "aller" --token-emb --no-abtt --allow-no-abtt --force
```

---

## Exemples de commandes (cas d'usage)

Toutes les commandes se lancent depuis la racine du projet. Les exemples supposent l'arborescence suivante :

```
corpus/
├── Camus.conllu
├── Sartre.xml
└── entretien.trs
```

> 💡 **Guillemets obligatoires autour des wildcards** (`"corpus/*"`). Sans eux, le shell développe le motif en plusieurs arguments et `argparse` répond `unrecognized arguments`.

### 1. Premiers pas : un seul fichier (mode phrase)

```bash
# Première exécution : parse, encode, indexe (ivfpq, ou flat si le corpus est trop petit), puis cherche.
# Crée corpus/Camus.npy, corpus/Camus.json et corpus/Camus.faiss
python main.py corpus/Camus.conllu "pas de souci"

# Exécutions suivantes : tout vient du cache, seule la requête est encodée
python main.py corpus/Camus.conllu "il fait beau aujourd'hui"

# Plus de résultats
python main.py corpus/Camus.conllu "pas de souci" --top-k 30

# Autres formats de corpus (le format est déduit de l'extension)
python main.py corpus/Sartre.xml "la liberté"
python main.py corpus/entretien.trs "je ne sais pas"

# Chercher directement dans un index déjà construit (nécessite corpus/Camus.json)
python main.py corpus/Camus.faiss "pas de souci"

# Afficher les journaux détaillés (par défaut, seules les erreurs des modules sont affichées)
python main.py corpus/Camus.conllu "pas de souci" --log
python main.py corpus/Camus.conllu "pas de souci" --warn      # avertissements uniquement
```

### 2. Mode token : chercher des lemmes proches

```bash
# Première exécution : crée corpus/Camus_token.npy, Camus_token_lemma_index.json,
# Camus_token_abtt.json et Camus_token.faiss (corpus/Camus.json est partagé avec le mode phrase)
python main.py corpus/Camus.conllu "aller" --token-emb

# Nombre de lemmes et nombre d'occurrences affichées par lemme
python main.py corpus/Camus.conllu "aller" --token-emb --top-k 20 --max-token-ids-occ 3
python main.py corpus/Camus.conllu "aller" --token-emb --max-token-ids-occ 0    # toutes les occurrences

# Une requête de plusieurs mots est encodée mot par mot puis moyennée en un seul vecteur
python main.py corpus/Camus.conllu "avoir peur" --token-emb

# Exporter les résultats en CSV (une ligne par occurrence : lemme, score, token, sent_id)
python main.py corpus/Camus.conllu "aller" --token-emb --output-csv resultats_aller.csv

# Le mode phrase et le mode token coexistent pour un même corpus : deux jeux de fichiers indépendants
python main.py corpus/Camus.conllu "pas de souci"
python main.py corpus/Camus.conllu "aller" --token-emb
```

> En mode token, passez le fichier de **corpus source** (`Camus.conllu`) et non l'index `Camus_token.faiss` (voir [Limitations](#limitations-connues-et-points-dattention)).

### 3. Plusieurs corpus : dossier ou wildcard

La recherche sur un dossier ou un wildcard **ne construit rien** sauf avec `--force` : il faut donc précalculer les index d'abord.

```bash
# Étape 1 : précalcul, sans requête (incrémental : ne fait que ce qui manque)
python main.py "corpus/*" --encode-only
python main.py "corpus/*" --encode-only --token-emb            # idem pour le mode token

# Étape 2 : recherche agrégée, résultats triés par score sur l'ensemble des fichiers
python main.py "corpus/*" "pas de souci" --top-k 20
python main.py corpus "pas de souci" --folder --top-k 20       # équivalent, avec un nom de dossier
python main.py "corpus/*" "aller" --token-emb --top-k 20

# Restreindre à un sous-ensemble de fichiers
python main.py "corpus/*Camus*" "pas de souci"
python main.py "corpus/ESLO*" "pas de souci" --output-csv resultats.csv

# Gain de vitesse : suppose que tous les index existent, saute les vérifications de cache
python main.py corpus "pas de souci" --folder --search-only

# Tout recalculer (embeddings, index) puis chercher
python main.py "corpus/*Camus*" "pas de souci" --force
```

> ⚠️ `--encode-only` doit recevoir un **wildcard** (`"corpus/*"`), pas un simple nom de dossier : avec `corpus --folder`, le motif ne correspond à aucun fichier de corpus et rien n'est traité.
> ⚠️ En mode `--encode-only`, un fichier `.npy` supprimé est **recalculé** (contrairement à une recherche en fichier unique, qui n'a besoin que du `.faiss`).

### 4. Choisir le type d'index et gérer l'espace disque

```bash
# Index exact (idéal pour un corpus de taille moyenne)
python main.py corpus/Camus.conllu "pas de souci" --index-type flat

# Index HNSW (rapide, gourmand en mémoire)
python main.py corpus/Camus.conllu "pas de souci" --index-type hnsw

# Changer le type d'un index existant : --index-type n'agit qu'à la CRÉATION de l'index.
# Supprimer le .faiss (le .npy doit exister) puis relancer
rm corpus/Camus.faiss
python main.py corpus/Camus.conllu "pas de souci" --index-type flat

# Embeddings en float16 : fichiers .npy environ 2 fois plus petits (uniquement lors du calcul des embeddings)
python main.py corpus/Camus.conllu "pas de souci" --reduce-precision

# Une fois le .faiss construit, le .npy peut être supprimé : la recherche n'en a plus besoin
# (mais il faudra recalculer les embeddings pour reconstruire l'index)
rm corpus/Camus.npy
python main.py corpus/Camus.conllu "pas de souci"
```

### 5. Choisir la stratégie d'encodage

```bash
# Par défaut : daemon persistant (démarré automatiquement, le premier appel charge le modèle)
python main.py corpus/Camus.conllu "pas de souci"

# Sans daemon : le modèle est chargé dans le processus courant (débogage, requête isolée)
python main.py corpus/Camus.conllu "pas de souci" --no-daemon
python main.py corpus/Camus.conllu "aller" --token-emb --no-daemon

# Ollama (plus lent, expérimental ; adresse et modèle à régler via OLLAMA_HOST / OLLAMA_MODEL en tête de main.py)
python main.py "corpus/*Camus*" "pas de souci" --use-ollama --force
```

> ⚠️ `--use-ollama` n'est transmis à l'encodage que **en mode dossier / wildcard** (requête et `--force`). En fichier unique, il est sans effet. Il est de plus incompatible avec `--token-emb`, et les vecteurs Ollama n'ont pas la même dimension que `bge-m3` : un index construit avec un modèle ne peut pas être interrogé avec l'autre (liste vide + avertissement).

### 6. Maintenance : mettre à jour ou reconstruire

```bash
# Régénérer uniquement les métadonnées (.json) après une mise à jour du parseur,
# ou pour ajouter la clé "lemmas" à d'anciens fichiers avant d'utiliser le mode token
python main.py "corpus/*" --regenerate-metadata
python main.py "corpus/*Camus*" --regenerate-metadata
python main.py corpus/Camus.conllu --regenerate-metadata     # fichier unique : corpus/Camus.faiss doit exister

# Tout reconstruire pour un fichier (métadonnées, embeddings, index)
python main.py corpus/Camus.conllu "pas de souci" --force
python main.py corpus/Camus.conllu "aller" --token-emb --force

# Message « index/metadata mismatch » : l'index et les métadonnées ne correspondent plus
python main.py corpus/Camus.conllu "pas de souci" --force
```

> ⛔ **N'utilisez pas `--folder` avec `--regenerate-metadata`** : en mode dossier, tous les fichiers du dossier (y compris les `.json`, `.npy`, `.faiss`) sont parcourus sans filtrage par extension, et les fichiers non reconnus **écrasent** leur `.json` avec `null`. Utilisez toujours un wildcard (`"corpus/*"`), qui ne retient que les `.conllu`, `.xml` et `.trs`.

### 7. Valider les résultats et tester des variantes

```bash
# Comparer FAISS à un calcul exact : produit scalaire direct sur les .npy, sans index
python main.py corpus/Camus.conllu "pas de souci" --no-faiss --top-k 30
python main.py corpus/Camus.conllu "pas de souci" --top-k 30           # à comparer avec la commande précédente
python main.py "corpus/*" "pas de souci" --no-faiss --top-k 30         # plusieurs fichiers (wildcard, pas de nom de dossier seul)

# Mode token SANS ABTT (test A/B). Les fichiers sont écrasés : une seule variante à la fois par corpus.
# --allow-no-abtt est nécessaire pour la recherche en fichier unique quand aucun _abtt.json n'existe
python main.py corpus/Camus.conllu "aller" --token-emb --no-abtt --allow-no-abtt --force

# Retour à la variante avec ABTT
python main.py corpus/Camus.conllu "aller" --token-emb --force
```

> ⚠️ Chercher avec `--allow-no-abtt` dans un index construit **avec** ABTT donne des scores faux (la requête et le corpus ne sont pas dans le même espace).
> `--no-faiss` associe les scores aux phrases de `corpus.json` : il n'est fiable qu'en mode phrase.

### 8. Gérer le daemon

```bash
# Démarrage manuel (les journaux s'affichent dans le terminal, utile pour déboguer) ; sinon il démarre tout seul
python -m utils.embed_daemon

# Vérifier qu'il écoute sur le port 6000 (Linux)
ss -ltnp | grep 6000

# L'arrêter (Linux/macOS)
pkill -f embed_daemon

# Surveiller la mémoire GPU pendant un gros encodage
watch -n 1 nvidia-smi
```

### 9. Utilisation depuis Python

```python
import faiss
from calcEmbeddings import calcEmbeddings, save_metadata
from makeIndex import makeIndex
from searchEmbedding import load_index, load_metadata, search, search_folder

# --- Pipeline complet, mode phrase, pas à pas ---
embeddings, metadata = calcEmbeddings("corpus/Camus.conllu", "corpus/Camus.npy")
save_metadata(metadata, "corpus/Camus.json")
makeIndex(embeddings=embeddings, metric_type=faiss.METRIC_INNER_PRODUCT,
          index_type="flat", output_file_path="corpus/Camus.faiss")

# --- Recherche dans un index existant ---
index = load_index("corpus/Camus.faiss")
metadata = load_metadata("corpus/Camus.json")
for sent_id, text, score in search(query_str="pas de souci", index=index,
                                   metric_type=faiss.METRIC_INNER_PRODUCT,
                                   top_k=5, metadata=metadata):
    print(sent_id, score, text)

# --- Recherche sur plusieurs corpus, avec export CSV ---
search_folder("corpus/*", query_str="pas de souci", top_k=20, output_csv="resultats.csv")
search_folder("corpus/*", query_str="aller", token_mode=True, top_k=20)

# --- Contexte autour d'une phrase trouvée (10 phrases avant/après par défaut) ---
from main import get_sent_context
print(get_sent_context("corpus/Camus.json", sent_id=134, context_size=5))
```

Pour le mode token pas à pas (`calcEmbeddings(..., token_mode=True)`, puis `search(..., token_mode=True, lemma_list=...)`), voir les exemples de [`calcEmbeddings`](#module-calcembeddingspy) et de [`search`](#module-searchembeddingpy).

### 10. Messages d'erreur fréquents

| Message | Cause probable | Solution |
|---|---|---|
| `unrecognized arguments: ...` | Wildcard non protégé par des guillemets | Écrire `"corpus/*"` |
| `index/metadata mismatch` ou `index/lemma mismatch` | Index et métadonnées (ou index de lemmes) générés à des moments différents | Relancer avec `--force` |
| `... _abtt.json not found` | Corpus token construit sans ABTT, ou fichier supprimé | Relancer avec `--token-emb --force`, ou chercher avec `--allow-no-abtt` (tests) |
| `Assert error: query was probably encoded using a different model...` | Requête et index de modes/modèles différents (ex. `--token-emb` sur un index de phrases, ou Ollama vs `bge-m3`) | Utiliser le même mode et le même modèle pour le corpus et la requête |
| Aucun résultat en mode dossier | Index non construits | Lancer d'abord `--encode-only` (avec un wildcard), ou ajouter `--force` |
| `KeyError: 'lemmas'` (ou lemmes absents) en mode token | `.json` généré avant l'ajout du mode token | `--regenerate-metadata` avec un wildcard |
| `Embedding daemon failed to start` | Le daemon n'a pas démarré en ~60 s (modèle lent à charger, port 6000 occupé) | Le lancer à la main (`python -m utils.embed_daemon`) et lire ses journaux |

---

## Choix de conception et garde-fous

Cette section récapitule les décisions structurantes du projet et les raisons qui les motivent :

- **Daemon persistant plutôt que rechargement systématique du modèle** : le coût de chargement d'un modèle Transformer (plusieurs secondes à dizaines de secondes) est trop élevé pour être payé à chaque script et à chaque requête ; le daemon amortit ce coût sur toute une session de travail.
- **Traitement par lots dynamique côté daemon** (fenêtre de 15 ms, taille max 64) : permet de mutualiser efficacement le GPU entre plusieurs requêtes concurrentes sans imposer à un client isolé d'attendre qu'un très gros lot d'un autre client soit terminé.
- **Chargement fainéant des modèles** (phrase vs token) : chaque modèle n'est chargé que lorsqu'il est demandé par le client ; une fois chargé, il est gardé en mémoire pour traiter les requêtes futures. Cela permet d'économiser de la VRAM.
- **Libération explicite du cache CUDA après chaque lot** : réponse directe à des fuites de mémoire GPU constatées lors de l'encodage de plusieurs corpus consécutifs.
- **Normalisation L2 systématique et symétrique** (côté indexation *et* côté requête) : condition nécessaire pour que la métrique `METRIC_INNER_PRODUCT` de FAISS soit mathématiquement équivalente à une similarité cosinus.
- **Valeur par défaut de `m=512` pour `ivfpq`** (paramètre exposé dans `makeIndex()` et `makeIndex_folder()`) : choix directement issu de tests comparatifs contre une recherche exacte, qui ont montré que `nprobe` n'avait aucun impact mesurable sur la qualité des résultats, alors que `m` en avait un très net — voir [Tests](#tests--impact-de-nprobe-et-de-m).
- **Bascule automatique IVFPQ → flat** lorsque le corpus est trop petit pour un entraînement statistiquement fiable : évite de construire silencieusement un index de mauvaise qualité, au prix d'un avertissement explicite dans les logs.
- **Vérification d'intégrité index/métadonnées** (`index.ntotal == len(metadata)`) avant toute recherche : détecte les désynchronisations entre fichiers dérivés générés à des moments différents.
- **Cache incrémental à trois niveaux** (embeddings → métadonnées → index) : évite de recalculer des embeddings coûteux en GPU/CPU si seule l'étape d'indexation ou de reconstruction des métadonnées doit être rejouée.
- **Suffixe `_token` systématique** sur les fichiers dérivés du mode token : évite d'écraser accidentellement les embeddings/index « phrase » d'un même corpus.
- **Mode `--no-faiss` de validation** : permet de comparer, sur un même corpus, les résultats d'une recherche FAISS (potentiellement approximative selon le type d'index) à un calcul de similarité cosinus exact, pour détecter une éventuelle perte de qualité liée à l'indexation.
- **Reconstruction systématique depuis les formes de surface (CoNLL-U)** : le texte, les tokens et les lemmes proviennent des mêmes lignes d'annotation, ce qui garantit leur alignement position par position — condition nécessaire pour associer chaque vecteur de token à son lemme en mode token.
- **Un vecteur par lemme en mode token** : moyenner les vecteurs contextuels de toutes les occurrences d'un lemme donne un point unique par lemme, ce qui réduit fortement la taille de l'index et permet de chercher des mots proches d'un mot, tout en conservant l'accès aux occurrences réelles (`token`, `sent_id`) via les métadonnées.
- **ABTT appliquée à la matrice de lemmes, avec paramètres persistés** (`_abtt.json`) : appliquer la transformation aux embeddings de tokens individuels demanderait une mémoire prohibitive ; sauvegarder `mu` et `P` permet de transformer la requête exactement comme le corpus, condition pour qu'elle soit dans le même espace.
- **Requête moyennée en mode token** : une requête de plusieurs mots est réduite à un seul vecteur, dans le même espace que l'index de lemmes.
- **Cache sensible à l'ABTT** : changer `--no-abtt` invalide le cache d'embeddings de mode token, pour ne jamais mélanger des espaces transformés et non transformés.
- **Étiquetage explicite des phrases manquantes** (`[phrase manquante]`) plutôt que leur suppression : préserve l'alignement entre indices FAISS, métadonnées et phrases d'origine.

---

## Limitations connues et points d'attention

Points relevés en relisant le code ; ils ne sont pas tous corrigés à ce jour.

**Mode token**
- **Alignement `tokens` / `lemmas` pour XML et TRS** : les tokens viennent de `get_tokens` (découpage sur les espaces et les apostrophes) et les lemmes de `simplemma.text_lemmatizer` (qui isole aussi la ponctuation). Les deux listes n'ont pas toujours la même longueur ; `build_lemma_embeddings` ignore alors la phrase (avertissement `mismatch`). Le CoNLL-U, lui, est aligné par construction.
- **Troncature** : les phrases sont tronquées à 512 sous-tokens par le tokenizer. Une phrase plus longue produit moins de vecteurs que de lemmes et est ignorée de la même façon.
- **Ollama** : `use_ollama` ne renvoie pas de vecteurs par mot ; ne pas l'utiliser avec `--token-emb`.
- **`--no-faiss` en mode token** : les résultats sont associés aux phrases de `corpus.json` alors que les scores portent sur des lemmes ; l'affichage n'est fiable qu'en mode phrase.
- **Fichier `.faiss` passé directement avec `--token-emb`** : `main()` ajoute `_token` au nom de base, sans retirer celui déjà présent dans `corpus_token.faiss`. Les chemins dérivés (`corpus_token_token.faiss`, `corpus_token.json`, …) peuvent alors être erronés. Préférer passer le fichier de corpus source (`corpus.conllu`) plutôt que l'index.
- **`encode_no_daemon` en mode token** n'utilise pas `torch.no_grad()` : sur un gros corpus, la consommation mémoire est plus élevée que via le daemon.
- **Anciens `.json`** sans clé `lemmas` : la recherche en mode token échoue tant qu'ils ne sont pas régénérés (`--regenerate-metadata`).

**Ligne de commande**
- `--regenerate-metadata` avec `--folder` ne filtre pas les extensions et écrase avec `null` les `.json` de tous les fichiers non reconnus : utiliser un wildcard.
- `--encode-only` avec un nom de dossier seul (sans wildcard) ne traite aucun fichier ; il faut `"dossier/*"`.
- `--encode-only` recalcule un `.npy` supprimé, même si l'index `.faiss` existe.
- `--use-ollama` n'est transmis à l'encodage qu'en mode dossier / wildcard ; en fichier unique et en `--encode-only`, il est ignoré.
- Avec `--no-abtt` sans `--allow-no-abtt`, la recherche finale en fichier unique s'arrête sur une erreur après avoir construit les fichiers.

**Daemon**
- Le worker regroupe les requêtes de plusieurs clients sans tenir compte de leur `token_mode` : deux clients demandant simultanément un mode phrase et un mode token peuvent être mélangés dans un même lot. Le daemon est prévu pour être utilisé par un seul mode à la fois.
- `handle_timeout` n'est toujours pas branchée : les modèles ne sont jamais déchargés automatiquement.
- `handle_client` affiche `token_mode` avec un `print` de débogage à chaque requête.
- Le daemon écoute en local sans authentification (voir la section de `main()` de `utils/embed_daemon.py`) et n'est pas géré comme un service système : l'arrêt se fait à la main (`kill`).

**Index et métadonnées**
- Un index FAISS n'est valable que pour la version des métadonnées qui l'a produit : `main()` détecte un décalage de taille (`index.ntotal`), pas un changement d'ordre. Après un changement de parseur, utiliser `--force`.
- Avec plusieurs fichiers contenant le même texte, les doublons occupent plusieurs places dans le top-k de `search_folder`.
