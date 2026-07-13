## Vue d'ensemble des scripts
- **`embed_daemon.py`** : processus persistant qui garde le modèle SentenceTransformer (Sentence-CamemBERT) chargé en mémoire en permanence afin d'éviter de le recharger à chaque requête. Les scripts `calcEmbedding.py` et `searchEmbedding.py` communiquent avec ce processus pour encoder les phrases avec le modèle SentenceTransformer. Il faut donc lancer ce processus au préalable avant toute autre opération (à partir du dossier utils).
- **`calcEmbeddings.py`** : parse le corpus (CONLLU ou XML), extrait les phrases + métadonnées (`sent_id`, texte), encode via CamemBERT (par le daemon), sauvegarde les embeddings (`.npy`) + métadonnées (`.json`). Seul le mode CONLLU a été testé pour l'instant, le mode xml ne fonctionne pas encore. La fonction qui parse le CONLLU est très simple: elle cherche les éléments #text_raw en utilisant une regex, afin d'optimiser les performances.
- **`makeIndex.py`** : construit un index FAISS (flat/HNSW/IVFPQ) à partir des embeddings normalisés, écrit sur disque.
- **`searchEmbedding.py`** : encode une phrase requête, cherche les k plus proches voisins (cosine via inner product), renvoie `(sent_id, texte, score)`.
- **`main.py`** : permet de tester tout le workflow en une seule ligne de commande.

## Pour tester

1. Lancer `utils/embed_daemon.py` et garder le processus ouvert.
2. Lancer la commande :
   ```bash
   ./main.py test/HS36-6030v2tv8.conllu "la vie est belle"
   ```

## Exemples d'utilisation des scripts

### `calcEmbeddings.py`

La fonction principale est `calcEmbeddings()` :

```python
from calcEmbeddings import calcEmbeddings, save_metadata

# Calcule les embeddings du corpus + les métadonnées associées (sent_id, texte...)
embeddings, metadata = calcEmbeddings(
    collection_file_path="test/HS36-6030v2tv8.conllu",  # fichier source (CONLLU ou XML)
    output_file_path="test/HS36-6030v2tv8.npy",          # où sauvegarder les embeddings
    mode="conllu",                                       # format du corpus
    reduce_precision=False                                # garder la précision float d'origine
)

# Sauvegarde les métadonnées correspondant aux embeddings
save_metadata(metadata, "test/HS36-6030v2tv8.json")
```

### `makeIndex.py`

La fonction principale est `makeIndex()` :

```python
import faiss
import numpy as np
from makeIndex import makeIndex

# Charge les embeddings précédemment calculés
embeddings = np.load("corpus_embeddings.npy")

# Construit l'index FAISS à partir des embeddings
index = makeIndex(
    embeddings=embeddings,
    embedding_file_path=None,                # non utilisé ici car embeddings déjà en mémoire
    metric_type=faiss.METRIC_INNER_PRODUCT,   # similarité cosinus via produit scalaire
    index_type="ivfpq",                       # type d'index FAISS (flat/HNSW/IVFPQ)
    output_file_path="test/HS36-6030v2tv8.faiss"  # chemin de sauvegarde de l'index
)
```

### `searchEmbedding.py`

La fonction principale est `search()` :

```python
import faiss
from searchEmbedding import search, load_index, load_metadata

# Charge l'index FAISS et les métadonnées associées
index = load_index("corpus.faiss")
metadata = load_metadata("test/HS36-6030v2tv8.json")

# Encode la requête et cherche les k plus proches voisins
results = search(
    query_str="il fait chaud",
    index=index,
    metric_type=faiss.METRIC_INNER_PRODUCT,
    top_k=10,
    metadata=metadata
)

# Affiche les résultats : identifiant, score de similarité, texte
for sent_id, text, score in results:
    print(sent_id, score, text)
```
