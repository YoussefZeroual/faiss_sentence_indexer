#!/usr/bin/env python3
"""
Interface en ligne de commande (CLI) pour la recherche sémantique sur corpus linguistiques.
Elle permet d'orchestrer l'utilisation des autres modules:
- makeIndex.py
- calcEmbeddings.py
- searchEmbedding.py
Ce script centralise toutes les opérations du pipeline : extraction de textes, génération
d'embeddings, création d'index FAISS et exécution de requêtes de similarité.
Concernant la création d'index (.faiss), embeddings (.npy) ou métadonnées (JSON) il permet de gérer un système de cache (ne recalculer que ce qui est manquant à la demande de l'utilisateur)
et propose de nombreux modes d'exécution (fichier unique, dossier complet, encodage seul,
recherche sans FAISS pour débogage, etc.).
"""
import argparse
import os
import sys
import glob
import faiss
import time
import numpy as np
from calcEmbeddings import calcEmbeddings, save_metadata, parse_sentences,encode_folder,fix_punctuation_spaces
from makeIndex import makeIndex,makeIndex_folder,load_embeddings
from utils.embed_client import encode
from searchEmbedding import search, load_metadata, load_index, search_folder,embedd_query
import logging

# serveur Ollama par défaut
OLLAMA_HOST = 'localhost:11434'
# model Ollama par défaut
OLLAMA_MODEL = 'nomic-embed-text-v2-moe:latest'
# différents modes d'exécution selon le type de fichier
MODE_BY_EXT = {".conllu": "conllu", ".xml": "xml",".trs":"trs",".faiss":"faiss"}
# configuration du logger pour la journalisation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #filename='/home/miai_guest/zeroualy/module_faiss/app.log'
)
logger = logging.getLogger(__name__)


def get_sent_context(f,sent_id,context_size=10):
    """
    Récupère le contexte textuel autour d'une phrase spécifique (phrases précédentes et suivantes).
    Utile pour examiner les correspondances de recherche dans leur environnement discursif d'origine.

    Args:
        f (str): Chemin vers le fichier de métadonnées (.json) correspondant au corpus.
        sent_id (int/str): L'identifiant ou l'index de la phrase cible.
        context_size (int, optional): Le nombre de phrases à inclure avant et après la cible (fenêtre de contexte).

    Valeur de retour:
        str ou None: Le bloc de texte concaténé contenant le contexte, ou None si introuvable.
    """
    context_size = int(context_size)
    sent_id = int(sent_id)
    # Chargement des phrases depuis le fichier JSON
    data = load_metadata(f)
    s = data['raw_text']
    target = None

    # Vérification des limites
    if (sent_id > len(s)):
        logger.warning("sent_id %s out of range, max %s",sent_id,len(s))
        return None
    if not data:
        return None

    # Recherche de l'index exact correspondant au sent_id
    for i,sent in zip(data['sent_id'],data['raw_text']):
        i=int(i)
        sent_id = int(sent_id)
        if(i==sent_id):
            # Concaténation des phrases se trouvant dans la fenêtre [i-context_size : i+context_size]
            target = ' '.join([s_ for i_,s_ in enumerate(s[i-context_size:i+context_size])])
            break
    # Nettoyage typographique final du bloc de contexte extrait
    try:
        target = fix_punctuation_spaces(target)
    except TypeError as e:
        logger.warning("%s, content:%s...",e,target[:20] if target else "[None]")
        return target
    return target
def process_no_faiss(args):
    """
    Effectue une recherche de similarité par produit matriciel direct (dot product) sur les vecteurs normalisés L2 (équivalent de la similarité cosinus)
    sans utiliser d'index FAISS. Sert à vérifier la fiabilité des recherches basées sur FAISS en comparant leur résultats à une recherche directe dans les embeddings.

    Args:
        args (argparse.Namespace): Les arguments de la ligne de commande.

    Valeur retournée:
        bool: False si l'argument no_faiss n'est pas trouvé dans l'objet argparse sinon complète l'exécution par un affichage.
    """

    t0 = time.perf_counter()   # chronomètre pour calculer le temps d'exécution'
    print("Using no faiss mode: calculating cosine sim directly on embeddings")
    print("-----------")
    results = []
    token_ext = ""
    if args.token_emb:
        token_ext = "_token"
    if not args.no_faiss:
        return False
    # Encodage de la requête de l'utilisateur
    query_embs = embedd_query(query_str=args.query,token_mode=args.token_emb,no_daemon=args.no_daemon)
    # Mode traitement par lots (dossier ou wildcard glob)
    if args.folder or '*' in args.input_file:
        files = glob.glob(args.input_file)
        files = [f for f in files if os.path.splitext(f)[1] in ['.conllu','.xml','.trs']]


        len_f = len(files)
        print(f"processing {len_f} files")
        for i,f in enumerate(files):
            #query_embs = embedd_query(query_str=args.query,token_mode=args.token_emb)
            base,ext = os.path.splitext(f)
            print(f"Processing file {i}/{len_f}: {base+token_ext+'.npy'}")
            try:
                # Chargement des embeddings bruts
                embs = load_embeddings(base+token_ext+".npy")
                #combined = np.vstack([embs,query_embs])
                #from utils.embed_daemon import all_but_the_top
                #combined = all_but_the_top(combined,2)
                #embs = combined[len(embs):]
                #query_embs = combined[:len(embs)]

                # Normalisation L2 indispensable pour que le produit scalaire équivaille à la similarité cosinus
                faiss.normalize_L2(embs)
                faiss.normalize_L2(query_embs)
            except FileNotFoundError as e:
                print("File not found:",base+token_ext+".npy","skipping")
                continue
            try:
                # Calcul de similarité vectorielle brute (produit matriciel)
                result = query_embs@embs.T
            except ValueError as e:
                print("Embedding and query dimension mismatch, skipping file:",base+token_ext+".npy")
                print("query dim",query_embs.shape,"embs dim",embs.shape)
                continue
            result = result[0]
            # Arrondi des scores pour un affichage lisible
            result = [np.round(float(r),3) for r in result]
            metadata = load_metadata(base+".json")
            # Agrégation des résultats (fichier, id, texte, score)
            results.extend(zip([f]*len(result),metadata["sent_id"],metadata["raw_text"],result))
    # Mode fichier unique
    else:
        base,ext = os.path.splitext(args.input_file)
        try:
            embs = load_embeddings(base+token_ext+".npy")
            #combined = np.vstack([embs,query_embs])
            #from utils.embed_daemon import all_but_the_top
            #combined = all_but_the_top(combined,5)
            #embs = combined[:len(embs)]
            #query_embs = combined[len(embs):]
            faiss.normalize_L2(embs)
            faiss.normalize_L2(query_embs)
        except FileNotFoundError as e:
            print(f"File not found {base+token_ext+'.npy'}")
            return False
        try:
            result = query_embs@embs.T
        except ValueError as e:
            print("Embedding and query dimension mismatch, skipping file")
            print("query dim",query_embs.shape,"embs dim",embs.shape)
            return False
        result = result[0]
        # Arrondi des scores pour un affichage lisible
        result = [np.round(float(r),3) for r in result]
        # chargement des métadonnées
        metadata = load_metadata(base+".json")
        # Agrégation des résultats (fichier, id, texte, score)
        results.extend(zip([args.input_file]*len(result),metadata["sent_id"],metadata["raw_text"],result))
    # Tri global de tous les résultats par score décroissant (indice 3)
    results.sort(key=lambda x:x[3],reverse=True)
    # Affichage final
    print("--------------")
    print("Query results")
    print("--------------")
    print("Collection file                             | sent_id         | Sentence     | SImilarity score")
    for r in results[:args.top_k]:
        print(r[0],"|   ",r[1],"   ",r[2],"|   ",r[3])
    t1 = time.perf_counter()
    exec_time = np.round(t1-t0,3)
    print(f"Execution time: {exec_time}s")
def parse_args():
    """
    Définit et analyse les arguments de l'interface en ligne de commande (CLI).

    Valeur retournée:
        argparse.Namespace: Un objet contenant tous les arguments parsés.
    """


    parser = argparse.ArgumentParser(description="Semantic search over a corpus using FAISS.")

    # Arguments obligatoires
    parser.add_argument("input_file", help="Path to the corpus file (.conllu, .xml or .trs)")
    parser.add_argument("query", help="Query string to search for")

    # Paramètres de l'index FAISS et de la recherche
    parser.add_argument("--index-type", choices=["flat", "hnsw", "ivfpq"], default="ivfpq",
                         help="FAISS index type (default: flat)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return (default: 10)")

    # Optimisations et forçage
    parser.add_argument("--reduce-precision", action="store_true",
                         help="Save embeddings as float16 to save disk space")
    parser.add_argument("--force", action="store_true",
                         help="Force recomputation of embeddings/metadata/index even if cached files exist")

    # Modes d'exécution globaux
    parser.add_argument("--folder", action="store_true",
                         help="process a folder")
    parser.add_argument("--log",action="store_true",help="enables logging info")
    parser.add_argument("--encode-only",action="store_true",help="encodes files and creates indexes without running queries (only in folder mode)")
    parser.add_argument("--search-only",action="store_true",help="searches presuming index files already exist to skip checking and improve speed (only in folder mode)")
    parser.add_argument("--regenerate-metadata",action="store_true",help="regenerate metadata of a folder or a file")
    parser.add_argument("--warn",action="store_true",help="enables log for warnings only")

    # Paramètres liés à l'architecture d'embedding
    parser.add_argument("--token-emb",action="store_true",help="enables token level embedding mode instead of sentence embedding")
    parser.add_argument("--no-faiss",action="store_true",help="processes a query using models directly without FAISS indexation, for testing purpose")
    parser.add_argument("--no-daemon",action="store_true",help="loads embedding models locally and use them instead of calling the daemon")
    parser.add_argument("--use-ollama",action="store_true",help="Use ollama for embeddings")
    return parser.parse_args()
def process(args,index, metric_type=faiss.METRIC_INNER_PRODUCT, metadata=None):
    """
    Exécute une requête de recherche sémantique sur un index FAISS unique et affiche les résultats.

    Args:
        args (argparse.Namespace): objet arguments de la ligne de commande contenant la requête et les options.
        index (faiss.Index): L'index FAISS chargé en mémoire.
        metric_type (int): Type de métrique de distance (défaut: produit scalaire).
        metadata (dict): Métadonnées associées aux vecteurs (ID, phrases, tokens).
    """
    query_str=args.query
    top_k=args.top_k
    import time
    t0 = time.perf_counter()

    # Exécution de la recherche vectorielle via le module searchEmbedding
    result = search(query_str=args.query, index=index, metric_type=faiss.METRIC_INNER_PRODUCT, top_k=args.top_k, metadata=metadata,token_mode=args.token_emb,no_daemon=args.no_daemon)
    t1 = time.perf_counter() # chronométrage du temps d'exécution
    exec_time = t1 - t0

    # Affichage des résultats dans la console
    print("Sent id               | Sentence    | similarity score")
    for r in result:
        print(f"{r[0]} | {r[1]} |  {r[2]}")
    print("temps d'exécution de la requête Faiss:",np.round(exec_time,2))
def process_folder(args,input_file):
    """
    Exécute une recherche sémantique sur un répertoire entier ou un lot de fichiers (via un wildcard).

    Optimisation majeure : la requête de l'utilisateur est encodée en vecteur une seule fois
    au début, puis ce vecteur est passé à la fonction de recherche globale pour interroger
    tous les index du dossier.

    Args:
        args (argparse.Namespace): Arguments de la ligne de commande.
        input_file (str): Chemin du dossier cible ou motif de recherche (ex: 'corpus/*.conllu').
    """
        # 1. Encodage unique de la requête pour gagner du temps lors du parcours des multiples fichiers
    query_vector = embedd_query(args.query,args.token_emb,no_daemon=args.no_daemon,use_ollama=args.use_ollama,ollama_host=OLLAMA_HOST,ollama_model=OLLAMA_MODEL)
    base, ext = os.path.splitext(input_file)
    # 2. récupération des chemins des fichiers
    if '*' in input_file:
            files = glob.glob(input_file)
    else:
        # Si un dossier classique est fourni, on cible tous ses fichiers .faiss
        files = glob.glob(base+"/"+"*faiss")
        # Filtrage pour ne conserver que les fichiers index
        files = [f for f in files if os.path.splitext(f)[1] in ['.faiss']]
        print("Processing folder: searching similarity in",len(files),"index files")
        # 3. Mode Force : recalcule tout le dossier (embeddings et index) avant la recherche
    if args.force:
        encode_folder(input_file,overwrite=True,token_mode=args.token_emb,no_daemon=args.no_daemon,use_ollama=args.use_ollama,ollama_host=OLLAMA_HOST,ollama_model=OLLAMA_MODEL)
        makeIndex_folder(input_folder=input_file,metric_type=faiss.METRIC_INNER_PRODUCT,index_type=args.index_type,overwrite=True,token_mode= args.token_emb)
    print("------------")
        # 4. Lancement de la recherche globale sur le répertoire avec le vecteur pré-calculé
    search_folder(input_file,query_vector=query_vector,metric_type=faiss.METRIC_INNER_PRODUCT,top_k=args.top_k,token_mode=args.token_emb,verbose=True)
def main():
    """
    Point d'entrée principal du script CLI.

    Orchestre la logique conditionnelle du pipeline :
    1. Analyse les arguments et configure le niveau de journalisation (logging).
    2. Détermine l'état du cache (quels fichiers .npy, .json, .faiss manquent).
    3. Renvoie vers des modes spécifiques si besoin (sans-faiss, encodage seul, etc.).
    4. Gère le traitement par lots (dossiers/wildcards) ou le traitement d'un fichier unique.
    5. Reconstruit les index ou les embeddings manquants si nécessaire.
    6. Lance la requête de recherche finale.
    """
    args = parse_args()
    input_file = args.input_file
    folder = args.folder

    # Définition des chemins de fichiers de sortie attendus
    base, ext = os.path.splitext(input_file)
    output_index = base + ".faiss"
    output_metadata = base + ".json"
    output_embeddings = base + ".npy"
    # Évaluation de l'état du cache (fichiers manquants ou écrasement forcé)
    # Règle : Si les embeddings sont recalculés, l'index doit être reconstruit.
    embeddings_missing = args.force or not os.path.exists(output_embeddings) or not os.path.exists(output_metadata)
    index_missing = args.force or (embeddings_missing) or (not os.path.exists(output_index))
    metadata_messing = not os.path.exists(output_metadata)

    # Configuration du niveau de journalisation pour tous les modules utilisés
    if not args.log:
        logging.getLogger("searchEmbedding").setLevel(logging.ERROR)
        logging.getLogger("calcEmbeddings").setLevel(logging.ERROR)
        logging.getLogger("makeIndex").setLevel(logging.ERROR)
        logging.getLogger("utils.embed_client").setLevel(logging.ERROR)

    if args.log:
        logging.getLogger("searchEmbedding").setLevel(logging.INFO)
        logging.getLogger("calcEmbeddings").setLevel(logging.INFO)
        logging.getLogger("makeIndex").setLevel(logging.INFO)
        logging.getLogger("utils.embed_client").setLevel(logging.INFO)
    elif args.warn:
        logging.getLogger("searchEmbedding").setLevel(logging.WARNING)
        logging.getLogger("calcEmbeddings").setLevel(logging.WARNING)
        logging.getLogger("makeIndex").setLevel(logging.WARNING)
        logging.getLogger("utils.embed_client").setLevel(logging.WARNING)


    # Renvoi vers les modes d'exécution spéciaux
    # Mode 1 : Recherche mathématique directe sur les fichiers embeddings (.npy) (sans FAISS), afin de tester les performances des index faiss en comparant leurs résultats à ceux de ce mode.
    if args.no_faiss:
        process_no_faiss(args)
        return True
    if '*' in input_file:
        files = glob.glob(input_file)
    elif not os.path.exists(input_file):
        sys.exit(f"Error: input file not found: {input_file}")
    # Mode 2 : Mode qui limite le traitement à regénérer les métadonnées (JSON) (utile si le script de parsing a été mis à jour, afin de mettre à jour les fichiers JSON)
    if args.regenerate_metadata:
        files = []
        if '*' in input_file:
            files = glob.glob(input_file)
            files = [f for f in files if ".conllu" in f or '.xml' in f or '.trs' in f]
        if folder:
            files = glob.glob(input_file+"/*")
        metadata = None
        len_f = len(files)
        # Traitement par lot des métadonnées
        if args.folder or "*" in input_file:
            logger.info("regenerating metadata for %s files",len(files))
            for i,f in enumerate(files):
                logger.info("regenerating metadata for file %s/%s filename=%s",i,len_f,f)
                _,metadata = parse_sentences(f)
                base, ext = os.path.splitext(f)
                save_metadata(metadata,base+".json")
        # Traitement d'un fichier unique
        elif not args.folder:
            logger.info("regenerating metadata for %s",input_file)
            _,metadata = parse_sentences(input_file)
            index = load_index(base+".faiss")
            save_metadata(metadata,base+".json")

        return True
    # Mode 3 : L'utilisateur pointe directement vers un index FAISS existant
    if ext == ".faiss" :
        metadata = load_metadata(output_metadata)
        index = load_index(output_index)
        process(args,index=index,metadata=metadata)
        return True
    # Mode 4 : Mode pré-calcul (encodage + indexation), sans lancer de requête utilisateur
    if args.encode_only and (folder or "*" in input_file):
        print("Encoding files")
        files = glob.glob(input_file)
        files = [f for f in files if os.path.splitext(f)[1] in ['.conllu','.xml','.trs']]
        token_ext = ""
        if args.token_emb:
            token_ext = "_token"
        print("checking and encoding",len(files),"files")
        cnt = 1
        len_f = len(files)
        for f in files:
            print(f"processing file {cnt}/{len_f} {f}")
            print("------")
            base,ext = os.path.splitext(f)
            index_file = base+token_ext+'.faiss'
            metadata_file = base+'.json'
            embs_file = base+token_ext+'.npy'
            print(embs_file)
            # Étape A : Métadonnées
            if not os.path.exists(metadata_file) or args.force:
                if args.force:
                    print("force mode enabled")
                else:
                    print("metadata file not found",metadata_file)
                _,metadata = parse_sentences(f)
                save_metadata(metadata,base+".json")
            else:
                print("Found metadata file",metadata_file)
            # Étape B : Embeddings
            if not os.path.exists(embs_file) or args.force:
                if args.force:
                    print("force mode enabled")
                else:
                    print("Embeddings file not found, regenerating",metadata_file)
                embeddings, metadata = calcEmbeddings(collection_file_path=f, output_file_path=embs_file, mode=ext.replace('.','').strip(),
                                               reduce_precision=args.reduce_precision,overwrite=args.force,token_mode=args.token_emb,no_daemon=args.no_daemon)
            else:
                print("Found embeddings file",embs_file)
            # Étape C : Indexation
            if not os.path.exists(index_file) or args.force:
                if args.force:
                    print("force mode enabled")
                else:
                    print("Index file not found, building",index_file)
                index=makeIndex(embeddings=None, embedding_file_path=embs_file,
                               metric_type=faiss.METRIC_INNER_PRODUCT,
                               index_type=args.index_type, output_file_path=index_file)
            else:
                print("Found index file",index_file)
            print("------")
            cnt+=1
        return True
    # Mode 5 : Recherche seule sur un dossier existant (optimisation vitesse)
    if args.search_only and folder:
        query_vector = embedd_query(args.query)
        process_folder(args,base)
        return True
    # Mode 6 : Traitement standard de dossiers ou de wildcards
    if folder:
        process_folder(args,base)
        return True
    if'*' in input_file:
        process_folder(args,input_file)
        return True

    #Mode 7 : Traitement standard d'un fichier corpus unique
    mode = MODE_BY_EXT.get(ext.lower())
    # Cas 7.1 : Données déjà prêtes
    if embeddings_missing and not index_missing and not args.force:
        index = load_index(output_index)
        metadata = load_metadata(output_metadata)
    elif (mode is None) and (not folder) and (embeddings_missing) and (index_missing):
        sys.exit(f"Error: unsupported file extension '{ext}' (expected one of {list(MODE_BY_EXT)})")
    # Cas 7.3 : Tout reconstruire (Fichier index/embeddings absents ou mode forcé --force)
    elif (embeddings_missing and index_missing) or args.force:
        embeddings, metadata = calcEmbeddings(input_file, output_embeddings, mode,
                                               reduce_precision=args.reduce_precision,overwrite=args.force,token_mode=args.token_emb,no_daemon=args.no_daemon)
        save_metadata(metadata, output_metadata)
        index=makeIndex(embeddings=embeddings, embedding_file_path=None,
                               metric_type=faiss.METRIC_INNER_PRODUCT,
                               index_type=args.index_type, output_file_path=output_index)
        embeddings_missing = False
        index_missing = False
    # Cas 7.4 : Charger les embeddings bruts (.npy) pour refaire juste l'index
    else:
        embeddings = np.load(output_embeddings)
        metadata = load_metadata(output_metadata)

    # Re-création isolée de l'index si manquant
    if index_missing:
        try:
            index = makeIndex(embeddings=embeddings, embedding_file_path=None,
                               metric_type=faiss.METRIC_INNER_PRODUCT,
                               index_type=args.index_type, output_file_path=output_index)
        except ValueError as e:
            # Alternative automatique : si IVFPQ manque de données d'entraînement, on bascule sur 'flat'
            if args.index_type == "ivfpq":
                print(f"Warning: {e}. Falling back to 'flat' index.", file=sys.stderr)
                index = makeIndex(embeddings=embeddings, embedding_file_path=None,
                                   metric_type=faiss.METRIC_INNER_PRODUCT,
                                   index_type="flat", output_file_path=output_index)
            else:
                raise
    # Validation d'intégrité finale : vérifie que l'index FAISS et le JSON ont le même nombre d'entrées
    if not index_missing:
        index = load_index(output_index)
        if index.ntotal != len(metadata["raw_text"]):
            sys.exit(f"Error: index/metadata mismatch (index has {index.ntotal} vectors, "
                    f"metadata has {len(metadata['raw_text'])} entries). "
                    f"Re-run with --force to rebuild.")
    # Lancement de la requête utilisateur sur l'index préparé
    process(args,index=index,metadata=metadata)


if __name__ == "__main__":
    main()
