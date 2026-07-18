from searchEmbedding import search_folder

import numpy as np
import time
import logging
logging.disable(logging.WARNING)
import threading
import glob
def run_folder(t):
    print(f"launched thread No.{t}")
    t0 = time.perf_counter()
    search_folder(input_folder="./test",query_str="allah",top_k=10,verbose=True)
    t1 = time.perf_counter()

    exec_time = np.round(t1-t0,2)

    print(f"exec_time for thread {t}:{exec_time}")
threads = []
files = glob.glob("./test/*faiss")
len_f = len(files)
print(f"n files = {len_f}*2")
t0 = time.perf_counter()
for i in range(5):
    t = threading.Thread(target=run_folder,args=(i,))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
t1 = time.perf_counter()
exec_time = np.round(t1-t0,2)
print(f"Total exec time:{exec_time}")
