from searchEmbedding import search_folder

import numpy as np
import time
import logging
logging.disable(logging.WARNING)
import threading
import glob
N_THREADS = 3
def run_folder(t,data):
    print(f"launched thread No.{t}")
    t0 = time.perf_counter()
    search_folder(input_folder="./index",query_str="la vie est belle",top_k=10,verbose=False)
    t1 = time.perf_counter()

    exec_time = np.round(t1-t0,2)
    data["exec_time"].append(exec_time)
    data["thread"].append(float(t))
threads = []
data = {"exec_time":[],
        "thread":[]}
files = glob.glob("./index/*.faiss")
len_f = len(files)

t0 = time.perf_counter()
for i in range(N_THREADS):
    t = threading.Thread(target=run_folder,args=(i,data,))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
t1 = time.perf_counter()
exec_time = np.round(t1-t0,2)
print(f"-----------\nTotal exec time:{exec_time}\nNumber of threads: {N_THREADS}")
print(f"n files = {len_f}*2")
for time,thread in zip(data["exec_time"],data["thread"]):
    print("Execution time for thread",thread,": ",time)
