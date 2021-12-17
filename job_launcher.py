import subprocess
import time

import torch

job_list = []
file_list = []

for i in range(5):
    for problem in ["mopta", "lunar"]:
        job_list.append(["python", "turbo.py", "--seed", str(i), "--problem", problem, "--gpu_idx"])
        file_list.append(f"{problem}_scbo_{i}.txt")

        for search in ["ets", "hts"]:
            job_list.append(["python", "mtgp.py", "--seed", str(i), "--problem", problem, "--search", search, "--gpu_idx"])
            file_list.append(f"{problem}_{search}_{i}.txt")

n_gpus = torch.cuda.device_count()
gpu_procs = [None] * n_gpus

while job_list:
    for j in range(n_gpus):
        if (gpu_procs[j] is None) or (gpu_procs[j].poll() is not None):
            job = job_list.pop()
            f = open(file_list.pop(), "w")
            gpu_procs[j] = subprocess.Popen(job + [str(j)], stdout=f, stderr=f)
            continue
    
    time.sleep(3)
