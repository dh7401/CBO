import os
import random
import subprocess
import torch

from lunar_lander.lunar_lander import demo_heuristic_lander


def mopta_evaluate(X):
    '''
    input: torch.tensor \in [0, 1]^124
    output: torch.tensor([f, c_1, \dots, c_68])
    '''
    suffix = str(random.getrandbits(64))
    os.mkdir(os.getcwd() + "/tmp_" + suffix)
    os.chdir(os.getcwd() + "/tmp_" + suffix)

    f = open("input.txt", "w")
    f.write("\n".join([str(x.item()) for x in X]))
    f.close()
    subprocess.Popen("../mopta_fortran_unix/run", stdout=subprocess.DEVNULL).wait()
    f = open("output.txt", "r")
    lines = f.readlines()
    f.close()

    os.remove(os.getcwd() + "/input.txt")
    os.remove(os.getcwd() + "/output.txt")
    os.chdir("../")
    os.rmdir(os.getcwd() + "/tmp_" + suffix)
    return torch.tensor([float(line.strip()) for line in lines])

def lunar_lander_evaluate(X, m=50):
    '''
    input: torch.tensor \in [0, 1]^12
    output: torch.tensor([f, c_1, \dots, c_m])
    '''
    results = [0.]
    for i in range(m):
        reward = demo_heuristic_lander(seed=i, params=X.tolist())
        results[0] += reward
        results.append(200 - reward)
    results[0] /= m
    return torch.tensor(results)
