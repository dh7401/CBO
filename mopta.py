import os
import random
import subprocess
import torch

def mopta_evaluate(X):
  '''
  input: torch.tensor \in [0, 1]^124
  output: torch.tensor([f, c_1, \dots, c_68])
  '''
  suffix = str(random.getrandbits(64))
  os.mkdir(os.getcwd() + '/tmp' + suffix)
  os.chdir(os.getcwd() + '/tmp' + suffix)

  f = open('input.txt', 'w')
  f.write('\n'.join([str(x.item()) for x in X]))
  f.close()
  subprocess.Popen('../mopta_fortran_unix/run', stdout=subprocess.DEVNULL).wait()
  f = open('output.txt', 'r')
  lines = f.readlines()
  f.close()

  os.remove(os.getcwd() + '/input.txt')
  os.remove(os.getcwd() + '/output.txt')
  os.chdir('../')
  os.rmdir(os.getcwd() + '/tmp' + suffix)
  print(os.getcwd())
  return torch.tensor([float(line.strip()) for line in lines])
  