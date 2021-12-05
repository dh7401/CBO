import subprocess
import torch

def mopta_evaluate(X):
  '''
  input: torch.tensor \in [0, 1]^124
  output: torch.tensor([f, c_1, \dots, c_68])
  '''
  f = open('input.txt', 'w')
  f.write('\n'.join([str(x.item()) for x in X]))
  f.close()

  subprocess.Popen('#!/usr/bin/env bash ./mopta_fortran_unix/run', stdout=subprocess.DEVNULL).wait()
  f = open('output.txt', 'r')
  lines = f.readlines()
  f.close()

  return torch.tensor([float(line.strip()) for line in lines])


def mopta_evaluate1(X):
  '''
  input: torch.tensor \in [0, 1]^124
  output: torch.tensor([f, c_1, \dots, c_68])
  '''
  f = open('input1.txt', 'w')
  f.write('\n'.join([str(x.item()) for x in X]))
  f.close()

  subprocess.Popen('./mopta_fortran_unix/run1', stdout=subprocess.DEVNULL).wait()
  f = open('output1.txt', 'r')
  lines = f.readlines()
  f.close()

  return torch.tensor([float(line.strip()) for line in lines])


def mopta_evaluate2(X):
  '''
  input: torch.tensor \in [0, 1]^124
  output: torch.tensor([f, c_1, \dots, c_68])
  '''
  f = open('input2.txt', 'w')
  f.write('\n'.join([str(x.item()) for x in X]))
  f.close()

  subprocess.Popen('./mopta_fortran_unix/run2', stdout=subprocess.DEVNULL).wait()
  f = open('output2.txt', 'r')
  lines = f.readlines()
  f.close()

  return torch.tensor([float(line.strip()) for line in lines])
