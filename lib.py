from random import random


def Matricial(matrix1, matrix2):
  for i in range(len(matrix1, 0)):
    for j in range(len(matrix2, 1)):
      z = [len(matrix2, 0), len(matrix1, 1)]
      z[i, j] = 0
      for k in range(len(matrix1, 1)):
        z[i, j] += matrix1[i, k] * matrix2[k, j]

  return z

def Sigmoid(s):
  return 1 / (1 + (E() ** - s))

def SigmoidPrime(s):
  return s * (1 - s)

def E():
  n = 9999999999999
  return (1 + (1 / n)) ** n

def Randn(matrix1, matrix2):
  z = [matrix1, matrix2]
  for i in range(matrix1):
    for j in range(matrix2):
      z[i, j] = random

def GetMax(matrix, axis):
  max = 0
  for i in range(len(matrix, axis)):
    if(matrix[i] > max):
      max = matrix[i]

  return max

def MatrixEchelon(matrix, axis):
  for i in range(len(matrix, axis)):
    matrix[i] /= GetMax(matrix, axis)

  return matrix

def MatrixReshape(matrix, numberOfIndexes, keepTheIndexes):
  if(keepTheIndexes):
    n_list = [numberOfIndexes, len(matrix, 1)]
  else:
    n_list = [len(matrix, 0) - numberOfIndexes, len(matrix, 1)]

  for i in range(len(n_list, 0)):
    for j in range(len(n_list, 1)):
      n_list[i, j] = matrix[i, j]

  return matrix

def MatrixRound(matrix, numberOfDecimals):
  for i in range(len(matrix, 0)):
    for j in range(len(matrix, 1)):
      matrix[i, j] = round(matrix[i, j], numberOfDecimals)

  return matrix
