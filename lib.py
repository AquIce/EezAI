from random import random

def Matricial(matrix1, matrix2):
  for i in range(GetLen(matrix1, 0)):
    for j in range(GetLen(matrix2, 1)):
      z = [GetLen(matrix2, 0), GetLen(matrix1, 1)]
      z[i, j] = 0
      for k in range(GetLen(matrix1, 1)):
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
  for i in range(GetLen(matrix, axis)):
    if(matrix[i, 0] > max):
      max = matrix[i]

  return max

def MatrixEchelon(matrix, axis):
  for i in range(GetLen(matrix, axis)):
    matrix[i] /= GetMax(matrix, axis)

  return matrix

def MatrixReshape(matrix, numberOfIndexes, keepTheIndexes):
  if(keepTheIndexes):
    n_list = [numberOfIndexes, GetLen(matrix, 1)]
  else:
    n_list = [GetLen(matrix, 0) - numberOfIndexes, GetLen(matrix, 1)]

  for i in range(GetLen(n_list, 0)):
    for j in range(GetLen(n_list, 1)):
      n_list[i, j] = matrix[i, j]

  return matrix

def MatrixRound(matrix, numberOfDecimals):
  for i in range(GetLen(matrix, 0)):
    for j in range(GetLen(matrix, 1)):
      matrix[i, j] = round(matrix[i, j], numberOfDecimals)

  return matrix

def GetLen(array, axis):
  if(axis==1):
    return len(array)
  else:
    return len(array[axis-1])
