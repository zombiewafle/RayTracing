# Archivo creado para almanecar todas las operaciones
#matematicas usadas en el documento gl.py


#producto punto

#Variables de prueba

import numpy as np
import struct
from math import cos, sin, pi, tan
from collections import namedtuple
V4 = namedtuple('Point4', ['x', 'y', 'z', 'w'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])

a = [1,2,3]
b = [4,5,6]
matrix = [1,2,3,4,5,6,7,8,9,10]
matrix1 = [2,3,4,5,6,7,8,9,10,11]


def dotProduct(a,b):
    #dotProduct = 0
    #for x,y in zip(a,b):
    #    dotProduct = dotProduct + (x*y)
    #return dotProduct
    return a.x * b.x + a.y * b.y + a.z * b.z

    #print(dotProduct)


#dotProduct(a,b)

#Funcion para sumar vectores de 3 elementos
#Notas: Los vectores con una mayor cantidad de elementos, ejemplo:
# a = [1,2,3,4]
# El elemento 4 sera ignorado

def sumVectors(a,b):
    return V3(a.x + b.x, a.y + b.y, a.z + b.z)

   # print(i)

#sumVectors(a,b)


def subVectors(a,b):
    return V3(a.x - b.x, a.y - b.y, a.z - b.z)

    #print(sub)

#subVectors(a,b)

def crossProduct(a,b):
    #A × B = (bz – cy)i + (cx – az)j + (ay – bx)k

    crossProduct1 = ((a[1])*(b[2])) - ((a[2])*(b[1]))
    crossProduct2 = ((a[2])*(b[0])) - ((a[0])*(b[2]))
    crossProduct3 = ((a[0])*(b[1])) - ((a[1])*(b[0]))

    crossProduct = ([crossProduct1 , crossProduct2 , crossProduct3])

    return crossProduct

    #print(np.array(crossProduct))

#crossProduct(a,b)

#def normalize(a):
#    max_valueA = max(a)
#    min_valueA = min(a)
#    for i in range(0, len(a)):
#        a[i] = (a[i] - min_valueA) / (max_valueA - min_valueA)
#
#    return a

#def normalize(a):
#    for i in range(0, len(a)):
#        sqrt = (a[i])**2
#        result = (sqrt)**0.5
#    return result#, print(result)


#norm(a)
#b = np.linalg.norm(a)
#print(b)
#normalize_list(a)

def createMatrix(row, col, listOfLists, multi = 1):
    matrix = []
    for i in range(row):

        rowList = []
        for j in range(col):

            # you need to increment through dataList here, like this:
            rowList.append((listOfLists[row * i + j]) * multi)

        matrix.append(rowList)

    return matrix

#def createMatrix(row, col, listOfLists):
#    matrix = []
#    for i in range(row):
#
#        rowList = []
#        for j in range(col):
#
#            # you need to increment through dataList here, like this:
#            rowList.append(listOfLists[row * i + j])
#
#        matrix.append(rowList)
#
#
#
#        #for i in range(len(matrix)):
#        #    for j in range(len(matrix[0])):
#        #        print('%3d'%matrix[i][j],end='')
#        #    print()
#    return matrix#,print(matrix)


#createMatrix(3,4,matrix)


def deg2rads(degNum):
    radNum = (degNum * 3.1415926535897932384626433)/180
    #print(radNum)
    return radNum

def vectorMultiplication(v0, v1):
  
  return V3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z)


def multiVecMatrix(Vector, Matrix):
    matrix1Row = len(Matrix)
    matrixColumns = len(Matrix[0])
    newVector = []
    for y in range(matrix1Row):
        newNumber = 0
        vectorCol = 0
        for x in range(matrixColumns):
            #print(Matrix[y][x], Vector[vectorCol])
            newNumber = (Matrix[y][x] * Vector[vectorCol]) + newNumber
            vectorCol += 1
        newVector.append(newNumber)
    return newVector#, print(newVector)

#hola = [[1,0,3,4],
#        [3,1,2,1],
#        [2,3,1,5],
#        [6,0,3,1]]
#hola5 = V4(7,9,11,2)

#createMatrix(3,4,matrix)
#multiVecMatrix(hola5,hola)


def multyMatrix (Matrix, Matrix2):
    matrix1Row = len(Matrix)
    matrix2RowLimit = len(Matrix2[0])
    newMatrix = []
    for y in range(matrix1Row):
        newRow = []
        matrix2Row = 0
        matrix2Col = len(Matrix2)
        column1 = 0
        for x in range(matrix1Row):
            for i in range(matrix2Col):
                #print(Matrix[y][(x+i) % matrix2Col],  Matrix2[(x+i) % matrix2Col][matrix2Row])
                column1 = (Matrix[y][(x+i) % matrix2Col] * Matrix2[(x+i) % matrix2Col][matrix2Row]) + column1
            #print(column1)
            if matrix2RowLimit == 1:
                newMatrix.append(column1)
                break
            matrix2Row += 1
            newRow.append(column1)
            column1 = 0
        if matrix2RowLimit != 1:
            newMatrix.append(newRow)
    #print(newMatrix)
    return newMatrix

#
#def multyMatrix4X4 (Matrix, Matrix2):
#    matrix1Row = len(Matrix)
#    matrix1Col = len(Matrix[0])
#    newMatrix = []
#    for y in range(matrix1Col):
#        newRow = []
#        matrix2Row = 0
#        column1 = 0
#        column2 = 0
#        column3 = 0
#        column4 = 0
#        for x in range(matrix1Row):
#            column1 = (Matrix[y][x] * Matrix2[x][matrix2Row]) + column1
#            column2 = (Matrix[y][x] * Matrix2[x][matrix2Row + 1]) + column2
#            column3 = (Matrix[y][x] * Matrix2[x][matrix2Row + 2]) + column3
#            column4 = (Matrix[y][x] * Matrix2[x][matrix2Row + 3]) + column4
#        newRow.extend([column1, column2, column3, column4])
#        newMatrix.append(newRow)
#    return newMatrix, print(newMatrix)
#
#hola2 = [[1,3,0,0],
#         [0,1,6,7],
#         [0,3,1,0],
#         [3,0,4,1]]
#
#
##multyMatrix4X4(hola2,hola2)


def transposeMatrix(m):
    return map(list,zip(*m))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
           # cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors

#def pi():
#    pi = 3.1415926535897932384626433
#    return pi

#pi()

def transpose(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    matrix_T = []
    for j in range(columns):
        row = []
        for i in range(rows):
           row.append(matrix[i][j])
        matrix_T.append(row)

    return matrix_T

#Obtiene la determinate de una matriz 3X3
def determinante3X3(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    newMatrix = []
    for y in range(rows):
        newRow = []
        for x in range(columns):
            if x == 2:
                newRow.extend([matrix[y][x], matrix[y][(x + 1) % columns], matrix[y][(x + 2) % columns]])
                break
            newRow.append(matrix[y][x])
        newMatrix.append(newRow)
    diagonal1 = 0
    diagonal2 = 0
    for x in range(columns):
        diagonal1 = (newMatrix[0][x] * newMatrix[1][x+1] * newMatrix[2][x+2]) + diagonal1
        diagonal2 = -(newMatrix[0][x+2] * newMatrix[1][x+1] * newMatrix[2][x]) + diagonal2
    determinante = diagonal1 + diagonal2
    return determinante

def inversa4X4(Matrix):
    newMatrix = transpose(Matrix)
    row = len(Matrix[0])
    column = len(Matrix)
    determinant = 0
    cofactorList = []
    for y in range(row):
        exponent1 = y + 1
        for x in range(column):
            exponent2 = x + 1
            exponentT = exponent2 + exponent1
            cofactorM = []
            if y == 0:
                detM = []
            verificador = False
            for i in range(row):
                if y == 0:
                    rowDe = []
                rowCo = []
                for k in range(column):
                    if i != y and x != k:
                        verificador = True
                        rowCo.append(newMatrix[i][k])
                        if y == 0:
                            rowDe.append(Matrix[i][k])
                if verificador:
                    if y == 0:
                        detM.append(rowDe)
                    cofactorM.append(rowCo)
                    verificador = False
            deter = ((-1) ** exponentT) * determinante3X3(cofactorM)
            cofactorList.append(deter)
            if y == 0:
               deter2 = ((-1) ** exponentT) * determinante3X3(detM)
               determinant = (Matrix[y][x] * deter2) + determinant
    Inverse = createMatrix(4, 4, cofactorList, (1/determinant))
    return Inverse


def length(v0):
  return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

def norm(v0):
  v0length = length(v0)
  if not v0length:
    return V3(0, 0, 0)

  return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)

def kMul(v0, k):
  return V3(v0.x * k, v0.y * k, v0.z *k)


def firstItemFunction(a):
    firstInternalList = a[0]
    firstItem = firstInternalList[2]


    return firstItem

def secondItemFunction(a):
    secondInternalList = a[1]

    secondtItem = secondInternalList[2]

    return secondtItem

def thirdItemFunction(a):
    thirdInternalList = a[2]

    thirdItem = thirdInternalList[2]

    return thirdItem

def fourthItemFunction(a):
    fourthInternalLists = a[3]

    fourthItem = fourthInternalLists[2]

    return fourthItem


#def arrayCreation(v0):
#    #counter = 3
#    #length = length(v0)
#
#    firstElement = v0(0)
#    secondElement = v0(1)
#    thirdElement = v0(2)
#
#    listOfElements = [firstElement, secondElement, thirdElement]
#
#    return listOfElements, print(listOfElements)
#
#arrayd = V3(0,1,0)
#
#
#
#arrayCreation(arrayd)


    



