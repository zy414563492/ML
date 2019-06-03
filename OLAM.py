#!/usr/bin/env python3
# -*- coding: utf-8 -*-

DATASIZE = 150
SETSIZE = 75

import numpy as np
import random

# Randomly dividing data sets into two lists
def getListsFromFileRandomly(fd: np.ndarray) -> (list, list):
    setA = []; setB = []
    num1 = random.sample(range(DATASIZE), SETSIZE)
    num2 = random.sample([item for item in range(0, DATASIZE) if item not in num1], SETSIZE)
    # print("num1:\n", num1, "\nnum2:\n", num2)
    for i in num1:
        setA.append(fd[i])
    for i in num2:
        setB.append(fd[i])
    return setA, setB

# read lists into two matrices, X represents the matrix of features, U represents the matrix of labels
# [1, 0, 0] --> Iris - setosa
# [0, 1, 0] --> Iris - versicolor
# [0, 0, 1] --> Iris - virginica
def listsToMatrices(dividedList: list) -> (np.ndarray, np.ndarray):
    X = np.zeros((4, SETSIZE))
    U = np.zeros((3, SETSIZE))
    for i in range(SETSIZE):
        X.T[i] = [dividedList[i][0], dividedList[i][1], dividedList[i][2], dividedList[i][3]]
        if dividedList[i][4].decode('utf-8') == "Iris-setosa":
            U.T[i] = [1, 0, 0]
        elif dividedList[i][4].decode('utf-8') == "Iris-versicolor":
            U.T[i] = [0, 1, 0]
        elif dividedList[i][4].decode('utf-8') == "Iris-virginica":
            U.T[i] = [0, 0, 1]
        else:
            print("illegal label")
    return X, U

# the minimum distance the highest possibility, so return the label vector by this principle
def minNumToVector(a: int, b: int, c: int) -> np.ndarray:
    temp = min(a, b, c)
    if temp == a:
        return np.array([1, 0, 0])
    elif temp == b:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


def multipleCalculate(fd: np.ndarray, times: int) -> (float, float):
    accuracy = []
    for item in range(times):
        setforTraining, setForTesting = getListsFromFileRandomly(fd)
        # print("setforTraining:\n", setforTraining, "\nsetForTesting:\n", setForTesting)

        X, U = listsToMatrices(setforTraining)
        # print('X:\n', X, '\nU:\n', U)

        # 因为是行向量独立，所以A伪逆=A转*(A*A转)逆；列向量独立的话才是A伪逆=(A转*A)逆*A转
        # X_pes = np.dot(X.T, np.linalg.inv(X.dot(X.T)))

        # this way can get pseudo-inverse matrix directly
        X_pes = np.linalg.pinv(X)
        # print('X_pseudo-inverse:\n', X_pes)

        # M = normalize(U.dot(X_pes))
        M = U.dot(X_pes)
        # print("The matrix M get by training is:\n", M)

        X_test, U_test = listsToMatrices(setForTesting)
        # print("X_test:\n", X_test, "\nU_test:\n", U_test)
        y_test = M.dot(X_test)  # get prediction matrix
        # print("y_test:\n", y_test)

        correctCount = 0
        for i in range(SETSIZE):
            # Calculate the distances between each column and three unit vectors
            delt_x0 = np.square(y_test[0][i] - 1) + np.square(y_test[1][i]) + np.square(y_test[2][i])
            delt_x1 = np.square(y_test[0][i]) + np.square(y_test[1][i] - 1) + np.square(y_test[2][i])
            delt_x2 = np.square(y_test[0][i]) + np.square(y_test[1][i]) + np.square(y_test[2][i] - 1)
            # print(delt_x0, delt_x1, delt_x2, minNumToVector(delt_x0, delt_x1, delt_x2), U_test.astype(int).T[i])

            # compare the prediction vector to the real label vector(need to transform it firstly)
            if (minNumToVector(delt_x0, delt_x1, delt_x2) == U_test.T[i]).all():
                correctCount += 1   # count the number of correct classification

        accuracy.append(correctCount/SETSIZE)
    # calculate average accuracy; calculate the variance after converting the list to a matrix
    return sum(accuracy)/times, np.var(np.array(accuracy))


def main():
    initData = np.genfromtxt('iris.data', delimiter=',', dtype=None)
    averageAccuracy, loss = multipleCalculate(initData, 20)
    print("The average accuracy of OLAM is: ", averageAccuracy, "\tLoss is: ", loss)


if __name__ == '__main__':
    main()
