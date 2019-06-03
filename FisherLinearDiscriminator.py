#!/usr/bin/env python3
# -*- coding: utf-8 -*-

DATASIZE = 150
SETSIZE = 75

import numpy as np
import random


# Randomly dividing data sets into two lists
def getListsFromFileRandomly(fd: np.ndarray) -> (list, list):
    setA = []
    setB = []
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
def listsToMatrices(dividedList: list) -> (np.ndarray, np.ndarray, np.ndarray):
    A = []; B = []; C = []
    for i in range(SETSIZE):
        tempList = [dividedList[i][0], dividedList[i][1], dividedList[i][2], dividedList[i][3]]
        if dividedList[i][4].decode('utf-8') == "Iris-setosa":
            A.append(tempList)
        elif dividedList[i][4].decode('utf-8') == "Iris-versicolor":
            B.append(tempList)
        elif dividedList[i][4].decode('utf-8') == "Iris-virginica":
            C.append(tempList)
        else:
            print("illegal label")
    return np.array(A).T, np.array(B).T, np.array(C).T


# calculate dispersion within classes and mean of class
def getCovAndAvg(samples: np.ndarray) -> (np.ndarray, np.ndarray):
    mean = np.mean(samples, axis=1)
    # get sample dimensions and quantities
    dimens, nums = samples.shape
    # all samples subtract the matrix of mean vector
    samples_mean = samples - np.tile(mean, (nums, 1)).T
    # initialize the in-class dispersion matrix
    s_in = np.zeros((dimens, dimens))
    # print("samples_mean:\n", samples_mean)
    for i in range(nums):
        x = np.zeros((dimens, 1))
        for j in range(dimens):
            x[j] = samples_mean[j, i]
        s_in += x.dot(x.T)
    return s_in, mean


def compareRatio(g1, g_1, g2, g_2, g3, g_3):
    delta_A = abs(g1 / (g1 - g_1))
    delta_B = abs(g2 / (g2 - g_2))
    delta_C = abs(g3 / (g3 - g_3))
    minRatio = min(delta_A, delta_B, delta_C)
    if minRatio == delta_A:
        return 0
    elif minRatio == delta_B:
        return 1
    else:
        return 2


def compareDistance(a, b, c):
    minValue = min(abs(a), abs(b), abs(c))
    if minValue == abs(a):
        return 0
    elif minValue == abs(b):
        return 1
    else:
        return 2


def multipleCalculate(fd: np.ndarray, times: int) -> (float, float):
    accuracy = []
    for item in range(times):
        setforTraining, setForTesting = getListsFromFileRandomly(fd)
        # print("setforTraining:\n", setforTraining, "\nsetForTesting:\n", setForTesting)

        # write the training set into three matrices
        C_1, C_2, C_3 = listsToMatrices(setforTraining)
        # print("C_1: (", len(C_1[0]), ")\n", C_1)
        # print("C_2: (", len(C_2[0]), ")\n", C_2)
        # print("C_3: (", len(C_3[0]), ")\n", C_3)

        # compute the complement of class 1, 2, 3 respectively
        C_23 = np.column_stack((C_2, C_3))
        C_13 = np.column_stack((C_1, C_3))
        C_12 = np.column_stack((C_1, C_2))
        # print("C_23: (", len(C_23[0]), ")\n", C_23)
        # print("C_13: (", len(C_13[0]), ")\n", C_13)
        # print("C_12: (", len(C_12[0]), ")\n", C_12)

        # dispersion within classes and mean of class 1, 2, 3
        s_in1, mean1 = getCovAndAvg(C_1)
        s_in2, mean2 = getCovAndAvg(C_2)
        s_in3, mean3 = getCovAndAvg(C_3)
        # print("s_in1 :\n", s_in1, "\nmean1 :\n", mean1)
        # print("s_in2 :\n", s_in2, "\nmean2 :\n", mean2)
        # print("s_in3 :\n", s_in3, "\nmean3 :\n", mean3)

        # dispersion within classes and mean of the complement of class 1, 2, 3
        s_in23, mean23 = getCovAndAvg(C_23)
        s_in13, mean13 = getCovAndAvg(C_13)
        s_in12, mean12 = getCovAndAvg(C_12)
        # print("s_in23 :\n", s_in23, "\nmean23 :\n", mean23)
        # print("s_in13 :\n", s_in13, "\nmean13 :\n", mean13)
        # print("s_in12 :\n", s_in12, "\nmean12 :\n", mean12)

        # get three in-class dispersion matrix
        sw1, sw2, sw3 = s_in1 + s_in23, s_in2 + s_in13, s_in3 + s_in12
        # print("sw1 :\n", sw1, "\nsw2 :\n", sw2, "\nsw3 :\n", sw3)

        # calculate classifiers
        w1 = np.dot(np.linalg.inv(sw1), mean1 - mean23)
        w2 = np.dot(np.linalg.inv(sw2), mean2 - mean13)
        w3 = np.dot(np.linalg.inv(sw3), mean3 - mean12)
        print("w1 :", w1, "\nw2 :", w2, "\nw3 :", w3)

        # write the testing set into three matrices
        T_1, T_2, T_3 = listsToMatrices(setForTesting)
        # print("T_1: (", len(T_1[0]), ")\n", T_1, "\nT_2: (", len(T_2[0]), ")\n", T_2, "\nT_3: (", len(T_3[0]), ")\n", T_3)

        correctList = [0, 0, 0, 0]
        for area, k in enumerate([T_1, T_2, T_3]):
            for i in range(len(k[0])):
                test = np.array(k.T[i])
                g1 = np.dot(w1.T, test.T) - np.dot(w1.T, mean1)
                g2 = np.dot(w2.T, test.T) - np.dot(w2.T, mean2)
                g3 = np.dot(w3.T, test.T) - np.dot(w3.T, mean3)
                g_1 = np.dot(w1.T, test.T) - np.dot(w1.T, mean23)
                g_2 = np.dot(w2.T, test.T) - np.dot(w2.T, mean13)
                g_3 = np.dot(w3.T, test.T) - np.dot(w3.T, mean12)
                # print("g(x) in T_%i:\t%f\t%f\t%f" % (area + 1, g1, g2, g3))

                # If the ratio is judged separately, the correct rate is about 82%.
                # if compareRatio(g1, g_1, g2, g_2, g3, g_3) == 0 and area == 0:
                #     correctList[0] += 1
                # elif compareRatio(g1, g_1, g2, g_2, g3, g_3) == 1 and area == 1:
                #     correctList[1] += 1
                # elif compareRatio(g1, g_1, g2, g_2, g3, g_3) == 2 and area == 2:
                #     correctList[2] += 1
                # else:
                #     correctList[3] += 1

                # If the distance is judged separately, the correct rate is about 55%.
                # if compareDistance(g1, g2, g3) == 0 and area == 0:
                #     correctList[0] += 1
                # elif compareDistance(g1, g2, g3) == 1 and area == 1:
                #     correctList[1] += 1
                # elif compareDistance(g1, g2, g3) == 2 and area == 2:
                #     correctList[2] += 1
                # else:
                #     correctList[3] += 1

                # If the ratio and distance are judged at the same time, the correct rate will increase to about 92%.
                if (compareRatio(g1, g_1, g2, g_2, g3, g_3) == 0 or compareDistance(g1, g2, g3) == 0) and area == 0:
                    correctList[0] += 1
                elif (compareRatio(g1, g_1, g2, g_2, g3, g_3) == 1 or compareDistance(g1, g2, g3) == 1) and area == 1:
                    correctList[1] += 1
                elif (compareRatio(g1, g_1, g2, g_2, g3, g_3) == 2 or compareDistance(g1, g2, g3) == 2) and area == 2:
                    correctList[2] += 1
                else:
                    correctList[3] += 1

        accuracy.append(sum(correctList[:3])/SETSIZE)
    # calculate average accuracy and loss
    return sum(accuracy) / times, np.var(np.array(accuracy))


def main():
    initData = np.genfromtxt('iris.data', delimiter=',', dtype=None)
    averageAccuracy, loss = multipleCalculate(initData, 20)
    print("The average accuracy of Fisher's Linear Discriminator is: ", averageAccuracy, "\tLoss is: ", loss)


if __name__ == '__main__':
    main()