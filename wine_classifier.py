#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
import operator
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']
train_set, train_labels, test_set, test_labels = load_data()

reducedTrainSet = train_set[:, [10, 12]]
reducedTestSet = test_set[:, [10, 12]]

def showScatter(data_set, data_labels):
    # n_features = data_set.shape[1]
    # fig, ax = plt.subplots(n_features, n_features)
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
    class_colours = [CLASS_1_C, CLASS_2_C, CLASS_3_C]
    colours = np.zeros_like(data_labels, dtype = np.object)
    colours[data_labels == 1] = class_colours[0]
    colours[data_labels == 2] = class_colours[1]
    colours[data_labels == 3] = class_colours[2]
    # for x in range(n_features):
    #     for y in range(n_features):
    #         ax[x, y].scatter(data_set[:, x], data_set[:, y], c=colours)
    # ax[10, 12].scatter(data_set[:, 10], data_set[:, 12], c=colours)
    plt.scatter(data_set, data_labels, c = colours)
    plt.xlabel("Feature 11")
    plt.ylabel("Feature 13")
    plt.show()

def calculate_accuracy(test_labels, pred_labels):
    count = 0
    for i in range(len(test_labels)):
        if(test_labels[i] == pred_labels[i]):
            count += 1
    accuracy = (count * 100)/len(test_labels)
    return accuracy

def viewCovmatrix(training):
    x = np.cov(training, rowvar = False)
    return x

def seperateDatabyClass(train_set, train_labels):
    class1 = []
    class2 = []
    class3 = []
    final = []
    for i in range(len(train_labels)):
        if(train_labels[i] == 1):
            class1.append(reducedTrainSet[i])
        if(train_labels[i] == 2):
            class2.append(reducedTrainSet[i])
        if(train_labels[i] == 3):
            class3.append(reducedTrainSet[i])
    seperated = [class1, class2, class3]
    for j in range(len(seperated)):
        x = np.column_stack(seperated[j])
        final.append(x)
    return final

def meanAndSd():
    seperatedClasses = seperateDatabyClass(train_set, train_labels)
    feature1 = []
    feature2 = []
    for i in range(len(seperatedClasses)):
        meanFeat1 = np.mean(seperatedClasses[i][0])
        sd = np.std(seperatedClasses[i][0])
        temp1 = [meanFeat1, sd]
        meanFeat2 = np.mean(seperatedClasses[i][1])
        sd2 = np.std(seperatedClasses[i][1])
        temp2 = [meanFeat2, sd2]
        feature1.append(temp1) #prints array with each index containing mean and sd of class 1, 2 and 3
        feature2.append(temp2)
    return feature1, feature2

def calPdf(x,meanClass,sdClass):
    expononent = np.exp(-(np.power(x-meanClass,2)/(2*np.power(sdClass,2))))
    return (1/(np.sqrt(2*math.pi)*sdClass))*expononent

def bayesform(a, b, c):
    class1 = (a * 0.328)/((a * 0.328) + (b * 0.4) + (c * 0.272))
    class2 = (b * 0.4)/((a * 0.328) + (b * 0.4) + (c * 0.272))
    class3 = (c * 0.272)/((a * 0.328) + (b * 0.4) + (c * 0.272))
    return class1, class2, class3

def feature_selection(train_set, train_labels, **kwargs):
    return [10, 12]

def ratio(test_labels, solution, element, predictedE):
    count = 0
    correct = 0
    for i in range(len(test_labels)):
        if (test_labels[i] == element):
            count += 1
            if (solution[i] == predictedE):
                correct += 1
    a = round(correct/count, 3 )
    return a

def confMatrix(test_labels, solution):
    labels= np.unique(solution)
    matrix = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            matrix[i][j] = ratio(test_labels, solution, labels[i], labels[j])
    return matrix

def plot_matrix(matrix, ax=None):
    if ax is None:
        ax = plt.gca()
    handle = ax.imshow(matrix, cmap=plt.get_cmap('summer'))
    plt.colorbar(handle)
    for i in range(3):
        for j in range(3):
            plt.text(i, j, matrix[i][j])
    plt.show()

def knn(train_set, train_labels, test_set, k, **kwargs):
    distanceArray = []
    distCal = lambda x, y: np.sqrt(np.sum((x-y)**2))
    distance = lambda x: [distCal(x, b) for b in reducedTrainSet]
    solution = []
    for i in reducedTestSet:
        n1 = 0
        n2 = 0
        n3 = 0
        distfromdata = np.column_stack((distance(i), train_labels))
        distfromdata = distfromdata[np.argsort(distfromdata[:, 0])]
        kth = distfromdata[:k]
        kth = kth[:,1]
        for l in kth:
            if(l == 1):
                n1 += 1
            if(l == 2):
                n2 += 1
            if(l == 3):
                n3 += 1
        temp = [n1, n2, n3]
        solution.append(np.argmax(temp) + 1)
    X = calculate_accuracy(test_labels, solution)
    labels= np.unique(solution)
    m = confMatrix(test_labels, solution)
    # print("KNN accuracy------>", X)
    plot_matrix(m, ax=None)
    return solution

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    feat1, feat2 = meanAndSd()
    solution = []
    for i in reducedTestSet:
        feat1class1 = calPdf(i[0], feat1[0][0], feat1[0][1]) #pdf of feat1 given class1
        feat2class1 = calPdf(i[1], feat2[0][0], feat2[0][1]) #pdf of feat2 given class1
        pOfXC1 = feat1class1 * feat2class1 #probability of testval given class1
        feat1class2 = calPdf(i[0], feat1[1][0], feat1[1][1]) #pdf of feat1 given class3
        feat2class2 = calPdf(i[1], feat2[1][0], feat2[1][1]) #pdf of feat2 given class3
        pOfXC2 = feat1class2 * feat2class2 #probability of testval given class2
        feat1class3 = calPdf(i[0], feat1[2][0], feat1[2][1]) #pdf of feat1 given class3
        feat2class3 = calPdf(i[1], feat2[2][0], feat2[2][1]) #pdf of feat2 given class3
        pOfXC3 = feat1class3 * feat2class3 #probability of testval given class3
        class1, class2, class3 = bayesform(pOfXC1,pOfXC2,pOfXC3)
        somting = [class1, class2, class3]
        answer = np.argmax(somting) + 1
        solution.append(answer)
    X = calculate_accuracy(test_labels, solution)
    # print("Naive Bayes------>", X)
    return solution

def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    threetrain = train_set[:, [4,10,12]]
    threetest = test_set[:, [4,11,12]]
    distCal = lambda x, y: np.sqrt(np.sum((x-y)**2))
    distance = lambda x: [distCal(x, b) for b in threetrain]
    solution = []
    for i in threetest:
        n1 = 0
        n2 = 0
        n3 = 0
        distfromdata = np.column_stack((distance(i), train_labels))
        distfromdata = distfromdata[np.argsort(distfromdata[:, 0])]
        kth = distfromdata[:k]
        kth = kth[:,1]
        for l in kth:
            if(l == 1):
                n1 += 1
            if(l == 2):
                n2 += 1
            if(l == 3):
                n3 += 1
        temp = [n1, n2, n3]
        solution.append(np.argmax(temp) + 1)
    X = calculate_accuracy(test_labels, solution)
    m = confMatrix(test_labels, solution)
    # print("KNN-3d accuracy------>", X)
    # print(m)
    return solution

def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    cov = viewCovmatrix(train_set)
    evalue, evector = np.linalg.eig(cov)
    sort = np.flip(np.argsort(evalue, kind = 'quicksort'))
    W = evector[sort]
    W = W[:,:n_components]
    transformTrain = train_set.dot(W)
    transformTest = test_set.dot(W)
    solution = []
    distCal = lambda x, y: np.sqrt(np.sum((x-y)**2))
    distance = lambda x: [distCal(x, b) for b in transformTrain]
    for i in transformTest:
        n1 = 0
        n2 = 0
        n3 = 0
        distfromdata = np.column_stack((distance(i), train_labels))
        distfromdata = distfromdata[np.argsort(distfromdata[:, 0])]
        kth = distfromdata[:k]
        kth = kth[:,1]
        for l in kth:
            if(l == 1):
                n1 += 1
            if(l == 2):
                n2 += 1
            if(l == 3):
                n3 += 1
        temp = [n1, n2, n3]
        solution.append(np.argmax(temp) + 1)
    X = calculate_accuracy(test_labels, solution)
    # print("KNN-PCA accuracy------>", X)
    return solution

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')
    args = parser.parse_args()
    mode = args.mode[0]
    return args, mode

if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
