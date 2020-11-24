import mnist_reader
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


def divide_data(data, labels, size):
    # Cut down size of data whilst retaining proportions of classes
    sortedData = np.empty([10, 6000, 784]).astype(int)
    sortedLabels = np.empty([10, 6000]).astype(int)
    divData = np.empty([size, 784]).astype(int)
    divLables = np.empty([size]).astype(int)
    counters = np.zeros(10).astype(int)
    labelSize = int(size / 10)
    # Sort data into labels
    for i in range(len(data)):
        label = labels[i]
        sortedData[label, counters[label]] = data[i]
        sortedLabels[label, counters[label]] = labels[i]
        counters[label] += 1
    # Take 'size' from each label
    c = 0
    for i in range(10):
        for j in range(labelSize):
            index = np.random.randint(0, 6000)
            divData[c] = sortedData[i, index]
            divLables[c] = sortedLabels[i, index]
            c += 1
    return divData, divLables


def decision_tree(trainSet, trainLabels, testSet, testLables):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainSet, trainLabels)
    predictions = clf.predict(testSet)
    return get_accuracy(predictions, testLables)


def nearest_neighbors(trainSet, trainLables, testSet, testLables, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(trainSet, trainLables)
    predictions = neigh.predict(testSet)
    return get_accuracy(predictions, testLables)


def gaussian_bayes(trainSet, trainLables, testSet, testLables):
    bayes = GaussianNB()
    bayes.fit(trainSet, trainLables)
    predictions = bayes.predict(testSet)
    return get_accuracy(predictions, testLables)


def get_accuracy(predict, trueLables):
    # Check % of correct predictions
    accuracy = 0
    for i in range(len(predict)):
        if predict[i] == trueLables[i]:
            accuracy += 1
    accuracy = accuracy / len(predict)
    return accuracy


if __name__ == '__main__':

    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    X_reduced, y_reduced = divide_data(X_train, y_train, 10000)

    print(gaussian_bayes(X_reduced, y_reduced, X_test, y_test))

