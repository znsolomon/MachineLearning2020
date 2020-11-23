import mnist_reader
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import tree
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
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
    x = trainSet
    y = trainLabels
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)
    predictions = clf.predict(testSet)
    # Check % of correct predictions
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testLables[i]:
            accuracy += 1
    accuracy = accuracy / len(predictions)
    return accuracy


def dbscan(trainSet, trainLabels):

    X = trainSet
    labels_true = trainLabels

    X = StandardScaler().fit_transform(X)

    # Compute DBSCAN
    db = DBSCAN(eps=0.5, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    #print("Silhouette Coefficient: %0.3f"
    #      % metrics.silhouette_score(X, labels))

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


if __name__ == '__main__':

    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    X_reduced, y_reduced = divide_data(X_train, y_train, 10000)

    print(decision_tree(X_reduced, y_reduced, X_test, y_test))

