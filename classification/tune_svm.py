import pandas as pd
import numpy as np
import warnings
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import sys

sns.set_style('whitegrid')

sys.path.insert(0, "../feature_extraction/")
from pca_by_class import remove_correlated


def oversample_smote(X, y):
    sm = SMOTE(random_state=2)
    X, y = sm.fit_sample(X, y)
    return X, y


def tune_kernel(X, Y, kernels):
    K = 5
    KF = KFold(n_splits=K, shuffle=True)

    error_train = np.zeros((K, len(kernels)))
    error_test = np.zeros((K, len(kernels)))

    for i, kernel_ in enumerate(kernels):
        k = 0
        for train_index, test_index in KF.split(X):
            X_train = X[train_index, :]
            y_train = Y[train_index]

            X_test = X[test_index, :]
            y_test = Y[test_index]

            clf = SVC(kernel=kernel_)
            clf.fit(X_train, y_train)

            y_hat_train = clf.predict(X_train)
            y_hat = clf.predict(X_test)

            error_train[k][i] = accuracy_score(y_train, y_hat_train)
            error_test[k][i] = accuracy_score(y_test, y_hat)

            k += 1

    error_test = np.mean(error_test, axis=0)
    error_train = np.mean(error_train, axis=0)

    plt.figure()
    plt.plot([1, 2, 3, 4], error_train, label='train error', marker='x')
    plt.plot([1, 2, 3, 4], error_test, label='test error', marker='x')
    plt.title("Accuracy wrt kernel")
    plt.xlabel("kernel")
    plt.ylabel("accuracy")
    plt.legend()
    plt.xticks([1, 2, 3, 4], ['linear', 'rbf', 'sigmoid', 'poly'])
    plt.show()


def tune_gamma(X, Y, gammas):
    K = 5
    KF = KFold(n_splits=K, shuffle=True)

    error_train = np.zeros((K, len(gammas)))
    error_test = np.zeros((K, len(gammas)))

    for i, gamma_ in enumerate(gammas):
        k = 0
        for train_index, test_index in KF.split(X):
            X_train = X[train_index, :]
            y_train = Y[train_index]

            X_test = X[test_index, :]
            y_test = Y[test_index]

            clf = SVC(kernel='rbf', gamma=gamma_)
            clf.fit(X_train, y_train)

            y_hat_train = clf.predict(X_train)
            y_hat = clf.predict(X_test)

            error_train[k][i] = accuracy_score(y_train, y_hat_train)
            error_test[k][i] = accuracy_score(y_test, y_hat)

            k += 1

    error_test = np.mean(error_test, axis=0)
    error_train = np.mean(error_train, axis=0)

    plt.figure()
    plt.semilogx(gammas, error_train, label='train error', marker='x')
    plt.semilogx(gammas, error_test, label='test error', marker='x')
    plt.title("Accuracy wrt gamma")
    plt.xlabel("gamma")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def tune_c(X, Y, cs):
    K = 5
    KF = KFold(n_splits=K, shuffle=True)

    error_train = np.zeros((K, len(cs)))
    error_test = np.zeros((K, len(cs)))

    for i, c_ in enumerate(cs):
        k = 0
        for train_index, test_index in KF.split(X):
            X_train = X[train_index, :]
            y_train = Y[train_index]

            X_test = X[test_index, :]
            y_test = Y[test_index]

            clf = SVC(kernel='rbf', C=c_)
            clf.fit(X_train, y_train)

            y_hat_train = clf.predict(X_train)
            y_hat = clf.predict(X_test)

            error_train[k][i] = accuracy_score(y_train, y_hat_train)
            error_test[k][i] = accuracy_score(y_test, y_hat)

            k += 1

    error_test = np.mean(error_test, axis=0)
    error_train = np.mean(error_train, axis=0)

    plt.figure()
    plt.semilogx(cs, error_train, label='train error', marker='x')
    plt.semilogx(cs, error_test, label='test error', marker='x')
    plt.title("Accuracy wrt c")
    plt.xlabel("c")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def tune_degree(X, Y, degrees):
    K = 5
    KF = KFold(n_splits=K, shuffle=True)

    error_train = np.zeros((K, len(degrees)))
    error_test = np.zeros((K, len(degrees)))

    for i, degree_ in enumerate(degrees):
        k = 0
        for train_index, test_index in KF.split(X):
            X_train = X[train_index, :]
            y_train = Y[train_index]

            X_test = X[test_index, :]
            y_test = Y[test_index]

            clf = SVC(kernel='poly', degree=degree_)
            clf.fit(X_train, y_train)

            y_hat_train = clf.predict(X_train)
            y_hat = clf.predict(X_test)

            error_train[k][i] = accuracy_score(y_train, y_hat_train)
            error_test[k][i] = accuracy_score(y_test, y_hat)

            k += 1

    error_test = np.mean(error_test, axis=0)
    error_train = np.mean(error_train, axis=0)

    plt.figure()
    plt.plot(degrees, error_train, label='train error', marker='x')
    plt.plot(degrees, error_test, label='test error', marker='x')
    plt.title("Accuracy wrt degree")
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def main():
    df = pd.read_csv("../data/features_by_channel.csv", index_col=['File', 'Segment'], sep=';')

    Y = df['labels_jules'].values
    X = remove_correlated(df.drop("labels_jules", axis=1)).values

    X, Y = oversample_smote(X, Y)
    X = scale(X)

    kernels = ['linear', 'rbf', 'sigmoid', 'poly']
    gammas = [10e-2, .1, 1, 10]
    cs = [.1, 1, 10, 100, 1000]
    degrees = [0, 1, 2, 3, 4, 5, 6]

    tune_kernel(X, Y, kernels)
    tune_gamma(X, Y, gammas)
    tune_c(X, Y, cs)
    tune_degree(X, Y, degrees)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
