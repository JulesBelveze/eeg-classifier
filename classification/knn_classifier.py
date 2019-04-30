import pandas as pd
import numpy as np
import warnings
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import sys

sys.path.insert(0, "../feature_extraction/")
from pca_by_class import remove_correlated


def oversample_smote(X, y):
    sm = SMOTE(random_state=2)
    X, y = sm.fit_sample(X, y)
    return X, y


def main(args):
    df = pd.read_csv('../data/features_all_nochnnels.csv', index_col=['File', 'Segment'], sep=';')
    accuracy_baseline = df['labels_jules'].value_counts()[0] / sum(df['labels_jules'].value_counts())

    Y = df['labels_jules'].values

    if args.reduce:
        df = remove_correlated(df.drop("labels_jules", axis=1))
        X = df.values
    else:
        X = df.drop("labels_jules", axis=1).values

    if args.smote:
        accuracy_baseline = 0.5
        X, Y = oversample_smote(X, Y)

    X = scale(X)

    K = 10
    KF = KFold(n_splits=K, shuffle=True)

    if args.tune:
        n_neighbors = 20
        error_train = np.zeros((K, n_neighbors))
        error_test = np.zeros((K, n_neighbors))
        for n in range(1, n_neighbors + 1):
            k = 0
            for train_index, test_index in KF.split(X):
                X_train = X[train_index, :]
                Y_train = Y[train_index]

                X_test = X[test_index, :]
                Y_test = Y[test_index]

                neigh = KNeighborsClassifier(n_neighbors=n)
                neigh.fit(X_train, Y_train)

                Y_train_hat = neigh.predict(X_train)
                Y_test_hat = neigh.predict(X_test)

                error_train[k][n - 1] = accuracy_score(Y_train, Y_train_hat)
                error_test[k][n - 1] = accuracy_score(Y_test, Y_test_hat)

                k += 1
        error_train = np.mean(error_train, axis=0)
        error_test = np.mean(error_test, axis=0)

        plt.plot(range(1, n_neighbors + 1), error_train, marker='x', label="error train")
        plt.plot(range(1, n_neighbors + 1), error_test, marker='x', label="error test")
        plt.legend()
        plt.show()

    else:

        error_train = np.zeros(K)
        error_test = np.zeros(K)
        k = 0
        for train_index, test_index in KF.split(X):
            X_train = X[train_index, :]
            Y_train = Y[train_index]

            X_test = X[test_index, :]
            Y_test = Y[test_index]

            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(X_train, Y_train)

            Y_train_hat = neigh.predict(X_train)
            Y_test_hat = neigh.predict(X_test)

            error_train[k] = accuracy_score(Y_train, Y_train_hat)
            error_test[k] = accuracy_score(Y_test, Y_test_hat)

            k += 1
        error_train = np.mean(error_train)
        error_test = np.mean(error_test)

        print("Accuracy training: %f" % np.mean(error_train))
        print("Accuracy: %f / baseline: %f" % (np.mean(error_test), accuracy_baseline))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reduce',
        type=bool,
        default=False,
        help='Setting to true will proceed to feature selection'
    )
    parser.add_argument(
        '--smote',
        type=bool,
        default=False,
        help='Setting to true will upsample the minority class using the SMOTE algorithm'
    )
    parser.add_argument(
        '--tune',
        type=bool,
        default=False,
        help='Setting to true will display a graph with the test and training error with respect to the number of neighbors in KNN'
    )

    return parser.parse_intermixed_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    main(args)
