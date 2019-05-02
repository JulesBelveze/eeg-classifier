import pandas as pd
import numpy as np
import warnings
import argparse
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import sys

sys.path.insert(0, "../feature_extraction/")
from pca_by_class import remove_correlated


def oversample_smote(X, y):
    sm = SMOTE(random_state=2)
    X, y = sm.fit_sample(X, y)
    return X, y


def main(args):
    df = pd.read_csv('../data/features_by_channel.csv', index_col=['File', 'Segment'], sep=';')
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

    print("Baseline accuracy: %".format(accuracy_baseline))

    if args.CV:
        K = 5
        KF = KFold(n_splits=K, shuffle=True)
        error_test = np.zeros(K)
        error_train = np.zeros(K)
        k = 0
        for train_index, test_index in KF.split(X):
            X_train = X[train_index, :]
            Y_train = Y[train_index]

            X_test = X[test_index, :]
            Y_test = Y[test_index]

            clf = SVC()
            clf.fit(X_train, Y_train)

            Y_hat_train = clf.predict(X_train)
            Y_hat = clf.predict(X_test)

            error_train[k] = accuracy_score(Y_train, Y_hat_train)
            error_test[k] = accuracy_score(Y_test, Y_hat)

            k += 1

        error_test = np.mean(error_test)
        error_train = np.mean(error_train)

        print("Accuracy training: %f" % error_train)
        print("Accuracy testing: %f" % error_test)

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3, random_state=42)

        clf = SVC()
        clf.fit(X_train, y_train)

        Y_hat = clf.predict(X_test)

        print(" \n-------------- Test ---------------")
        print("Accuracy: %f" % accuracy_score(y_test, Y_hat))
        print(classification_report(y_test, Y_hat, target_names=['bad', 'good']))

        plt.figure()
        sns.heatmap(confusion_matrix(y_test, Y_hat),
                    annot=True,
                    cbar=False,
                    xticklabels=['bad', 'good'],
                    yticklabels=['bad', 'good'])
        plt.show()


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
        '--CV',
        type=bool,
        default=False,
        help='Setting to true will proceed to a cross-validation'
    )

    return parser.parse_intermixed_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    main(args)
