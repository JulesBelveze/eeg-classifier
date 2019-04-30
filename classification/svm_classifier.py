import pandas as pd
import numpy as np
import warnings
import argparse
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import sys

sys.path.insert(0, "../feature_extraction/")
from pca_by_class import remove_correlated


def oversample_smote(X, y):
    sm = SMOTE(random_state=2)
    X, y = sm.fit_sample(X, y)
    return X, y


def oversample(df):
    count = df['labels_jules'].value_counts()
    max_count = max(count)

    df_bad = df[df['labels_jules'] == 0]
    df_good_oversampled = df[df['labels_jules'] == 1].sample(n=max_count, replace=True)
    df_oversampled = pd.concat([df_bad, df_good_oversampled])

    return shuffle(df_oversampled)


def main(args):
    df = pd.read_csv('../data/features_all_nochnnels.csv', index_col=['File', 'Segment'], sep=';')
    accuracy_baseline = df['labels_jules'].value_counts()[0] / sum(df['labels_jules'].value_counts())

    if args.upsample:
        if args.smote:
            print("Incompatibility of --smote and --upsample flags")
            sys.exit()
        accuracy_baseline = 0.5
        df = oversample(df)

    if args.reduce:
        df = remove_correlated(df)

    X = df.drop('labels_jules', axis=1).values
    Y = df['labels_jules'].values

    if args.smote:
        accuracy_baseline = 0.5
        X, Y = oversample_smote(X, Y)

    X = scale(X)

    K = 5
    KF = KFold(n_splits=K, shuffle=True)
    error_test = np.zeros(K)
    k = 0
    for train_index, test_index in KF.split(X):
        X_train = X[train_index, :]
        Y_train = Y[train_index]

        X_test = X[test_index, :]
        Y_test = Y[test_index]

        clf = SVC()
        clf.fit(X_train, Y_train)

        Y_hat = clf.predict(X_test)
        error_test[k] = accuracy_score(Y_test, Y_hat)

        k += 1

    print("Accuracy: %f / baseline: %f" % (np.mean(error_test), accuracy_baseline))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--upsample',
        type=bool,
        default=False,
        help='Setting to true will randomly upsample the minority class'
    )
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

    return parser.parse_intermixed_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    main(args)
