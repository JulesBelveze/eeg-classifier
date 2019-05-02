import pandas as pd
import numpy as np
import warnings
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_curve
from imblearn.over_sampling import SMOTE
import sys

sys.path.insert(0, "../feature_extraction/")
from pca_by_class import remove_correlated


def oversample_smote(X, y):
    sm = SMOTE(random_state=2)
    X, y = sm.fit_sample(X, y)
    return X, y


def tune_nb_tree(X_train, y_train, X_test, y_test):
    '''Use to investigate the number of trees in the forest
    Too many trees will slow down the algorithm'''
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100]
    train_roc, test_roc = [], []

    for n in n_estimators:
        clf = RandomForestClassifier(n_estimators=n, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_hat_train = clf.predict(X_train)
        fp, tp, _ = roc_curve(y_train, y_hat_train)
        auc_train = auc(fp, tp)
        train_roc.append(auc_train)

        y_hat = clf.predict(X_test)
        fp, tp, _ = roc_curve(y_test, y_hat)
        auc_test = auc(fp, tp)
        test_roc.append(auc_test)

    plt.figure()
    plt.plot(n_estimators, train_roc, color='b', label="train AUC")
    plt.plot(n_estimators, test_roc, color='r', label="test AUC")
    plt.ylabel("AUC score")
    plt.xlabel("nb estimators")
    plt.legend()
    plt.show()


def tune_depth_tree(X_train, y_train, X_test, y_test):
    '''Use to investigate the ideal maximum depth of each tree'''
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_roc, test_roc = [], []

    for n in max_depths:
        clf = RandomForestClassifier(max_depth=n, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_hat_train = clf.predict(X_train)
        fp, tp, _ = roc_curve(y_train, y_hat_train)
        auc_train = auc(fp, tp)
        train_roc.append(auc_train)

        y_hat = clf.predict(X_test)
        fp, tp, _ = roc_curve(y_test, y_hat)
        auc_test = auc(fp, tp)
        test_roc.append(auc_test)

    plt.figure()
    plt.plot(max_depths, train_roc, color='b', label="train AUC")
    plt.plot(max_depths, test_roc, color='r', label="test AUC")
    plt.ylabel("AUC score")
    plt.xlabel("depth")
    plt.legend()
    plt.show()


def tune_min_samples_split(X_train, y_train, X_test, y_test):
    '''Use to investigate the ideal minimum nb of samples required to split
    an internal node'''
    min_samples_splits = np.linspace(.1, 1, 10, endpoint=True)
    train_roc, test_roc = [], []

    for n in min_samples_splits:
        clf = RandomForestClassifier(min_samples_split=n, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_hat_train = clf.predict(X_train)
        fp, tp, _ = roc_curve(y_train, y_hat_train)
        auc_train = auc(fp, tp)
        train_roc.append(auc_train)

        y_hat = clf.predict(X_test)
        fp, tp, _ = roc_curve(y_test, y_hat)
        auc_test = auc(fp, tp)
        test_roc.append(auc_test)

    plt.figure()
    plt.plot(min_samples_splits, train_roc, color='b', label="train AUC")
    plt.plot(min_samples_splits, test_roc, color='r', label="test AUC")
    plt.ylabel("AUC score")
    plt.xlabel("min sample splits")
    plt.legend()
    plt.show()


def tune_min_samples_leaf(X_train, y_train, X_test, y_test):
    '''Use to investigate the min number of samples required to be at a
    leaf node'''
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    train_roc, test_roc = [], []

    for n in min_samples_leafs:
        clf = RandomForestClassifier(min_samples_split=n, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_hat_train = clf.predict(X_train)
        fp, tp, _ = roc_curve(y_train, y_hat_train)
        auc_train = auc(fp, tp)
        train_roc.append(auc_train)

        y_hat = clf.predict(X_test)
        fp, tp, _ = roc_curve(y_test, y_hat)
        auc_test = auc(fp, tp)
        test_roc.append(auc_test)

    plt.figure()
    plt.plot(min_samples_leafs, train_roc, color='b', label="train AUC")
    plt.plot(min_samples_leafs, test_roc, color='r', label="test AUC")
    plt.ylabel("AUC score")
    plt.xlabel("min sample leafs")
    plt.legend()
    plt.show()


def randomized_search(X_train, y_train):
    hyperparameters = {
        'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
        'max_depth': np.linspace(1, 64, 64, endpoint=True),
        'min_samples_split': np.linspace(.1, 1, 10, endpoint=True),
        'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
        'max_features': ['auto', 'sqrt', None]
    }

    pre_gs_inst = RandomizedSearchCV(RandomForestClassifier(),
                                     param_distributions=hyperparameters,
                                     cv=10,
                                     n_iter=100,
                                     n_jobs=-1)

    pre_gs_inst.fit(X_train, y_train)
    print(pre_gs_inst.best_params_)


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42)

    tune_nb_tree(X_train, y_train, X_test, y_test)
    tune_depth_tree(X_train, y_train, X_test, y_test)
    tune_min_samples_split(X_train, y_train, X_test, y_test)
    tune_min_samples_leaf(X_train, y_train, X_test, y_test)

    randomized_search(X_train, y_train)


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

    return parser.parse_intermixed_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    main(args)
