import pandas as pd
import warnings
import argparse
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)

    Y_hat_train = kmeans.labels_
    Y_hat_test = kmeans.predict(X_test)

    print("accuracy baseline: %f" % accuracy_baseline)
    print(" \n--------- Training -----------")
    print("Accuracy: %f" % accuracy_score(y_train, Y_hat_train))
    print(confusion_matrix(y_train, Y_hat_train))
    print(" \n----------- Test -------------")
    print("Accuracy: %f" % accuracy_score(y_test, Y_hat_test))
    print(confusion_matrix(y_test, Y_hat_test))


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
        '--neighbors',
        type=int,
        default=2,
        help='Specify the number of clusters to use for the KMeans algorithm'
    )
    return parser.parse_intermixed_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    main(args)
