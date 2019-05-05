import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import sys

sys.path.insert(0, "../feature_extraction/")
from pca_by_class import remove_correlated


def oversample_smote(X, y):
    sm = SMOTE(random_state=2)
    X, y = sm.fit_sample(X, y)
    return X, y


if __name__ == '__main__':
    df = pd.read_csv("../data/features_all_nochnnels.csv", index_col=['File', 'Segment'], sep=";")

    Y = df['labels_jules'].values
    df = remove_correlated(df.drop("labels_jules", axis=1))
    X = df.values

    X, Y = oversample_smote(X, Y)
    X = scale(X)

    # classification models
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_train, Y_train)
    Y_hat = clf.predict(X_test)

    # printing metrics
    target_names = ['bad', 'good']
    print(classification_report(Y_test, Y_hat, target_names=target_names))
    print(confusion_matrix(Y_test, Y_hat))
    print("Accuracy: %s" % accuracy_score(Y_test, Y_hat))
