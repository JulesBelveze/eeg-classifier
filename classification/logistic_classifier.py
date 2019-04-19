import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import sys

sys.path.insert(0, "../dimension_reduction/")
from dimension_reduction import bonferonni_corr


def oversample(df):
    count = df['labels_jules'].value_counts()
    max_count = max(count)

    df_bad = df[df['labels_jules'] == 0]
    df_good_oversampled = df[df['labels_jules'] == 1].sample(n=max_count, replace=True)
    df_oversampled = pd.concat([df_bad, df_good_oversampled])

    return shuffle(df_oversampled)


if __name__ == '__main__':
    df = pd.read_csv("../data/features_all_nochnnels.csv", index_col=['File', 'Segment'], sep=";")
    df = oversample(df)

    Y = df['labels_jules'].values
    X = df.drop('labels_jules', axis=1).values

    # getting the most significant features and removing the others
    features_to_include = bonferonni_corr(X, Y)
    columns_to_include = df.columns[features_to_include].values
    X = df[columns_to_include].values

    # classification models
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_train, Y_train)
    Y_hat = clf.predict(X_test)


    # printing metrics
    target_names = ['bad', 'good']
    print(classification_report(Y_test, Y_hat, target_names=target_names))
    print(confusion_matrix(Y_test, Y_hat))
    print("Accuracy: %s" % accuracy_score(Y_test, Y_hat))
