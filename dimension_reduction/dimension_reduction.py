import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from regressors import stats
from scipy import stats
from sklearn import preprocessing
from scipy.linalg import svd
from sklearn.decomposition import PCA


def svd_extraction(X):
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    # PCA by computing SVD of Y
    U, s, V = svd(X, full_matrices=False)
    S = np.diag(s)

    # matrix of features after SVD
    A = U @ S @ V

    # Frobenius norm between features after and before SVD
    norm_ = np.linalg.norm(X - A)

    # Compute variance explained by principal components
    var_ = (s * s) / (s * s).sum()

    return A, var_


def pca_extraction(X, n_components):
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(X)

    X_new = pca.transform(X)

    var_ratio = pca.explained_variance_ratio_

    return X_new, var_ratio


def bonferonni_corr(X, Y, _alpha=.05):
    nb_features = len(X[0])
    PV = np.zeros((nb_features))  # array containing p-values

    for i in range(nb_features):
        X_sub = X[:, i]
        slope, intercept, r_value, PV[i], std_err = stats.linregress(X_sub.astype(float), Y.astype(float))

    features_to_include = [i for i, elt in enumerate(PV) if elt < _alpha / nb_features]
    print("According to Bonferroni correction we need to include {} features. \n".format(
        len(features_to_include)))

    return features_to_include


if __name__ == "__main__":
    df = pd.read_csv('../data/features_all_nochnnels.csv', index_col=['File', 'Segment'], sep=";")
    Y = df['labels_jules'].values
    X = df.drop('labels_jules', axis=1).values

    _, nb_features = X.shape

    # investigating the number of needed features
    features_to_include = bonferonni_corr(X, Y)
    print("The columns extracted by forward selection are: {}".format(df.columns[features_to_include].values))


