import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.linalg import svd

if __name__ == '__main__':
    df = pd.read_csv('../data/features_all_nochnnels.csv', index_col=['File', 'Segment'], sep=';')

    # FEATURES NAMES, RESPONSE AND CLASSES
    C = 2
    classNames = ['Bad', 'Good']
    y = df['labels_jules'].values
    df.drop('labels_jules', axis=1, inplace=True)
    column_names = df.columns

    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(df)

    # PCA by computing SVD of Y
    U, s, V = svd(x, full_matrices=False)
    S = np.diag(s)

    # matrix of features after SVD
    A = U @ S @ V

    # Frobenius norm between features after and before SVD
    norm_ = np.linalg.norm(x - A)

    # Compute variance explained by principal components
    var_ = (s * s) / (s * s).sum()

    # Plot variance explained
    plt.figure()
    plt.plot(range(1, len(var_) + 1), var_, 'o-')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.show()

    # # Project the centered data onto principal component space
    Z = x @ V

    # Indices of the principal components to be plotted
    i, j = 0, 1

    # Plot PCA of the data
    f = plt.figure()
    plt.title('EEG data: PCA')
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y == c
        plt.plot(Z[class_mask, i], Z[class_mask, j], 'o')
    plt.legend(classNames)
    plt.xlabel('PC{0}'.format(i + 1))
    plt.ylabel('PC{0}'.format(j + 1))

    plt.show()
