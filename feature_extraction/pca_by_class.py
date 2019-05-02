import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt


def remove_correlated(df):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.7
    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]

    # Drop features 
    df = df.drop(to_drop, axis=1)

    print('Features selected ' + str(df.columns))
    return df


if __name__ == '__main__':
    # DATA PREPARATION
    df = pd.read_csv('../data/features_standardized.csv', sep=';')
    Y = df.labels_jules.values  # Response

    df = remove_correlated(df)
    classNames = ['Bad', 'Good']

    C = 2  # Number of classes

    X0 = df[df['labels_jules'] == 0].drop(['labels_jules'], axis=1).values
    X1 = df[df['labels_jules'] == 1].drop(['labels_jules'], axis=1).values

    # PCA by computing SVD of X
    U0, S0, V0 = svd(X0, full_matrices=False)
    U1, S1, V1 = svd(X1, full_matrices=False)

    # Compute variance explained by principal components
    rho0 = (S0 * S0) / (S0 * S0).sum()
    rho1 = (S1 * S1) / (S1 * S1).sum()

    # Plot variance explained
    plt.figure()
    plt.plot(range(1, len(rho0) + 1), rho0, 'o-', color='b')
    plt.plot(range(1, len(rho1) + 1), rho1, 'o-', color='r')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.grid(False)
    plt.legend(classNames)
    plt.show()

    # Plot feature importance for 1st component
    plt.figure()
    plt.legend(classNames)
    plt.plot(df.columns[:-1], V0[0], 'o', color='b')
    plt.plot(df.columns[:-1], V1[0], 'o', color='r')
    plt.xticks(rotation=90)
    plt.legend(classNames)
    plt.show()

    # Plot feature importance for 2nd component
    plt.figure()
    plt.legend(classNames)
    plt.plot(df.columns[:-1], V0[1], 'o', color='b')
    plt.plot(df.columns[:-1], V1[1], 'o', color='r')
    plt.xticks(rotation=90)
    plt.legend(classNames)
    plt.show()

    # two Principal PCA projection
    V0 = V0.T
    V1 = V1.T

    # Project the centered data onto principal component space
    Z0 = X0 @ V0
    Z1 = X1 @ V1

    clas = ['Bad']
    color = 'b'
    for Z in [Z0, Z1]:
        # Indices of the principal components to be plotted
        i = 0
        j = 1

        # Plot PCA of the data
        f = plt.figure()
        plt.title('EEG data: 2 PCA')

        # Z = array(Z)

        plt.plot(Z[:, i], Z[:, j], 'o', color=color)

        plt.legend(clas)
        plt.xlabel('PC{0}'.format(i + 1))
        plt.ylabel('PC{0}'.format(j + 1))
        # Output result to screen
        plt.show()

'''
#two Principal PCA projection
V = V.T
# Project the centered data onto principal component space
Z = x @ V


# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('EEG data: PCA')

#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o')
    
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()
'''
