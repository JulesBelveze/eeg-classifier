import pywt
import scipy.stats as sp
from numpy import zeros, floor, log10, log, array, sqrt, cumsum, log2, std
from spectrum import *
from scipy import signal
from numpy.fft import fft
from numpy.linalg import lstsq
from nitime import algorithms as alg
from scipy.signal import argrelextrema


# ----------------------------- Hjorth parameters ----------------------------
def first_order_diff(X):
    D = []
    for i in range(1, len(X)):
        D.append(X[i] - X[i - 1])
    return D


def hjorth2(X, D=None):
    if D is None:
        D = first_order_diff(X)

    D.insert(0, X[0])  # pad the first difference
    D = array(D)
    n = len(X)
    M2 = float(sum(D ** 2)) / n
    TP = sum(array(X) ** 2)
    M4 = 0;
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
        M4 = M4 / n

    return sqrt(M2 / TP), sqrt(float(M4) * TP / M2 / M2)


def bin_power(X, Band, Fs):
    C = fft(X)
    C = abs(C)
    Power = zeros(len(Band))
    for Freq_Index in range(0, len(Band) - 1):
        Freq = Band[Freq_Index]
        Next_Freq = Band[Freq_Index + 1]
        Power[Freq_Index] = sum(C[int(floor(Freq / Fs * len(X))):int(floor(Next_Freq / Fs * len(X)))])
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio


# ---------------------- Kurtosis , 2nd Diff Mean, 2nd Diff Max -----------------------

def kurtosis(a):
    b = a  # Extracting the data from the 14 channels
    output = np.zeros(len(b))  # Initializing the output array with zeros (length = 14)
    k = 0;  # For counting the current row no.
    for i in b:
        mean_i = np.mean(i)  # Saving the mean of array i
        std_i = np.std(i)  # Saving the standard deviation of array i
        t = 0.0
        for j in i:
            t += (pow((j - mean_i) / std_i, 4) - 3)
        kurtosis_i = t / len(i)  # Formula: (1/N)*(summation(x_i-mean)/standard_deviation)^4-3
        output[k] = kurtosis_i  # Saving the kurtosis in the array created
        k += 1  # Updating the current row no.
    return np.sum(output) / 14


def secDiffMean(a):
    b = a  # Extracting the data of the 14 channels
    output = np.zeros(len(b))  # Initializing the output array with zeros (length = 14)
    temp1 = np.zeros(len(b[0]) - 1)  # To store the 1st Diffs
    k = 0;  # For counting the current row no.
    for i in b:
        t = 0.0
        for j in range(len(i) - 1):
            temp1[j] = abs(i[j + 1] - i[j])  # Obtaining the 1st Diffs
        for j in range(len(i) - 2):
            t += abs(temp1[j + 1] - temp1[j])  # Summing the 2nd Diffs
        output[k] = t / (len(i) - 2)  # Calculating the mean of the 2nd Diffs
        k += 1  # Updating the current row no.
    return np.sum(output) / 14


def secDiffMax(a):
    b = a  # Extracting the data from the 14 channels
    output = np.zeros(len(b))  # Initializing the output array with zeros (length = 14)
    temp1 = np.zeros(len(b[0]) - 1)  # To store the 1st Diffs
    k = 0;  # For counting the current row no.
    t = 0.0
    for i in b:
        for j in range(len(i) - 1):
            temp1[j] = abs(i[j + 1] - i[j])  # Obtaining the 1st Diffs

        t = temp1[1] - temp1[0]
        for j in range(len(i) - 2):
            if abs(temp1[j + 1] - temp1[j]) > t:
                t = temp1[j + 1] - temp1[j]  # Comparing current Diff with the last updated Diff Max

        output[k] = t  # Storing the 2nd Diff Max for channel k
        k += 1  # Updating the current row no.
    return np.sum(output) / 14


# ----------------------- Coefficient of variation -------------------

def coeff_var(a):
    b = a  # Extracting the data from the 14 channels
    output = np.zeros(len(b))  # Initializing the output array with zeros
    k = 0;  # For counting the current row no.
    for i in b:
        mean_i = np.mean(i)  # Saving the mean of array i
        std_i = np.std(i)  # Saving the standard deviation of array i
        output[k] = std_i / mean_i  # computing coefficient of variation
        k = k + 1
    return np.sum(output) / 14


# ----------------- Skewness , 1st Difference Mean, 1st Difference Max --------------

def skewness(arr):
    return sp.stats.skew(arr, axis=0, bias=True)


def first_diff_mean(arr):
    data = arr
    diff_mean_array = np.zeros(len(data))  # Initialinling the array as all 0s
    index = 0  # current cell position in the output array

    for i in data:
        sum = 0.0  # initializing the sum at the start of each iteration
        for j in range(len(i) - 1):
            sum += abs(i[j + 1] - i[j])  # Obtaining the 1st Diffs

        diff_mean_array[index] = sum / (len(i) - 1)
        index += 1  # updating the cell position
    return np.sum(diff_mean_array) / 14


def first_diff_max(arr):
    data = arr
    diff_max_array = np.zeros(len(data))  # Initialinling the array as all 0s
    first_diff = np.zeros(len(data[0]) - 1)  # Initialinling the array as all 0s
    index = 0  # current cell position in the output array

    for i in data:
        max = 0.0  # initializing at the start of each iteration
        for j in range(len(i) - 1):
            first_diff[j] = abs(i[j + 1] - i[j])  # Obtaining the 1st Diffs
            if first_diff[j] > max:
                max = first_diff[j]  # finding the maximum of the first differences
        diff_max_array[index] = max
        index += 1  # updating the cell position
    return np.sum(diff_max_array) / 14


# ------------------------- Wavelet transform features ------------------------

# Adapted Wavelet transform
def wavelet_features(epoch):
    cA_values = []
    cD_values = []
    cA_mean = []
    cA_std = []
    cA_Energy = []
    cD_mean = []
    cD_std = []
    cD_Energy = []
    Entropy_D = []
    Entropy_A = []
    # For each channel I compute the discret wavelet transform
    for i in range(14):
        cA, cD = pywt.dwt(epoch[i][:], 'coif1')
        cA_values.append(cA)
        cD_values.append(cD)  # calculating the coefficients of wavelet transform.
    # I make the mean for each channel
    for x in range(14):
        cA_mean.append(np.mean(cA_values[x]))
        cA_std.append(np.std(cA_values[x]))
        cA_Energy.append(np.sum(np.square(cA_values[x])))
        cD_mean.append(
            np.mean(cD_values[x]))  # mean and standard deviation values of coefficents of each channel is stored .
        cD_std.append(np.std(cD_values[x]))
        cD_Energy.append(np.sum(np.square(cD_values[x])))
        Entropy_D.append(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x]))))
        Entropy_A.append(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x]))))
    return np.sum(cA_mean) / 14, np.sum(cA_std) / 14, np.sum(cD_mean) / 14, np.sum(cD_std) / 14, np.sum(
        cA_Energy) / 14, np.sum(cD_Energy) / 14, np.sum(Entropy_A) / 14, np.sum(Entropy_D) / 14


# --------------------- Variance and Mean of Vertex to Vertex Slope ---------------------

def first_diff(i):
    b = i
    out = np.zeros(len(b))

    for j in range(len(i)):
        out[j] = b[j - 1] - b[j]  # Obtaining the 1st Diffs

        j = j + 1
        c = out[1:len(out)]
    return c


def slope_mean(p):
    b = np.array(p)  # Extracting the data from the 14 channels
    output = np.zeros(len(b))  # Initializing the output array with zeros
    res = np.zeros(len(b) - 1)

    k = 0;  # For counting the current row no.
    for i in b:
        x = i
        amp_max = i[argrelextrema(x, np.greater)[0]]
        t_max = argrelextrema(x, np.greater)[0]
        amp_min = i[argrelextrema(x, np.less)[0]]
        t_min = argrelextrema(x, np.less)[0]
        t = np.concatenate((t_max, t_min), axis=0)
        t.sort()  # sort on the basis of time

        h = 0
        amp = np.zeros(len(t))
        res = np.zeros(len(t) - 1)
        for l in range(len(t)):
            amp[l] = i[t[l]]

        amp_diff = first_diff(amp)

        t_diff = first_diff(t)

        for q in range(len(amp_diff)):
            res[q] = amp_diff[q] / t_diff[q]
        output[k] = np.mean(res)
        k = k + 1
    return np.sum(output) / 14


def slope_var(p):
    b = np.array(p)  # Extracting the data from the 14 channels
    output = np.zeros(len(b))  # Initializing the output array with zeros
    res = np.zeros(len(b) - 1)

    k = 0;  # For counting the current row no.
    for i in b:
        x = i
        amp_max = i[argrelextrema(x, np.greater)[0]]  # storing maxima value
        t_max = argrelextrema(x, np.greater)[0]  # storing time for maxima
        amp_min = i[argrelextrema(x, np.less)[0]]  # storing minima value
        t_min = argrelextrema(x, np.less)[0]  # storing time for minima value
        t = np.concatenate((t_max, t_min), axis=0)  # making a single matrix of all matrix
        t.sort()  # sorting according to time

        h = 0
        amp = np.zeros(len(t))
        res = np.zeros(len(t) - 1)
        for l in range(len(t)):
            amp[l] = i[t[l]]

        amp_diff = first_diff(amp)

        t_diff = first_diff(t)

        for q in range(len(amp_diff)):
            res[q] = amp_diff[q] / t_diff[q]  # calculating slope

        output[k] = np.var(res)
        k = k + 1  # counting k
    return np.sum(output) / 14


# -------------------- FFT features(Max Power) ---------------------

def maxPwelch(data_win, Fs):
    BandF = [0.1, 3, 7, 12, 30]
    PMax = np.zeros([14, (len(BandF) - 1)])

    for j in range(14):
        f, Psd = signal.welch(data_win[j], Fs)

        for i in range(len(BandF) - 1):
            fr = np.where((f > BandF[i]) & (f <= BandF[i + 1]))
            PMax[j, i] = np.max(Psd[fr])

    return np.sum(PMax[:, 0]) / 14, np.sum(PMax[:, 1]) / 14, np.sum(PMax[:, 2]) / 14, np.sum(PMax[:, 3]) / 14


# ---------------------- Shanon Entropy and Entropy Spectral ----------------------

def spectral_entropy(Power_Ratio):
    Spectral_Entropy = 0
    for i in range(0, len(Power_Ratio) - 1):
        Spectral_Entropy += Power_Ratio[i] * log(Power_Ratio[i])
    Spectral_Entropy /= log(len(Power_Ratio))  # to save time, minus one is omitted
    return -1 * Spectral_Entropy


def in_range(Template, Scroll, Distance):
    for i in range(0, len(Template)):
        if abs(Template[i] - Scroll[i]) > Distance:
            return False
    return True


# ------------------ Autoregression model- Yule Walker Algorithm ------------------
def autogressiveModelParameters(epoch):
    feature = []
    # Order 11 for the autoregressive model, why?
    for i in range(14):
        coeff, sig = alg.AR_est_YW(np.array(epoch[i]), 11, )
        feature.append(coeff)

    return feature


# ------------------------- Autoregression model- Burg Algorithm --------------------

def autogressiveModelParametersBurg(labels):
    feature = []
    feature1 = []
    model_order = 3
    for i in range(14):
        AR, rho, ref = arburg(labels[i], model_order)
        feature.append(AR)

    for j in range(len(feature)):
        for i in range(model_order):
            feature1.append(feature[j][i])

    return feature1


# ----------------------------- Hurst exponent ------------------------------
def hurst(X):
    N = len(X)

    T = array([float(i) for i in range(1, N + 1)])
    Y = cumsum(X)
    Ave_T = Y / T

    S_T = zeros((N))
    R_T = zeros((N))
    for i in range(N):
        S_T[i] = std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = max(X_T[:i + 1]) - min(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = log(R_S)
    n = log(T).reshape(N, 1)
    H = lstsq(n[1:], R_S[1:])[0]
    return H[0]
