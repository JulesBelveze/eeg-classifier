import pandas as pd
import numpy as np
import os
from progress.bar import Bar
from features import hjorth2, skewness, autogressiveModelParametersBurg, autogressiveModelParameters, first_diff_mean, \
    first_diff_max, spectral_entropy, bin_power, maxPwelch, wavelet_features, slope_var, slope_mean, secDiffMax, \
    secDiffMean, kurtosis, coeff_var

if __name__ == '__main__':
    names = ['Activity', 'Mobility', 'Complexity', 'Kurtosis', '2nd Difference Mean', '2nd Difference Max',
             'Coeffiecient of Variation', 'Skewness', '1st Difference Mean', '1st Difference Max',
             'Wavelet Approximate Mean', 'Wavelet Approximate Std Deviation', 'Wavelet Detailed Mean',
             'Wavelet Detailed Std Deviation', 'Wavelet Approximate Energy', 'Wavelet Detailed Energy',
             'Wavelet Approximate Entropy', 'Wavelet Detailed Entropy', 'Variance', 'Mean of Vertex to Vertex Slope',
             'FFT Delta MaxPower', 'FFT Theta MaxPower', 'FFT Alpha MaxPower', 'FFT Beta MaxPower',
             'Autro Regressive Mode Order 3 Coefficients for each channel ->']
    channel = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14',
               'ch15']

    path_to_data_folder = "../real_EEG_data/csv_files/"
    data_files = os.listdir(path_to_data_folder)
    data_files = [f for f in data_files if f.endswith('.csv')]

    bar = Bar("Starting data processing...", max=len(data_files))
    for file in data_files:
        path_to_file = os.path.join(path_to_data_folder, file)
        df = pd.read_csv(path_to_file)

        channels = []
        for j in range(0, 14):
            ch = []
            channels.append(ch)

        # Creating an array of arrays with the data from each channel
        for j in range(14):
            for i in range(1, 23041):
                channels[j].append(df.iloc[j, i])

        # Extracting features
        mobility = []
        complexity = []
        skewness_list = []
        hurst_list = []

        # Features that are computed per channel (each feature has 14 components, one per channel)
        for j in range(0, 14):
            mobility.append(hjorth2(channels[j])[0])
            complexity.append(hjorth2(channels[j])[1])
            skewness_list.append(skewness(channels[j]))
            # hurst_list.append(hurst(channels[j]))

        autoregressive_burg = autogressiveModelParametersBurg(channels)
        autoregressive = autogressiveModelParameters(channels)

        Band = [0.1, 3, 7, 12, 30]
        a = first_diff_mean(channels)
        b = first_diff_max(channels)
        kurt = kurtosis(channels)
        secdiffmean = secDiffMean(channels)
        secdiffmax = secDiffMax(channels)
        coef_variation = coeff_var(channels)
        slopemean = slope_mean(channels)
        slopevar = slope_var(channels)
        wavelet_features_list = (wavelet_features(channels))
        w1 = wavelet_features_list[0]
        w2 = wavelet_features_list[1]
        w3 = wavelet_features_list[2]
        w4 = wavelet_features_list[3]
        w5 = wavelet_features_list[4]
        w6 = wavelet_features_list[5]
        w7 = wavelet_features_list[6]
        w8 = wavelet_features_list[7]

        maxPwelch_list = maxPwelch(channels, 256)
        welch1 = maxPwelch_list[0]
        welch2 = maxPwelch_list[1]
        welch3 = maxPwelch_list[2]
        welch4 = maxPwelch_list[3]
        binpower, binratio = bin_power(channels[0], Band, 256)
        power1 = binpower[0]
        power2 = binpower[1]
        power3 = binpower[2]
        power4 = binpower[3]
        ratio1 = binratio[0]
        ratio2 = binratio[1]
        ratio3 = binratio[2]
        ratio4 = binratio[3]

        spec_entropy = spectral_entropy(binratio)

        array_features = [a, b, kurt, secdiffmean, secdiffmax, coef_variation, slopemean, slopevar, w1, w2, w3, w4, w5, w6,
                          w7, w8, welch1, welch2, welch3, welch4, power1, power2, power3, power4, ratio1, ratio2, ratio3,
                          ratio4, spec_entropy]
        name_features = ['first_diff_mean', 'first_diff_max', 'kurtosis', 'sec_diff_mean', 'sec_diff_max', 'coef_variation',
                         'slopemean', 'slopevar', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'welch1', 'welch2',
                         'welch3', 'welch4', 'power1', 'power2', 'power3', 'power4', 'ratio1', 'ratio2', 'ratio3', 'ratio4',
                         'spec_entropy']

        df = pd.DataFrame(array_features)

        df.rename(index={0: 'first_diff_mean', 1: 'first_diff_max', 2: 'kurtosis', 3: 'sec_diff_mean', 4: 'sec_diff_max',
                         5: 'coef_variation', 6: 'slopemean', 7: 'slopevar', 8: 'w1', 9: 'w2', 10: 'w3', 11: 'w4', 12: 'w5',
                         13: 'w6', 14: 'w7', 15: 'w8', 16: 'welch1', 17: 'welch2', 18: 'welch3', 19: 'welch4', 20: 'power1',
                         21: 'power2', 22: 'power3', 23: 'power4', 24: 'ratio1', 25: 'ratio2', 26: 'ratio3', 27: 'ratio4',
                         28: 'spec_entropy'}, inplace=True)
        df.to_csv("../real_EEG_data/csv_features/features_per_recording_no_channel" + file)

        data = np.array([mobility, complexity, skewness_list])
        transpose = data.transpose()
        df2 = pd.DataFrame(transpose)
        df2.rename(
            index={0: 'ch1', 1: 'ch2', 2: 'ch3', 3: 'ch4', 4: 'ch5', 5: 'ch6', 6: 'ch7', 7: 'ch8', 8: 'ch9', 9: 'ch10',
                   10: 'ch11', 11: 'ch12', 12: 'ch13', 13: 'ch14'}, inplace=True)
        df2.columns = ['Mobility', 'Complexity', 'skewness']
        df2.to_csv("../real_EEG_data/csv_features/features_per_channel" + file)

        bar.next()
    bar.finish()


