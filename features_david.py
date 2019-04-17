

import mne.io  
import pandas as pd
import numpy as np
from features import hjorth2, skewness, autogressiveModelParametersBurg, autogressiveModelParameters, first_diff_mean, \
    first_diff_max, spectral_entropy, bin_power, maxPwelch, wavelet_features, slope_var, slope_mean, secDiffMax, \
    secDiffMean, kurtosis, coeff_var

columns = ['File','Segment','first_diff_mean', 'first_diff_max', 'kurtosis', 'sec_diff_mean', 'sec_diff_max',
                         'coef_variation',
                         'slopemean', 'slopevar', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'welch1',
                         'welch2',
                         'welch3', 'welch4', 'power1', 'power2', 'power3', 'power4', 'ratio1', 'ratio2', 'ratio3',
                         'ratio4',
                         'spec_entropy']
col_channels = ['Mobility1', 'Mobility2', 'Mobility3', 'Mobility4', 'Mobility5', 'Mobility6', 'Mobility7', 'Mobility8', 'Mobility9', 'Mobility10', 'Mobility11', 'Mobility12', 'Mobility13', 'Mobility14','Complexity1', 'Complexity2', 'Complexity3', 'Complexity4', 'Complexity5', 'Complexity6', 'Complexity7', 'Complexity8', 'Complexity9', 'Complexity10', 'Complexity11', 'Complexity12', 'Complexity13', 'Complexity14', 'Skewness1', 'Skewness2', 'Skewness3', 'Skewness4', 'Skewness5', 'Skewness6', 'Skewness7', 'Skewness8', 'Skewness9', 'Skewness10', 'Skewness11', 'Skewness12', 'Skewness13', 'Skewness14']
columns=columns+col_channels
df = pd.DataFrame(columns=columns)
time_segment = 180 #seconds - 3 min

for file in range(1,32):
    raw_fname = 'real_EEG_data/'+str(file)+'.edf'
    raw = mne.io.read_raw_edf(raw_fname,preload=True)
    data, times = raw[-1,-1:]
    total_time = float(times)
    
    t_start = 0
    t_end = t_start + time_segment
    segment = 1

    
    while t_end < total_time:
        try:
            start, stop = raw.time_as_index([t_start, t_end])
            
            #array d'arrays: outer array es el segment, inner array es el channel
            channels = raw.get_data(start=start,stop=stop)
            channels = channels[:-1]
            
            
            #another loopfor every channel
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
    
            array_features = [file,segment,a, b, kurt, secdiffmean, secdiffmax, coef_variation, slopemean, slopevar, w1, w2, w3, w4,
                              w5,
                              w6,
                              w7, w8, welch1, welch2, welch3, welch4, power1, power2, power3, power4, ratio1, ratio2,
                              ratio3,
                              ratio4, spec_entropy]
            
    
            #df = df.append(pd.Series(array_features,index=columns),ignore_index=True)
            
            
            # per channel Extracting features
            mobility = []
            complexity = []
            skewness_list = []

            # Features that are computed per channel (each feature has 14 components, one per channel)
            for j in range(0, 14):
                mobility.append(hjorth2(channels[j])[0])
                complexity.append(hjorth2(channels[j])[1])
                skewness_list.append(skewness(channels[j]))
            
            data = mobility+ complexity + skewness_list
            df = df.append(pd.Series(array_features + data,index=columns),ignore_index=True)


            
            
            
          
            
            print(str(file)+' segment '+str(segment)+' from '+str(t_start)+' to '+ str(t_end))
            
            
        except:
            print("\nError in file %s cannot extract features.\n" +str(file)+' segment '+str(segment))
        t_start = t_end
        t_end += time_segment
        segment += 1

df.to_csv('features_all.csv')