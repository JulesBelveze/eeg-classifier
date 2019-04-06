# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:03:58 2019

@author: mirab
"""
#mne.open_docs() to open the library

import mne.io  
import matplotlib.pyplot as plt
import pandas as pd
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
from pyeeg import *
import numpy as np

time_segment = 180 #seconds - 3 min
file=1
raw_fname = 'real_EEG_data/'+str(file)+'.edf'
raw = mne.io.read_raw_edf(raw_fname,preload=True)
segment =0  #0 for good segment 5 for bad 
raw.crop(180*segment,180+180*segment)
x=raw.get_data()
bins=[0.5,4,7,12,30]
for i in range(0,x.shape[0]):
    
    
    
    print(fisher_info(x[i],1,50)) #same as svd_entropy
    #print(svd_entropy(x[i],1,50)) # I don't know what the fucking numbers are
    
    #power,powerratio= bin_power(x[i],Band=bins,Fs=256)
    #print(spectral_entropy(x[i],Band=bins,Fs=256,Power_Ratio=powerratio)) #use power ratio from bin power to speed up
    
    #print(hjorth(x[i])) #tuples of max and min
    
    #print(hfd(x[i],50))#kmax=50 need to be ajusted

    #print(pfd(x[i]))
    #print(bin_power(x[i],Band=bins,Fs=256))
    #dfa(x[i])
    #hurst(x[i])
