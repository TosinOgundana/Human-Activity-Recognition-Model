import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class Preprocessing:
    def __init__(self):
        #self.data_df = data_df
        self.segments = None
        self.segments_label = None
        self.scaler = None #Saves the  MinMaxScaler object
        self.labelbinarizer = None #Saves the LabelBinarizer object
    
    def windows(self, data, size, overlap):
        start = 0
        while start < len(data):
            yield start, start + size
            start += int((size / int(1/overlap))) #Ensures overlapping based on the overlap fraction
        
    def segment_data(self, X, y, window_size = 90, overlap=0.5):#overlap: fraction of window_size that should overlap
        X, y = np.array(X), np.array(y) #Convert data to numpy array

        segments = np.empty((0,window_size,X.shape[1]))
        segments_label = np.empty((0)) 
        #Empty arays to store segments and values

        for start, end in self.windows(X, window_size, overlap):
            #segment = column[start:end]
            columns_in_segment = [X[start:end, i] for i in range(X.shape[1])]

            if(len(X[start:end]) == window_size): #Ensure that the last segment is dropped
                                                            #if it contains index that is out of range for the column
                segments = np.vstack([segments,np.dstack(columns_in_segment)])
                
                segments_label = np.append(segments_label, stats.mode(y[start:end])[0][0])
                #For each segment the label will be the highest occuring label (mode)
        self.segments, self.segments_label = segments, segments_label
        print('Done')
        return self.segments, self.segments_label
        
    def min_max_scale(self, X_train, X_test):
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        #Scale data, return the scaled data and the scaler object
        
        #Transform the test data
        X_test_scaled = self.scaler.transform(X_test)
        #The train data is usually fitted and transformed while the test data is only transformed to train data's scale
        return X_train_scaled, X_test_scaled
    
    def imbalance_handle(self, X_train, y_train):
        strategy1 = {k:2000 for k in np.unique(y_train)}
        strategy2 = {k:1000 for k in np.unique(y_train)}
        #First, the data is oversampled 
        #Afterwards undersampled
        
        oversample = SMOTE(sampling_strategy=strategy1, k_neighbors=1)
        undersample = RandomUnderSampler(sampling_strategy=strategy2)      
        
        X_train_sampled, y_train_sampled = oversample.fit_resample(X_train, y_train)
        X_train_sampled, y_train_sampled = undersample.fit_resample(X_train_sampled, y_train_sampled)
        
        return X_train_sampled, y_train_sampled
    
    def plot_target_distribution(self, y):
        plt.figure(figsize=(10,5))
        chart = sns.countplot(y, palette='Set1')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=70)

