import numpy as np
import pandas as pd

class Features:
    def __init__(self, data):
        self.data = data
    
    def mean_compute(self):
        return np.mean(self.data)

    def variance_compute(self):
        return np.var(self.data)

    def median_compute(self):
        return np.median(self.data)

    def stdev_compute(self):
        return np.std(self.data)

    def root_mean_square_compute(self):
        return np.sqrt(np.mean(np.square(self.data)))

    def sum_of_squares_compute(self):
        return np.sum((np.array(self.data) - np.mean(self.data)) ** 2)

    def covariance_compute(self):
        #Use pandas to calculate covariance if data is a DataFrame or numpy otherwise
        cov = pd.DataFrame(self.data).cov()# if type(self.data)==pd.core.frame.DataFrame else np.cov(self.data)
        cov = np.array(cov)
        return cov[np.triu_indices(n=cov.shape[1])] #triu_indices returns indices for upper triangle

    def zero_crossing_compute(self):
        data = np.array(self.data)
        zero_cross = ((data[:-1] * data[1:]) < 0).sum()
        return zero_cross
    
    def write_dict_to_csv(self, write_data, file_name, row_label=None, header=True):
        pd.DataFrame(write_data, index=row_label).to_csv(file_name)
        print("Write to {} successful".format(file_name))

