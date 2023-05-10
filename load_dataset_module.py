import pandas as pd
import numpy as np

from feature_module import Features

class LoadDataset:
    def __init__(self):
        self.data = None
        self.features = None
        self.function_names = {'Variance(Var)':'variance_compute', 'Median':'median_compute', 'Mean':'mean_compute',
                               'Standard Deviation(STD)':'stdev_compute', 'Root Mean Square':'root_mean_square_compute',
                               'Sum of Squares':'sum_of_squares_compute', 'Zero Crossing':'zero_crossing_compute'}
            #Variable is made global in order to be accessed when calling write_dict_to_csv in main

            # Features to be computed are listed here and works provided the corresponding function appears in feature_module
            # and that it requires activity_data as the only compulsory argument
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        return self.data

    def get_features(self, activity_data): #data must be dictionary or Pandas dataframe
        activity_features = {}
        activity_data = {k:list(activity_data[k]) for k in activity_data.columns} if type(activity_data)==pd.core.frame.DataFrame else activity_data
        #Convert activity_data to a dictionary if it is a DataFrame

        for key, value in activity_data.items():
            feature = []
            #activity_features[key] = []
            features = Features(value)
            for func in self.function_names.values():
                feature += list(np.array(getattr(features, func)()).flatten())
                activity_features[key] = feature

        cov = Features(activity_data).covariance_compute() #Compute covariance separately since it requires comparing two features and not evaluated from a single feature.
        feat_part = np.array(list(activity_features.values())).flatten()
        return np.append(feat_part, cov)