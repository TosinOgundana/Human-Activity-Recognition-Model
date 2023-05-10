import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from  sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt
#%matplotlib inline

class Models:
    
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = [None]*4 #Set all to None
        
        self.result_dict = {} #To save data transformations, train scores,
                                                                         #test scores, object of models etc to be pickled
        self.result_dict['Accuracy Score'] = {}
        self.result_dict['Model Objects'] = {}
        self.result_dict['Parameters'] = {}
        self.result_dict['Predictions'] = {}

        #To plot the plot the graph of the accuracies for the used algorithms
    def plot_evaluations(self):
        pd.DataFrame(self.result_dict['Accuracy Score']).plot.bar(figsize=(10,8), grid=True, title='Accuracy Scores')

    def plot_conf_mat(self, y_true, y_pred):
            CM = confusion_matrix(y_true, y_pred)
            fig, ax = plot_confusion_matrix(conf_mat=CM , figsize=(20, 10))
            plt.show()


    def result(self, model_object, model_name, X_train, X_test,
               y_train, y_test, print_summary=True):
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        y_train_pred = model_object.predict(self.X_train)
        y_test_pred = model_object.predict(self.X_test)
 #Save
        train_acc_score = accuracy_score(self.y_train, y_train_pred)

        test_acc_score = accuracy_score(self.y_test, y_test_pred)
        self.result_dict['Accuracy Score'][model_name] = {}
        self.result_dict['Accuracy Score'][model_name]['Train'] = train_acc_score
        self.result_dict['Accuracy Score'][model_name]['Test'] = test_acc_score


        choice = input('Press 1 to save Model object and parameters to result_dict, any other key to cancel.\nPrevious save with same model name is overwritten: ')
        if choice == '1':
            param = model_object.__dict__ 
            self.result_dict['Model Objects'].update({model_name:model_object})
            self.result_dict['Parameters'].update({model_name:param})
            self.result_dict['Predictions'][model_name] = {}
            self.result_dict['Predictions'][model_name]['Train'] = y_train_pred
            self.result_dict['Predictions'][model_name]['Test'] = y_test_pred
            self.result_dict['Datasets'] = {'X_train':self.X_train, 'X_test':self.X_test,
                                            'y_train':self.y_train, 'y_test':self.y_test}

        if print_summary:
            print('\n--------------------------------Train Set-----------------------------------------')
            print(classification_report(self.y_train, y_train_pred))
            print('\n\n                        Train Confusion Matrix')
            self.plot_conf_mat(self.y_train, y_train_pred)

            print('\n--------------------------------Test Set------------------------------------------')
            print(classification_report(self.y_test, y_test_pred))
            #pd.DataFrame(self.result_dict['Evaluations'][result_name])
            #self.print_metrics(result_name)
            print('\n\n                          Test Confusion Matrix')
            self.plot_conf_mat(self.y_test, y_test_pred)

