import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

class FeatureSelector:

    def __init__(self, scoring_function, kbest):
        """ Initializes the class with the scoring function type and the target number of features to be selected.
        
        Args:
            scoring_function (str): Scoring for feature selection mechanism. Valid entries are 'f_classif' and 'chi2'.
            kbest (int): Number of features to select.
        """

        self.valid_scoring_funcs = ['f_classif', 'chi2', 'mutual_info_classif']
        self.kbest = kbest

        self.scoring_func = None
        if scoring_function == 'f_classif':
            self.scoring_func = f_classif
        elif scoring_function == 'chi2':
            self.scoring_func = chi2
        elif scoring_function == 'mutual_info_classif':
            self.scoring_func = mutual_info_classif
        else:
            print('Invalid scoring function \"%s\" given' % scoring_function)
            exit()

        self.selector = SelectKBest(self.scoring_func, kbest)
        self.selected_ind = None


    def fit(self, X, Y, label_set):
        """ Fits the data by training the feature selection model compomnent.
        
        Args:
            X (numpy matrix): The data matrix.
            y (integer list): The labels for the data.
            label_set (str): Either 'purpose' or 'field'.

        Return:
            None
        """
        selected_features = []
        feature_indices = []
        for label in label_set:
            selector = SelectKBest(self.scoring_func, k='all')
            selector.fit(X, Y[label])
            selected_features.append(list(selector.scores_))
            feature_indices.append(selector.get_support(indices=True))

        selected_features = np.mean(selected_features, axis=0)


    def transform(self, X):
        """ Transforms the data, retaining only features learned in the "fit" process.
        
        Args:
            X (numpy matrix): The data matrix.
            y (integer list): The labels for the data.
        
        Return:
            (numpy matrix): Transformed data matrix.
        """
        X = self.selector.transform(X)
        return X
