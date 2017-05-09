#    Copyright (C) 2017  Joseph St.Amand

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
import numpy as np

class MultiFeatureSelector:

    def __init__(self, scoring_function, kbest, multi_integrator):
        """ Feature selection with regard to multiple labels values.
         
         This class is made to perform feature selection with regard to multiple label values. Typically, feature selection
         in sklearn is done with respect to a single 0-1 valued label vector. This class runs multiple feature selectors,
         then integrates the selected features using a specified multi_integrator function.
         
         Args:
             scoring_function: Valid scoring functions are 'f_classif', 'chi2', 'mutual_info_classif'
             kbest (int): The total number of selected features.
             multi_integrator: Function to combine multiple sets of selected features. valid options are 'mean', 'max'.       
        """

        self.valid_scoring_funcs = ['f_classif', 'chi2', 'mutual_info_classif']
        self.kbest = kbest
        self.multi_integrator = None
        if multi_integrator == 'mean':
            self.multi_integrator = np.mean
        elif multi_integrator == 'max':
            self.multi_integrator = np.max
        else:
            print('Invalid multi_integrator function \"%s\" given' % scoring_function)
            exit()

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

        self.selected_ind = None


    def fit(self, X, Y, label_set):
        """ Trains the feature selection process
        
        Args:
            X (numpy matrix): Training samples.
            Y (numpy matrix): Training Labels.
            label_set (str): Denotes if label set is of the 'purpose' or 'field' type.
        """
        # score all feature in regards to each label
        selected_features = []
        feature_indices = []
        (rows, cols) = Y.shape
        for i in range(0, cols):
            selector = SelectKBest(self.scoring_func, k='all')
            selector.fit(X, Y[:, i])
            selected_features.append(list(selector.scores_))
            feature_indices.append(selector.get_support(indices=True))

        # feature selection as mean across all datasets
        selected_features = self.multi_integrator(selected_features, axis=0)
        self.selected_ind = np.argsort(selected_features)
        self.selected_ind = self.selected_ind[-self.kbest:]
        return


    def transform(self, X):
        """ Tranforms the data by selecting the features learned in the training or "fit" process.
        
        Args:
            X (numpy matrix): Data samples to run feature selection on.
        
        Return:
            (numpy matrix): Data with only the selected features.
        """
        X = X[:, self.selected_ind]
        return X
