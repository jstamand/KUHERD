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

from collections import namedtuple
import nltk
from nltk import BigramAssocMeasures, TrigramAssocMeasures
from nltk.collocations import *
from nltk.stem import *
from nltk.tokenize import wordpunct_tokenize
import numpy
import scipy
from sklearn.feature_extraction.text import TfidfTransformer
import pandas
import pickle
import numpy as np
import sys
from KUHERD.HerdVectorizer import HerdVectorizer
from KUHERD.LabelTransformer import LabelTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier


class ClassificationModel:


    def __init__(self, config):
        """This class is a wrapper for a model which targets either the Purpose or Field category set.
        """
        self.target_set = config['target_set']
        self.model_config = config['model_config']
        self.model_name = config['model']
        self.model = None

        # look for a non-default label set
        self.labelset = None

        if 'labels' in config.keys():
            self.labelset = LabelTransformer(self.target_set, config['labels'])
        else:
            self.labelset = LabelTransformer(self.target_set, LabelTransformer.default_labels(self.target_set))

        if self.target_set != 'purpose' and self.target_set != 'field':
            raise ValueError('Unknown target_set configuration value: %s \n' % config['target_set'])


        # sort out which model the configuration contains
        if self.model_name == 'LogisticRegression':
            self.model = LogisticRegression()
        elif self.model_name == 'DecisionTree':
            self.model = DecisionTreeClassifier()
        elif self.model_name == 'SVC':
            self.model = SVC()
        elif self.model_name == 'Naivebayes':
            self.model = naive_bayes()
        elif self.model_name == 'RandomForest':
            self.model = RandomForestClassifier()
        else:
            sys.exit('Invalid config, model given is unknown: %s' % self.model)

        self.model.set_params(**self.model_config)

        return


    def fit(self, X, Y):
        """ Trains the model.

        Fitting or "training" must be done before the model is able to make predictions.

        Args:
            X (numpy matrix): Training samples.
            Y (numpy matrix): Training labels.

        Returns:
            None: No return value.
        """

        self.model.fit(X, Y)
        return

    def predict(self, X):
        """ Make predictions.

        Args:
            X (numpy matrix): Training samples.

        Returns:
            numpy matrix: predicted label values.
        """

        Y = self.model.predict(X)
        return Y

    def get_config(self):
        """ Returns the configuration used to build this model.

        Returns:
            dict: dictionary containing target label set, internal model configuration, and model name.
        """

        config = dict()
        config['target_set'] = self.target_set
        config['model_config'] = self.model_config
        config['model_name'] = self.model_name
        config['labelset'] = self.labelset

        return config


class PurposeFieldModel:


    def __init__(self, config):
        """ Combines the purpose and field vectorizers and prediction models into a single "composite" model.

        Takes configurations for building the vectorizers and classification models. The 'purpose' and 'field' categories each get their own
        vectorizer and model. The vectorizer configs can be retrieved from HerdVectorizer class, the model configs may be retrieved from the
        ClassificationModel class.
    
        Args:
            config (dictionary): Dictionary containing the following configuration: 'purpose_vectorizer', 'field_vectorizer', 'purpose_model', 'field_model'.
    
        """
        self.purpose_vectorizer = HerdVectorizer(config['purpose_vectorizer'])
        self.field_vectorizer = HerdVectorizer(config['field_vectorizer'])

        self.purpose_model = ClassificationModel(config['purpose_model'])
        self.field_model = ClassificationModel(config['field_model'])

        return

    def fit(self, abstracts, Y_purpose, Y_field):
        """ Trains the model.

        Input arguments must all be the same length.

        Args:
            abstracts (list): A list of documents, each document is represented as a list of words.
            Y_purpose (list): A list of labels of the 'purpose' variety.
            Y_field (list): A list of labels of the 'field' variety.
        """
        self.purpose_vectorizer.train(abstracts, Y_purpose, 'purpose')
        self.field_vectorizer.train(abstracts, Y_field, 'field')

        X_purpose = self.purpose_vectorizer.transform_data(abstracts)
        X_field = self.field_vectorizer.transform_data(abstracts)

        Y_purpose = self.purpose_model.labelset.mat2vec(Y_purpose)
        Y_field = self.field_model.labelset.mat2vec(Y_field)

        self.purpose_model.fit(X_purpose, Y_purpose)
        self.field_model.fit(X_field, Y_field)

        return

    def predict(self, abstracts):
        """ Make predictions on the input data.

        The list of documents input is vectorized and input to the prediction model, which generates label predictions.
        This process is done separately for generating both purpose and field label predictions.

        Args:
            abstracts (list): A list of documents, each document is represented as a list of words.

        Returns:
            dictionary: dictionary containing two lists of predictions, dictionary keys are 'purpose' and 'field'.
        """
        X_purpose = self.purpose_vectorizer.transform_data(abstracts)
        X_field = self.field_vectorizer.transform_data(abstracts)

        predictions = dict()
        predictions['purpose'] = self.purpose_model.predict(X_purpose)
        predictions['field'] = self.field_model.predict(X_field)

        return predictions

    def get_config(self):
        """ Returns the configuration used to build this model.

        Returns:
            dict: dictionary containing the following keys, 'purpose_vectorizer', 'field_vectorizer', 'purpose_model', 'field_model'. Each entry is the configuration required to build the model.
        """
        config = dict()
        config['purpose_vectorizer'] = self.purpose_vectorizer.get_config()
        config['field_vectorizer'] = self.field_vectorizer.get_config()
        config['purpose_model'] = self.purpose_model.get_config()
        config['field_model'] = self.field_model.get_config()
        return config

