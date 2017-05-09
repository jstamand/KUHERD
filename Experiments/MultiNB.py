import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..",".."))

import argparse
import numpy as np
import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as skmetrics
from sklearn import preprocessing
from KUHERD import LabelSets
import yaml

from KUHERD.HerdVectorizer import HerdVectorizer
from KUHERD.LabelTransformations import vec2mat, label2mat, mat2vec

def MultiNB():
    """ Program for running an experiment using Naive Bayes classifier."""
    parser = argparse.ArgumentParser(description='Simple Decision Tree Search')
    parser.add_argument('-o', dest='fout', type=str, help='output file name')
    parser.add_argument('-p', dest='parameters', type=str, help='parameters to exercise')
    args = parser.parse_args()

    # options for the preprocessor
    clf_parser = argparse.ArgumentParser(description='representation testing options')
    clf_parser.add_argument('-g', action='store_true')
    clf_parser.add_argument('-d', dest='dataset', type=str, help='dataset(xlsx or hdf5)')

    clf_parser.add_argument('--target_set', dest='target_set', type=str, default='purpose', help='target set for prediction, {purpose, field, both}')
    clf_parser.add_argument('--stemmer', dest='stemmer', type=str, default=None)
    clf_parser.add_argument('--tok_min_df', dest='token_min_df', type=int, default=10, help='Min. doc. freq. of tokens')
    clf_parser.add_argument('--tok_max_df', dest='token_max_df', type=int, default=200, help='Max. doc. freq. of tokens')
    clf_parser.add_argument('--bigrams', dest='bigrams', action='store_true')
    clf_parser.add_argument('--bigram_window_size', dest='bigram_window_size', type=int, default=2,
                            help='bigram window size')
    clf_parser.add_argument('--bigram_filter_size', dest='bigram_filter_size', type=int, default=5,
                            help='bigram filter size')
    clf_parser.add_argument('--bigram_nbest', dest='bigram_nbest', type=int, default=2000, help='bigram N best')
    clf_parser.add_argument('--trigrams', dest='trigrams', action='store_true')
    clf_parser.add_argument('--trigram_window_size', dest='trigram_window_size', type=int, default=2,
                            help='trigram window size')
    clf_parser.add_argument('--trigram_filter_size', dest='trigram_filter_size', type=int, default=10,
                            help='trigram filter size')
    clf_parser.add_argument('--trigram_nbest', dest='trigram_nbest', type=int, default=1000, help='trigram N best')
    clf_parser.add_argument('--fselect', dest='fselect', type=str, default='chi2', help='feature selection scoring function')
    clf_parser.add_argument('--kbest', dest='kbest', type=int, default=500, help='feature selection option: Kbest feature to keep')
    clf_parser.add_argument('--selectfunc', dest='selectfunc', type=str, default='mean', help='function to integrate selected features in multi-class scenario')

    clf_args = clf_parser.parse_args([x for x in args.parameters.strip().split(' ') if x != ''])

    # load data
    filename, file_ext = os.path.splitext(clf_args.dataset)
    if file_ext == '.hdf5':
        df = pandas.read_hdf(clf_args.dataset)
    elif file_ext == '.xlsx':
        df = pandas.read_excel(clf_args.dataset)

    # split into train/test sets
    df_train = df[df.cv_index != 0]
    df_test = df[df.cv_index == 0]

    # retrieve the label set
    label_set = []
    if clf_args.target_set == 'purpose':
        label_set = LabelSets.purpose
    elif clf_args.target_set == 'field':
        label_set = LabelSets.field
    elif clf_args.target_set == 'both':
        label_set = LabelSets.purpose + LabelSets.field
    else:
        print('target_set invalid!')
        exit()

    Y_train = list(df_train[clf_args.target_set])
    Y_test = list(df_test[clf_args.target_set])

    # prepare the vectorizer and vectorize the data
    myVec = HerdVectorizer()
    if clf_args.bigrams and clf_args.bigram_nbest != 0:
        myVec.set_bigrams(True, clf_args.bigram_window_size, clf_args.bigram_filter_size, clf_args.bigram_nbest)

    if clf_args.trigrams and clf_args.trigram_nbest != 0:
        myVec.set_trigrams(True, clf_args.trigram_window_size, clf_args.trigram_filter_size, clf_args.trigram_nbest)

    myVec.set_stemmer(clf_args.stemmer)

    if clf_args.fselect and clf_args.kbest and clf_args.selectfunc:
        myVec.set_feature_selector(clf_args.fselect, clf_args.kbest, clf_args.selectfunc)

    YY = label2mat(Y_train, clf_args.target_set)

    myVec.train(df_train['sow'], YY, label_set)
    X_train_validate = myVec.transform_data(df_train['sow'])

    # return an index as a single vector with values 1-5,
    # Need to convert this to an iterable object with train,test folds
    cv_folds = df_train['cv_index'].values
    train_folds = []
    test_folds = []
    for i in [1, 2, 3, 4, 5]:
        train_fold = [ind for ind, val in enumerate(cv_folds) if val != i]
        test_fold = [ind for ind, val in enumerate(cv_folds) if val == i]
        train_folds.append(train_fold)
        test_folds.append(test_fold)

    cv_folds = zip(train_folds, test_folds)

    # initialize nested dictionary to store CV metrics
    metrics = {}
    for label in label_set:
        metrics[label] = {}
        metrics[label]['accuracy'] = []
        metrics[label]['recall'] = []
        metrics[label]['precision'] = []
        metrics[label]['f1'] = []
        metrics[label]['mcc'] = []
    metrics['f1-micro'] = []
    metrics['f1-macro'] = []

    y = mat2vec(YY)

    for train_ind, test_ind in cv_folds:
        model = GaussianNB()
        model.fit(X_train_validate[train_ind, :].toarray(), y[train_ind])
        y_predict = model.predict(X_train_validate[test_ind].toarray())
        y_test = y[test_ind]

        y_multi_test = vec2mat(y_test, clf_args.target_set)
        y_multi_predict = vec2mat(y_predict, clf_args.target_set)

        for i, label in enumerate(label_set):
            metrics[label]['accuracy'].append(skmetrics.accuracy_score(y_multi_test[:, i], y_multi_predict[:, i]))
            metrics[label]['recall'].append(skmetrics.recall_score(y_multi_test[:, i], y_multi_predict[:, i]))
            metrics[label]['precision'].append(skmetrics.precision_score(y_multi_test[:, i], y_multi_predict[:, i]))
            metrics[label]['f1'].append(skmetrics.f1_score(y_multi_test[:, i], y_multi_predict[:, i]))
            metrics[label]['mcc'].append(skmetrics.matthews_corrcoef(y_multi_test[:, i], y_multi_predict[:, i]))
        metrics['f1-micro'].append(skmetrics.f1_score(y_multi_test, y_multi_predict, average='micro'))
        metrics['f1-macro'].append(skmetrics.f1_score(y_multi_test, y_multi_predict, average='macro'))

    for label in label_set:
        metrics[label]['accuracy'] = np.mean(metrics[label]['accuracy'])
        metrics[label]['recall'] = np.mean(metrics[label]['recall'])
        metrics[label]['precision'] = np.mean(metrics[label]['precision'])
        metrics[label]['f1'] = np.mean(metrics[label]['f1'])
        metrics[label]['mcc'] = np.mean(metrics[label]['mcc'])

    metrics['f1-micro'] = np.mean(metrics['f1-micro'])
    metrics['f1-macro'] = np.mean(metrics['f1-macro'])

    # need to combine all stats and scores and save as a yaml file
    results = {}
    results['classifier'] = 'Naive_Bayes'
    results['metrics'] = metrics  # metrics as calculated by this script
    results['clf_parameters'] = vars(clf_args)  # parameters passed into this script
    print(results)

    with open(args.fout, 'w') as fout:
        yaml.dump(results, fout, explicit_start=True, default_flow_style=False)

if __name__ == "__main__":
    MultiNB()
