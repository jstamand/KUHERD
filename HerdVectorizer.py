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
from KUHERD.FeatureSelector import FeatureSelector
from KUHERD.MultiFeatureSelector import MultiFeatureSelector
from gensim import utils


class HerdVectorizer:
    """ Main class responsible for vectorization of the text data. This class is extremely configurable, with many
        options for each preprocessing behavior, bigrams, stemmers, stopwords, and feature selection. Use of this
        class is done in the following manner:
        - Set configuration options for preproc_config, bigram_config, stemmer, stopwords, and feature selection.
        - Train the vectorizer on a set of documents and their corresponding labels.
        - After training is complete, the documents may be given to the transform function to convert to TFIDF form.
    """

    def __init__(self, config):
        """ Initilize class with specific configuration options."""

        # configuration for learning
        self.preproc_config = config['preproc_config']
        self.bigram_config = config['bigram_config']
        self.feature_selector_config = config['feature_selector']

        # parameters learned in training
        self.tok_index_map = {}
        self.tfidf_config = {}
        self.bigram_index_map = {}

        # stopwords always the same
        self.stopwords = nltk.corpus.stopwords.words('english')

        # feature selection mechanism
        if self.feature_selector_config != None:
            self.feature_selector = MultiFeatureSelector(self.feature_selector_config['fselect'],
                                                         self.feature_selector_config['kbest'],
                                                         self.feature_selector_config['multi_integrator'])
        else:
            self.feature_selector = None


    def getConfig(self):
        """ Retrieve the complete configuration needed to build a vectorizer.

        Returns the complete configuration needed to build a vectorizer. Note that the config returned may only be used
        to train a new vectorizer. The config does NOT give model persistance.
        """
        config = {}
        config['preproc_config'] = self.preproc_config
        config['bigram_config'] = self.bigram_config
        config['trigram_config'] = self.trigram_config
        config['feature_selector'] = self.feature_selector
        return config


    def set_feature_selector(self, scoring_func, kbest, multi_type):
        """Set the feature selection configuration values. """
        if multi_type != None:
            self.feature_selector = MultiFeatureSelector(scoring_func, kbest, multi_type)
        else:
            self.feature_selector = FeatureSelector(scoring_func, kbest)


    def set_stemmer(self, the_stemmer):
        """Set the stemmer configuration values."""
        valid_stemmers = ['none', 'porter', 'lancaster', 'snowball']
        if the_stemmer in valid_stemmers:
            self.preproc_config['stemmer'] = the_stemmer
        elif the_stemmer is None:
            self.preproc_config['stemmer'] = None
        else:
            print('Invalid Stemmer Given: %s' % the_stemmer)
            exit()


    def set_preproc_config(self, name, value):
        """Set the preprocessor configuration value."""
        if name in self.preproc_config.keys():
            self.preproc_config[name] = value
        return


    def set_bigrams(self, bigrams, bigram_window_size, bigram_filter_size, bigram_nbest):
        """Set the bigram configuration."""
        self.bigram_config['bigrams'] = bigrams
        self.bigram_config['bigram_window_size'] = bigram_window_size
        self.bigram_config['bigram_filter_size'] = bigram_filter_size
        self.bigram_config['bigram_nbest'] = bigram_nbest
        return


    def set_bigram_config(self, name, value):
        """Set the bigram configuration."""
        if name in self.bigram_config.keys():
            self.preproc_config[name] = value
        return


    def get_preproc_config(self):
        """Retrieve the preprocessor configuration."""
        return self.preproc_config


    def get_bigram_config(self):
        """Retrieve the bigram configuration."""
        return self.bigram_config


    def train(self, docs, y, label_set):
        """ Takes a list of documents, and the corresponding labels and trains the preprocessor(including feature selection).

        Args:
            docs (list): A list of documents, where each document is represented as a string.
            y (list): A list of integers representing the label for each document.
            label_set (str): Specifies the label set so that the input 'y' may be interpreted. Valiud entries are either 'purpose' or 'field'.

        @param docs The list of documents
        @param y A vector of labels which correspond to each document
        """

        tokenized_docs = self.tokenize_docs(docs)

        self.tok_index_map = self.create_token_index_map(tokenized_docs)

        # now filter the documents on the new token-index map
        tokenized_docs = self.filter_docs(tokenized_docs, self.tok_index_map)

        csr_mat = self.form_count_matrix(tokenized_docs).tocsr()

        if self.bigram_config['bigrams'] == True:
            self.bigram_index_map = self.create_bigram_index_map(tokenized_docs)
            bigram_count_csr_mat = self.form_bigram_count_matrix(tokenized_docs)
            csr_mat = scipy.sparse.hstack([csr_mat, bigram_count_csr_mat])

        # now we find the TF-IDF representation
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.fit(csr_mat)
        X = tfidf_transformer.transform(csr_mat)
        if self.feature_selector != None:
            self.feature_selector.fit(X, y, label_set)

        self.tfidf_config['transformer'] = tfidf_transformer

        return


    def transform_data(self, docs):
        """ Tranforms documents into a sparse matrix. 

        Must be called after the preprocessor has been trained on some data. Process is as follows:
            -tokenize documents
            -search for bigrams
            -transform to TFIDF representation
            -select features

        Args:
            docs (list): A list of documents, each document is represented as a string.

        Return:
            (sparse numpy matrix): A sparse CSR formatted matrix, each row corresponds to a document, ordering of documents is preserved.
        """

        # tokenize and vectorize using bag of words representation
        tokenized_docs = self.tokenize_docs(docs)

        # extract individual words
        filtered_docs = self.filter_docs(tokenized_docs, self.tok_index_map)
        csr_mat = self.form_count_matrix(filtered_docs).tocsr()

        # search for bigrams
        if self.bigram_config['bigrams'] == True:
            bigram_count_csr_mat = self.form_bigram_count_matrix(tokenized_docs)
            csr_mat = scipy.sparse.hstack([csr_mat, bigram_count_csr_mat])

        # now transform into TF-IDF representation
        tfidf_transformer = self.tfidf_config['transformer']
        X = tfidf_transformer.transform(csr_mat)
        if self.feature_selector != None:
            X = self.feature_selector.transform(X)
        return X

    @staticmethod
    def filter_docs(tokenized_docs, tok_index_map):
        """Filters tokenized documents, removing all tokens which are not recognized by the specified token index map.

        Args:
            tokenized_docs (list): A list of documents, each document is represented as a single long string
            tok_index_map (dictionary): A mapping from tokens to their index in the feature matrix.

        Return:
            (list): A list of documents, where each document is a list of (filtered) tokens.
        """
        filtered_docs = []
        for doc in tokenized_docs:
            new_doc = []
            for token in doc:
                if token in tok_index_map:
                    new_doc.append(token)
            filtered_docs.append(new_doc)
        return filtered_docs

    def create_bigram_index_map(self, tokenized_docs):
        """ Creates a mapping from each bigram to a column index.

        Args:
            tokenized_docs (list): A list of documents, each document is represented as a single long string

        Return:
            (dictionary): Dictionary where keys are tokens, values are the index into a feature matrix.
        """

        flattened_tokens = [item for sublist in tokenized_docs for item in sublist]

        finder = BigramCollocationFinder.from_words(flattened_tokens)
        finder.apply_freq_filter(self.bigram_config['bigram_filter_size'])  # key to getting good results
        if self.bigram_config['bigram_measure'] == 'pmi':
            bigram_measure = BigramAssocMeasures.pmi
        elif self.bigram_config['bigram_measure'] == 'chi_sq':
            bigram_measure = BigramAssocMeasures.chi_sq
        elif self.bigram_config['bigram_measure'] == 'jaccard':
            bigram_measure = BigramAssocMeasures.jaccard
        elif self.bigram_config['bigram_measure'] == 'likelihood_ratio':
            bigram_measure = BigramAssocMeasures.likelihood_ratio
        elif self.bigram_config['bigram_measure'] == 'mi_like':
            bigram_measure = BigramAssocMeasures.mi_like
        elif self.bigram_config['bigram_measure'] == 'poisson_stirling':
            bigram_measure = BigramAssocMeasures.poisson_stirling
        elif self.bigram_config['bigram_measure'] == 'student_t':
            bigram_measure = BigramAssocMeasures.student_t
        elif self.bigram_config['bigram_measure'] == 'raw_freq':
            bigram_measure = BigramAssocMeasures.raw_freq
        elif self.bigram_config['bigram_measure'] == 'phi_sq':
            bigram_measure = BigramAssocMeasures.phi_sq
        elif self.bigram_config['bigram_measure'] == 'dice':
            bigram_measure = BigramAssocMeasures.dice
        else:
            exit('Invalid bigram_measure given!')

        nbest_bigrams = finder.nbest(bigram_measure, self.bigram_config['bigram_nbest'])

        bigram_index = {}
        i = 0
        for bigram in nbest_bigrams:
            bigram_index[bigram] = i
            i += 1

        return bigram_index

    def form_bigram_count_matrix(self, tokenized_docs):
        """Calculates a bigram count matrix from a list of tokenized documents.

        Args:
            tokenized_docs (list): A list of documents, each document is represented as a single long string

        Return:
            (sparse numpy matrix): A sparse count matrix in COO format.
        """
        data = []
        doc_index = []
        bigram_index = []

        doc_num = 0
        for doc in tokenized_docs:
            finder = BigramCollocationFinder.from_words(doc)
            for bigram in list(finder.ngram_fd.keys()):
                if bigram in self.bigram_index_map:
                    data.append(1)
                    doc_index.append(doc_num)
                    bigram_index.append(self.bigram_index_map[bigram])
            doc_num += 1

        return scipy.sparse.coo_matrix((data, (doc_index, bigram_index)),
                                       shape=(len(tokenized_docs), len(self.bigram_index_map)),
                                       dtype=numpy.float16).tocsr()

    def create_token_index_map(self, tokenized_docs):
        """Given the tokenized documents, finds all unique tokens and forms an index map

        Args:
            tokenized_docs (list): A list of documents, each document is represented as a single long string

        Return:
            (dictionary): Dictionary where keys are tokens, values are the index into a feature matrix.
        """

        df_count = {}
        for doc in tokenized_docs:
            unique_doc_tokens = set(doc)
            for tok in unique_doc_tokens:
                if tok in df_count:
                    df_count[tok] += 1
                else:
                    df_count[tok] = 1

        # remove the words not in the document freq range
        for key in list(df_count.keys()):
            if df_count[key] < self.preproc_config['token_min_df'] or df_count[key] > self.preproc_config[
                'token_max_df']:
                del (df_count[key])

        # flatten the list of list of tokenized docs
        flattened_tokens = [item for sublist in tokenized_docs for item in sublist]

        # form a complete list of unique words and create mapping
        index_keys = list(df_count.keys())
        index_map = {}
        index = 0
        for key in index_keys:
            index_map[key] = index
            index += 1

        return index_map

    def tokenize_docs(self, docs):
        """ Breaks each document down into a list of words(tokens).

        Converts a list of documents(each document is given as a single string) and converts them to their tokenized
        form in the following manner(some steps may be skipped if configured as such in the configuration settings)
         - break document into tokens
         - remove punctuation
         - stem tokens

        Args:
            docs (list): A list of documents, each document is represented as a single long string

        Return:
            (list): The tokenized documents as a list of lists, each item of the outer list is a document, which is represented as a list of words.
        """
        tokenized_docs = []
        i = 0
        for d in docs:
            try:
                d = str(d).lower()
            except AttributeError:
                print(d)
                continue
            for punct in self.preproc_config['puncts']:
                d = d.replace(punct, ' ')
            d = wordpunct_tokenize(d)
            tokens = [utils.to_unicode(str(token)) for token in d if token not in self.stopwords and len(token) > 2]
            tokenized_docs.append(tokens)
            i += 1

        # stem the docs if enabled
        if self.preproc_config['stemmer'] == 'porter':
            tokenized_docs = self.porter_stemmer(tokenized_docs)
        elif self.preproc_config['stemmer'] == 'lancaster':
            tokenized_docs = self.lancaster_stemmer(tokenized_docs)
        elif self.preproc_config['stemmer'] == 'snowball':
            tokenized_docs = self.snowball_stemmer(tokenized_docs)
        elif self.preproc_config['stemmer'] == 'none':
            pass
        elif self.preproc_config['stemmer'] == None:
            pass
        else:
            raise ValueError('unrecognized stemmer specified: %s' % self.preproc_config['stemmer'])

        return tokenized_docs

    def form_count_matrix(self, tokenized_docs):
        """Forms the count matrix from the tokenized documents

        Args:
            tokenized_docs (list): A list of lists representing the tokenized documents. Each document is a list of tokens.

        Return:
            (sparse numpy matrix): A sparse count matrix in COO format.
        """

        coo_doc_index = []
        coo_token_index = []
        coo_data = []
        doc_index = 0
        for doc in tokenized_docs:
            for token in doc:
                try:
                    token_index = self.tok_index_map[token]
                    coo_doc_index.append(doc_index)
                    coo_token_index.append(token_index)
                    coo_data.append(1)
                except ValueError:
                    continue
                except TypeError:
                    continue
            doc_index += 1

        token_coo_mat = scipy.sparse.coo_matrix((coo_data, (coo_doc_index, coo_token_index)),
                                                shape=(len(tokenized_docs), len(self.tok_index_map)),
                                                dtype=numpy.float16)
        return token_coo_mat

    @staticmethod
    def lancaster_stemmer(docs):
        """Lancaster stemming algorithm"""
        stemmer = LancasterStemmer()
        docs = [[stemmer.stem(word) for word in abstract] for abstract in docs]
        return docs

    @staticmethod
    def snowball_stemmer(docs):
        """Snowball stemming algorithm"""
        stemmer = SnowballStemmer('english')
        docs = [[stemmer.stem(word) for word in abstract] for abstract in docs]
        return docs

    @staticmethod
    def porter_stemmer(docs):
        """Porter stemming algorithm"""
        stemmer = PorterStemmer()
        docs = [[stemmer.stem(word) for word in abstract] for abstract in docs]
        return docs


def main():
    herd = HerdVectorizer()

    filename = '/users/j817s517/PycharmProjects/HERD/Eager_HERDAttribs_wDataElements_032616.xlsx'
    # filename = '/users/j817s517/PycharmProjects/HERD/test.xlsx'
    dataframe = pandas.read_excel(filename)
    abstracts = dataframe['SOW']

    herd.train(abstracts)
    herd.save_configuration('test.hdf5')

    tfidf1 = herd.transform_data(abstracts)

    herd2 = HerdVectorizer()
    herd2.load_configuration('test.hdf5')
    tfidf2 = herd2.transform_data(abstracts)


if __name__ == "__main__":
    main()
