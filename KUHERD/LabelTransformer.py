
import numpy as np

class LabelTransformer:

    def __init__(self, label_set_name, labels):

        self.label_set_name = label_set_name

        if labels is None:
            labels = LabelTransformer.default_labels(label_set_name)

        # remove duplicate labels and sort them alphabetically
        labels = list(set(labels))
        labels.sort()

        # form a dictionary of labels (and the inverse mapping)
        self.label_map = {}
        self.index_map = {}
        for idx, label in enumerate(labels):
            self.label_map[label] = idx
            self.index_map[idx] = label
        return

    def vec2string(self, label_vec):
        """Converts vector containing integers to a string representation using the label set dictionaries.
            Args:
                label_vec (list): A vector containing integer values mapping to members of the label_type.

            Return:
                (list): A list of strings that are members of the label_type.
            """

        converted_labels = [self.index_map[x] for x in label_vec]
        return converted_labels


    def label2mat(self, x):
        """ Converts a vector of strings to a matrix of of zero-one valued columns.
    
        Args:
            x (list): A vector containing string values representing the labels.
    
        Return:
            (numpy matrix): A matrix of zero-one valued columns.
    
        """

        n = len(x)
        p = len(self.label_map)

        M = np.zeros([n, p])
        for i, val in enumerate(x):
            val = val.lower().strip()
            try:
                label_value = self.label_map[val]
            except KeyError:
                raise ValueError('Unknown label given: %s \n' % val)

            M[i, label_value] = 1

        return M


    def vec2mat(self, x):
        """ Converts a vector of integers to a matrix of of zero-one valued columns.
        
        Args:
            x (list): A vector containing integer values mapping to members of the label_type.

        Return:
            (numpy matrix): A matrix of zero-one valued columns.

        """

        # verify that the vector is in the valid range

        n = len(x)
        p = len(self.label_map)
        max_value = max(x)
        if max_value > p:
            raise ValueError('More labels given in x than in this LabelSet!\n')


        M = np.zeros([n, p])
        for i, val in enumerate(x):
            M[i, int(val)] = 1

        return M

    def mat2vec(self, M):
        """ Converts a zero-one valued label matrix to an integer valued label vector.

        Args:
            M (numpy mat): A zero-one valued label matrix.
        """

        (n, p) = M.shape
        (r, c) = M.nonzero()
        if len(r) > n:
            raise ValueError('convert convert to label vector, more than one label per instance!')

        y = np.zeros([n])
        for row, col in zip(r, c):
            y[row] = int(col)
        return y


    def default_labels(target_set):

        if target_set == 'field':
            return {'a1': 1, 'a2': 2, 'a3': 3, 'a4': 4, 'a5': 5, 'a6': 6, 'a7': 7, 'a8': 8, 'a9': 9, 'b1': 10, 'b2': 11, 'b3': 12,
         'b4': 13, 'b5': 14, 'c1': 15, 'c2': 16, 'c3': 17, 'c4': 18, 'd1': 19, 'd2': 20,  'e': 21, 'f1': 22,
         'f2': 23, 'f3': 24, 'f4': 25, 'f5': 26, 'g': 27, 'h1': 28, 'h2': 29, 'h3': 30, 'h4': 31, 'h5': 32, 'i': 33,
         'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41}

        elif target_set == 'purpose':
            return {'applied': 1, 'basic': 2, 'development': 3, 'other': 4, 'researchtr': 5, 'service': 6, 'training': 7}

        else:
            raise ValueError('Unknown target_set given: %s \n' % target_set)