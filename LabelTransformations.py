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


# Contains a listing of valid labels and utilities which convert between label strings, a vector and a matrix


import KUHERD.LabelSets as label_sets
import numpy as np


def vec2string(label_vec, label_type):
    """Converts vector containing integers to a string representation using the label set dictionaries.
    
    Args:
        label_vec (list): A vector containing integer values mapping to members of the label_type.
        label_type (str): Specifies label_type of label_vec, either 'purpose' and 'field'.
    
    Return:
        (list): A list of strings that are members oif the label_type.
    """

    mapping = None
    if label_type == 'field':
        mapping = {v:k for k, v in label_sets.field.items()}
    elif label_type == 'purpose':
        mapping = {v:k for k, v in label_sets.purpose.items()}
    else:
        raise ValueError('Only valid options are field or purpose')


    converted_labels = [mapping[x] for x in label_vec]
    return converted_labels


def label2mat(x, label_type):
    """ Converts a vector of integers to a matrix of of zero-one valued columns.
    
    Args:
        label_vec (list): A vector containing integer values mapping to members of the label_type.
        label_type (str): Specifies label_type of label_vec, either 'purpose' and 'field'.

    Return:
        (numpy matrix): A matrix of zero-one valued columns.
    
    """

    n = len(x)
    label_set = None
    if label_type == 'field':
        p = len(label_sets.field)
        label_set = label_sets.field
    elif label_type =='purpose':
        p = len(label_sets.purpose)
        label_set = label_sets.purpose
    else:
        raise ValueError('Only valid options are field or purpose')

    M = np.zeros([n, p])
    for i, val in enumerate(x):
        val = val.lower().strip()
        label_value = label_set[val]-1
        M[i, label_value] = 1

    return M


def vec2mat(x, label_type):
    """ Converts a vector of integers to a matrix of of zero-one valued columns.

    Args:
        x (list): A vector containing integer values mapping to members of the label_type.
        label_type (str): Specifies label_type of label_vec, either 'purpose' and 'field'.

    Return:
        (numpy matrix): A matrix of zero-one valued columns.

    """


    n = len(x)
    label_set = None
    if label_type == 'field':
        p = len(label_sets.field)
        label_set = label_sets.field
    elif label_type =='purpose':
        p = len(label_sets.purpose)
        label_set = label_sets.purpose
    else:
        raise ValueError('Only valid options are field or purpose')

    M = np.zeros([n, p])
    for i, val in enumerate(x):
        M[i, int(val)-1] = 1

    return M


def mat2vec(M):
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
        y[row] = int(col+1)
    return y


