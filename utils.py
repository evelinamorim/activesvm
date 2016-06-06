import scipy
from scipy import sparse
import numpy as np
from bisect import bisect_left
import array
from nltk.corpus import verbnet as vn
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

def make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))

def create_matrix_csr(m, dim):

    j_indices = make_int_array()
    idx = make_int_array()

    idx.append(0)
    for doc in m:
        j_indices = j_indices + array.array("i", doc)
        idx.append(len(j_indices))

    values = np.ones(len(j_indices))

    X = sparse.csr_matrix((values, j_indices, idx), shape = (len(idx) - 1, dim))
    X.sum_duplicates()    
    return X


def first_level_class(classes_list):
    """
    get only the first level classes from a list of levin class verb
    """
    first_classes = []
    for c in classes_list:
        first_classes.append(c.split('-')[0])

    return set(first_classes)


# print(binary_search([2,4,7,8,10], 1))
# print(binary_search([2,4,7,8,10], 8))

# x = create_row_csr([22,41,101,52,33,93,512], 600)
# y = create_matrix_csr([[522,412,11,5,3,97,51],[22,41,101,52,33,93,93, 512]], 600)
# print(y)
