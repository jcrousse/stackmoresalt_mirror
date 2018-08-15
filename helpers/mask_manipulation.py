import numpy as np
from functools import reduce
from math import ceil, sqrt

def load_csv_mask():
    pass

def mask_to_binary(rle_mask, dim=(101,101), index_zero=False, col_encoding=True):
    """
    :param binary_mask: run-length encoded mask
    :param dim: target: mask matrix dimension
    :param index_zero: flag whether encoding starts at index 0 (True) or 1 (False)
    :param col_encoding: flag on whether to the RLE is by column (True) or by row
    :return: mask matrix as numpy array
    """

    # split input in two: starts points and lengths:
    encoded_starts, encoded_lengths = rle_mask[::2], rle_mask[1::2]
    encoded_ends = [sum(x) for x in zip(encoded_starts, encoded_lengths)]

    # if dim is none: dim = ceil(sqrt(max))
    if not dim:
        min_dim = ceil(sqrt(max(encoded_ends)))
        dim = (min_dim, min_dim)

    # if single value given as dim, assume it is dimension of square output
    if type(dim) is int:
        dim = (dim,dim)
    try:
        if len(dim)==1:
            dim=(dim[0], dim[0])
    except:
        pass

    # out = np.array zeroes, dim ()
    n_cells = reduce((lambda x, y: x * y), dim)
    decoded_array = np.zeros(n_cells)



    if not index_zero:
        encoded_starts = [x-1 for x in encoded_starts]
        encoded_ends = [x-1 for x in encoded_ends]

    for lower_b, upper_b in zip(encoded_starts,encoded_ends):
        decoded_array[lower_b:upper_b] = 1

    decoded_matrix = np.reshape(decoded_array, dim)

    if col_encoding:
        decoded_matrix = np.transpose(decoded_matrix)

    return decoded_matrix

def binary_to_mask(binary_mask, index_zero=False, col_encoding=True):
    array_mask = np.array(binary_mask)

    if col_encoding:
        array_mask = np.transpose(array_mask)

    index_shift = 1
    if not index_zero:
        index_shift = 2

    flat_mask = array_mask.flatten()

    rle = np.where(flat_mask[1:] != flat_mask[:-1])[0] + index_shift
    if flat_mask[0]:
        rle = np.append([1], rle)
    if flat_mask[-1]:
        rle=np.append(rle, [len(flat_mask)+1])
    rle[1::2] = rle[1::2] - rle[::2]

    return rle

