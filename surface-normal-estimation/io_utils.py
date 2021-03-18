#!/usr/bin/env python3

import pickle
import gzip
import os


def _get_open_lambda(filename, is_save):
    '''
    with file extension pklz, we compress the pickle using gzip.
    '''
    extension = os.path.splitext(filename)[1]
    compress = extension == '.pklz'
    if compress:
        open_mode = 'w' if is_save else 'r'
        # note: I tried using bz2 instead of gzip. It compresses better
        # but it is slower.

        def open_lambda(x):
            return gzip.GzipFile(x, open_mode)

    else:
        open_mode = 'wb' if is_save else 'rb'

        def open_lambda(x):
            return open(x, open_mode)

    return open_lambda


def load_pickle(filename):
    '''
    load a pickle file
    '''
    open_lambda = _get_open_lambda(filename, is_save=False)
    with open_lambda(filename) as file:
        data = pickle.load(file)
    return data


def save_pickle(filename, data):
    '''
    save a pickle file.
    Note that pickle is incompatible across Python versions 2.7.x and 3.x
    with file extension pklz, we compress the pickle using gzip
    '''
    open_lambda = _get_open_lambda(filename, is_save=True)
    with open_lambda(filename) as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
