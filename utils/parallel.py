#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Parallel computing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import multiprocessing as mp


def make_parallel(func, args, core=mp.cpu_count()):
    """
    Args:
        func (function):
        args (tuple): tuples of arguments for func
    Returns:
        result_tuple (tuple): tuple of returns
    """
    p = mp.Pool(core - 1)
    result_tuple = p.map(func, args)
    return result_tuple
