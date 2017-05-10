#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os


def mkdir(path):
    """Make directory if the directory does not exist.
    Args:
        path: path to directory
    Returns:
        path: path to directory
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    return path
