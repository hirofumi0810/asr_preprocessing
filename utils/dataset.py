#! /usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd


def add_element(df, elem_list):
    series = pd.Series(elem_list, index=df.columns)
    df = df.append(series, ignore_index=True)
    return df
