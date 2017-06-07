#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def map_phone2phone(phone_list, label_type, map_file_path):
    """Map from 61 phones to 39 or 48 phones.
    Args:
        phone_list: list of 61 phones (string)
        label_type: phone39 or phone48 or phone61
        map_file_path: path to the phone2phone mapping file
    Returns:
        phone_list: list of phones (string)
    """
    if label_type == 'phone61':
        return phone_list

    # read a mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[1] != 'nan':
                if label_type == 'phone48':
                    map_dict[line[0]] = line[1]
                elif label_type == 'phone39':
                    map_dict[line[0]] = line[2]
            else:
                map_dict[line[0]] = ''

    # mapping from 61 phones to 39 or 48 phones
    for i in range(len(phone_list)):
        if phone_list[i] in map_dict.keys():
            phone_list[i] = map_dict[phone_list[i]]

    # ignore "q"
    while '' in phone_list:
        phone_list.remove('')

    return phone_list
