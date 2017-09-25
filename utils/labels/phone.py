#! /usr/bin/env python
# -*- coding: utf-8 -*-


def phone2idx(phone_list, map_file_path):
    """Convert from phone to index.
    Args:
        phone_list (list): phones (string)
        map_file_path (string): path to the mapping file
    Returns:
        phone_list (list): phone indices
    """
    # read the mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split('  ')
            map_dict[str(line[0])] = int(line[1])

    # convert from phone to index
    for i in range(len(phone_list)):
        phone_list[i] = map_dict[phone_list[i]]

    return phone_list


def idx2phone(index_list, map_file_path):
    """Convert from index to phone.
    Args:
        index_list (list): phone indices
        map_file_path (string): path to the mapping file
    Returns:
        phone_list (list): phones (string)
    """
    # read the mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[int(line[1])] = line[0]

    # convert from indices to the corresponding phones
    phone_list = []
    for i in range(len(index_list)):
        phone_list.append(map_dict[index_list[i]])

    return phone_list
