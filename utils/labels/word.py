#! /usr/bin/env python
# -*- coding: utf-8 -*-


def word2num(word_list, map_file_path):
    """Convert from word to number.
    Args:
        word_list: list of words (string)
        map_file_path: path to the mapping file
    Returns:
        word_list: list of word indices (int)
    """
    # read the mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split('  ')
            map_dict[str(line[0])] = int(line[1])

    # convert from word to number
    for i in range(len(word_list)):
        if word_list[i] in map_dict.keys():
            word_list[i] = map_dict[word_list[i]]
        else:
            # Pad by UNK
            word_list[i] = len(map_dict.keys())

    return word_list
