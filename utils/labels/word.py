#! /usr/bin/env python
# -*- coding: utf-8 -*-


def word2idx(word_list, map_file_path):
    """Convert from word to index.
    Args:
        word_list (list): words (string)
        map_file_path (string): path to the mapping file
    Returns:
        word_list (lisr): word indices
    """
    # read the mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split('  ')
            map_dict[str(line[0])] = int(line[1])

    # convert from word to index
    for i in range(len(word_list)):
        if word_list[i] in map_dict.keys():
            word_list[i] = map_dict[word_list[i]]
        else:
            # Pad by UNK
            word_list[i] = len(map_dict.keys())

    return word_list
