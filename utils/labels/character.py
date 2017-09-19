#! /usr/bin/env python
# -*- coding: utf-8 -*-


def char2index(str_char, map_file_path, double_letter=False):
    """Convert from character to index.
    Args:
        str_char (string): string of characters
        map_file_path (string): path to the mapping file
        double_letter (bool, optional): if True, group repeated letters
    Returns:
        char_list (list): character indices
    """
    char_list = list(str_char)

    # Lead the mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[line[0]] = int(line[1])

    # Convert from character to index
    if double_letter:
        skip_flag = False
        for i in range(len(char_list) - 1):
            if skip_flag:
                char_list[i] = ''
                skip_flag = False
                continue

            if char_list[i] + char_list[i + 1] in map_dict.keys():
                char_list[i] = map_dict[char_list[i] + char_list[i + 1]]
                skip_flag = True
            else:
                char_list[i] = map_dict[char_list[i]]

        # Final character
        if skip_flag:
            char_list[-1] = ''
        else:
            char_list[-1] = map_dict[char_list[-1]]

        # Remove skipped characters
        while '' in char_list:
            char_list.remove('')
    else:
        for i in range(len(char_list)):
            char_list[i] = map_dict[char_list[i]]
    return char_list


def kana2index(str_char, map_file_path):
    """Convert from kana character to index.
    Args:
        str_char (string): string of kana characters
        map_file_path (string): path to the mapping file
    Returns:
        index_list (list): kana character indices
    """
    kana_list = list(str_char)
    index_list = []

    # Lead the mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[line[0]] = int(line[1])

    i = 0
    while i < len(kana_list):
        # Check whether next kana character is a double consonant
        if i != len(kana_list) - 1:
            if kana_list[i] + kana_list[i + 1] in map_dict.keys():
                index_list.append(int(map_dict[kana_list[i] + kana_list[i + 1]]))
                i += 1
            elif kana_list[i] in map_dict.keys():
                index_list.append(int(map_dict[kana_list[i]]))
            else:
                raise ValueError(
                    'There are no kana character such as %s' % kana_list[i])
        else:
            if kana_list[i] in map_dict.keys():
                index_list.append(int(map_dict[kana_list[i]]))
            else:
                raise ValueError(
                    'There are no kana character such as %s' % kana_list[i])
        i += 1

    return index_list


def index2char(index_list, map_file_path):
    """Convert from index to character.
    Args:
        index_list (list): list of character indices
        map_file_path (string): path to the mapping file
    Returns:
        str_char (string): string of characters
    """
    # Lead the mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[int(line[1])] = line[0]

    # Convert from indices to the corresponding characters
    char_list = []
    for i in range(len(index_list)):
        char_list.append(map_dict[index_list[i]])

    str_char = ''.join(char_list)
    return str_char
