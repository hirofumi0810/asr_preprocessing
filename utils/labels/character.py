#! /usr/bin/env python
# -*- coding: utf-8 -*-


def char2num(str_char, map_file_path):
    """Convert from character to number.
    Args:
        str_char: string of characters
        map_file_path: path to the mapping file
    Returns:
        char_list: list of character indices
    """
    char_list = list(str_char)

    # Lead the mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[line[0]] = int(line[1])

    # Convert from character to number
    for i in range(len(char_list)):
        char_list[i] = map_dict[char_list[i]]

    return char_list


def kana2num(str_char, map_file_path):
    """Convert from kana character to number.
    Args:
        str_char: string of kana characters
        map_file_path: path to the mapping file
    Returns:
        num_list: list of kana character indices
    """
    kana_list = list(str_char)
    num_list = []

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
                num_list.append(int(map_dict[kana_list[i] + kana_list[i + 1]]))
                i += 1
            elif kana_list[i] in map_dict.keys():
                num_list.append(int(map_dict[kana_list[i]]))
            else:
                raise ValueError(
                    'There are no kana character such as %s' % kana_list[i])
        else:
            if kana_list[i] in map_dict.keys():
                num_list.append(int(map_dict[kana_list[i]]))
            else:
                raise ValueError(
                    'There are no kana character such as %s' % kana_list[i])
        i += 1

    return num_list


def num2char(num_list, map_file_path):
    """Convert from number to character.
    Args:
        num_list: list of character indices
        map_file_path: path to the mapping file
    Returns:
        str_char: string of characters
    """
    # Lead the mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[int(line[1])] = line[0]

    # Convert from indices to the corresponding characters
    char_list = []
    for i in range(len(num_list)):
        char_list.append(map_dict[num_list[i]])

    str_char = ''.join(char_list)
    return str_char
