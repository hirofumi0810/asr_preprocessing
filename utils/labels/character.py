#! /usr/bin/env python
# -*- coding: utf-8 -*-


class Char2idx(object):
    """Convert from character to index.
    Args:
        map_file_path (string): path to the mapping file
        double_letter (bool, optional): if True, group repeated letters
    """

    def __init__(self, map_file_path, double_letter=False):

        self.double_letter = double_letter

        # Read the mapping file
        self.map_dict = {}
        with open(map_file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.map_dict[line[0]] = int(line[1])

    def __call__(self, str_char):
        """
        Args:
            str_char (string): string of characters
        Returns:
            char_list (list): character indices
        """
        char_list = list(str_char)

        # Convert from character to index
        if self.double_letter:
            skip_flag = False
            for i in range(len(char_list) - 1):
                if skip_flag:
                    char_list[i] = ''
                    skip_flag = False
                    continue

                if char_list[i] + char_list[i + 1] in self.map_dict.keys():
                    char_list[i] = self.map_dict[char_list[i] +
                                                 char_list[i + 1]]
                    skip_flag = True
                else:
                    char_list[i] = self.map_dict[char_list[i]]

            # Final character
            if skip_flag:
                char_list[-1] = ''
            else:
                char_list[-1] = self.map_dict[char_list[-1]]

            # Remove skipped characters
            while '' in char_list:
                char_list.remove('')
        else:
            for i in range(len(char_list)):
                char_list[i] = self.map_dict[char_list[i]]

        return char_list


class Idx2char(object):
    """Convert from index to character.
    Args:
        map_file_path (string): path to the mapping file
    """

    def __init__(self, map_file_path):
        # Read the mapping file
        self.map_dict = {}
        with open(map_file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.map_dict[int(line[1])] = line[0]

    def __call__(self, index_list):
        """Convert from index to character.
        Args:
            index_list (list): list of character indices
        Returns:
            str_char (string): string of characters
        """
        # Convert from indices to the corresponding characters
        char_list = []
        for i in range(len(index_list)):
            char_list.append(self.map_dict[index_list[i]])

        str_char = ''.join(char_list)
        return str_char
