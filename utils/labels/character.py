#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Char2idx(object):
    """Convert from character to index.
    Args:
        vocab_file_path (string): path to the vocablary file
        double_letter (bool, optional): if True, group repeated letters
        remove_list (list, optional): characters to neglect
    """

    def __init__(self, vocab_file_path, double_letter=False, remove_list=[]):

        self.double_letter = double_letter

        # Read the vocablary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                char = line.strip()
                if char in remove_list:
                    continue
                self.map_dict[char] = vocab_count
                vocab_count += 1

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
        vocab_file_path (string): path to the vocablary file
    """

    def __init__(self, vocab_file_path):
        # Read the vocablary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                char = line.strip()
                self.map_dict[vocab_count] = char
                vocab_count += 1

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
