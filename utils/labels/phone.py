#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Phone2idx(object):
    """Convert from phone to index.
    Args:
        vocab_file_path (string): path to the vocablary file
        remove_list (list, optional): phones to neglect
    """

    def __init__(self, vocab_file_path, remove_list=[]):
        # Read the vocablary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                phone = line.strip()
                if phone in remove_list:
                    continue
                self.map_dict[phone] = vocab_count
                vocab_count += 1

    def __call__(self, phone_list):
        """
        Args:
            phone_list (list): phones (string)
        Returns:
            index_list (list): phone indices
        """
        # Convert from phone to index
        index_list = []
        for i in range(len(phone_list)):
            index_list.append(self.map_dict[phone_list[i]])
        return index_list


class Idx2phone(object):
    """Convert from index to phone.
    Args:
        vocab_file_path (string): path to the vocablary file
    """

    def __init__(self, vocab_file_path):
        # Read the vocablary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                phone = line.strip()
                self.map_dict[vocab_count] = phone
                vocab_count += 1

    def __call__(self, index_list):
        """Convert from index to phone.
        Args:
            index_list (list): phone indices
        Returns:
            phone_list (list): phones (string)
        """
        # convert from indices to the corresponding phones
        phone_list = []
        for i in range(len(index_list)):
            phone_list.append(self.map_dict[index_list[i]])
        return phone_list
