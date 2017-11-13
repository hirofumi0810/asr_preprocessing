#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Word2idx(object):
    """Convert from word to index.
    Args:
        vocab_file_path (string): path to the vocablary file
    """

    def __init__(self, vocab_file_path):
        # Read the vocablary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                word = line.strip()
                self.map_dict[word] = vocab_count
                vocab_count += 1

    def __call__(self, word_list):
        """Convert from word to index.
        Args:
            word_list (list): words (string)
        Returns:
            index_list (lisr): word indices
        """
        # Convert from word to index
        index_list = []
        for word in word_list:
            if word in self.map_dict.keys():
                index_list.append(self.map_dict[word])
            else:
                # Replace with <UNK>
                index_list.append(self.map_dict['OOV'])
        return index_list
