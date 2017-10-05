#! /usr/bin/env python
# -*- coding: utf-8 -*-


class Word2idx(object):
    """Convert from word to index.
    Args:
        map_file_path (string): path to the mapping file
    """

    def __init__(self, map_file_path):
        # Read the mapping file
        self.map_dict = {}
        with open(map_file_path, 'r') as f:
            for line in f:
                line = line.strip().split('  ')
                self.map_dict[str(line[0])] = int(line[1])

    def __call__(self, word_list):
        """Convert from word to index.
        Args:
            word_list (list): words (string)
        Returns:
            index_list (lisr): word indices
        """
        # Convert from word to index
        index_list = []
        for i in range(len(word_list)):
            if word_list[i] in self.map_dict.keys():
                index_list.append(self.map_dict[word_list[i]])
            else:
                # Pad by <UNK> (the last index)
                index_list.append(len(self.map_dict.keys()))
            # TODO: add unk index clearly
        return index_list
