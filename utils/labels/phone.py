#! /usr/bin/env python
# -*- coding: utf-8 -*-


class Phone2idx(object):
    """Convert from phone to index.
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

    def __call__(self, phone_list):
        """
        Args:
            phone_list (list): phones (string)
        Returns:
            phone_list (list): phone indices
        """
        # Convert from phone to index
        for i in range(len(phone_list)):
            phone_list[i] = self.map_dict[phone_list[i]]

        return phone_list


class Idx2phone(object):
    """Convert from index to phone.
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
