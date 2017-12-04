#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Read word boundary information (LDC97S62 corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re


def read_segmentation(word_boundary_path):
    """
    Args:
        word_boundary_paths (list): list of paths to word boundary files
    Returns:
        segmentation_dict (dict):
            key (string): utt_index
            value (list): list of [start_frame, end_frame, word]
    """
    segmentation_dict = {}
    with open(word_boundary_path, 'r') as f:
        for line in f:
            line = re.sub(r'[\s]+', ' ', line.strip().lower().expandtabs(1))
            line = line.split(' ')
            # speaker = line[0].split('-')[0]
            utt_index = line[0].split('-')[-1]
            start_frame = int(float(line[1]) * 100 + 0.05)
            end_frame = int(float(line[2]) * 100 + 0.05)
            word = line[3].replace('[silence]', '')

            if utt_index not in segmentation_dict.keys():
                segmentation_dict[utt_index] = []

            segmentation_dict[utt_index].append([start_frame, end_frame, word])

    return segmentation_dict
