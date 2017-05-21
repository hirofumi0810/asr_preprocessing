#! /usr/bin/env python
# -*- coding: utf-8 -*-


import re
from .regular_expression import *


def fix_transcript(kana_seq):
    if 'R' in kana_seq or '×' in kana_seq:
        return ''

    # remove <雑音>, <笑>, <息>, <咳>, <泣>, <フロア発話>, <フロア笑>
    kana_seq = re.sub(r'<雑音>', '', kana_seq)
    kana_seq = re.sub(r'<笑>', '', kana_seq)
    kana_seq = re.sub(r'<息>', '', kana_seq)
    kana_seq = re.sub(r'<咳>', '', kana_seq)
    kana_seq = re.sub(r'<泣>', '', kana_seq)
    kana_seq = re.sub(r'<拍手>', '', kana_seq)
    kana_seq = re.sub(r'<フロア発話>', '', kana_seq)
    kana_seq = re.sub(r'<フロア笑>', '', kana_seq)
    kana_seq = re.sub(r'<ベル>', '', kana_seq)
    kana_seq = re.sub(r'<デモ>', '', kana_seq)
    kana_seq = re.sub(r'<朗読間違い>', '', kana_seq)

    # convert (?) => ?, <FV> => V
    kana_seq = re.sub(r'\(\?\)', '?', kana_seq)
    kana_seq = re.sub(r'<FV>', 'V', kana_seq)  # vocal fly

    # decompose hierarchical structure
    kana_seq = remove_pose(kana_seq)
    kana_seq = remove_question_which(kana_seq)
    kana_seq = remove_question(kana_seq)
    kana_seq = remove_bwhich(kana_seq)
    kana_seq = remove_disfluency(kana_seq)
    kana_seq = remove_filler(kana_seq)
    kana_seq = remove_X(kana_seq)
    kana_seq = remove_cry(kana_seq)
    kana_seq = remove_question_which(kana_seq)
    kana_seq = remove_cough(kana_seq)
    kana_seq = remove_which(kana_seq)
    kana_seq = remove_question_which(kana_seq)
    kana_seq = remove_laughing(kana_seq)
    kana_seq = remove_which(kana_seq)  # error processing
    kana_seq = remove_O(kana_seq)
    kana_seq = remove_M(kana_seq)

    # remove <FV>, <H>, <Q>, ?, )
    kana_seq = re.sub(r'<FV>', '', kana_seq)  # vocal fly
    kana_seq = re.sub(r'<H>', '', kana_seq)
    kana_seq = re.sub(r'<Q>', '', kana_seq)
    kana_seq = re.sub(r'\?', '', kana_seq)
    kana_seq = re.sub(r'V', '', kana_seq)

    # convert space to underbar
    kana_seq = re.sub(r'\s', '_', kana_seq)

    return kana_seq
