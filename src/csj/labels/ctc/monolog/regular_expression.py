#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions of regular expression for hierarchical structure."""

import re


def remove_pose(kana_seq):

    pose = re.match(r'(.*)<P:\d{5}\.\d{3}-\d{5}\.\d{3}>(.*)', kana_seq)
    while pose is not None:
        kana_seq = pose.group(1) + ' ' + pose.group(2)
        pose = re.match(r'(.*)<P:\d{5}\.\d{3}-\d{5}\.\d{3}>(.*)', kana_seq)
    return kana_seq


def remove_question_which(kana_seq):

    qw = re.match(r'(.*)\(\? ([^)]+),([^)]+)\)(.*)', kana_seq)
    while qw is not None:
        if 'W' in qw.group(2) or 'W' in qw.group(3):
            return kana_seq
        if 'F' in qw.group(2) or 'F' in qw.group(3):
            return kana_seq
        if 'D' in qw.group(2) or 'D' in qw.group(3):
            return kana_seq

        # select latter
        kana_seq = qw.group(1) + qw.group(3) + qw.group(4)
        qw = re.match(r'(.*)\(\? ([^)]+),([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_question(kana_seq):

    question = re.match(r'(.*)\(\? ([^)FWD]+)\)(.*)', kana_seq)
    while question is not None:
        kana_seq = question.group(1) + question.group(2) + question.group(3)
        question = re.match(r'(.*)\(\? ([^)FWD]+)\)(.*)', kana_seq)
    return kana_seq


def remove_bwhich(kana_seq):

    bwhich = re.match(r'(.*)\(B ([^)]+);([^)]+)\)(.*)', kana_seq)
    while bwhich is not None:
        # select latter
        kana_seq = bwhich.group(1) + bwhich.group(3) + bwhich.group(4)
        bwhich = re.match(r'(.*)\(B ([^)]+);([^)]+)\)(.*)', kana_seq)

    return kana_seq


def remove_disfluency(kana_seq):

    disfluency = re.match(r'(.*)\(D[\d]* ([^)]+)\)(.*)', kana_seq)
    while disfluency is not None:
        kana_seq = disfluency.group(
            1) + disfluency.group(2) + disfluency.group(3)
        disfluency = re.match(r'(.*)\(D[\d]* ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_X(kana_seq):

    X = re.match(r'(.*)\(X ([^)]+)\)(.*)', kana_seq)
    while X is not None:
        kana_seq = X.group(1) + X.group(2) + X.group(3)
        X = re.match(r'(.*)\(X ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_filler(kana_seq):

    filler = re.match(r'(.*)\(F ([^)]+)\)(.*)', kana_seq)
    while filler is not None:
        kana_seq = filler.group(1) + filler.group(2) + filler.group(3)
        filler = re.match(r'(.*)\(F ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_cry(kana_seq):

    cry = re.match(r'(.*)\(泣 ([^)]+)\)(.*)', kana_seq)
    while cry is not None:
        kana_seq = cry.group(1) + cry.group(2) + cry.group(3)
        cry = re.match(r'(.*)\(泣 ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_cough(kana_seq):

    cough = re.match(r'(.*)\(咳 ([^)]+)\)(.*)', kana_seq)
    while cough is not None:
        kana_seq = cough.group(1) + cough.group(2) + cough.group(3)
        cough = re.match(r'(.*)\(咳 ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_which(kana_seq):

    which = re.match(r'(.*)\(W ([^)]+);([^)]+)\)(.*)', kana_seq)
    # which2 = re.match(r'(.*)\(W (.+);(.+)(.*)', kana_seq)
    while which is not None:
        if 'L' in which.group(2) or 'L' in which.group(3):
            return kana_seq
        if '笑' in which.group(2) or '笑' in which.group(3):
            return kana_seq
        if ',' in which.group(2) or ',' in which.group(3):
            return kana_seq

        # select latter
        kana_seq = which.group(1) + which.group(3) + which.group(4)
        which = re.match(r'(.*)\(W ([^)]+);([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_laughing(kana_seq):

    laughing = re.match(r'(.*)\([L|笑] ([^)]+)\)(.*)', kana_seq)
    while laughing is not None:
        kana_seq = laughing.group(1) + laughing.group(2) + laughing.group(3)
        laughing = re.match(r'(.*)\([L|笑] ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_O(kana_seq):

    O = re.match(r'(.*)\(O ([^)]+)\)(.*)', kana_seq)
    while O is not None:
        kana_seq = O.group(1) + O.group(2) + O.group(3)
        O = re.match(r'(.*)\(O ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_M(kana_seq):

    M = re.match(r'(.*)\(M ([^)]+)\)(.*)', kana_seq)
    while M is not None:
        kana_seq = M.group(1) + M.group(2) + M.group(3)
        M = re.match(r'(.*)\(M ([^)]+)\)(.*)', kana_seq)
    return kana_seq
