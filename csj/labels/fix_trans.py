#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from csj.labels.regular_expression import remove_pause
from csj.labels.regular_expression import remove_question_which
from csj.labels.regular_expression import remove_question
from csj.labels.regular_expression import remove_Btag
from csj.labels.regular_expression import remove_disfluency
from csj.labels.regular_expression import remove_filler
from csj.labels.regular_expression import remove_Xtag
from csj.labels.regular_expression import remove_Atag
from csj.labels.regular_expression import remove_Ktag
from csj.labels.regular_expression import remove_cry
from csj.labels.regular_expression import remove_cough
from csj.labels.regular_expression import remove_which
from csj.labels.regular_expression import remove_Ltag
from csj.labels.regular_expression import remove_laughing
from csj.labels.regular_expression import remove_Otag
from csj.labels.regular_expression import remove_Mtag

NOISE = 'NZ'


def fix_transcript(kana_seq):
    if 'R' in kana_seq or '×' in kana_seq:
        return ''

    # Replace to Noise
    kana_seq = re.sub(r'<雑音>', NOISE, kana_seq)
    kana_seq = re.sub(r'<笑>', NOISE, kana_seq)
    kana_seq = re.sub(r'<息>', NOISE, kana_seq)
    kana_seq = re.sub(r'<咳>', NOISE, kana_seq)
    kana_seq = re.sub(r'<泣>', NOISE, kana_seq)
    kana_seq = re.sub(r'<拍手>', NOISE, kana_seq)
    kana_seq = re.sub(r'<フロア発話>', NOISE, kana_seq)
    kana_seq = re.sub(r'<フロア笑>', NOISE, kana_seq)
    kana_seq = re.sub(r'<ベル>', NOISE, kana_seq)
    kana_seq = re.sub(r'<デモ>', NOISE, kana_seq)

    # Remove
    kana_seq = re.sub(r'<朗読間違い>', '', kana_seq)

    # Convert (?) => ?, <FV> => V
    kana_seq = re.sub(r'\(\?\)', '?', kana_seq)
    kana_seq = re.sub(r'<FV>', '<>', kana_seq)  # vocal fly

    # Decompose hierarchical structure
    for _ in range(kana_seq.count('(') + kana_seq.count('<')):
        kana_seq = remove_pause(kana_seq)
        kana_seq = remove_question(kana_seq)
        kana_seq = remove_which(kana_seq)
        kana_seq = remove_question_which(kana_seq)

        kana_seq = remove_cry(kana_seq)
        kana_seq = remove_cough(kana_seq)
        kana_seq = remove_laughing(kana_seq)
        kana_seq = remove_filler(kana_seq)
        kana_seq = remove_disfluency(kana_seq)

        kana_seq = remove_Atag(kana_seq)
        kana_seq = remove_Btag(kana_seq)
        kana_seq = remove_Ktag(kana_seq)
        kana_seq = remove_Ltag(kana_seq)
        kana_seq = remove_Mtag(kana_seq)
        kana_seq = remove_Otag(kana_seq)
        kana_seq = remove_Xtag(kana_seq)

    # Remove
    kana_seq = re.sub(r'<H>', '', kana_seq)  # extended voise
    kana_seq = re.sub(r'<Q>', '', kana_seq)  # exytended voise
    kana_seq = re.sub(r'\?', '', kana_seq)
    kana_seq = re.sub(r'<>', '', kana_seq)

    return kana_seq


def is_hiragana(char):
    if "ぁ" <= char <= "ん":
        return True
    return False


def is_katakana(char):
    if "ァ" <= char <= "ン":
        return True
    return False


def is_kanji(char):
    if '亜' <= char <= '話':
        return True
    return False


def is_alphabet(char):
    if 'a' <= char <= 'z' or 'A' <= char <= 'Z' or 'ａ' <= char <= 'ｚ' or 'Ａ' <= char <= 'Ｚ':
        return True
    return False
