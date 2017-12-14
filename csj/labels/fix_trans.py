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
NOISES = ['<雑音>', '<笑>', '<息>', '<咳>', '<泣>', '<拍手>', '<フロア発話>',
          '<フロア笑>', '<ベル>', '<デモ>']


def fix_transcript(kana_seq):

    # Ignore utterances (R ×××...)
    if 'R' in kana_seq or '×' in kana_seq:
        return ''

    # Replace noises to a single class
    for noise in NOISES:
        kana_seq = kana_seq.replace(noise, NOISE)

    # Remove
    kana_seq = re.sub(r'<朗読間違い>', '', kana_seq)

    # Convert (?) -> ?, <FV> -> <>
    kana_seq = re.sub(r'\(\?\)', '?', kana_seq)
    kana_seq = re.sub(r'<FV>', '<>', kana_seq)  # vocal fly
    # NOTE: 先に完全に消すわけにはいかない

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

    # Convert number to kanji character
    # kana_seq = kana_seq.replace('１', '一')
    # kana_seq = kana_seq.replace('２', '二')
    # kana_seq = kana_seq.replace('３', '三')
    # kana_seq = kana_seq.replace('４', '四')
    # kana_seq = kana_seq.replace('５', '五')
    # kana_seq = kana_seq.replace('６', '六')
    # kana_seq = kana_seq.replace('７', '七')
    # kana_seq = kana_seq.replace('８', '八')
    # kana_seq = kana_seq.replace('９', '九')
    # kana_seq = kana_seq.replace('０', '零')
    # \十,\百,\千

    # Remove
    kana_seq = re.sub(r'<H>', '', kana_seq)  # extended voise
    kana_seq = re.sub(r'<Q>', '', kana_seq)  # exytended voise
    kana_seq = re.sub(r'\?', '', kana_seq)  # (?)
    kana_seq = re.sub(r'<>', '', kana_seq)  # <FV>

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
