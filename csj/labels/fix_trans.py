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

NOISES = ['<雑音>', '<息>', '<笑>', '<咳>', '<泣>', '<拍手>', '<フロア発話>',
          '<フロア笑>', '<ベル>', '<デモ>', '<朗読間違い>']


def fix_transcript(transcript):

    # Ignore utterances (R ×××...)
    if 'R' in transcript or '×' in transcript:
        return ''

    # Remove noises
    for noise in NOISES:
        transcript = transcript.replace(noise, '')

    # Convert (?) -> ?, <FV> -> <>
    transcript = re.sub(r'\(\?\)', '?', transcript)
    transcript = re.sub(r'<FV>', '<>', transcript)  # vocal fly
    # NOTE: 先に完全に消さない

    # Decompose hierarchical structure
    for _ in range(transcript.count('(') + transcript.count('<')):
        transcript = remove_pause(transcript)
        transcript = remove_question(transcript)
        transcript = remove_which(transcript)
        transcript = remove_question_which(transcript)

        transcript = remove_cry(transcript)
        transcript = remove_cough(transcript)
        transcript = remove_laughing(transcript)
        transcript = remove_filler(transcript)
        transcript = remove_disfluency(transcript)

        transcript = remove_Atag(transcript)
        transcript = remove_Btag(transcript)
        transcript = remove_Ktag(transcript)
        transcript = remove_Ltag(transcript)
        transcript = remove_Mtag(transcript)
        transcript = remove_Otag(transcript)
        transcript = remove_Xtag(transcript)

    # Remove
    transcript = re.sub(r'<H>', '', transcript)  # extended voise
    transcript = re.sub(r'<Q>', '', transcript)  # exytended voise
    transcript = re.sub(r'\?', '', transcript)  # (?)
    transcript = re.sub(r'<>', '', transcript)  # <FV>

    return transcript


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
