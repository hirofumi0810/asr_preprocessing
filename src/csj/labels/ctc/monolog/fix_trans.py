#! /usr/bin/env python
# -*- coding: utf-8 -*-


import re
from .regular_expression import *


def fix_transcript(kana_seq, speaker_name):
    if 'R' in kana_seq or '×' in kana_seq:
        return ''

    # replace to Noise
    kana_seq = re.sub(r'<雑音>', 'NZ', kana_seq)
    kana_seq = re.sub(r'<笑>', 'NZ', kana_seq)
    kana_seq = re.sub(r'<息>', 'NZ', kana_seq)
    kana_seq = re.sub(r'<咳>', 'NZ', kana_seq)
    kana_seq = re.sub(r'<泣>', 'NZ', kana_seq)
    kana_seq = re.sub(r'<拍手>', 'NZ', kana_seq)
    kana_seq = re.sub(r'<フロア発話>', 'NZ', kana_seq)
    kana_seq = re.sub(r'<フロア笑>', 'NZ', kana_seq)
    kana_seq = re.sub(r'<ベル>', 'NZ', kana_seq)
    kana_seq = re.sub(r'<デモ>', 'NZ', kana_seq)

    # remove
    kana_seq = re.sub(r'<朗読間違い>', '', kana_seq)

    # convert (?) => ?, <FV> => V
    kana_seq = re.sub(r'\(\?\)', '?', kana_seq)
    kana_seq = re.sub(r'<FV>', '<>', kana_seq)  # vocal fly

    # decompose hierarchical structure
    kana_seq = remove_pause(kana_seq)
    kana_seq = remove_question_which(kana_seq)
    kana_seq = remove_question(kana_seq)
    kana_seq = remove_Btag(kana_seq)
    kana_seq = remove_disfluency(kana_seq)
    kana_seq = remove_filler(kana_seq)
    kana_seq = remove_Xtag(kana_seq)
    kana_seq = remove_Atag(kana_seq)
    kana_seq = remove_Ktag(kana_seq)
    kana_seq = remove_cry(kana_seq)
    kana_seq = remove_question_which(kana_seq)
    kana_seq = remove_cough(kana_seq)
    kana_seq = remove_which(kana_seq, speaker_name)
    kana_seq = remove_question_which(kana_seq)
    kana_seq = remove_Ltag(kana_seq)
    kana_seq = remove_laughing(kana_seq)
    kana_seq = remove_which(kana_seq, speaker_name)
    kana_seq = remove_Otag(kana_seq)
    kana_seq = remove_Mtag(kana_seq)

    # remove
    kana_seq = re.sub(r'<H>', '', kana_seq)  # extended voise
    kana_seq = re.sub(r'<Q>', '', kana_seq)  # exytended voise
    kana_seq = re.sub(r'\?', '', kana_seq)
    kana_seq = re.sub(r'<>', '', kana_seq)

    # convert space to underbar
    kana_seq = re.sub(r'\s', '_', kana_seq)

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
