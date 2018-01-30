#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions of regular expression for hierarchical structure."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re


def remove_pause(transcript):
    # 200ms以上のポーズ
    expr = r'(.*)<P:\d{5}\.\d{3}-\d{5}\.\d{3}>(.*)'
    pause = re.match(expr, transcript)
    while pause is not None:
        transcript = pause.group(1) + pause.group(2)
        pause = re.match(expr, transcript)
    return transcript


def remove_question_which(transcript):
    expr = r'(.*)\(\?[\s]+([^()]+),([^()]+)\)(.*)'
    qw = re.match(expr, transcript)
    while qw is not None:
        # NOTE: Select the FORMER
        transcript = qw.group(1) + qw.group(2) + qw.group(4)
        # transcript = qw.group(1) + qw.group(3) + qw.group(4)
        qw = re.match(expr, transcript)
    return transcript


def remove_question(transcript):
    # 聞き取りや語彙の判断に自信がない場合
    # expr = r'(.*)\(\?[\s]+([^,()]+)\)(.*)'
    expr = r'(.*)\(\?[\s]+([^()]+)\)(.*)'
    question = re.match(expr, transcript)
    while question is not None:
        if ',' in question.group(2):
            transcript = question.group(
                1) + question.group(2).split(',')[0] + question.group(3)
            # NOTE: (? 等,十,当)
        else:
            transcript = question.group(
                1) + question.group(2) + question.group(3)
        question = re.match(expr, transcript)
    return transcript


def remove_Btag(transcript):
    # 語の読みに関する知識レベルの言い間違い
    expr = r'(.*)\(B[\s]+([^()]+);([^()]+)\)(.*)'
    Btag = re.match(expr, transcript)
    while Btag is not None:
        # former: observed pronunciation
        # latter: true pronunciation
        # NOTE: Select the FORMER. This tag is observed only in trans_kana
        transcript = Btag.group(1) + Btag.group(2) + Btag.group(4)
        # transcript = Btag.group(1) + Btag.group(3) + Btag.group(4)
        Btag = re.match(expr, transcript)
    return transcript


def remove_disfluency(transcript):
    # D:言い直し,言い淀み等による語断片
    # D2:助詞,助動詞,接辞の言い直し
    expr = r'(.*)\(D[\d]*[\s]+([^()]+)\)(.*)'
    disfluency = re.match(expr, transcript)
    while disfluency is not None:
        transcript = disfluency.group(
            1) + disfluency.group(2) + disfluency.group(3)
        disfluency = re.match(expr, transcript)
    return transcript


def remove_filler(transcript):
    # フィラー,感情表出系感動詞
    expr = r'(.*)\(F[\s]+([^()]+)\)(.*)'
    filler = re.match(expr, transcript)
    while filler is not None:
        transcript = filler.group(1) + filler.group(2) + filler.group(3)
        filler = re.match(expr, transcript)
    return transcript


def remove_Xtag(transcript):
    # 非朗読対象発話 ( 朗読における言い間違い等 )
    expr = r'(.*)\(X[\s]+([^()]+)\)(.*)'
    Xtag = re.match(expr, transcript)
    while Xtag is not None:
        transcript = Xtag.group(1) + Xtag.group(2) + Xtag.group(3)
        Xtag = re.match(expr, transcript)
    return transcript


def remove_Atag(transcript):
    # アルファベットや算用数字,記号の表記
    expr = r'(.*)\(A[\s]+([^()]+);([^()]+)\)(.*)'
    Atag = re.match(expr, transcript)
    while Atag is not None:
        # NOTE: Select the LATTER. This means pick up alphabet or number.
        # This tag is observed only in trans_kanji
        # transcript = Atag.group(1) + Atag.group(2) + Atag.group(4)
        transcript = Atag.group(1) + Atag.group(3) + Atag.group(4)
        Atag = re.match(expr, transcript)
    return transcript


def remove_Ktag(transcript):
    expr = r'(.*)\(K[\s]+([^()]+);([^()]+)\)(.*)'
    Ktag = re.match(expr, transcript)
    while Ktag is not None:
        transcript = Ktag.group(1) + Ktag.group(3) + Ktag.group(4)
        Ktag = re.match(expr, transcript)
    return transcript


def remove_cry(transcript):
    # 泣きながら発話
    expr = r'(.*)\(泣[\s]+([^()]+)\)(.*)'
    cry = re.match(expr, transcript)
    while cry is not None:
        transcript = cry.group(1) + cry.group(2) + cry.group(3)
        cry = re.match(expr, transcript)
    return transcript


def remove_cough(transcript):
    # 咳をしながら発話
    expr = r'(.*)\(咳[\s]+([^()]+)\)(.*)'
    cough = re.match(expr, transcript)
    while cough is not None:
        transcript = cough.group(1) + cough.group(2) + cough.group(3)
        cough = re.match(expr, transcript)
    return transcript


def remove_which(transcript):
    # 転訛,発音の怠けなど ,一時的な発音エラー
    expr = r'(.*)\(W[\s]+([^()]+);([^()]+)\)(.*)'
    which = re.match(expr, transcript)
    while which is not None:
        # former: observed pronunciation
        # latter: true pronunciation
        # NOTE: Select the LATTER. This tag is observed only in trans_kana
        transcript = which.group(1) + which.group(3) + which.group(4)
        which = re.match(expr, transcript)
    return transcript


def remove_which_Ltag(transcript):
    expr = r'(.*)\(W[\s]+([^()]*)\(L[\s]+([^()]+);([^()]+)\)\)(.*)'
    which_Ltag = re.match(expr, transcript)
    while which_Ltag is not None:
        transcript = which_Ltag.group(
            1) + which_Ltag.group(4) + which_Ltag.group(5)
        which_Ltag = re.match(expr, transcript)
    return transcript


def remove_which_laughing(transcript):
    expr = r'(.*)\(W[\s]+([^()]*)\(笑[\s]+([^()]+);([^()]+)\)\)(.*)'
    which_laughing = re.match(expr, transcript)
    while which_laughing is not None:
        transcript = which_laughing.group(
            1) + which_laughing.group(4) + which_laughing.group(5)
        which_laughing = re.match(expr, transcript)
    return transcript


def remove_Ltag(transcript):
    # ささやき声や独り言などの小さな声
    expr = r'(.*)\(L[\s]+([^()]+)\)(.*)'
    Ltag = re.match(expr, transcript)
    while Ltag is not None:
        transcript = Ltag.group(1) + Ltag.group(2) + Ltag.group(3)
        Ltag = re.match(expr, transcript)
    return transcript


def remove_laughing(transcript):
    # 発話笑い
    expr = r'(.*)\(笑[\s]+([^()]+)\)(.*)'
    laughing = re.match(expr, transcript)
    while laughing is not None:
        transcript = laughing.group(1) + laughing.group(2) + laughing.group(3)
        laughing = re.match(expr, transcript)
    return transcript


def remove_Otag(transcript):
    # 外国語や古語,方言など
    expr = r'(.*)\(O[\s]+([^()]+)\)(.*)'
    Otag = re.match(expr, transcript)
    while Otag is not None:
        transcript = Otag.group(1) + Otag.group(2) + Otag.group(3)
        Otag = re.match(expr, transcript)
    return transcript


def remove_Mtag(transcript):
    # 音や言葉に関するメタ的な引用
    expr = r'(.*)\(M[\s]+([^()]+)\)(.*)'
    Mtag = re.match(expr, transcript)
    while Mtag is not None:
        transcript = Mtag.group(1) + Mtag.group(2) + Mtag.group(3)
        Mtag = re.match(expr, transcript)
    return transcript
