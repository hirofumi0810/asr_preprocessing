#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions of regular expression for hierarchical structure."""

import re


def remove_pause(kana_seq):
    # 0.2 秒以上のポーズ
    pause = re.match(r'(.*)<P:\d{5}\.\d{3}-\d{5}\.\d{3}>(.*)', kana_seq)
    while pause is not None:
        kana_seq = pause.group(1) + ' ' + pause.group(2)
        pause = re.match(r'(.*)<P:\d{5}\.\d{3}-\d{5}\.\d{3}>(.*)', kana_seq)
    return kana_seq


def remove_question_which(kana_seq):

    qw = re.match(r'(.*)\(\? ([^)]+),([^)]+)\)(.*)', kana_seq)
    while qw is not None:
        # より深い階層にW,F，Dタグがあればそちらを先に除去するため，ここでは処理をパスする
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
    # 聞き取りや語彙の判断に自信がない場合
    question = re.match(r'(.*)\(\? ([^)FWD]+)\)(.*)', kana_seq)
    while question is not None:
        kana_seq = question.group(1) + question.group(2) + question.group(3)
        question = re.match(r'(.*)\(\? ([^)FWD]+)\)(.*)', kana_seq)
    return kana_seq


def remove_Btag(kana_seq):
    # 語の読みに関する知識レベルの言い間違い
    Btag = re.match(r'(.*)\(B ([^)]+);([^)]+)\)(.*)', kana_seq)
    while Btag is not None:
        # select latter
        kana_seq = Btag.group(1) + Btag.group(3) + Btag.group(4)
        Btag = re.match(r'(.*)\(B ([^)]+);([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_disfluency(kana_seq):
    # D:言い直し,言い淀み等による語断片
    # D2:助詞,助動詞,接辞の言い直し
    disfluency = re.match(r'(.*)\(D[\d]* ([^)]+)\)(.*)', kana_seq)
    while disfluency is not None:
        kana_seq = disfluency.group(
            1) + disfluency.group(2) + disfluency.group(3)
        disfluency = re.match(r'(.*)\(D[\d]* ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_Xtag(kana_seq):
    # 非朗読対象発話 ( 朗読における言い間違い等 )
    Xtag = re.match(r'(.*)\(X ([^)]+)\)(.*)', kana_seq)
    while Xtag is not None:
        kana_seq = Xtag.group(1) + Xtag.group(2) + Xtag.group(3)
        Xtag = re.match(r'(.*)\(X ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_filler(kana_seq):
    # フィラー,感情表出系感動詞
    filler = re.match(r'(.*)\(F ([^)]+)\)(.*)', kana_seq)
    while filler is not None:
        kana_seq = filler.group(1) + filler.group(2) + filler.group(3)
        filler = re.match(r'(.*)\(F ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_cry(kana_seq):
    # 泣きながら発話
    cry = re.match(r'(.*)\(泣 ([^)]+)\)(.*)', kana_seq)
    while cry is not None:
        kana_seq = cry.group(1) + cry.group(2) + cry.group(3)
        cry = re.match(r'(.*)\(泣 ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_cough(kana_seq):
    # 咳をしながら発話
    cough = re.match(r'(.*)\(咳 ([^)]+)\)(.*)', kana_seq)
    while cough is not None:
        kana_seq = cough.group(1) + cough.group(2) + cough.group(3)
        cough = re.match(r'(.*)\(咳 ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_which(kana_seq, speaker_name):
    # 転訛,発音の怠けなど ,一時的な発音エラー
    which = re.match(r'(.*)\(W ([^)]+);([^)]+)\)(.*)', kana_seq)
    while which is not None:
        if 'L' in which.group(2) or 'L' in which.group(3):
            # WとLは階層構造に優先順位がないための対処
            kana_seq = remove_which_Ltag(kana_seq)
            return kana_seq
        if '笑' in which.group(2) or '笑' in which.group(3):
            # Wと笑は階層構造に優先順位がないための対処
            kana_seq = remove_which_laughing(kana_seq)
            return kana_seq

        # select latter
        kana_seq = which.group(1) + which.group(3) + which.group(4)
        which = re.match(r'(.*)\(W ([^)]+);([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_which_Ltag(kana_seq):
    which_Ltag = re.match(
        r'(.*)\(W ([^)]*)\(L ([^)]+);([^)]+)\)\)(.*)', kana_seq)
    while which_Ltag is not None:
        kana_seq = which_Ltag.group(
            1) + which_Ltag.group(4) + which_Ltag.group(5)
        which_Ltag = re.match(r'(.*)\(W \(L ([^)]+);([^)]+)\)\)(.*)', kana_seq)
    return kana_seq


def remove_which_laughing(kana_seq):
    which_laughing = re.match(
        r'(.*)\(W ([^)]*)\(笑 ([^)]+);([^)]+)\)\)(.*)', kana_seq)
    while which_laughing is not None:
        kana_seq = which_laughing.group(
            1) + which_laughing.group(4) + which_laughing.group(5)
        which_laughing = re.match(
            r'(.*)\(W \(笑 ([^)]+);([^)]+)\)\)(.*)', kana_seq)
    return kana_seq


def remove_Ltag(kana_seq):
    # ささやき声や独り言などの小さな声
    Ltag = re.match(r'(.*)\(L ([^)]+)\)(.*)', kana_seq)
    while Ltag is not None:
        kana_seq = Ltag.group(1) + Ltag.group(2) + Ltag.group(3)
        Ltag = re.match(r'(.*)\(L ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_laughing(kana_seq):
    # 笑いながら発話
    laughing = re.match(r'(.*)\(笑 ([^)]+)\)(.*)', kana_seq)
    while laughing is not None:
        kana_seq = laughing.group(1) + laughing.group(2) + laughing.group(3)
        laughing = re.match(r'(.*)\(笑 ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_Otag(kana_seq):
    # 外国語や古語,方言など
    Otag = re.match(r'(.*)\(O ([^)]+)\)(.*)', kana_seq)
    while Otag is not None:
        kana_seq = Otag.group(1) + Otag.group(2) + Otag.group(3)
        Otag = re.match(r'(.*)\(O ([^)]+)\)(.*)', kana_seq)
    return kana_seq


def remove_Mtag(kana_seq):
    # 音や言葉に関するメタ的な引用
    Mtag = re.match(r'(.*)\(M ([^)]+)\)(.*)', kana_seq)
    while Mtag is not None:
        kana_seq = Mtag.group(1) + Mtag.group(2) + Mtag.group(3)
        Mtag = re.match(r'(.*)\(M ([^)]+)\)(.*)', kana_seq)
    return kana_seq
