#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

LAUGHTER = 'LA'
NOISE = 'NZ'
VOCALIZED_NOISE = 'VN'
HESITATION = ['uh', 'um', 'eh', 'mm', 'hm', 'ah', 'huh', 'ha', 'er', 'oof',
              'hee', 'ach', 'eee', 'ew']


def fix_transcript(transcript, speaker):

    # Remove silence, <b_aside>, <e_aside>, and so on...
    transcript = re.sub(r'\[silence\]', '', transcript)
    transcript = re.sub(r'\<b_aside\>', '', transcript)
    transcript = re.sub(r'\<e_aside\>', '', transcript)
    transcript = re.sub(r'\[noise\]', '', transcript)
    transcript = re.sub(r'\[vocalized-noise\]', '', transcript)
    transcript = re.sub(r'\[laughter\]', '', transcript)
    transcript = re.sub(r'\[right\]', '', transcript)

    # Replace with special symbols
    transcript = re.sub(r'\[noise\]', NOISE, transcript)
    transcript = re.sub(r'\[vocalized-noise\]', VOCALIZED_NOISE, transcript)
    transcript = re.sub(r'\[laughter\]', LAUGHTER, transcript)

    # TODO: check in kaldi
    transcript = re.sub(r'\[noise-good\]', NOISE, transcript)
    transcript = re.sub(r'\[uh\]', 'uh', transcript)

    ####################
    # laughter
    ####################
    # ex.) [laughter-story] -> story
    laughter_expr = re.compile(r'(.*)\[laughter-([\S]+)\](.*)')
    while re.match(laughter_expr, transcript) is not None:
        laughter = re.match(laughter_expr, transcript)
        transcript = laughter.group(1) + laughter.group(2) + laughter.group(3)
        # transcript = laughter.group(
        #     1) + ' ' + LAUGHTER + laughter.group(2) + laughter.group(3)

    ####################
    # abbreviation
    ####################
    # ex.) i'm -> i am (2 words)
    abbr_expr2 = re.compile(
        r'(.*)<contraction e_form=\"\[[\S]+=>([\S]+)\]\[[\S]+=>([\S]+)\]\">([\S]+)(.*)')
    while re.match(abbr_expr2, transcript) is not None:
        abbr = re.match(abbr_expr2, transcript)
        transcript = abbr.group(1) + abbr.group(2) + \
            ' ' + abbr.group(3) + abbr.group(5)

    # ex.) can't -> cannot (1 word)
    abbr_expr1 = re.compile(
        r'(.*)<contraction e_form=\"\[[\S]+=>([\S]+)\]\">([\S]+)(.*)')
    while re.match(abbr_expr1, transcript) is not None:
        abbr = re.match(abbr_expr1, transcript)
        transcript = abbr.group(1) + abbr.group(2) + abbr.group(4)

    # TODO: check in kaldi

    ####################
    # double bracket
    ####################
    # ex.) ((yeah)) -> (yeah)
    bracket_expr = re.compile(r'(.*)\(\(([^()\s]+)\)\)(.*)')
    while re.match(bracket_expr, transcript) is not None:
        bracket = re.match(bracket_expr, transcript)
        transcript = bracket.group(
            1) + '(' + bracket.group(2) + ')' + bracket.group(3)

    # ex.) ((is => is
    transcript = re.sub(r'\(\(', '(', transcript)
    # transcript = re.sub(r'\)', '', transcript)
    # TODO: compare with the stm file

    ####################
    # partial word
    ####################
    # forward
    # y[ou]i- -> yi-
    partial_forward_expr = re.compile(
        r'(.*)([^\[\]\s]+)\[([^\[\]\s]+)\]([^\[\]\s]+)-(.*)')
    while re.match(partial_forward_expr, transcript) is not None:
        # print(transcript)
        forward = re.match(partial_forward_expr, transcript)
        transcript = forward.group(1) + forward.group(2) + \
            forward.group(4) + '-' + forward.group(5)
        # print(transcript)

    # backward
    # ex.) -[w]here -> -here
    # ex.) -[a]nd -> -nd
    backward_expr = re.compile(r'(.*)-\[([^\[\]\s]+)\](.*)')
    while re.match(backward_expr, transcript) is not None:
        # print(transcript)
        backward = re.match(backward_expr, transcript)
        transcript = backward.group(1) + '-' + backward.group(3)
        # print(transcript)

    ####################
    # exception
    ####################
    # ex.) ju[st] -> ju-
    # ex.) rein[carnating] -> rein-
    partial_expr = re.compile(r'(.*)([^\[\]\s]+)\[([^\[\]\s]+)\](.*)')
    while re.match(partial_expr, transcript) is not None:
        # print(transcript)
        partial = re.match(partial_expr, transcript)
        transcript = partial.group(
            1) + partial.group(2) + '-' + partial.group(4)
        # print(transcript)

    # Remove consecutive spaces
    while '  ' in transcript:
        transcript = re.sub(r'[\s]+', ' ', transcript)

    if transcript in ['', ' ']:
        return transcript

    # Remove first and last space
    if transcript[0] == ' ':
        transcript = transcript[1:]
    if transcript[-1] == ' ':
        transcript = transcript[:-1]

    # Remove ()
    transcript = re.sub(r'[\(\)]+', '', transcript)

    # Convert hesitation
    word_list = []
    for word in transcript.split(' '):
        if word in HESITATION:
            word = '%hesitation'
        word_list.append(word)
    transcript = ' '.join(word_list)

    # Convert to stm file-like transcription
    # final_transcript = ''
    # word_list = transcript.split(' ')
    # for i, word in enumerate(word_list):
    #     # partial word
    #     if word[0] == '-' or word[-1] == '-':
    #         word = '(' + word + ')'
    #
    #     if i != 0:
    #         final_transcript += ' ' + word
    #     else:
    #         final_transcript += word

    return transcript
