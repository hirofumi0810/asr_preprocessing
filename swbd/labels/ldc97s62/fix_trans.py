#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

LAUGHTER = '@'
NOISE = '#'
VOCALIZED_NOISE = '$'


def fix_transcript(transcript):

    # Remove silence, <b_aside> and <e_aside>
    transcript = re.sub(r'\[silence\]', '', transcript)
    transcript = re.sub(r'\<b_aside\>', '', transcript)
    transcript = re.sub(r'\<e_aside\>', '', transcript)

    # Replace with special symbols
    transcript = re.sub(r'\[noise\]', NOISE, transcript)
    transcript = re.sub(r'\[vocalized-noise\]', VOCALIZED_NOISE, transcript)
    transcript = re.sub(r'\[laughter\]', LAUGHTER, transcript)
    transcript = re.sub('&', ' and ', transcript)

    ####################
    # laughter
    ####################
    # exception (sw3845A): [laughter-okay] laughter -> okay L
    if transcript == '[laughter-okay] laughter':
        transcript = re.sub(
            r'\[laughter-okay\] laughter', 'okay ' + LAUGHTER, transcript)

    # ex.) [laughter-story] -> story
    laughter_expr = re.compile(r'(.*)\[laughter-([\S]+)\](.*)')
    while re.match(laughter_expr, transcript) is not None:
        laughter = re.match(laughter_expr, transcript)
        transcript = laughter.group(1) + laughter.group(2) + laughter.group(3)
        # transcript = laughter.group(
        #     1) + ' ' + LAUGHTER + laughter.group(2) + laughter.group(3)

    ####################
    # which
    ####################
    # 1st part may include partial-word stuff, which we process further below
    # ex.) [it'n/isn't] -> it'n
    # ex.) [lem[guini]-/linguini] -> lem[guini]-
    which_expr = re.compile(r'(.*)\[([\S]+)/([\S]+)\](.*)')
    while re.match(which_expr, transcript) is not None:
        which = re.match(which_expr, transcript)
        transcript = which.group(1) + which.group(2) + which.group(4)
        # NOTE: the forward word is adopted

    #############################
    # partial word
    #############################
    # backward
    # ex.) -[an]y -> -y
    partial_backward_expr = re.compile(r'(.*)-\[([^\[\]\s]+)\](.*)')
    while re.match(partial_backward_expr, transcript) is not None:
        backward = re.match(partial_backward_expr, transcript)
        transcript = backward.group(1) + '-' + backward.group(3)

    # forward
    # ex.) ab[solute]- -> ab-
    # ex.) ex[specially]-/especially] -> ex-
    partial_forward_expr = re.compile(r'(.*)\[([^\[\]\s]+)\]-(.*)')
    while re.match(partial_forward_expr, transcript) is not None:
        forward = re.match(partial_forward_expr, transcript)
        transcript = forward.group(1) + '-' + forward.group(3)

    ####################
    # exception
    ####################
    # ex.) {yuppiedom} -> yuppiedom
    wave_expr = re.compile(r'(.*)\{([\S]+)\}(.*)')
    while re.match(wave_expr, transcript) is not None:
        wave = re.match(wave_expr, transcript)
        transcript = wave.group(1) + wave.group(2) + wave.group(3)

    # ex.) ammu[n]it- -> ammu-it- (sw2434A)
    # ex.) [-can]sego -> -sego (sw2105A)
    middle_expr = re.compile(r'(.*)\[([\S]+)\](.*)')
    while re.match(middle_expr, transcript) is not None:
        kaku_kakko = re.match(middle_expr, transcript)
        transcript = kaku_kakko.group(1) + '-' + kaku_kakko.group(3)

    # ex.) them_1 -> them
    transcript = re.sub(r'_\d', '', transcript)

    # Remove "/"
    transcript = re.sub('/', '', transcript)

    # Remove double spaces
    while '  ' in transcript:
        transcript = re.sub(r'[\s]+', ' ', transcript)

    # Remove double --
    transcript = re.sub('--', '-', transcript)

    return transcript
