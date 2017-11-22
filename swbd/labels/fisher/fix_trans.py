#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

LAUGHTER = 'LA'
NOISE = 'NZ'
VOCALIZED_NOISE = 'VN'


def fix_transcript(transcript):

    # Replace with special symbols
    transcript = re.sub(r'\[laughter\]', LAUGHTER, transcript)
    transcript = re.sub(r'\[laugh\]', LAUGHTER, transcript)
    transcript = re.sub(r'\[noise\]', NOISE, transcript)
    transcript = re.sub(r'\[sigh\]', NOISE, transcript)
    transcript = re.sub(r'\[cough\]', NOISE, transcript)
    transcript = re.sub(r'\[mn\]', NOISE, transcript)
    transcript = re.sub(r'\[breath\]', NOISE, transcript)
    transcript = re.sub(r'\[lipsmack\]', NOISE, transcript)
    transcript = re.sub(r'\[sneeze\]', NOISE, transcript)
    transcript = re.sub('&', ' and ', transcript)

    # Remove
    transcript = re.sub(r'\[pause\]', '', transcript)
    transcript = re.sub(r'\[\[skip\]\]', '', transcript)
    transcript = re.sub(r'\?', '', transcript)
    transcript = re.sub(r'\*', '', transcript)
    transcript = re.sub(r'~', '', transcript)
    transcript = re.sub(r'\,', '', transcript)
    transcript = re.sub(r'\.', '', transcript)

    # Remove sentences which include german words
    german = re.match(r'(.*)<german (.+)>(.*)', transcript)
    if german is not None:
        transcript = ''

    # Remove ((  ))
    transcript = re.sub(r'\(\([\s]+\)\)', '', transcript)
    kakko_expr = re.compile(r'(.*)\(\( ([^(]+) \)\)(.*)')
    while re.match(kakko_expr, transcript) is not None:
        kakko = re.match(kakko_expr, transcript)
        transcript = kakko.group(1) + kakko.group(2) + kakko.group(3)

    # remove "/"
    # transcript = re.sub('/', '', transcript)

    # Remove double spaces
    while '  ' in transcript:
        transcript = re.sub(r'[\s]+', ' ', transcript)

    return transcript
