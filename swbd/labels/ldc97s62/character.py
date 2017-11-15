#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level target labels for the End-to-End model (LDC97S62 corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import re
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from swbd.labels.ldc97s62.fix_trans import fix_transcript
from utils.labels.character import Char2idx
from utils.labels.word import Word2idx
from utils.util import mkdir_join

# NOTE:
############################################################
# [character]
# 26 alphabets(a-z), space(_), apostorophe('), hyphen(-)
# laughter(L), noise(N), vocalized-noise(V)
# = 32 labels

# [character_capital_divide]
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 22 special double-letters, apostorophe('), hyphen(-),
# laughter(L), noise(N), vocalized-noise(V)
# = 92 labels

# [word, threshold == 1]
# Original:  labels + OOV
############################################################

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
HYPHEN = '-'
APOSTROPHE = '\''
LAUGHTER = '@'
NOISE = '#'
VOCALIZED_NOISE = '$'
OOV = 'OOV'


def read_trans(label_paths, run_root_path, vocab_file_save_path,
               save_path=None):
    """Read transcripts (*_trans.txt) & save files (.npy).
    Args:
        label_paths (list): list of paths to label files
        run_root_path (string):
        vocab_file_save_path (string): path to vocabulary files
        save_path (string, optional): path to save target labels.
            If None, don't save labels
    Returns:
        speaker_dict: dictionary of speakers
            key (string) => speaker
            value (dict) => dictionary of utterance infomation of each speaker
                key (string) => utterance index
                value (list) => [start_frame, end_frame, transcript, transcript_capital_divide]
    """
    print('===> Reading target labels...')
    speaker_dict = OrderedDict()
    char_set, char_capital_set = set([]), set([])
    word_count_dict = {}
    vocab_set = set([])

    # for debug
    fp_original = open(join(run_root_path, 'labels',
                            'ldc97s62', 'trans_original.txt'), 'w')
    fp_fixed = open(join(run_root_path, 'labels',
                         'ldc97s62', 'trans_fixed.txt'), 'w')
    fp_fixed_capital = open(
        join(run_root_path, 'labels', 'ldc97s62', 'trans_fixed_capital.txt'), 'w')

    for label_path in tqdm(label_paths):
        utterance_dict = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                speaker = line[0].split('-')[0]
                # Fix speaker name
                speaker = speaker.replace('sw0', 'sw').replace(
                    'a', '-A').replace('b', '-B')
                utt_index = line[0].split('-')[-1]
                start_frame = int(float(line[1]) * 100 + 0.05)
                end_frame = int(float(line[2]) * 100 + 0.05)
                transcript = ' '.join(line[3:])

                transcript_original = transcript
                transcript = fix_transcript(transcript)

                # Skip silence
                if transcript == '':
                    continue

                # Remove the first and last space
                if transcript[0] == ' ':
                    transcript = transcript[1:]
                if transcript[-1] == ' ':
                    transcript = transcript[:-1]

                # Count words
                for word in transcript.split(' '):
                    vocab_set.add(word)
                    if word not in word_count_dict.keys():
                        word_count_dict[word] = 0
                    word_count_dict[word] += 1

                # Capital-divided
                transcript_capital = ''
                for word in transcript.split(' '):
                    if len(word) == 1:
                        char_capital_set.add(word)
                        transcript_capital += word
                    else:
                        # Replace the first character with the capital letter
                        word = word[0].upper() + word[1:]

                        # Check double-letters
                        for i in range(0, len(word) - 1, 1):
                            if word[i:i + 2] in DOUBLE_LETTERS:
                                char_capital_set.add(word[i:i + 2])
                            else:
                                char_capital_set.add(word[i])
                        transcript_capital += word

                # Convert space to "_"
                transcript = re.sub(r'\s', SPACE, transcript)

                for c in list(transcript):
                    char_set.add(c)

                utterance_dict[utt_index.zfill(4)] = [
                    start_frame, end_frame, transcript, transcript_capital]

                fp_original.write('%s  %s  %s\n' %
                                  (speaker, utt_index, transcript_original))
                fp_fixed.write('%s  %s  %s\n' %
                               (speaker, utt_index, transcript))
                fp_fixed_capital.write('%s  %s  %s\n' % (
                    speaker, utt_index, transcript_capital))

                # for debug
                # print(transcript_original)
                # print(transcript)
                # print(transcript_capital)

            speaker_dict[speaker] = utterance_dict

    fp_original.close()
    fp_fixed.close()
    fp_fixed_capital.close()

    # Make vocabulary files
    char_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'character_swbd.txt')
    char_capital_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'character_capital_divide_swbd.txt')
    word_freq1_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq1_swbd.txt')
    word_freq5_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq5_swbd.txt')
    word_freq10_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq10_swbd.txt')
    word_freq15_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq15_swbd.txt')

    # Reserve some indices
    char_set.discard(SPACE)
    char_set.discard(LAUGHTER)  # laughter
    char_set.discard(NOISE)  # noise
    char_set.discard(VOCALIZED_NOISE)  # vocalized-noise
    char_capital_set.discard(LAUGHTER)  # laughter
    char_capital_set.discard(NOISE)  # noise
    char_capital_set.discard(VOCALIZED_NOISE)  # vocalized-noise

    # for debug
    # print(sorted(list(char_set)))
    # print(sorted(list(char_capital_set)))

    # character-level
    with open(char_vocab_file_path, 'w') as f:
        char_list = sorted(list(char_set)) + \
            [APOSTROPHE, HYPHEN, LAUGHTER, NOISE, VOCALIZED_NOISE]
        for char in char_list:
            f.write('%s\n' % char)

    # character-level (capital-divided)
    with open(char_capital_vocab_file_path, 'w') as f:
        char_capital_list = [SPACE] + sorted(list(char_capital_set)) + \
            [APOSTROPHE, HYPHEN, LAUGHTER, NOISE, VOCALIZED_NOISE]
        for char in char_capital_list:
            f.write('%s\n' % char)

    # word-level (threshold == 1)
    with open(word_freq1_vocab_file_path, 'w') as f:
        vocab_list = sorted(list(vocab_set)) + [OOV]
        for word in vocab_list:
            f.write('%s\n' % word)
    original_vocab_num = len(vocab_list) - 1
    print('Original vocab: %d' % original_vocab_num)

    # word-level (threshold == 5)
    with open(word_freq5_vocab_file_path, 'w') as f:
        vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                             if freq >= 5]) + [OOV]
        for word in vocab_list:
            f.write('%s\n' % word)
    print('Word (freq5):')
    print('  Restriced vocab: %d' % len(vocab_list))
    print('  OOV rate (train): %f %%' %
          (((original_vocab_num - len(vocab_list) + 1) / original_vocab_num) * 100))

    # word-level (threshold == 10)
    with open(word_freq10_vocab_file_path, 'w') as f:
        vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                             if freq >= 10]) + [OOV]
        for word in vocab_list:
            f.write('%s\n' % word)
    print('Word (freq10):')
    print('  Restriced vocab: %d' % len(vocab_list))
    print('  OOV rate (train): %f %%' %
          (((original_vocab_num - len(vocab_list) + 1) / original_vocab_num) * 100))

    # word-level (threshold == 15)
    with open(word_freq15_vocab_file_path, 'w') as f:
        vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                             if freq >= 15]) + [OOV]
        for word in vocab_list:
            f.write('%s\n' % word)
    print('Word (freq15):')
    print('  Restriced vocab: %d' % len(vocab_list))
    print('  OOV rate (train): %f %%' %
          (((original_vocab_num - len(vocab_list) + 1) / original_vocab_num) * 100))

    char2idx = Char2idx(char_vocab_file_path)
    char2idx_capital = Char2idx(char_capital_vocab_file_path,
                                double_letter=True)
    word2idx_freq1 = Word2idx(word_freq1_vocab_file_path)
    word2idx_freq5 = Word2idx(word_freq5_vocab_file_path)
    word2idx_freq10 = Word2idx(word_freq10_vocab_file_path)
    word2idx_freq15 = Word2idx(word_freq15_vocab_file_path)

    # Save target labels
    print('===> Saving target labels...')
    for speaker, utterance_dict in tqdm(speaker_dict.items()):
        for utt_index, utt_info in utterance_dict.items():
            start_frame, end_frame, transcript, transcript_capital = utt_info
            save_file_name = speaker + '_' + utt_index + '.npy'

            # TODO: Convert to word to phone

            if save_path is not None:
                word_list = transcript.split(SPACE)

                # Save target labels as index
                np.save(mkdir_join(save_path, 'character' 'train', speaker, save_file_name),
                        char2idx(transcript))
                np.save(mkdir_join(save_path, 'character_capital_divide' 'train', speaker, save_file_name),
                        char2idx_capital(transcript))
                np.save(mkdir_join(save_path, 'word_freq1', 'train', speaker, save_file_name),
                        word2idx_freq1(word_list))
                np.save(mkdir_join(save_path, 'word_freq5', 'train', speaker, save_file_name),
                        word2idx_freq5(word_list))
                np.save(mkdir_join(save_path, 'word_freq10', 'train', speaker, save_file_name),
                        word2idx_freq10(word_list))
                np.save(mkdir_join(save_path, 'word_freq15', 'train', speaker, save_file_name),
                        word2idx_freq15(word_list))

    return speaker_dict
