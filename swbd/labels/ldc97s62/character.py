#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level target labels for the End-to-End model (LDC97S62)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import re
import numpy as np
from tqdm import tqdm

from swbd.labels.ldc97s62.fix_trans import fix_transcript
# from utils.labels.phone import Phone2idx
from utils.labels.character import Char2idx
from utils.labels.word import Word2idx
from utils.util import mkdir_join

# NOTE:
############################################################
# CTC model

# [phone]

# [character]
# 26 alphabets(a-z), space(_), apostorophe('), hyphen(-)
# laughter(L), noise(N), vocalized-noise(V)
# = 32 labels

# [character_capital_divide]
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 22 special double-letters, apostorophe('), hyphen(-),
# laughter(L), noise(N), vocalized-noise(V)
# = 92 labels

# [word]

############################################################

############################################################
# Attention-based model

# [phone]

# [character]
# 26 alphabets(a-z), space(_), apostorophe('),
# laughter(L), noise(N), vocalized-noise(V), <SOS>, <EOS>
# = 34 labels

# [character_capital_divide]
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 22 special double-letters, apostorophe('), hyphen(-),
# laughter(L), noise(N), vocalized-noise(V), <SOS>, <EOS>
# = 94 labels

# [word]

############################################################

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
HYPHEN = '-'
APOSTROPHE = '\''
SOS = '<'
EOS = '>'
LAUGHTER = '@'
NOISE = '#'
VOCALIZED_NOISE = '$'


def read_char(label_paths, run_root_path,
              ctc_phone_save_path=None, att_phone_save_path=None,
              ctc_char_save_path=None, att_char_save_path=None,
              ctc_char_capital_save_path=None, att_char_capital_save_path=None,
              ctc_word_save_path=None, att_word_save_path=None,
              frequency_threshold=5):
    """Read transcripts (*_trans.txt) & save files (.npy).
    Args:
        label_paths (list): list of paths to label files
        run_root_path (string): absolute path of make.sh
        phone_save_path (string, optional): path to save phone-level labels.
            If None, don't save labels
        ctc_char_save_path (string, optional): path to save character-level labels.
            If None, don't save labels

        ctc_char_capital_save_path (string, optional): path to save capital-divided
            character-level labels. If None, don't save labels

        ctc_word_save_path (string, optional): path to save word-level labels.
            If None, don't save labels

        frequency_threshold (int, optional): the vocabulary is restricted to
            words which appear more than 'frequency_threshold' in the training
            set
    Returns:
        speaker_dict: dictionary of speakers
            key (string) => speaker
            value (dict) => dictionary of utterance infomation of each speaker
                key (string) => utterance index
                value (list) => [start_frame, end_frame, transcript, transcript_capital]
    """
    print('===> Reading target labels...')
    speaker_dict = {}
    char_set, char_capital_set = set([]), set([])
    word_count_dict = {}
    vocab_set = set([])
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

                # Remove first and last space
                if transcript[0] == ' ':
                    transcript = transcript[1:]
                if transcript[-1] == ' ':
                    transcript = transcript[:-1]

                # Count word frequency
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

    # Make mapping file to index
    phone_map_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files', 'phone.txt')
    char_map_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files', 'character.txt')
    char_capital_map_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files', 'character_capital_divide.txt')
    word_map_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files',
        'word_freq' + str(frequency_threshold) + '.txt')

    # Reserve some indices
    char_set.discard(SPACE)
    char_set.discard(LAUGHTER)  # laughter
    char_set.discard(NOISE)  # noise
    char_set.discard(VOCALIZED_NOISE)  # vocalized-noise
    char_capital_set.discard(LAUGHTER)  # laughter
    char_capital_set.discard(NOISE)  # noise
    char_capital_set.discard(VOCALIZED_NOISE)  # vocalized-noise

    # Restrict the vocabulary
    oov_list = [word for word, freq in word_count_dict.items()
                if freq < frequency_threshold]
    original_vocab_num = len(vocab_set)
    vocab_set -= set(oov_list)

    # for debug
    print(sorted(list(char_set)))
    print(sorted(list(char_capital_set)))

    print('Original vocab: %d' % original_vocab_num)
    print('Restriced vocab: %d' % (len(vocab_set) + 1))  # + OOV
    total_word_count = np.sum(list(word_count_dict.values()))
    total_oov_word_count = np.sum(
        [count for word, count in word_count_dict.items() if word in oov_list])
    print('OOV rate %f %%' % ((total_oov_word_count / total_word_count) * 100))

    # phone-level
    # with open(phone_map_file_path, 'w') as f:
    #     raise NotImplementedError

    # character-level
    with open(char_map_file_path, 'w') as f:
        char_list = sorted(list(char_set)) + \
            [APOSTROPHE, HYPHEN, LAUGHTER, NOISE, VOCALIZED_NOISE, SOS, EOS]
        for i, char in enumerate(char_list):
            f.write('%s  %s\n' % (char, str(i)))

    # character-level (capital-divided)
    with open(char_capital_map_file_path, 'w') as f:
        char_capital_list = [SPACE] + sorted(list(char_capital_set)) + \
            [APOSTROPHE, HYPHEN, LAUGHTER, NOISE, VOCALIZED_NOISE, SOS, EOS]
        for i, char in enumerate(char_capital_list):
            f.write('%s  %s\n' % (char, str(i)))

    # word-level
    with open(word_map_file_path, 'w') as f:
        word_list = sorted(list(vocab_set)) + ['OOV', SOS, EOS]
        for i, word in enumerate(word_list):
            f.write('%s  %s\n' % (word, str(i)))

    # phone2idx = Word2idx(map_file_path=phone_map_file_path)
    char2idx = Char2idx(map_file_path=char_map_file_path)
    char2idx_capital = Char2idx(map_file_path=char_capital_map_file_path)
    word2idx = Word2idx(map_file_path=word_map_file_path)

    # Save target labels
    print('===> Saving target labels...')
    for speaker, utterance_dict in tqdm(speaker_dict.items()):
        for utt_index, utt_info in utterance_dict.items():
            start_frame, end_frame, transcript, transcript_capital = utt_info
            save_file_name = speaker + '_' + utt_index + '.npy'

            # if ctc_phone_save_path is not None:
            #     raise NotImplementedError
            #
            #     # Convert from phone to index
            #
            #     # Save as npy file
            #     np.save(mkdir_join(ctc_phone_save_path, 'phone', 'ctc', speaker, save_file_name),
            #             ctc_phone_index_list)
            #     np.save(mkdir_join(att_phone_save_path, 'phone', 'attention', speaker, save_file_name),
            #             att_phone_index_list)

            if ctc_char_save_path is not None:
                # Convert from character to index
                index_list = char2idx(transcript, double_letter=False)

                # Save as npy file
                np.save(mkdir_join(ctc_char_save_path, speaker, save_file_name),
                        index_list)

            if att_char_save_path is not None:
                # Convert from character to index
                char_index_list = char2idx(
                    SOS + transcript + EOS, double_letter=False)

                # Save as npy file
                np.save(mkdir_join(att_char_save_path,  speaker, save_file_name),
                        char_index_list)

            if ctc_char_capital_save_path is not None:
                # Convert from character to index
                index_list = char2idx_capital(transcript, double_letter=True)

                # Save as npy file
                np.save(mkdir_join(ctc_char_capital_save_path, speaker, save_file_name),
                        index_list)

            if att_char_capital_save_path is not None:
                # Convert from character to index
                index_list = char2idx_capital(
                    SOS + transcript + EOS, double_letter=True)

                # Save as npy file
                np.save(mkdir_join(att_char_capital_save_path, speaker, save_file_name),
                        index_list)

            if ctc_word_save_path is not None:
                # Convert to OOV
                word_list = [
                    word if word in vocab_set else 'OOV' for word in word_list]

                # Convert from word to index
                index_list = word2idx(word_list)

                # Save as npy file
                np.save(mkdir_join(ctc_word_save_path, 'word_freq' + str(frequency_threshold), speaker, save_file_name),
                        index_list)

            if att_word_save_path is not None:
                # Convert to OOV
                word_list = [
                    word if word in vocab_set else 'OOV' for word in word_list]

                # Convert from word to index
                index_list = word2idx([SOS] + word_list + [EOS])

                # Save as npy file
                np.save(mkdir_join(att_word_save_path, 'word_freq' + str(frequency_threshold), speaker, save_file_name),
                        index_list)

    return speaker_dict
