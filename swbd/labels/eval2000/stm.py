#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make target labels from stm files (eval2000 corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import re
from collections import OrderedDict

SPACE = '_'
HESITATION = ['uh', 'um', 'eh', 'mm', 'hm', 'ah', 'huh', 'ha', 'er', 'oof',
              'hee', 'ach', 'eee', 'ew']


def read_stm(stm_path, pem_path, glm_path, run_root_path):
    """Read transcripts (.stm) & save files (.npy).
    Args:
        stm_path (string): path to the transcription file
        pem_path (string): path to the segmentation file
        glm_path (string): path to the GLM file
        run_root_path (string): absolute path of make.sh
    Returns:
        speaker_dict_swbd (dict): dictionary of speakers in eval2000 (swbd)
            key (string) => speaker
            value (dict) => dictionary of utterance infomation of each speaker
                key (string) => utterance index
                value (list) => [start_frame, end_frame, transcript]
        speaker_dict_ch (dict): dictionary of speakers in eval2000 (ch)
            key (string) => speaker
            value (dict) => dictionary of utterance infomation of each speaker
                key (string) => utterance index
                value (list) => [start_frame, end_frame, transcript]
    """
    print('===> Reading the segmentation file...')
    segmentation_info = {}
    speaker_pre = ''
    utt_index = 0
    with open(pem_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line[0] == ';':
                continue

            # Remove double spaces
            while '  ' in line:
                line = re.sub(r'[\s]+', ' ', line)

            # NOTE: pem file has lines like:
            # en_4156 A unknown_speaker 301.85 302.48

            speaker = line.split(' ')[0].replace(
                '_', '') + '-' + line.split(' ')[1]
            # ex.) speaker: en4156-A

            if speaker != speaker_pre:
                utt_index = 0

            if speaker not in segmentation_info.keys():
                segmentation_info[speaker] = {}

            start_time = float(line.split(' ')[3])
            end_time = float(line.split(' ')[4])
            segmentation_info[speaker][utt_index] = [start_time, end_time]

            utt_index += 1
            speaker_pre = speaker

    # print('===> Reading the GLM file...')
    # misspelling_map_dict = {}
    # with open(glm_path, 'r')as f:
    #     for line in f:
    #         line = line.strip()
    #         if len(line) == 0 or line[0] in [';', '*', '\'']:
    #             continue
    #         before, after = line.split('=>')
    #         before = re.sub(r'[\[\]\s]+', '', before).lower()
    #         after = after.split('/')[0]
    #         after = after.split(';')[0]
    #         after = re.sub(r'[\[\]{}]+', '', after).lower()
    #         print(before + '  ' + after)
    #         # NOTE: use the first word
    #         misspelling_map_dict[before] = after
    # # Fix misspelling based on glm
    # word_list = transcript.split(' ')
    # word_list_fixed = []
    # for word in word_list:
    #     if word in misspelling_map_dict.keys():
    #         word_fixed = misspelling_map_dict[word]
    #         word_list_fixed.extend(word_fixed.split(' '))
    #         print('fixed: %s => %s' % (word, word_fixed))
    #     else:
    #         word_list_fixed.append(word)
    # transcript = ' '.join(word_list_fixed)

    print('===> Reading target labels...')
    speaker_dict_swbd = OrderedDict()
    speaker_dict_ch = OrderedDict()
    char_set = set([])
    fp_swbd_original = open(join(run_root_path, 'labels',
                                 'eval2000', 'trans_swbd_stm_original.txt'), 'w')
    fp_ch_original = open(join(run_root_path, 'labels',
                               'eval2000', 'trans_ch_stm_original.txt'), 'w')
    fp_swbd_fixed = open(join(run_root_path, 'labels',
                              'eval2000', 'trans_swbd_stm_fixed.txt'), 'w')
    fp_ch_fixed = open(join(run_root_path, 'labels',
                            'eval2000', 'trans_ch_stm_fixed.txt'), 'w')
    speaker_pre = ''
    utt_index = 0
    utterance_dict = OrderedDict()
    with open(stm_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line[0] == ';':
                continue

            # Remove double spaces
            while '  ' in line:
                line = re.sub(r'[\s]+', ' ', line)

            # NOTE: sgm file has lines like:
            # en_4156 A en_4156_A 357.64 359.64 <O,en,F,en-F>  HE IS A POLICE
            # OFFICER

            speaker = line.split(' ')[2].replace(
                '_', '').replace('A1', 'A').replace('B1', 'B').replace('A', '-A').replace('B', '-B')
            # ex.) speaker: en4156-A

            if speaker != speaker_pre:
                if speaker_pre != '':
                    if speaker[:2] == 'sw':
                        speaker_dict_swbd[speaker] = utterance_dict
                    elif speaker[:2] == 'en':
                        speaker_dict_ch[speaker] = utterance_dict
                    else:
                        raise ValueError

                # reset
                utt_index = 0
                utterance_dict = OrderedDict()

            start_time = float(line.split(' ')[3])
            end_time = float(line.split(' ')[4])
            transcript = ' '.join(line.split(' ')[6:]).lower()
            transcript_original = transcript

            # Error check of segmentation time
            assert segmentation_info[speaker][utt_index][0] == start_time
            assert segmentation_info[speaker][utt_index][1] == end_time

            ##################################################
            # Clean transcript
            ##################################################
            # Remove <b_aside> and <e_aside>
            transcript = re.sub(r'\<b_aside\>', '', transcript)
            transcript = re.sub(r'\<e_aside\>', '', transcript)

            # Remove consecutive spaces
            while '  ' in transcript:
                transcript = re.sub(r'[\s]+', ' ', transcript)

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

            # Convert space to "_"
            transcript = re.sub(r'\s', SPACE, transcript)

            # Write to text files for debug
            if speaker[:2] == 'sw':
                fp_swbd_original.write('%s  %d  %.2f  %.2f  %s\n' %
                                       (speaker, utt_index, start_time, end_time, transcript_original))
                fp_swbd_fixed.write('%s  %d  %.2f  %.2f  %s\n' %
                                    (speaker, utt_index, start_time, end_time, transcript))
            elif speaker[:2] == 'en':
                fp_ch_original.write('%s  %d  %.2f  %.2f  %s\n' %
                                     (speaker, utt_index, start_time, end_time, transcript_original))
                fp_ch_fixed.write('%s  %d  %.2f  %.2f  %s\n' %
                                  (speaker, utt_index, start_time, end_time, transcript))

            for char in list(transcript.lower()):
                char_set.add(char)

            # for debug
            # print(transcript)

            start_frame = int(start_time * 100 + 0.5)
            end_frame = int(end_time * 100 + 0.5)
            utterance_dict[str(utt_index).zfill(4)] = [
                start_frame, end_frame, transcript]

            utt_index += 1
            speaker_pre = speaker

    fp_swbd_original.close()
    fp_swbd_fixed.close()
    fp_ch_original.close()
    fp_ch_fixed.close()

    # for debug
    print(sorted(list(char_set)))

    word_freq1_vocab_file_path = join(
        run_root_path, 'config/vocab_files/word_freq1_300h.txt')
    word_freq5_vocab_file_path = join(
        run_root_path, 'config/vocab_files/word_freq5_300h.txt')
    word_freq10_vocab_file_path = join(
        run_root_path, 'config/vocab_files/word_freq10_300h.txt')
    word_freq15_vocab_file_path = join(
        run_root_path, 'config/vocab_files/word_freq15_300h.txt')

    # Compute OOV rate
    # NOTE: these are not corrct because many %hesitation are included.
    with open(join(run_root_path, 'config/oov_rate_eval2000_swbd_stm.txt'), 'w') as f:

        # word-level (threshold == 1)
        oov_rate = compute_oov_rate(
            speaker_dict_swbd, word_freq1_vocab_file_path)
        f.write('Word (freq1):\n')
        f.write('  OOV rate (eval2000, swbd, from stm): %f %%\n' % oov_rate)

        # word-level (threshold == 5)
        oov_rate = compute_oov_rate(
            speaker_dict_swbd, word_freq5_vocab_file_path)
        f.write('Word (freq5):\n')
        f.write('  OOV rate (eval2000, swbd, from stm): %f %%\n' % oov_rate)

        # word-level (threshold == 10)
        oov_rate = compute_oov_rate(
            speaker_dict_swbd, word_freq10_vocab_file_path)
        f.write('Word (freq10):\n')
        f.write('  OOV rate (eval2000, swbd, from stm): %f %%\n' % oov_rate)

        # word-level (threshold == 15)
        oov_rate = compute_oov_rate(
            speaker_dict_swbd, word_freq15_vocab_file_path)
        f.write('Word (freq15):\n')
        f.write('  OOV rate (eval2000, swbd, from stm): %f %%\n' % oov_rate)

    with open(join(run_root_path, 'config/oov_rate_eval2000_ch_stm.txt'), 'w') as f:

        # word-level (threshold == 1)
        oov_rate = compute_oov_rate(
            speaker_dict_ch, word_freq1_vocab_file_path)
        f.write('Word (freq1):\n')
        f.write('  OOV rate (eval2000, ch, from stm): %f %%\n' % oov_rate)

        # word-level (threshold == 5)
        oov_rate = compute_oov_rate(
            speaker_dict_ch, word_freq5_vocab_file_path)
        f.write('Word (freq5):\n')
        f.write('  OOV rate (eval2000, ch, from stm): %f %%\n' % oov_rate)

        # word-level (threshold == 10)
        oov_rate = compute_oov_rate(
            speaker_dict_ch, word_freq10_vocab_file_path)
        f.write('Word (freq10):\n')
        f.write('  OOV rate (eval2000, ch, from stm): %f %%\n' % oov_rate)

        # word-level (threshold == 15)
        oov_rate = compute_oov_rate(
            speaker_dict_ch, word_freq15_vocab_file_path)
        f.write('Word (freq15):\n')
        f.write('  OOV rate (eval2000, ch, from stm): %f %%\n' % oov_rate)

    return speaker_dict_swbd, speaker_dict_ch


def compute_oov_rate(speaker_dict, vocab_file_path):

    with open(vocab_file_path, 'r') as f:
        vocab_set = set([])
        for line in f:
            word = line.strip()
            vocab_set.add(word)

    oov_count = 0
    word_num = 0
    for speaker_dict, utt_dict in speaker_dict.items():
        for utt_name, utt_info in utt_dict.items():
            transcript = utt_info[2]
            word_list = transcript.split(SPACE)
            word_num += len(word_list)

            for word in word_list:
                if word not in vocab_set:
                    oov_count += 1

    oov_rate = oov_count * 100 / word_num

    return oov_rate
