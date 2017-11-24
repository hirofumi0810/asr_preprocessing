#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make target labels from txt files (eval2000, swbd)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import re
from tqdm import tqdm
from collections import OrderedDict

from swbd.labels.eval2000.fix_trans_text import fix_transcript
from swbd.labels.eval2000.stm import compute_oov_rate

SPACE = '_'


def read_text(label_paths, pem_path, glm_path, run_root_path,
              data_size='300h'):
    """Read transcripts (.txt) & save files (.npy).
    Args:
        label_paths (list): list of paths to label files
        pem_path (string): path to the segmentation file
        glm_path (string): path to the GLM file
        run_root_path (string): absolute path of make.sh
        data_size (string): 300h or 2000h
    Returns:
        speaker_dict: dictionary of speakers
            key (string) => speaker
            value (dict) => dictionary of utterance infomation of each speaker
                key (string) => utterance index
                value (list) => [start_frame, end_frame, transcript * 6]
    """
    print('=====> Processing target labels...')
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

    speaker_dict = OrderedDict()
    char_set = set([])
    fp_original = open(join(run_root_path, 'labels',
                            'eval2000', 'trans_swbd_text_original.txt'), 'w')
    fp_fixed = open(join(run_root_path, 'labels',
                         'eval2000', 'trans_swbd_text_fixed.txt'), 'w')
    for label_path in tqdm(label_paths):
        utterance_dict = OrderedDict()
        flag_speaker_b = False
        utt_index = 0
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                start_time = float(line[0])
                end_time = float(line[1])

                if line[2] == 'B:' and not flag_speaker_b:
                    # The last utterance of speaker A
                    flag_speaker_b = True
                    speaker_a = basename(label_path).split('.')[0]
                    # Fix speaker name
                    speaker_a = speaker_a.replace('sw_', 'sw') + '-A'
                    speaker_dict[speaker_a] = utterance_dict
                    # reset
                    utterance_dict = OrderedDict()
                    utt_index = 0

                speaker = basename(label_path).split(
                    '.')[0].replace('sw_', 'sw') + '-' + line[2][0]

                # Convert to lowercase
                transcript = ' '.join(line[3:]).lower()
                transcript_original = transcript

                # Clean transcript
                transcript = fix_transcript(transcript, speaker)

                # Skip silence
                if transcript in ['', ' ']:
                    continue

                # Error check of segmentation time
                assert segmentation_info[speaker][utt_index][0] == start_time
                assert segmentation_info[speaker][utt_index][1] == end_time

                # Convert space to "_"
                transcript = re.sub(r'\s', SPACE, transcript)

                # Write to text files for debug
                fp_original.write('%s  %d  %.2f  %.2f  %s\n' %
                                  (speaker, utt_index, start_time, end_time, transcript_original))
                fp_fixed.write('%s  %d  %.2f  %.2f  %s\n' %
                               (speaker, utt_index, start_time, end_time, transcript))

                for char in list(transcript.lower()):
                    char_set.add(char)

                # for debug
                # print(transcript)

                start_frame = int(float(line[0]) * 100 + 0.5)
                end_frame = int(float(line[1]) * 100 + 0.5)
                utterance_dict[str(utt_index).zfill(4)] = [
                    start_frame, end_frame, transcript, transcript,
                    transcript, transcript, transcript, transcript]
                utt_index += 1
            speaker_dict[speaker] = utterance_dict

    fp_original.close()
    fp_fixed.close()

    # for debug
    # print(sorted(list(char_set)))

    word_freq1_vocab_file_path = join(
        run_root_path, 'config/vocab_files/word_freq1_' + data_size + '.txt')
    word_freq5_vocab_file_path = join(
        run_root_path, 'config/vocab_files/word_freq5_' + data_size + '.txt')
    word_freq10_vocab_file_path = join(
        run_root_path, 'config/vocab_files/word_freq10_' + data_size + '.txt')
    word_freq15_vocab_file_path = join(
        run_root_path, 'config/vocab_files/word_freq15_' + data_size + '.txt')

    # Compute OOV rate
    with open(join(run_root_path, 'config/oov_rate_eval2000_swbd_txt_' + data_size + '.txt'), 'w') as f:

        # word-level (threshold == 1)
        oov_rate = compute_oov_rate(
            speaker_dict, word_freq1_vocab_file_path)
        f.write('Word (freq1):\n')
        f.write('  OOV rate (eval2000, swbd, from txt): %f %%\n' % oov_rate)

        # word-level (threshold == 5)
        oov_rate = compute_oov_rate(
            speaker_dict, word_freq5_vocab_file_path)
        f.write('Word (freq5):\n')
        f.write('  OOV rate (eval2000, swbd, from txt): %f %%\n' % oov_rate)

        # word-level (threshold == 10)
        oov_rate = compute_oov_rate(
            speaker_dict, word_freq10_vocab_file_path)
        f.write('Word (freq10):\n')
        f.write('  OOV rate (eval2000, swbd, from txt): %f %%\n' % oov_rate)

        # word-level (threshold == 15)
        oov_rate = compute_oov_rate(
            speaker_dict, word_freq15_vocab_file_path)
        f.write('Word (freq15):\n')
        f.write('  OOV rate (eval2000, swbd, from txt): %f %%\n' % oov_rate)

    return speaker_dict
