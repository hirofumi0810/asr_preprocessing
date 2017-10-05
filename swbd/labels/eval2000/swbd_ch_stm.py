#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import re

SPACE = '_'


def read_stm(stm_path, pem_path, run_root_path, save_path=None):

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

    print('===> Reading target labels...')
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
                utt_index = 0

            start_time = float(line.split(' ')[3])
            end_time = float(line.split(' ')[4])
            transcript = ' '.join(line.split(' ')[6:]).lower()
            transcript_original = transcript

            # Error check of segmentation time
            if segmentation_info[speaker][utt_index][0] != start_time:
                raise ValueError
            if segmentation_info[speaker][utt_index][1] != end_time:
                raise ValueError

            # Clean transcript
            # Remove <b_aside> and <e_aside>
            transcript = re.sub(r'\<b_aside\>', '', transcript)
            transcript = re.sub(r'\<e_aside\>', '', transcript)

            # Convert space to "_"
            transcript = re.sub(r'\s', SPACE, transcript)

            # Write to text files
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

            # for debug
            # print(transcript)

            utt_index += 1
            speaker_pre = speaker

    fp_swbd_original.close()
    fp_swbd_fixed.close()
    fp_ch_original.close()
    fp_ch_fixed.close()

    if save_path is not None:
        # save target labels
        print('===> Saving target labels...')
