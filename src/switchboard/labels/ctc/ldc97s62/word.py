#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make word-level labels for CTC model (LDC97S62 corpus)."""

import os
import re
import numpy as np
import time
from progressbar import ProgressBar

from prepare_path import Prepare


# TODO:
# 笑いのマスクを作る
# word embedding用の特徴量

def read_word(label_paths, save_path=None):
    """Read word segmentation files & save as npy files.
    Args:
        label_paths: list of paths to label files
        save_path: path to save labels. If None, don't save labels
    Returns:
        speaker_dict: dictionary of speakers
            key => speaker index
            value => dictionary of utterance infomation of each speaker
                key => utterance index
                value => [start_time, end_time, transcript]
    """
    print('===> Reading word segmentation...')
    p, i = ProgressBar(max_value=len(label_paths)), 0
    speaker_dict = {}
    word_set, char_set = set([]), set([])
    for label_path in p(label_paths):
        utterance_dict = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                speaker_index = line[0].split('-')[0]
                utt_index = line[0].split('-')[-1]
                start_time = float(line[1])
                end_time = float(line[2])
                # convert to lowercase
                word_original = ' '.join(line[3:]).lower()

                # clean transcript
                transcript = fix_transcript(word_original, speaker_index)

                # skip silence
                if transcript == '':
                    continue

                # remove head & last space
                # if transcript[0] == ' ':
                #     transcript = transcript[1:]
                # if transcript[-1] == ' ':
                #     transcript = transcript[:-1]

                # convert from character to phone
                # phone_list = ['SIL']
                # keys = pronounce_dict.keys()
                # for word in transcript.split(' '):
                #     if word in keys:
                #         phone_list.append(pronounce_dict[word])
                #         phone_list.append('SIL')
                #     else:
                #         print(transcript.split(' '))

                # convert to phone list where each element is phone (remove ' ')
                # phone_seq = ' '.join(phone_list)
                # phone_list = phone_seq.split(' ')

                for char in list(word):
                    if char == '':
                        print(list(word))
                    char_set.add(char)

            #     utterance_dict[utt_index] = [start_time, end_time, phone_list]
            # speaker_dict[speaker_index] = utterance_dict

        p.update(i + 1)
        i += 1
        time.sleep(0.01)

    char_set = sorted(list(char_set))
    print(len(char_set))

    # make mapping file (from phone to number)
    # if not os.path.isfile(os.path.abspath('../phone2num.txt')):
    #     with open(os.path.abspath('../phone2num.txt'), 'w') as f:
    #         index = 0
    #         for phone in phones:
    #             f.write('%s  %s\n' % (phone, str(index)))
    #             index += 1
    #
    # if save_path is not None:
    #     print('Saving target labels...')
    #     p_save = ProgressBar(max_value=len(label_paths))
    #     i_save = 0
    #     for speaker_index, utterance_dict in p_save(speaker_dict.items()):
    #         save_path_speaker = mkdir(os.path.join(save_path, speaker_index))
    #         for utt_index,  utt_info in utterance_dict.items():
    #             start_time, end_time, phone_list = utt_info
    #             save_file_name = speaker_index + '_' + utt_index + '.npy'
    #
    #             # convert from phone to number
    #             phone_index_list = phone2num(phone_list)
    #
    #             # save as npy file
    #             np.save(os.path.join(save_path_speaker, save_file_name), phone_index_list)
    #
    #         p_save.update(i_save + 1)
    #         i_save += 1
    #         time.sleep(0.01)

    return speaker_dict


def fix_transcript(word, speaker_index):

    # remove <b_aside>, <e_aside>, [silence]
    word = re.sub(r'\<b_aside\>', '', word)
    word = re.sub(r'\<e_aside\>', '', word)
    word = re.sub(r'\[silence\]', '', word)

    # exception (sw3845A)
    # if ' laughter' in transcript:
    #     transcript = re.sub(r' laughter', ' [laughter]', transcript)

    ####################
    # exception
    ####################
    # remove double space
    # transcript = re.sub('  ', ' ', transcript)

    return word


def test():
    prep = Prepare()
    words = prep.label_train(label_type='word')

    read_word(label_paths=words)


if __name__ == '__main__':
    test()
