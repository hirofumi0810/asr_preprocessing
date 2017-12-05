#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys
import unittest
from glob import glob
from collections import Counter

sys.path.append('../../')
from swbd.input_data import read_audio
from swbd.labels.ldc97s62.character import read_trans as read_trans_swbd
from swbd.labels.fisher.character import read_trans as read_trans_fisher
from utils.measure_time_func import measure_time
from utils.util import mkdir_join

swbd_trans_path = '/n/sd8/inaguma/corpus/swbd/swb_ms98_transcriptions'
fisher_path = '/n/sd8/inaguma/corpus/swbd/data/fisher'
htk_save_path = '/n/sd8/inaguma/corpus/swbd/htk'
wav_save_path = '/n/sd8/inaguma/corpus/swbd/wav'

# Search paths to transcript
label_paths_swbd = []
for trans_path in glob(join(swbd_trans_path, '*/*/*.text')):
    if trans_path.split('.')[0][-5:] == 'trans':
        label_paths_swbd.append(trans_path)
label_paths_swbd = sorted(label_paths_swbd)

label_paths_fisher = []
for trans_path in glob(join(fisher_path, 'data/trans/*/*.txt')):
    label_paths_fisher.append(trans_path)
label_paths_fisher = sorted(label_paths_fisher)


# Search paths to audio files
wav_paths_swbd = [wav_path for wav_path in glob(
    join(wav_save_path, 'swbd/*.wav'))]
htk_paths_swbd = [htk_path for htk_path in glob(
    join(htk_save_path, 'swbd/*.htk'))]

wav_paths_fisher = [wav_path for wav_path in glob(
    join(wav_save_path, 'fisher/*/*.wav'))]
htk_paths_fisher = [htk_path for htk_path in glob(
    join(htk_save_path, 'fisher/*/*.htk'))]

wav_paths = wav_paths_swbd + wav_paths_fisher
htk_paths = htk_paths_swbd + htk_paths_fisher

CONFIG = {
    'feature_type': 'logmelfbank',
    'channels': 40,
    'sampling_rate': 8000,  # NOTE: 8000Hz
    'window': 0.025,
    'slide': 0.01,
    'energy': False,
    'delta': True,
    'deltadelta': True
}


class TestInput2000h(unittest.TestCase):

    def test(self):

        speaker_dict_a, char_set_a, char_capital_set_a, word_count_dict_a = read_trans_fisher(
            label_paths=label_paths_fisher,
            target_speaker='A')
        speaker_dict_b, char_set_b, char_capital_set_b, word_count_dict_b = read_trans_fisher(
            label_paths=label_paths_fisher, target_speaker='B')

        # Meage 2 dictionaries
        speaker_dict_fisher = merge_dicts([speaker_dict_a, speaker_dict_b])
        char_set = char_set_a | char_set_b
        char_capital_set = char_capital_set_a | char_capital_set_b
        word_count_dict_fisher = dict(
            Counter(word_count_dict_a) + Counter(word_count_dict_b))

        self.speaker_dict = read_trans_swbd(
            label_paths=label_paths_swbd,
            run_root_path='../',
            vocab_file_save_path=mkdir_join('../config/vocab_files'),
            save_vocab_file=True,
            speaker_dict_fisher=speaker_dict_fisher,
            char_set=char_set,
            char_capital_set=char_capital_set,
            word_count_dict=word_count_dict_fisher)

        self.check(normalize='global', tool='htk')
        self.check(normalize='speaker', tool='htk')
        self.check(normalize='utterance', tool='htk')

        # self.check(normalize='global', tool='python_speech_features')
        # self.check(normalize='speaker', tool='python_speech_features')
        # self.check(normalize='utterance', tool='python_speech_features')

        # self.check(normalize='global', tool='librosa')
        # self.check(normalize='speaker', tool='librosa')
        # self.check(normalize='utterance', tool='librosa')

    @measure_time
    def check(self, normalize, tool):

        print('=' * 50)
        print('  normalize: %s' % normalize)
        print('  tool: %s' % tool)
        print('=' * 50)

        audio_paths = htk_paths if tool == 'htk' else wav_paths

        read_audio(audio_paths=audio_paths,
                   tool=tool,
                   config=CONFIG,
                   normalize=normalize,
                   speaker_dict=self.speaker_dict,
                   is_training=True)


def merge_dicts(dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


if __name__ == '__main__':
    unittest.main()
