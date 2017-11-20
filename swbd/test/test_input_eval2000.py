#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys
import unittest
from glob import glob
import numpy as np

sys.path.append('../../')
from swbd.input_data import read_audio
from swbd.labels.eval2000.stm import read_stm
from utils.measure_time_func import measure_time
from utils.util import mkdir_join

eval2000_trans_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43'
eval2000_stm_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43/reference/hub5e00.english.000405.stm'
eval2000_pem_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002S09/english/hub5e_00.pem'
eval2000_glm_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43/reference/en20000405_hub5.glm'
htk_save_path = '/n/sd8/inaguma/corpus/swbd/htk'
wav_save_path = '/n/sd8/inaguma/corpus/swbd/wav'

# Search paths to audio files
wav_paths_swbd = [wav_path for wav_path in glob(
    join(wav_save_path, 'eval2000/swbd/*.wav'))]
htk_paths_swbd = [htk_path for htk_path in glob(
    join(htk_save_path, 'eval2000/swbd/*.htk'))]
wav_paths_ch = [wav_path for wav_path in glob(
    join(wav_save_path, 'eval2000/callhome/*.wav'))]
htk_paths_ch = [htk_path for htk_path in glob(
    join(htk_save_path, 'eval2000/callhome/*.htk'))]

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


class TestInputEval2000(unittest.TestCase):

    def test(self):

        self.speaker_dict_swbd, self.speaker_dict_ch = read_stm(
            stm_path=eval2000_stm_path,
            pem_path=eval2000_pem_path,
            glm_path=eval2000_glm_path,
            run_root_path='../')

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

        # Load statistics over train dataset
        global_mean = np.load(
            '/n/sd8/inaguma/dataset/300h/train/train_mean.npy')
        global_std = np.load('dataset/300h/train/train_std.npy')

        # swbd
        audio_paths_swbd = htk_paths_swbd if tool == 'htk' else wav_paths_swbd
        read_audio(audio_paths=audio_paths_swbd,
                   tool=tool,
                   config=CONFIG,
                   speaker_dict=self.speaker_dict_swbd,
                   normalize=normalize,
                   is_training=False,
                   global_mean=None,
                   global_std=None)

        # ch
        audio_paths_ch = htk_paths_ch if tool == 'htk' else wav_paths_ch
        read_audio(audio_paths=audio_paths_ch,
                   tool=tool,
                   config=CONFIG,
                   speaker_dict=self.speaker_dict_ch,
                   normalize=normalize,
                   is_training=False,
                   global_mean=None,
                   global_std=None)


if __name__ == '__main__':
    unittest.main()
