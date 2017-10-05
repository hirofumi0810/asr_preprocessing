#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys
import unittest
from glob import glob

sys.path.append('../../')
from swbd.inputs.input_data import read_audio
from swbd.labels.ldc97s62.character import read_char
from utils.measure_time_func import measure_time

swbd_trans_path = '/n/sd8/inaguma/corpus/swbd/dataset/swb_ms98_transcriptions'
htk_save_path = '/n/sd8/inaguma/corpus/swbd/htk'
wav_save_path = '/n/sd8/inaguma/corpus/swbd/wav'

# Search paths to transcript
label_paths = []
for trans_path in glob(join(swbd_trans_path, '*/*/*.text')):
    if trans_path.split('.')[0][-5:] == 'trans':
        label_paths.append(trans_path)

# Search paths to audio files
wav_paths = [wav_path for wav_path in glob(join(wav_save_path, 'swbd/*.wav'))]
htk_paths = [htk_path for htk_path in glob(join(htk_save_path, 'swbd/*.htk'))]

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


class TestInputLDC97S62(unittest.TestCase):

    def test(self):

        self.speaker_dict = read_char(label_paths=label_paths,
                                      run_root_path='../')

        self.check_feature_extraction(normalize='global', tool='htk')
        self.check_feature_extraction(normalize='speaker', tool='htk')
        self.check_feature_extraction(normalize='utterance', tool='htk')

        # self.check_feature_extraction(
        #     normalize='global', tool='python_speech_features')
        # self.check_feature_extraction(
        #     normalize='speaker', tool='python_speech_features')
        # self.check_feature_extraction(
        #     normalize='utterance', tool='python_speech_features')

        # self.check_feature_extraction(normalize='global', tool='librosa')
        # self.check_feature_extraction(normalize='speaker', tool='librosa')
        # self.check_feature_extraction(normalize='utterance', tool='librosa')

    @measure_time
    def check_feature_extraction(self, normalize, tool):

        print('==================================================')
        print('  normalize: %s' % normalize)
        print('  tool: %s' % tool)
        print('==================================================')

        audio_paths = htk_paths if tool == 'htk' else wav_paths

        train_global_mean, train_global_std = read_audio(
            audio_paths=audio_paths,
            tool=tool,
            config=CONFIG,
            normalize=normalize,
            is_training=True,
            speaker_dict=self.speaker_dict)


if __name__ == '__main__':
    unittest.main()
