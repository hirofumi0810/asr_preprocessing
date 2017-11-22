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
from swbd.input_data import read_audio
from swbd.labels.ldc97s62.character import read_trans
from utils.measure_time_func import measure_time
from utils.util import mkdir_join

swbd_trans_path = '/n/sd8/inaguma/corpus/swbd/dataset/swb_ms98_transcriptions'
htk_save_path = '/n/sd8/inaguma/corpus/swbd/htk'
wav_save_path = '/n/sd8/inaguma/corpus/swbd/wav'

# Search paths to transcript
label_paths = []
for trans_path in glob(join(swbd_trans_path, '*/*/*.text')):
    if trans_path.split('.')[0][-5:] == 'trans':
        label_paths.append(trans_path)
label_paths = sorted(label_paths)

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

        self.speaker_dict = read_trans(
            label_paths=label_paths,
            run_root_path='../',
            vocab_file_save_path=mkdir_join('../config/vocab_files'),
            save_vocab_file=False)

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

        global_mean, global_std, _ = read_audio(
            audio_paths=audio_paths,
            tool=tool,
            config=CONFIG,
            normalize=normalize,
            speaker_dict=self.speaker_dict,
            is_training=True)


if __name__ == '__main__':
    unittest.main()
