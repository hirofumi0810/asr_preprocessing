#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys
import unittest
from glob import glob
import functools

sys.path.append('../../')
from swbd.input_data import read_audio
from swbd.labels.fisher.character import read_trans
from utils.measure_time_func import measure_time
from utils.util import mkdir_join

fisher_path = '/n/sd8/inaguma/corpus/swbd/data/fisher'
htk_save_path = '/n/sd8/inaguma/corpus/swbd/htk'
wav_save_path = '/n/sd8/inaguma/corpus/swbd/wav'

# Search paths to transcript
label_paths = []
for trans_path in glob(join(fisher_path, 'data/trans/*/*.txt')):
    label_paths.append(trans_path)
label_paths = sorted(label_paths)

# Search paths to audio files
wav_paths = [wav_path for wav_path in glob(
    join(wav_save_path, 'fisher/*/*.wav'))]
htk_paths = [htk_path for htk_path in glob(
    join(htk_save_path, 'fisher/*/*.htk'))]

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


class TestInputFisher(unittest.TestCase):

    def test(self):

        speaker_dict_a = read_trans(
            label_paths=label_paths, target_speaker='A')
        speaker_dict_b = read_trans(
            label_paths=label_paths, target_speaker='B')

        # Merge 2 dictionaries
        self.speaker_dict = functools.reduce(lambda first, second: dict(first, second),
                                             speaker_dict_a, speaker_dict_b)

        # TODO: Merge statistics of ldc97s62 and fisher

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
