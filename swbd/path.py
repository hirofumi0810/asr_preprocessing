#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
from glob import glob


class Path(object):
    """Prepare for making dataset.
    Args:
        swbd_audio_path (string): path to audio files of Switchboard corpus
        swbd_trans_path (string): path to transcipt files of Switchboard corpus
        eval2000_audio_path (string): path to audio files of eval2000 corpus
        eval2000_trans_path (string): path to trans files of eval2000 corpus
        fisher_path (string): path to Fisher corpus
        run_root_path (string): path to ./make.sh
    """

    def __init__(self, swbd_audio_path, swbd_trans_path, eval2000_audio_path,
                 eval2000_trans_path, run_root_path, fisher_path=None,
                 wav_save_path=None, htk_save_path=None):

        self.swbd_audio_path = swbd_audio_path
        self.swbd_trans_path = swbd_trans_path
        self.eval2000_audio_path = eval2000_audio_path
        self.eval2000_trans_path = eval2000_trans_path
        self.fisher_path = fisher_path
        self.wav_save_path = wav_save_path
        self.htk_save_path = htk_save_path
        self.pem_path = None
        # NOTE: hub5e_00.pem file is a segmentation file
        self.stm_path = None
        # NOTE* stm is a transcription file of swbd and callhome

        # Absolute path to this directory
        self.run_root_path = run_root_path

        self.__make()

    def __make(self):

        self._sph_paths = {
            'swbd': [],
            'fisher': [],
            'eval2000_swbd': [],
            'eval2000_ch': []
        }

        self._trans_paths = {
            'swbd': [],
            'fisher': [],
            'eval2000_swbd': [],
            'eval2000_ch': []
        }

        self._word_paths = {
            'swbd': [],
            'fisher': [],
            'eval2000_swbd': [],
            'eval2000_ch': []
        }

        ####################
        # train (LDC97S62)
        ####################
        if self.swbd_audio_path is not None:
            self.word_dict_path = join(
                self.swbd_audio_path, 'sw-ms98-dict.text')
            for sph_path in glob(join(self.swbd_audio_path, '*/data/*.sph')):
                self._sph_paths['swbd'].append(sph_path)

        if self.swbd_trans_path is not None:
            for trans_path in glob(join(self.swbd_trans_path,
                                        '*/*/*.text')):
                if trans_path.split('.')[0][-4:] == 'word':
                    self._word_paths['swbd'].append(trans_path)
                elif trans_path.split('.')[0][-5:] == 'trans':
                    self._trans_paths['swbd'].append(trans_path)

        ####################
        # train (Fisher)
        ####################
        if self.fisher_path is not None:
            for sph_path in glob(join(self.fisher_path, 'audio/*/*.sph')):
                self._sph_paths['fisher'].append(sph_path)

            for trans_path in glob(join(self.fisher_path, 'data/trans/*/*.txt')):
                self._trans_paths['fisher'].append(trans_path)

        ########################################
        # test (eval2000)
        ########################################
        if self.eval2000_audio_path is not None:
            for file_path in glob(join(self.eval2000_audio_path, 'english/*')):
                file_name = basename(file_path)
                if file_name[:2] == 'sw':
                    self._sph_paths['eval2000_swbd'].append(file_path)
                elif file_name[:2] == 'en':
                    self._sph_paths['eval2000_ch'].append(file_path)
                elif file_name == 'hub5e_00.pem':
                    self.pem_path = file_path

        if self.eval2000_trans_path is not None:
            for file_path in glob(join(self.eval2000_trans_path, 'reference/english/*')):
                file_name = basename(file_path)
                if file_name[:2] == 'sw':
                    self._trans_paths['eval2000_swbd'].append(file_path)
                elif file_name[:2] == 'en':
                    self._trans_paths['eval2000_ch'].append(file_path)
            self.stm_path = join(self.eval2000_trans_path,
                                 'reference', 'hub5e00.english.000405.stm')
            self.glm_path = join(self.eval2000_trans_path,
                                 'reference', 'en20000405_hub5.glm')

    def sph(self, corpus):
        """Get paths to sph files of training data.
        Args:
            corpus (string): swbd or fisher or eval2000_swbd or
                eval2000_ch
        Returns:
            paths to sph files
        """
        return sorted(self._sph_paths[corpus])

    def wav(self, corpus):
        """Get paths to wav files of training data.
        Args:
            corpus (string): swbd or fisher or eval2000_swbd or
                eval2000_ch
        Returns:
            paths to wav files
        """
        if self.wav_save_path is None:
            raise ValueError('Set path to wav files.')

        if corpus == 'swbd':
            return [p for p in glob(join(self.wav_save_path, 'swbd/*.wav'))]
            # ex.) wav/swbd/
        elif corpus == 'fisher':
            if self.fisher_path is None:
                raise ValueError('Set path to fisher corpus.')

            return [p for p in glob(join(self.wav_save_path, 'fisher/*/*.wav'))]
            # ex.) wav/fisher/speaker/*.wav
        elif corpus == 'eval2000_swbd':
            return [p for p in glob(join(self.wav_save_path, 'eval2000/swbd/*.wav'))]
            # ex.) wav/eval2000/swbd/*.wav
        elif corpus == 'eval2000_ch':
            return [p for p in glob(join(self.wav_save_path, 'eval2000/callhome/*.wav'))]
            # ex.) wav/eval2000/callhome/*.wav
        else:
            raise TypeError

    def htk(self, corpus):
        """Get paths to htk files of training data.
        Args:
            corpus (string): swbd or fisher or eval2000_swbd or
                eval2000_ch
        Returns:
            paths to htk files
        """
        if self.htk_save_path is None:
            raise ValueError('Set path to htk files.')

        if corpus == 'swbd':
            return [p for p in glob(join(self.htk_save_path, 'swbd/*.htk'))]
            # ex.) htk/swbd/
        elif corpus == 'fisher':
            if self.fisher_path is None:
                raise ValueError('Set path to fisher corpus.')

            return [p for p in glob(join(self.htk_save_path, 'fisher/*/*.htk'))]
            # ex.) htk/fisher/speaker/*.htk
        elif corpus == 'eval2000_swbd':
            return [p for p in glob(join(self.htk_save_path, 'eval2000/swbd/*.htk'))]
            # ex.) htk/eval2000/swbd/*.htk
        elif corpus == 'eval2000_ch':
            return [p for p in glob(join(self.htk_save_path, 'eval2000/callhome/*.htk'))]
            # ex.) htk/eval2000/callhome/*.htk
        else:
            raise TypeError

    def trans(self, corpus):
        """Get paths to transcription files of the training data.
        Args:
            corpus (string): swbd or fisher or eval2000_swbd or
                eval2000_ch
        Returns:
            paths: paths to transcription files
        """
        return sorted(self._trans_paths[corpus])

    def word(self, corpus):
        """Get paths to word boundary files of the training data.
        Args:
            corpus (string): swbd
        Returns:
            paths: paths to transcription files
        """
        assert corpus == 'swbd'
        return sorted(self._word_paths['swbd'])


if __name__ == '__main__':

    path = Path(
        swbd_audio_path='/n/sd8/inaguma/corpus/swbd/data/LDC97S62',
        swbd_trans_path='/n/sd8/inaguma/corpus/swbd/swb_ms98_transcriptions',
        fisher_path='/n/sd8/inaguma/corpus/swbd/data/fisher',
        eval2000_audio_path='/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002S09',
        eval2000_trans_path='/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43',
        wav_save_path='/n/sd8/inaguma/corpus/swbd/wav',
        htk_save_path='/n/sd8/inaguma/corpus/swbd/htk',
        run_root_path='./')

    print('===== LDC97S62 ====')
    print(len(path.sph(corpus='swbd')))  # 2ch
    print(len(path.wav(corpus='swbd')))
    print(len(path.htk(corpus='swbd')))
    print(len(path.trans(corpus='swbd')))
    print(len(path.word(corpus='swbd')))

    print('==== Fisher ====')
    print(len(path.sph(corpus='fisher')))
    print(len(path.wav(corpus='fisher')))
    print(len(path.htk(corpus='fisher')))
    print(len(path.trans(corpus='fisher')))

    print('==== eval2000 (SWB) ====')
    print(len(path.sph(corpus='eval2000_swbd')))
    print(len(path.wav(corpus='eval2000_swbd')))
    print(len(path.htk(corpus='eval2000_swbd')))
    print(len(path.trans(corpus='eval2000_swbd')))

    print('==== eval2000 (CH) ====')
    print(len(path.sph(corpus='eval2000_ch')))
    print(len(path.wav(corpus='eval2000_ch')))
    print(len(path.htk(corpus='eval2000_ch')))
    print(len(path.trans(corpus='eval2000_ch')))
