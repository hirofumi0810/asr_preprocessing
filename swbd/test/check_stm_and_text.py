#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import unittest


class TestCTCLabelFisherChar(unittest.TestCase):

    def test(self):

        self.trans_file_path_swbd_stm = '../labels/eval2000/trans_swbd_stm_fixed.txt'
        self.trans_file_path_swbd_text = '../labels/eval2000/trans_swbd_text_fixed.txt'
        self.trans_file_path_ch_stm = '../labels/eval2000/trans_ch_stm_fixed.txt'
        self.trnas_file_path_ch_text = '../labels/eval2000/trans_ch_text_fixed.txt'

        self.check_eval2000_swbd()
        # self.check_eval2000_ch()

    def check_eval2000_swbd(self):
        with open(self.trans_file_path_swbd_stm, 'r') as f_stm:
            with open(self.trans_file_path_swbd_text, 'r') as f_text:
                for line_stm, line_text in zip(f_stm, f_text):
                    line_stm = line_stm.strip()
                    line_text = line_text.strip()

                    # Remove hesitation, noise, hyphen
                    line_stm = line_stm.replace(
                        '%hesitation', '').replace('NZ', '').replace('-', '')
                    line_text = line_text.replace(
                        '%hesitation', '').replace('NZ', '').replace('-', '')

                    # Remove consecutive spaces
                    while '__' in line_stm:
                        line_stm = re.sub(r'[_]+', '_', line_stm)
                    while '__' in line_text:
                        line_text = re.sub(r'[_]+', '_', line_text)

                    # Remove first and last space
                    if line_stm[0] == '_':
                        line_stm = line_stm[1:]
                    if line_stm[-1] == '_':
                        line_stm = line_stm[:-1]
                    if line_text[0] == '_':
                        line_text = line_text[1:]
                    if line_text[-1] == '_':
                        line_text = line_text[:-1]

                    if line_stm != line_text:
                        print('stm: %s' % line_stm)
                        print('txt: %s' % line_text)
                        print('=' * 30)

    def check_eval2000_ch(self):
        with open(self.trans_file_path_ch_stm, 'r') as f_stm:
            with open(self.trans_file_path_ch_text, 'r') as f_text:
                for line_stm, line_text in zip(f_stm, f_text):
                    line_stm = line_stm.strip()
                    line_text = line_text.strip()

                    # Remove hesitation, noise, hyphen
                    line_stm = line_stm.replace(
                        '%hesitation', '').replace('NZ', '').replace('-', '')
                    line_text = line_text.replace(
                        '%hesitation', '').replace('NZ', '').replace('-', '')

                    # Remove consecutive spaces
                    while '__' in line_stm:
                        line_stm = re.sub(r'[_]+', '_', line_stm)
                    while '__' in line_text:
                        line_text = re.sub(r'[_]+', '_', line_text)

                    # Remove first and last space
                    if line_stm[0] == '_':
                        line_stm = line_stm[1:]
                    if line_stm[-1] == '_':
                        line_stm = line_stm[:-1]
                    if line_text[0] == '_':
                        line_text = line_text[1:]
                    if line_text[-1] == '_':
                        line_text = line_text[:-1]

                    if line_stm != line_text:
                        print('stm: %s' % line_stm)
                        print('txt: %s' % line_text)
                        print('==========')


if __name__ == '__main__':
    unittest.main()
