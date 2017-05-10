#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from struct import unpack


def read(htk_path):
    """Read each HTK file.
    Args:
        htk_path: path to a HTK file
    Returns:
        input_data: np.ndarray, (frame_num, feature_dim)
    """
    with open(htk_path, "rb") as fh:
        spam = fh.read(12)
        frame_num, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
        # print(frame_num)  # frame num
        # print(sampPeriod)  # 10ms
        # print(sampSize)  # feature dim * 4 (byte)
        # print(parmKind)
        veclen = int(sampSize / 4)
        fh.seek(12, 0)
        input_data = np.fromfile(fh, 'f')
        # input_data = input_data.reshape(int(len(input_data) / veclen), veclen)
        input_data = input_data.reshape(-1, veclen)
        input_data.byteswap(True)

    return input_data
