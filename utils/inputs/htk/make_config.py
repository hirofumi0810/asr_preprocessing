#! /usr/bin/env python
# -*- coding: utf-8 -*-


def save(audio_file_type, feature_type, channels, config_path,
         sampling_rate=16000, window=0.025, slide=0.01,
         energy=False, delta=True, deltadelta=True):
    """Save a configuration file for HTK.
    Args:
        audio_file_type (string): nist or
        feature_type (string): the type of features, logmelfbank or mfcc or linearmelfbank
        channels (int): the number of frequency channels
        config_path (string): path to save the config file
        sampling_rate (float, optional):
        window (float, optional): window width to extract features
        slide (float, optional): extract features per 'slide'
        energy (bool, optional): if True, add the energy feature
        delta (bool, optional): if True, delta features are also extracted
        deltadelta (bool, optional): if True, double delta features are also extracted
"""
    with open(config_path, 'w') as f:
        f.write('SOURCEFORMAT = %s\n' % audio_file_type.upper())

        # Sampling rate
        if sampling_rate == 16000:
            f.write('SOURCERATE = 625\n')
        elif sampling_rate == 8000:
            f.write('SOURCERATE = 1250\n')

        # Target features
        if feature_type == 'logmelfbank':
            feature_type = 'FBANK'
        elif feature_type == 'mfcc':
            feature_type = 'MFCC'
        elif feature_type == 'linearmelfbank':
            feature_type = 'MELSPEC'
        else:
            raise ValueError('feature_type must be logmelfbank or mfcc or linearmelfbank.')

        if energy:
            feature_type += '_E'
        if delta:
            feature_type += '_D'
        if deltadelta:
            feature_type += '_A'
        f.write('TARGETKIND = %s\n' % feature_type)

        # f.write('DELTAWINDOW = 2')
        # f.write('ACCWINDOW = 2')

        # Extract features per slide
        f.write('TARGETRATE = %.1f\n' % (slide * 10000000))

        f.write('SAVECOMPRESSED = F\n')
        f.write('SAVEWITHCRC = F\n')

        # Window size
        f.write('WINDOWSIZE = %.1f\n' % (window * 10000000))

        f.write('USEHAMMING = T\n')
        f.write('PREEMCOEF = 0.97\n')
        f.write('NUMCHANS = %d\n' % channels)
        f.write('ENORMALISE = F\n')
        f.write('ZMEANSOURCE = T\n')
