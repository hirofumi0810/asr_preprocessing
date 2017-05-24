#!/bin/bash

HTK_PATH='/misc/local/htk-3.4/bin/HCopy'
CONFIG_PATH="./config"

# convert from sph to wav files
# python sph2wav.py

# make a mapping file from wav to htk
# python make_scp.py

# convert from wav to htk files
$HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_train.scp
# $HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_train_fisher.scp
# $HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_test.scp
# $HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_test_callhome.scp
