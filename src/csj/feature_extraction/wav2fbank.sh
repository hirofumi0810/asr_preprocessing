#!/bin/bash

HTK_PATH='/misc/local/htk-3.4/bin/HCopy'
CONFIG_PATH="./config"

# make a mapping file from wav to htk
python make_scp.py

# convert from wav to htk files
# $HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_train.scp
# $HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_train_all.scp
# $HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_eval1.scp
# $HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_eval2.scp
# $HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_eval3.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_dialog_train.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_dialog_dev.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S wav2fbank_dialog_test.scp
