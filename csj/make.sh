#!/bin/bash

echo '----------------------------------------------------'
echo '|                       CSJ                         |'
echo '----------------------------------------------------'
RUN_ROOT_PATH=`pwd`

# Set the root path to CSJ corpus
CSJ_PATH='/n/sd8/inaguma/corpus/csj/data/'

# Set the path to save dataset
DATASET_SAVE_PATH='/n/sd8/inaguma/corpus/csj/dataset/'

# Set the path to save input features (fbank or MFCC)
INPUT_FEATURE_SAVE_PATH='/n/sd8/inaguma/corpus/csj/fbank/'


echo '--------------------------------'
echo '|      Feature extraction       |'
echo '--------------------------------'
# Set the path to HTK
HTK_PATH='/misc/local/htk-3.4/bin/HCopy'

# Make a mapping file from wav to htk
python make_scp.py $CSJ_PATH $INPUT_FEATURE_SAVE_PATH $RUN_ROOT_PATH
CONFIG_PATH="./config/config_fbank"

# Convert from wav to htk files
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train_subset.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train_fullset.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_dev.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_eval1.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_eval2.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_eval3.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_dialog.scp


echo '--------------------------------'
echo '|             CTC               |'
echo '--------------------------------'
# Make dataset for CTC model
# python make_ctc.py $CSJ_PATH $DATASET_SAVE_PATH $INPUT_FEATURE_SAVE_PATH $RUN_ROOT_PATH


echo '--------------------------------'
echo '|           Attention           |'
echo '--------------------------------'
# Make dataset for Attention-based model
# python make_attention.py $CSJ_PATH $DATASET_SAVE_PATH $INPUT_FEATURE_SAVE_PATH $RUN_ROOT_PATH
