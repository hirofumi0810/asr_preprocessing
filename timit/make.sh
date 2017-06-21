#!/bin/bash

echo '----------------------------------------------------'
echo '|                      TIMIT                        |'
echo '----------------------------------------------------'

# Set the root path to TIMIT corpus
TIMIT_PATH='/n/sd8/inaguma/corpus/timit/original/'

# Set the path to save dataset
DATASET_SAVE_PATH='/n/sd8/inaguma/corpus/timit/dataset/'

# Set the path to save input features (fbank or MFCC)
INPUT_FEATURE_SAVE_PATH='/n/sd8/inaguma/corpus/timit/fbank/'

RUN_ROOT_PATH=`pwd`

# Feature extraction
echo '--------------------------'
echo '|   Feature extraction    |'
echo '--------------------------'
# Set the path to HTK
HTK_PATH='/misc/local/htk-3.4/bin/HCopy'

# Make a mapping file from wav to htk
python make_scp.py $TIMIT_PATH $INPUT_FEATURE_SAVE_PATH $RUN_ROOT_PATH
CONFIG_PATH="./config/config_fbank"

# Convert from wav to htk files
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_dev.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_test.scp


# Make dataset for CTC
echo '--------------------------'
echo '|          CTC            |'
echo '--------------------------'
python make_ctc.py $TIMIT_PATH $DATASET_SAVE_PATH $INPUT_FEATURE_SAVE_PATH $RUN_ROOT_PATH


# Make dataset for Attention Mechanism
echo '--------------------------'
echo '|        Attention        |'
echo '--------------------------'
python make_attention.py $TIMIT_PATH $DATASET_SAVE_PATH $INPUT_FEATURE_SAVE_PATH $RUN_ROOT_PATH
