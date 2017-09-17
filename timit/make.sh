#!/bin/bash

echo '----------------------------------------------------'
echo '|                      TIMIT                        |'
echo '----------------------------------------------------'

### Set paths
TIMIT_PATH='/n/sd8/inaguma/corpus/timit/original'
DATASET_SAVE_PATH='/n/sd8/inaguma/corpus/timit/dataset'
HTK_SAVE_PATH='/n/sd8/inaguma/corpus/timit/htk'
HTK_PATH='/home/lab5/inaguma/htk-3.4/bin/HCopy'

### Select one tool to extract features (HTK is the fastest)
TOOL='htk'
# TOOL='python_speech_features'
# TOOL='librosa'

### Configuration (Set by yourself)
FEATURE_TYPE='logmelfbank'  # or mfcc
CHANNELS=40
SAMPLING_RATE=16000
WINDOW=0.025
SLIDE=0.01
ENERGY=False
DELTA=True
DELTADELTA=True
# NORMALIZE='global'
NORMALIZE='speaker'
# NORMALIZE='utterance'

##############################
# Don't change from here ↓↓↓
##############################
set -eu

if [ ! -e $TIMIT_PATH ]; then
    echo "TIMIT directory was not found."
    exit 1
fi
if [ ! -e $DATASET_SAVE_PATH ]; then
    mkdir $DATASET_SAVE_PATH
fi
if [ ! -e $HTK_SAVE_PATH ] && [ $TOOL = 'htk']; then
    mkdir $HTK_SAVE_PATH
fi

RUN_ROOT_PATH=`pwd`


echo ============================================================================
echo "                           Feature extraction                             "
echo ============================================================================

if [ $TOOL = 'htk' ]; then
    # Set the path to HTK (optional, set only when using HTK toolkit)
    CONFIG_PATH="./config/config_file"

    # Make a config file to covert from wav to htk file
    python make_config.py --data_path $TIMIT_PATH  \
                          --htk_save_path $HTK_SAVE_PATH \
                          --run_root_path $RUN_ROOT_PATH \
                          --feature_type $FEATURE_TYPE \
                          --channels $CHANNELS \
                          --sampling_rate $SAMPLING_RATE \
                          --window $WINDOW \
                          --slide $SLIDE \
                          --energy $ENERGY \
                          --delta $DELTA \
                          --deltadelta $DELTADELTA \
                          --config_path $CONFIG_PATH

    # Convert from wav to htk files
    $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train.scp
    $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_dev.scp
    $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_test.scp
fi

# Make input features
python make_input.py --data_path $TIMIT_PATH  \
                     --dataset_save_path $DATASET_SAVE_PATH \
                     --run_root_path $RUN_ROOT_PATH \
                     --tool $TOOL \
                     --htk_save_path $HTK_SAVE_PATH \
                     --feature_type $FEATURE_TYPE \
                     --channels $CHANNELS \
                     --sampling_rate $SAMPLING_RATE \
                     --window $WINDOW \
                     --slide $SLIDE \
                     --energy $ENERGY \
                     --delta $DELTA \
                     --deltadelta $DELTADELTA \
                     --normalize $NORMALIZE


echo ============================================================================
echo "                         Process transcriptions                           "
echo ============================================================================

# Make transcripts for the End-to-End models (CTC and Attention)
python make_label_end2end.py --data_path $TIMIT_PATH  \
                             --dataset_save_path $DATASET_SAVE_PATH \
                             --run_root_path $RUN_ROOT_PATH


echo 'Successfully completed!!!'
