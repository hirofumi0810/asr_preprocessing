#!/bin/bash

echo ============================================================================
echo "                                   CSJ                                     "
echo ============================================================================

# Set paths
CSJ_PATH='/n/sd8/inaguma/corpus/csj/data'
DATASET_SAVE_PATH='/n/sd8/inaguma/corpus/csj/dataset'
HTK_SAVE_PATH='/n/sd8/inaguma/corpus/csj/htk'
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
ENERGY=0
DELTA=1
DELTADELTA=1
# NORMALIZE='global'
NORMALIZE='speaker'
# NORMALIZE='utterance'


########################################
# ↓↓↓ Don't change from here ↓↓↓
########################################
set -eu

if [ ! -e $CSJ_PATH ]; then
    echo "CSJ directory was not found."
    exit 1
fi
if [ ! -e $DATASET_SAVE_PATH ]; then
    mkdir $DATASET_SAVE_PATH
fi
if [ ! -e $HTK_SAVE_PATH ] && [ $TOOL = 'htk' ]; then
    mkdir $HTK_SAVE_PATH
fi

RUN_ROOT_PATH=`pwd`


echo ============================================================================
echo "                   Feature extraction by HTK toolkit                      "
echo ============================================================================
if [ $TOOL = 'htk' ]; then
    # Set the path to HTK (optional, set only when using HTK toolkit)
    CONFIG_PATH="./config/config_file"

    # Make a config file to covert from wav to htk file
    python make_config.py \
        --data_path $CSJ_PATH \
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
    $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train_subset.scp
    $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train_fullset.scp
    $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_dev.scp
    $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_eval1.scp
    $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_eval2.scp
    $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_eval3.scp
fi


echo ============================================================================
echo "                                  Main                                    "
echo ============================================================================
python main.py \
    --data_path $CSJ_PATH \
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


echo 'Successfully completed!!!'
