#!/bin/bash

echo '----------------------------------------------------'
echo '|                   Librispeech                     |'
echo '----------------------------------------------------'
RUN_ROOT_PATH=`pwd`

# Set the root path to LIBRI corpus
LIBRI_PATH='/n/sd8/inaguma/corpus/librispeech/data/'

# Set the path to save dataset
DATASET_SAVE_PATH='/n/sd8/inaguma/corpus/librispeech/dataset/'

# Set the path to save input features (fbank or MFCC)
INPUT_FEATURE_SAVE_PATH='/n/sd8/inaguma/corpus/librispeech/fbank/'
WAV_SAVE_PATH='/n/sd8/inaguma/corpus/librispeech/wav/'


echo '--------------------------------'
echo '|   Convert from FLAC to WAV    |'
echo '--------------------------------'
Convert from flac to wav files (remove flac files)
flac_paths=$(find $LIBRI_PATH -type f)
for flac_path in $flac_paths ; do
    dir_path=$(dirname $flac_path)
    file_name=$(basename $flac_path)
    base=${file_name%.*}
    extention=${file_name##*.}
    if [ $extention = "flac" ]; then
        wav_path=$dir_path"/"$base".wav"
        sox $flac_path $wav_path
        rm -f $flac_path
    fi
done


echo '--------------------------------'
echo '|      Feature extraction       |'
echo '--------------------------------'
# Set the path to HTK
HTK_PATH='/misc/local/htk-3.4/bin/HCopy'

# Make a mapping file from wav to htk (wav -> HTK)
python make_scp.py $LIBRI_PATH $INPUT_FEATURE_SAVE_PATH $RUN_ROOT_PATH
CONFIG_PATH="./config/config_fbank"

# Convert from wav to htk files
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train_clean100.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train_clean360.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train_other500.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_dev_clean.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_dev_other.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_test_clean.scp
$HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_test_other.scp


echo '--------------------------------'
echo '|             CTC               |'
echo '--------------------------------'
# Make dataset for CTC model
# python make_ctc.py $LIBRI_PATH $DATASET_SAVE_PATH $INPUT_FEATURE_SAVE_PATH $RUN_ROOT_PATH


echo '--------------------------------'
echo '|           Attention           |'
echo '--------------------------------'
# Make dataset for Attention-based model
# python make_attention.py $LIBRI_PATH $DATASET_SAVE_PATH $INPUT_FEATURE_SAVE_PATH $RUN_ROOT_PATH
