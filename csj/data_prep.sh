#!/bin/bash

echo ============================================================================
echo "                                   CSJ                                     "
echo ============================================================================

# Set paths
CSJ_PATH='/n/sd8/inaguma/corpus/csj/data'
DATASET_ROOT_PATH='/n/sd8/inaguma/corpus/csj'
HCOPY_PATH='/home/lab5/inaguma/htk-3.4/bin/HCopy'

### Select one tool to extract features (HTK is the fastest)
TOOL='htk'
# TOOL='python_speech_features'
# TOOL='librosa'
# TOOL='kaldi'  # under implementation

### Configuration (Set by yourself)
FEATURE_TYPE='fbank'  # (logmel) fbank or mfcc
CHANNELS=40
WINDOW=0.025
SLIDE=0.01
ENERGY=1
DELTA=1
DELTADELTA=1
# NORMALIZE='global'
NORMALIZE='speaker'
# NORMALIZE='utterance'
# NORMALIZE='no'

# SAVE_FORMAT='numpy'
SAVE_FORMAT='htk'
# SAVE_FORMAT='wav'
# NOTE: normalization will not be conducted in case of wav

### Data size
# subset (about 240h)
subset=1

# fullset (about 586h)
fullset=1


########################################
# ↓↓↓ Don't change from here ↓↓↓
########################################
set -eu
DATASET_SAVE_PATH=$DATASET_ROOT_PATH/dataset
HTK_SAVE_PATH=$DATASET_ROOT_PATH/htk
FEATURE_SAVE_PATH=$DATASET_ROOT_PATH/feature
RUN_ROOT_PATH=`pwd`

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

declare -A file_number
file_number["train_subset"]=986
file_number["train_fullset"]=3212
file_number["dev"]=39
file_number["eval1"]=10
file_number["eval2"]=10
file_number["eval3"]=10
file_number["dialog"]=162


if [ $TOOL = 'htk' ]; then
  echo ============================================================================
  echo "                   Feature extraction by HTK toolkit                      "
  echo ============================================================================

  mkdir -p $HTK_SAVE_PATH

  # Make a config file to covert from wav to htk file
  python make_config.py \
      --data_path $CSJ_PATH \
      --htk_save_path $HTK_SAVE_PATH \
      --feature_type $FEATURE_TYPE \
      --channels $CHANNELS \
      --window $WINDOW \
      --slide $SLIDE \
      --energy $ENERGY \
      --delta $DELTA \
      --deltadelta $DELTADELTA \
      --subset $subset \
      --fullset $fullset

  # Convert from wav to htk files
  for data_type in train_subset train_fullset dev eval1 eval2 eval3 ; do

    htk_paths=$(find $HTK_SAVE_PATH/$data_type/ -iname '*.htk')
    htk_file_num=$(find $HTK_SAVE_PATH/$data_type/ -iname '*.htk' | wc -l)

    if [ $htk_file_num -ne ${file_number[$data_type]} ]; then
      $HCOPY_PATH -T 1 -C ./config/$FEATURE_TYPE.conf -S ./config/wav2htk_$data_type.scp
    fi
  done
else
  if ! which sox >&/dev/null; then
    echo "This script requires you to first install sox";
    exit 1;
  fi
fi


echo ============================================================================
echo "                                  Main                                    "
echo ============================================================================

python main.py \
    --data_path $CSJ_PATH \
    --dataset_save_path $DATASET_SAVE_PATH \
    --feature_save_path $FEATURE_SAVE_PATH \
    --tool $TOOL \
    --htk_save_path $HTK_SAVE_PATH \
    --feature_type $FEATURE_TYPE \
    --channels $CHANNELS \
    --window $WINDOW \
    --slide $SLIDE \
    --energy $ENERGY \
    --delta $DELTA \
    --deltadelta $DELTADELTA \
    --normalize $NORMALIZE \
    --save_format $SAVE_FORMAT \
    --subset $subset \
    --fullset $fullset


echo 'Successfully completed!!!'
