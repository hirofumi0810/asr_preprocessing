#!/bin/bash

echo ============================================================================
echo "                                   CSJ                                     "
echo ============================================================================

# Set paths
CSJ_PATH='/n/sd8/inaguma/corpus/csj/data'
DATASET_SAVE_PATH='/n/sd8/inaguma/corpus/csj/dataset'
HTK_SAVE_PATH='/n/sd8/inaguma/corpus/csj/htk'
HCOPY_PATH='/home/lab5/inaguma/htk-3.4/bin/HCopy'

### Select one tool to extract features (HTK is the fastest)
TOOL='htk'
# TOOL='python_speech_features'
# TOOL='librosa'
# TOOL='kaldi'  # under implementation

### Configuration (Set by yourself)
FEATURE_TYPE='fbank'  # (logmel) fbank or mfcc
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

### data size to create
# subset (240h)
subset=true
# subset=false

# fullset (586h)
fullset=true
# fullset=false


########################################
# ↓↓↓ Don't change from here ↓↓↓
########################################
set -eu
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
file_number["dev"]=19
file_number["eval1"]=10
file_number["eval2"]=10
file_number["eval3"]=10


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
      --sampling_rate $SAMPLING_RATE \
      --window $WINDOW \
      --slide $SLIDE \
      --energy $ENERGY \
      --delta $DELTA \
      --deltadelta $DELTADELTA \
      --fullset $fullset

  # Convert from wav to htk files
  for data_type in train_subset train_fullset dev eval1 eval2 eval3 ; do

    htk_paths=$(find $HTK_SAVE_PATH/$data_type/ -iname '*.htk')
    htk_file_num=$(find $HTK_SAVE_PATH/$data_type/ -iname '*.htk' | wc -l)

    if [ $htk_file_num -ne ${file_number[$data_type]} ]; then
      # Make parallel
      $HCOPY_PATH -T 1 -C ./config/$FEATURE_TYPE.config -S ./config/wav2htk_$data_type.scp &
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
    --normalize $NORMALIZE \
    --fullset $fullset


echo 'Successfully completed!!!'
