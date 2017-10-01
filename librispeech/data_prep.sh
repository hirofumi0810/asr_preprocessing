#!/bin/bash

echo ============================================================================
echo "                              Librispeech                                 "
echo ============================================================================

### Set paths
DATA_SAVE_PATH='/n/sd8/inaguma/corpus/librispeech/data'
DATASET_SAVE_PATH='/n/sd8/inaguma/corpus/librispeech/dataset'
WAV_SAVE_PATH='/n/sd8/inaguma/corpus/librispeech/wav'
HTK_SAVE_PATH='/n/sd8/inaguma/corpus/librispeech/htk'
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

if [ ! -e $DATA_SAVE_PATH ]; then
  mkdir $DATA_SAVE_PATH
fi
if [ ! -e $DATASET_SAVE_PATH ]; then
  mkdir $DATASET_SAVE_PATH
fi
if [ ! -e $WAV_SAVE_PATH ]; then
  mkdir $WAV_SAVE_PATH
fi
if [ ! -e $HTK_SAVE_PATH ] && [ $TOOL = 'htk' ]; then
  mkdir $HTK_SAVE_PATH
fi

RUN_ROOT_PATH=`pwd`


echo ============================================================================
echo "                           Download tha data                              "
echo ============================================================================
for part in train-clean-100 \
            train-clean-360 \
            train-other-500 \
            dev-clean \
            dev-other \
            test-clean \
            test-other; do
  if [ -d $DATA_SAVE_PATH/$part ]; then
    echo file exists: $part
  else
    wget http://www.openslr.org/resources/12/$part.tar.gz -P $DATA_SAVE_PATH
    tar xvfz $DATA_SAVE_PATH/LibriSpeech/$part.tar.gz -C $DATA_SAVE_PATH
    rm $DATA_SAVE_PATH/LibriSpeech/$part.tar.gz
  fi
done

# Move directories
if [ -d $DATA_SAVE_PATH/LibriSpeech ]; then
  mv $DATA_SAVE_PATH/LibriSpeech/* $DATA_SAVE_PATH
  rm -rf $DATA_SAVE_PATH/LibriSpeech
fi

# Download the LM resources
for part in 3-gram \
            3-gram.pruned.1e-7 \
            3-gram.pruned.3e-7 \
            4-gram; do
  if [ -e $DATA_SAVE_PATH/$part.arpa ]; then
    echo file exists: $part
  else
    wget http://www.openslr.org/resources/11/$part.arpa.gz -P $DATA_SAVE_PATH
    gunzip $DATA_SAVE_PATH/$part.arpa.gz
  fi
done
if [ ! -e $DATA_SAVE_PATH/librispeech-lm-corpus ]; then
  wget http://www.openslr.org/resources/11/librispeech-lm-corpus.tgz -P $DATA_SAVE_PATH
  tar xzvf $DATA_SAVE_PATH/librispeech-lm-corpus.tgz -C $DATA_SAVE_PATH
  rm $DATA_SAVE_PATH/librispeech-lm-corpus.tgz
fi
if [ ! -e $DATA_SAVE_PATH/librispeech-lm-norm.txt ]; then
  wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P $DATA_SAVE_PATH
  gunzip $DATA_SAVE_PATH/librispeech-lm-norm.txt.gz
fi
for part in g2p-model-5 \
            librispeech-lexicon.txt \
            librispeech-vocab.txt; do
  if [ -e $DATA_SAVE_PATH/$part ]; then
    echo file exists: $part
  else
    wget http://www.openslr.org/resources/11/$part -P $DATA_SAVE_PATH
  fi
done

# Remove the rest of files
for path in $DATA_SAVE_PATH/*.gz; do
  if [ -e $path ]; then
    rm $path
  fi
done
for path in $DATA_SAVE_PATH/*.tgz; do
  if [ -e $path ]; then
    rm $path
  fi
done
for path in $DATA_SAVE_PATH/*.1; do
  if [ -e $path ]; then
    rm $path
  fi
done


echo ============================================================================
echo "                        Convert from flac to wav                          "
echo ============================================================================
flac_paths=$(find $DATA_SAVE_PATH -type f)
for flac_path in $flac_paths ; do
  dir_path=$(dirname $flac_path)
  file_name=$(basename $flac_path)
  base=${file_name%.*}
  ext=${file_name##*.}
  wav_path=$dir_path"/"$base".wav"
  if [ $ext = "flac" ]; then
    echo "Converting from"$flac_path" to "$wav_path
    sox $flac_path -t wav $wav_path
    rm -f $flac_path
  else
    echo "Already converted: "$wav_path
  fi
done


if [ $TOOL = 'htk' ]; then
  echo ============================================================================
  echo "                   Feature extraction by HTK toolkit                      "
  echo ============================================================================

  # Set the path to HTK (optional, set only when using HTK toolkit)
  if [ $FEATURE_TYPE = 'logmelfbank' ]; then
    CONFIG_PATH="./config/fbank.config"
  else
    CONFIG_PATH="./config/mfcc.config"
  fi

  # Make a config file to covert from wav to htk file
  python make_config.py \
    --data_path $DATA_SAVE_PATH  \
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
  $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train_clean100.scp
  $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train_clean360.scp
  $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_train_other500.scp
  $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_dev_clean.scp
  $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_dev_other.scp
  $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_test_clean.scp
  $HTK_PATH -T 1 -C $CONFIG_PATH -S config/wav2fbank_test_other.scp
fi


echo ============================================================================
echo "                                  Main                                    "
echo ============================================================================
python main.py \
  --data_path $DATA_SAVE_PATH \
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
