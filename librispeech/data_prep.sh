#!/bin/bash

echo ============================================================================
echo "                              Librispeech                                 "
echo ============================================================================

### Set paths
DOWNLOAD_DATA_SAVE_PATH='/n/sd8/inaguma/corpus/librispeech/data'
DATASET_SAVE_PATH='/n/sd8/inaguma/corpus/librispeech/dataset'
HTK_SAVE_PATH='/n/sd8/inaguma/corpus/librispeech/htk'
HCOPY_PATH='/home/lab5/inaguma/htk-3.4/bin/HCopy'

### Select one tool to extract features (default is HTK)
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
# 100h (train-clean-100)
small=true

# 460h (train-clean-100 + train-clean-360)
# medium=true
medium=false

# 960h (train-clean-100 + train-clean-360  +train-other-500)
# large=true
large=false


########################################
# ↓↓↓ Don't change from here ↓↓↓
########################################
set -eu

if [ ! -e $DOWNLOAD_DATA_SAVE_PATH ]; then
  mkdir $DOWNLOAD_DATA_SAVE_PATH
fi
if [ ! -e $DATASET_SAVE_PATH ]; then
  mkdir $DATASET_SAVE_PATH
fi
if [ ! -e $HTK_SAVE_PATH ] && [ $TOOL = 'htk' ]; then
  mkdir $HTK_SAVE_PATH
fi


echo ============================================================================
echo "                           Download tha data                              "
echo ============================================================================

if ! which wget >&/dev/null; then
  echo "This script requires you to first install wget";
  exit 1;
fi

# Download the LM resources
for part in 3-gram \
            3-gram.pruned.1e-7 \
            3-gram.pruned.3e-7 \
            4-gram; do
  if [ -e $DOWNLOAD_DATA_SAVE_PATH/$part.arpa ]; then
    echo file exists: $part
  else
    if [ ! -e $DOWNLOAD_DATA_SAVE_PATH/$part.arpa.gz ]; then
      wget http://www.openslr.org/resources/11/$part.arpa.gz -P $DOWNLOAD_DATA_SAVE_PATH
    fi
    gunzip $DOWNLOAD_DATA_SAVE_PATH/$part.arpa.gz
  fi
done
if [ ! -e $DOWNLOAD_DATA_SAVE_PATH/librispeech-lm-corpus ]; then
  if [ ! -e $DOWNLOAD_DATA_SAVE_PATH/librispeech-lm-corpus.tgz ]; then
    wget http://www.openslr.org/resources/11/librispeech-lm-corpus.tgz -P $DOWNLOAD_DATA_SAVE_PATH
  fi
  tar xzvf $DOWNLOAD_DATA_SAVE_PATH/librispeech-lm-corpus.tgz -C $DOWNLOAD_DATA_SAVE_PATH
  rm $DOWNLOAD_DATA_SAVE_PATH/librispeech-lm-corpus.tgz
fi
if [ ! -e $DOWNLOAD_DATA_SAVE_PATH/librispeech-lm-norm.txt ]; then
  if [ ! -e $DOWNLOAD_DATA_SAVE_PATH/librispeech-lm-norm.txt.gz ]; then
    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P $DOWNLOAD_DATA_SAVE_PATH
  fi
  gunzip $DOWNLOAD_DATA_SAVE_PATH/librispeech-lm-norm.txt.gz
fi
for part in g2p-model-5 \
            librispeech-lexicon.txt \
            librispeech-vocab.txt; do
  if [ -e $DOWNLOAD_DATA_SAVE_PATH/$part ]; then
    echo file exists: $part
  else
    wget http://www.openslr.org/resources/11/$part -P $DOWNLOAD_DATA_SAVE_PATH
  fi
done

# Download the AM resources
declare -a am_resources=("train-clean-100" "dev-clean" "dev-other" "test-clean" "test-other")

if [ $large ]; then
  am_resources+=("train-clean-360" "train-other-500")
elif [ $medium ]; then
  am_resources+=("train-clean-360")
fi

for part in ${am_resources[@]}; do
  if [ -d $DOWNLOAD_DATA_SAVE_PATH/$part ]; then
    echo file exists: $part
  else
    if [ ! -e $DOWNLOAD_DATA_SAVE_PATH/$part.tar.gz ]; then
      wget http://www.openslr.org/resources/12/$part.tar.gz -P $DOWNLOAD_DATA_SAVE_PATH
    fi
    tar xvfz $DOWNLOAD_DATA_SAVE_PATH/$part.tar.gz -C $DOWNLOAD_DATA_SAVE_PATH
    rm $DOWNLOAD_DATA_SAVE_PATH/$part.tar.gz
  fi
done

# Move directories
if [ -d $DOWNLOAD_DATA_SAVE_PATH/LibriSpeech ]; then
  mv $DOWNLOAD_DATA_SAVE_PATH/LibriSpeech/* $DOWNLOAD_DATA_SAVE_PATH
  rm -rf $DOWNLOAD_DATA_SAVE_PATH/LibriSpeech
fi

# Remove the rest of files
for path in $DOWNLOAD_DATA_SAVE_PATH/*.gz; do
  if [ -e $path ]; then
    rm $path
  fi
done
for path in $DOWNLOAD_DATA_SAVE_PATH/*.tgz; do
  if [ -e $path ]; then
    rm $path
  fi
done
for path in $DOWNLOAD_DATA_SAVE_PATH/*.1; do
  if [ -e $path ]; then
    rm $path
  fi
done

declare -A file_number
file_number["train-clean-100"]=28539
file_number["train-clean-360"]=104014
file_number["train-other-500"]=148688
file_number["dev-clean"]=2703
file_number["dev-other"]=2864
file_number["test-clean"]=2620
file_number["test-other"]=2939


echo ============================================================================
echo "                        Convert from flac to wav                          "
echo ============================================================================

# flac_paths=$(find $DOWNLOAD_DATA_SAVE_PATH -iname '*.flac')
# for flac_path in $flac_paths ; do
#   dir_path=$(dirname $flac_path)
#   file_name=$(basename $flac_path)
#   base=${file_name%.*}
#   ext=${file_name##*.}
#   wav_path=$dir_path"/"$base".wav"
#   if [ $ext = "flac" ]; then
#     echo "Converting from"$flac_path" to "$wav_path
#     sox $flac_path -t wav $wav_path
#     rm -f $flac_path
#   else
#     echo "Already converted: "$wav_path
#   fi
# done


if [ $TOOL = 'htk' ]; then
  echo ============================================================================
  echo "                   Feature extraction by HTK toolkit                      "
  echo ============================================================================

  mkdir -p $HTK_SAVE_PATH

  # Make a config file to covert from wav to htk file
  python make_config.py \
    --data_path $DOWNLOAD_DATA_SAVE_PATH  \
    --htk_save_path $HTK_SAVE_PATH \
    --feature_type $FEATURE_TYPE \
    --channels $CHANNELS \
    --sampling_rate $SAMPLING_RATE \
    --window $WINDOW \
    --slide $SLIDE \
    --energy $ENERGY \
    --delta $DELTA \
    --deltadelta $DELTADELTA \
    --config_save_path ./config/$FEATURE_TYPE.config \
    --medium $medium \
    --large $large

  # Convert from wav to htk files
  for part in ${am_resources[@]}; do
    mkdir -p $HTK_SAVE_PATH/$part

    htk_paths=$(find $HTK_SAVE_PATH/$part/ -iname '*.htk')
    htk_file_num=$(find $HTK_SAVE_PATH/$part/ -iname '*.htk' | wc -l)

    if [ $htk_file_num -ne ${file_number[$part]} ]; then
      # Make parallel
      $HCOPY_PATH -T 1 -C ./config/$FEATURE_TYPE.config -S ./config/wav2htk_$part.scp &
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
  --data_path $DOWNLOAD_DATA_SAVE_PATH \
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
  --medium $medium \
  --large $large


echo 'Successfully completed!!!'
