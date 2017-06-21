## TIMIT
About TIMIT corpus:

The TIMIT corpus is one of the original clean speech databases.
This corpus of read speech is designed to provide speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems.
TIMIT contains broadband recordings of 630 speakers of eight major dialects of American English, each reading ten phonetically rich sentences.
The TIMIT corpus includes time-aligned orthographic, phonetic and word transcriptions as well as a 16-bit, 16kHz speech waveform file for each utterance. Description of catalog from [here](http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC93S1). The LDC numbers are

    LDC93S1

The scripts to make datasets for CTC and Attention Mechanism are prepared now. The target labels are
- 61 phones
- 48 phones
- 39 phones
- 30 characters

In general, 61 phones are used for training stage, and the model is evaluated by using 39 phones.

## Usage
1. At first, please download & install HTK from [here](http://htk.eng.cam.ac.uk/download.shtml) and set the path to HCopy in make.sh. Feature extraction is based on HTK toolkit.
The non-HTK version will be added in the future.
The configure file is in ./config/config_fbank. The default input feature is 40 channel log-mel filterbank features and energy (+Δ, ΔΔ). Please change it or ./make_script.py by yourself.

2. Set to the path to TIMIT corpus and paths to save input features & dataset in make.sh and run
```
./make.sh
```
