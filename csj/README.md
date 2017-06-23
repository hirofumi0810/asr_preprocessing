## CSJ (Corpus of Spontaneous Japanese) corpus
About the CSJ corpus:

CSJ is a database of spoken
Japanese developed by the Japan's national priority area research project "Spontaneous Speech: Corpus and Processing Technology". It contains about 650 hours of speech consisting of approximately 7.5 million words that were provided by more than 1,400 speakers. For more details about the corpus, please visit the website of the National Institute for Japanese Language (NINJAL), which is available from the [here](http://pj.ninjal.ac.jp/corpus_center/csj/en/).


## Usage
1. At first, please download & install HTK from [here](http://htk.eng.cam.ac.uk/download.shtml) and set the path to HCopy in make.sh. Feature extraction is based on HTK toolkit.
The non-HTK version will be added in the future.
The configure file is in ./config/config_fbank. The default input feature is 40 channel log-mel filterbank features and energy (+Δ, ΔΔ). Please change it or ./make_script.py by yourself.

2. Set to the path to CSJ corpus and paths to save input features & dataset in make.sh and run
```
./make.sh
```
