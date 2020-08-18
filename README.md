# CTC segmentation

Segment a given audio into utterances using a trained end-to-end ASR model.

## Setup

Install ESPnet (tested v0.7.0)

https://espnet.github.io/espnet/installation.html

Clone this repo into ESPnets egs folder:

```
cd <espnet_folder>/egs
git clone https://github.com/cornerfarmer/ctc_segmentation
cd ctc_segmentation
```

Activate ESPnets virtual environment

```
source ../../tools/venv/bin/activate
```

## Setup data

In this example it is explained how to align the utterances in the TED-lium dev and test dataset, but one can easily align any custom data by providing `.wav` files inside `data/<dataset_name>` together with equally named `.txt` files which contain the corresponding transcriptions (one utterance per line).

```
wget https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release1.tar.gz
tar xvf TEDLIUM_release1.tar.gz
```

Convert TED-lium data format into .wav + .txt file:

```
python convert_ted.py TEDLIUM_release1/dev data/tedlium_dev
python convert_ted.py TEDLIUM_release1/test data/tedlium_test
```

## Download model

Prepare trained ESPnet model. In this example we're gonna use a RNN trained on TED-lium v2 (tedlium2.rnn.v2) provided by ESPNet.

Download the weights from https://drive.google.com/open?id=1cac5Uc09lJrCYfWkLQsF8eapQcxZnYdf and place the `.tar.gz` file into the ctc_segmentation folder.

Extract data and move the relavent parts into `exp/tedlium2_rnn/`. 

```
tar xvf model.streaming.v2.tar.gz --one-top-level
mkdir -p exp/tedlium2_rnn
mv model.streaming.v2/exp/train_trim_sp_pytorch_train2/results/* exp/tedlium2_rnn/
mv model.streaming.v2/data/train_trim_sp/cmvn.ark exp/tedlium2_rnn/
```

## Decode audio

In this step we apply the model to the prepared data. By using the ctc part of the network this gives us a prob distr across all possible symbols per audio frame which are stored in a `.npz` file.

If you want to speed up this step by using the GPU use the flag `--gpu`.

If you are using a transformer model it is recommended to use the split mode where the audio is split up into different parts which are decoded individually.
Use therefore the `--split <part_size>` flag. We found `2000` to work good as a part size.

```
python decode.py exp/tedlium2_rnn/model.acc.best exp/tedlium2_rnn/cmvn.ark data/tedlium_dev eval/tedlium_dev_tedlium2_rnn
```


## Segment audio

Finally align / segment the audio. This step will apply our ctc segmentation algorithm and store the results in `eval/tedlium_dev_tedlium2_rnn/*.txt`.

```
python align.py exp/tedlium2_rnn/model.acc.best data/tedlium_dev eval/tedlium_dev_tedlium2_rnn
```

@abhishekyana

## Pathfile:

When using the scipt with a path file, there is no need to move the data and text files into a single folder. So this path files explicitely informs the script to search for wav file, input text file at certain paths.

format: MetaData/data_fullpath.file format. 
```
<path to wav file>|<path to input text file>|<desired path where npz should be stored>|<desired path where output text file should be stored>

eg. input.wav|input.txt|input.npz|output.txt
```
input.wav - path to the input wav file. 
input.txt - path to the input text file. (one utterance per line)
input.npz - desired path where we need npz file to be saved. (probabilities are stored here in decode step)
output.txt - desired path where we need our output to be saved. (timestamps of corresponding utterance is saved per each line)

## Decode Audio with a pathfile. 

So to handle huge datasets instead of moving data here and there we can use this decodelite.py which expects the above mentioned pathfile. 
```
python decodelite.py --gpu --split 20000 exp/<model>/model.acc.best exp/<model>/cvmn.ark MetaData/data_fullpath.file
```

## Align Audio with a pathfile. 

So to handle huge datasets instead of moving data here and there we can use this alignlite.py or alignlitemp.py which expects the above mentioned pathfile. 
```
python alignlite.py exp/<model>/model.acc.best MetaData/data_fullpath.file # on single processor.
(or)
python alignlitemp.py exp/<model>/model.acc.best MetaData/data_fullpath.file # multiprocessing for using multiple cores. 
```


# Reference

CTC-segmentation is described in **CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition** at https://arxiv.org/abs/2007.09127 

```
@misc{ctcsegmentation,
    title={CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition},
    author={Ludwig Kürzinger and Dominik Winkelbauer and Lujun Li and Tobias Watzel and Gerhard Rigoll},
    year={2020},
    eprint={2007.09127},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
