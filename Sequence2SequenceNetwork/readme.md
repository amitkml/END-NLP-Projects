# Sequence to Sequence Network

![image](https://github.com/amitkml/END-NLP-Projects/blob/main/Sequence2SequenceNetwork/src/NMT_Seq2Seq.JPG?raw=true)


## Standard Seq2Seq - Training Time

![image](https://github.com/amitkml/END-NLP-Projects/blob/main/Sequence2SequenceNetwork/src/Sew2SeqTraining.png)

### Greedy Decoding
This strategy selects the most probable word (i.e. argmax) from the model’s vocabulary at each decoding time-step as the candidate to output sequence.
![image](https://github.com/amitkml/END-NLP-Projects/blob/main/Sequence2SequenceNetwork/src/Greedy_Decoding.JPG)
The problem with this approach is that once the output is chosen at any time-step t, we don’t get the flexibility to go back and change our choice. It is seen in practice that greedy decoding strategy is prone to have grammatical errors in the generated text. It will result in choosing the best at any time-step t but that might not necessarily give the best when considering the full sentence to be grammatically correct and sensible.
![image](https://miro.medium.com/max/344/1*84ND8QjSn_P0Lb_p4WZ97A.png)

### Beam Search with Fixed Beam Size
The beam search strategy tries to find an output sequence having a maximum likelihood. It does this by extending the greedy sampling to Top-k sampling strategy. At any time-step t, it considers top-k most probable words as the candidate words for that step. Here, k is called the beam size. 
![image](https://miro.medium.com/max/614/1*_y1AdUWcDMcMihTvFAI4CQ.png)
    When (k=1) — It behaves like Greedy search where argmax at any time-step t is fed to later consecutive step.
    When (k=Size of Vocabulary) — It behaves like an Exhaustive search where possible words at each time-step are whole of vocabulary and of which each gives probability distribution over the next set of vocabulary for later consecutive step.
    
## Standard Seq2Seq - Testing Time

![Image](https://github.com/amitkml/END-NLP-Projects/blob/main/Sequence2SequenceNetwork/src/Seq2Seq_Testing_Time.JPG?raw=true)


## Standard Seq2Seq Application

- Seq2Seq is versatile
- Many NLP task can be phrased as Seq2Se
  - Summarization (Long Text -> Short Text)
  - Dialogue (Previous Sentence -> Next Sentence)
  - Parsing (Input Text -> Output is parsed as Sequence)
  - Code Language (Natuaral Language -> Python Code)

## Business Problem

1. Change the languages from german<>english to french<>german
2. Change the model such that it has 3 layers instead of 2

## Solution

Solution has been loaded into github and link is [**END_NLP_Session_8_Assignment_B_Sequence_to_Sequence_Learning_with_Neural_Networks**](https://github.com/amitkml/END-NLP-Projects/blob/main/Sequence2SequenceNetwork/src/END_NLP_Session_8_Assignment_B_Sequence_to_Sequence_Learning_with_Neural_Networks.ipynb)

## Model Performance Measurement

A NLP model is being said to be a good model when it can assign higher probability to real/frequently observed sentence than rarely observed ones.

- **Perplexity** is a measurement of **how well a probability model predicts a test data**. In the context of Natural Language Processing, perplexity is one way to **evaluate language models**.
- Perplexity is just an *exponentiation of the entropy*!
- Low perplexity is good and high perplexity is bad since the perplexity is the exponentiation of the entropy

## Summary of Changes

- Added steps to download French

  ```
  python -m spacy download fr
  ```

- Load the models associated French. Here I have added spacy_fr which is going to load model associated with french

  ```
  spacy_de = spacy.load('de')
  spacy_en = spacy.load('en')
  spacy_fr = spacy.load('fr')
  ```

- Added Tokenization function for French and German. Here my source language is French and so it is being reversed.

  ```
  ## Change the languages from german<>english to french<>german
  
  def tokenize_de(text):
      """
      Tokenizes German text from a string into a list of strings (tokens)
      """
      return [tok.text for tok in spacy_de.tokenizer(text)]
  
  def tokenize_en(text):
      """
      Tokenizes English text from a string into a list of strings (tokens)
      """
      return [tok.text for tok in spacy_en.tokenizer(text)]
  
  
  def tokenize_fr(text):
      """
      Tokenizes french text from a string into a list of strings (tokens) and reverses it
      """
      return [tok.text for tok in spacy_fr.tokenizer(text)][::-1]
  ```

- Defined Tokenization function for French and German. Point to note here that SRC is referring french tokenization function and target is referring German tokenization function.

```
SRC = Field(tokenize = tokenize_fr, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TRG = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
```

- Downloaded Multi30K data from github repo multi30k

```
!git clone --recursive https://github.com/multi30k/dataset.git multi30k
```

- Moved all the required training, validation and test files into proper directory which is Multi30K. This is ensure the split function can look into default directory and read files.

  ```
  def prepare_file_Multi30k():
    import shutil
    import os
    source_dir = '/content/multi30k/data/task1/raw/'
    target_dir = '/content/multi30k/'
    ROOT_DIR = '/content/'
  
    file_names = os.listdir(source_dir)
    for file_name in file_names:
      shutil.move(os.path.join(source_dir, file_name), target_dir)
    !gzip -d /content/multi30k/*.gz
    for fileName in os.listdir(target_dir):
      if fileName.startswith('test_2016_flickr'):
        os.chdir(target_dir)
        os.rename(fileName, fileName.replace("test_2016_flickr", "test2016"))
    os.chdir(ROOT_DIR)
  ```

- Split function has been changed to ensure Source and Target are set to French and German.

  ```
  train_data, valid_data, test_data = Multi30k.splits(exts = ('.fr', '.de'),                                                   
                                                      fields = (SRC, TRG),
                                                      root = '')
  ```

- Increased no of LSTM layers to 3 from earlier value of 2.

```
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
# N_LAYERS = 2
N_LAYERS = 3 # changed as per assignment 1 instruction of Change the model such that it has 3 layers instead of 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
```

## Model Performance Log

```
Epoch: 01 | Time: 1m 43s
	Train Loss: 5.226 | Train PPL: 186.035
	 Val. Loss: 5.031 |  Val. PPL: 153.092
Epoch: 02 | Time: 1m 44s
	Train Loss: 4.730 | Train PPL: 113.335
	 Val. Loss: 4.845 |  Val. PPL: 127.150
Epoch: 03 | Time: 1m 44s
	Train Loss: 4.346 | Train PPL:  77.189
	 Val. Loss: 4.671 |  Val. PPL: 106.851
Epoch: 04 | Time: 1m 44s
	Train Loss: 4.104 | Train PPL:  60.588
	 Val. Loss: 4.601 |  Val. PPL:  99.605
Epoch: 05 | Time: 1m 44s
	Train Loss: 3.963 | Train PPL:  52.625
	 Val. Loss: 4.500 |  Val. PPL:  89.995
Epoch: 06 | Time: 1m 45s
	Train Loss: 3.849 | Train PPL:  46.942
	 Val. Loss: 4.421 |  Val. PPL:  83.168
Epoch: 07 | Time: 1m 44s
	Train Loss: 3.764 | Train PPL:  43.109
	 Val. Loss: 4.454 |  Val. PPL:  85.961
Epoch: 08 | Time: 1m 44s
	Train Loss: 3.651 | Train PPL:  38.497
	 Val. Loss: 4.390 |  Val. PPL:  80.614
Epoch: 09 | Time: 1m 44s
	Train Loss: 3.553 | Train PPL:  34.906
	 Val. Loss: 4.337 |  Val. PPL:  76.486
Epoch: 10 | Time: 1m 44s
	Train Loss: 3.462 | Train PPL:  31.886
	 Val. Loss: 4.264 |  Val. PPL:  71.098
Epoch: 11 | Time: 1m 44s
	Train Loss: 3.372 | Train PPL:  29.136
	 Val. Loss: 4.168 |  Val. PPL:  64.599
Epoch: 12 | Time: 1m 44s
	Train Loss: 3.267 | Train PPL:  26.243
	 Val. Loss: 4.108 |  Val. PPL:  60.812
Epoch: 13 | Time: 1m 43s
	Train Loss: 3.188 | Train PPL:  24.236
	 Val. Loss: 4.066 |  Val. PPL:  58.350
Epoch: 14 | Time: 1m 44s
	Train Loss: 3.100 | Train PPL:  22.195
	 Val. Loss: 4.046 |  Val. PPL:  57.190
Epoch: 15 | Time: 1m 44s
	Train Loss: 3.027 | Train PPL:  20.639
	 Val. Loss: 3.988 |  Val. PPL:  53.973
Epoch: 16 | Time: 1m 44s
	Train Loss: 2.958 | Train PPL:  19.267
	 Val. Loss: 3.938 |  Val. PPL:  51.314
Epoch: 17 | Time: 1m 44s
	Train Loss: 2.891 | Train PPL:  18.011
	 Val. Loss: 3.927 |  Val. PPL:  50.778
Epoch: 18 | Time: 1m 44s
	Train Loss: 2.788 | Train PPL:  16.242
	 Val. Loss: 3.923 |  Val. PPL:  50.556
Epoch: 19 | Time: 1m 44s
	Train Loss: 2.735 | Train PPL:  15.417
	 Val. Loss: 3.879 |  Val. PPL:  48.381
Epoch: 20 | Time: 1m 44s
	Train Loss: 2.669 | Train PPL:  14.427
	 Val. Loss: 3.918 |  Val. PPL:  50.294
Epoch: 21 | Time: 1m 44s
	Train Loss: 2.593 | Train PPL:  13.366
	 Val. Loss: 3.916 |  Val. PPL:  50.211
Epoch: 22 | Time: 1m 43s
	Train Loss: 2.538 | Train PPL:  12.652
	 Val. Loss: 3.874 |  Val. PPL:  48.122
Epoch: 23 | Time: 1m 44s
	Train Loss: 2.486 | Train PPL:  12.008
	 Val. Loss: 3.925 |  Val. PPL:  50.649
Epoch: 24 | Time: 1m 44s
	Train Loss: 2.432 | Train PPL:  11.382
	 Val. Loss: 3.827 |  Val. PPL:  45.925
Epoch: 25 | Time: 1m 44s
	Train Loss: 2.374 | Train PPL:  10.741
	 Val. Loss: 3.863 |  Val. PPL:  47.594
Epoch: 26 | Time: 1m 44s
	Train Loss: 2.311 | Train PPL:  10.082
	 Val. Loss: 3.883 |  Val. PPL:  48.584
Epoch: 27 | Time: 1m 44s
	Train Loss: 2.279 | Train PPL:   9.772
	 Val. Loss: 3.921 |  Val. PPL:  50.442
Epoch: 28 | Time: 1m 44s
	Train Loss: 2.228 | Train PPL:   9.277
	 Val. Loss: 3.961 |  Val. PPL:  52.495
Epoch: 29 | Time: 1m 44s
	Train Loss: 2.194 | Train PPL:   8.967
	 Val. Loss: 3.915 |  Val. PPL:  50.124
Epoch: 30 | Time: 1m 44s
	Train Loss: 2.112 | Train PPL:   8.263
	 Val. Loss: 4.003 |  Val. PPL:  54.747
```

## Further Enhancement

- We can implement Beam Search method which allows us to choose from a set of word. This is quite better than Greedy Search Algorithm. Refer the video [Beam Search](https://www.youtube.com/watch?v=UXW6Cs82UKo) to understand more. ![Image](https://github.com/amitkml/END-NLP-Projects/blob/main/Sequence2SequenceNetwork/src/Beam_Search.JPG?raw=true)



# Input Data Generation for English to Python Code



## Solution

Solution has been loaded into github and link is [**END_NLP_Session_8_Assignment_PythonGeneration.ipynb**](https://github.com/amitkml/END-NLP-Projects/blob/main/Sequence2SequenceNetwork/src/END_NLP_Session_8_Assignment_PythonGeneration.ipynb)

## Further details

This repo contains the mapping of description and te the code to be written (starts with #), and the code it should generate. It will be later used for network training.

Here are some examples:
- provide the length of list, dict, tuple, etc
- write a function to sort a list
- write a function to test the time it takes to run a function
- write a program to remove stop words from a sentence provided,

Source files are available in src directory and their links are



