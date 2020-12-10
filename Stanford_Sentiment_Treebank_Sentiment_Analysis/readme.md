[TOC]

# Stanford Sentiment Treebank Business Problem

The main goal of this research is to build a sentiment analysis system which automatically determines user opinions of the Stanford Sentiment Treebank in terms of sentiments pattern.

I have looked into the data pattern and have decided to approach the problem into three ways.

**High Level Sentiment Analysis**

- sentiment sentences been marked to Positive or negative depending on my cutoff value of 0.5. So any sentence having sentiment value <= 0.5 being marked as Negative. Otherwise it is being marked as Positive.
- Three Level of sentiments such as positive, negative, and neutral. 
  - 0 to 0.4 - Negative
  - 0.4 to 0.6 - Neutral
  - 0.6 to 1 - Positive

**Fine Grained Sentiment Analysis**

- Did the mapping of positivity probability using the following cut-offs: [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0] for very negative, negative, neutral, positive, very positive, respectively".

Our Download the StanfordSentimentAnalysis Dataset is available from this [link](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip)

- Use "datasetSentences.txt" and "sentiment_labels.txt" files from the zip you just downloaded as your dataset. 

- This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes. 

- The sentiments are rated between 1 and 25, where one is the most negative and 25 is the most positive.

  

> **Our target is : Train your model and try and achieve 60%+ accuracy.**



## Dataset Analysis

**This file includes:**

- original_rt_snippets.txt contains 10,605 processed snippets from the original pool of Rotten Tomatoes HTML files. Please note that some snippet may contain multiple sentences.
- dictionary.txt contains all phrases and their IDs, separated by a vertical line |
- sentiment_labels.txt contains all phrase ids and the corresponding sentiment labels, separated by a vertical line.
  Note that you can recover the 5 classes by mapping the positivity probability using the following cut-offs:
  [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
  for very negative, negative, neutral, positive, very positive, respectively.
  Please note that phrase ids and sentence ids are not the same.
- datasetSentences.txt contains the sentence index, followed by the sentence string separated by a tab. These are the sentences of the train/dev/test sets.
- datasetSplit.txt contains the sentence index (corresponding to the index in datasetSentences.txt file) followed by the set label separated by a comma:
  	1 = train
  	2 = test
  	3 = dev

## Brief Intro on Data Augmentation

Data augmenting of text always far more complex w.r.t to other type of data due to high complexity of language. Following are few of the key NLP data augmentation approach.

- Thesaurus

  -  replacing words or phrases with their synonyms. Leverage existing thesaurus help to generate lots of data in a short time.

- Word Embeddings

  - we can leverages pre-trained [classic word embeddings](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) such as word2vec, GloVe and fasttext to perform similarity search. Following  table represent “Most similar words of “fox” among classical word embeddings models”.

    ![image](https://miro.medium.com/max/655/1*eatzgOd9njd6mv8pwXZ9LA.png)

- [EDA (Easy Data Augmentation)](https://arxiv.org/abs/1901.11196).

  - **Synonym Replacement**: Randomly choose *n* words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random.

    - For example, given the sentence:

      *This **article** will focus on summarizing data augmentation **techniques** in NLP.*

      The method randomly selects n words (say two), the words *article* and *techniques*, and replaces them with *write-up* and *methods* respectively.

      *This **write-up** will focus on summarizing data augmentation **methods** in NLP.*

  - **Random Insertion**

    - Find a random synonym of a random word in the sentence that is not a 
      stop word. Insert that synonym into a random position in the sentence. 
      Do this *n* times. 

    - For example, given the sentence:

      *This **article** will focus on summarizing data augmentation **techniques** in NLP.*

      The method randomly selects n words (say two), the words *article* and *techniques* find the synonyms as *write-up* and *methods* respectively. Then these synonyms are inserted at a random position in the sentence.

      *This article will focus on **write-up** summarizing data augmentation techniques in NLP **methods**.*

  - **Random Swap**

    - Randomly choose two words in the sentence and swap their positions. Do this *n* times. 

    - For example, given the sentence

      *This **article** will focus on summarizing data augmentation **techniques** in NLP.*

      The method randomly selects n words (say two), the words *article* and *techniques* and swaps them to create a new sentence.

      *This **techniques** will focus on summarizing data augmentation **article** in NLP.*

  - **Random Deletion**

    - Randomly remove each word in the sentence with probability *p*. 

      For example, given the sentence

      *This **article** will focus on summarizing data augmentation **techniques** in NLP.*

      The method selects n words (say two), the words *will* and *techniques*, and removes them from the sentence.

      *This* ***article** focus on summarizing data augmentation in NLP.*

- [NLP Albumentation.](https://github.com/albumentations-team/albumentations)

  - **Shuffle Sentences Transform**

    - In this transformation, if the given text sample contains multiple  sentences these sentences are shuffled to create a new sample. 

      For example:

      **text = ‘<Sentence1>. <Sentence2>. <Sentence4>. <Sentence4>. <Sentence5>. <Sentence5>.’**

      Is transformed to:

      **text = ‘<Sentence2>. <Sentence3>. <Sentence1>. <Sentence5>. <Sentence5>. <Sentence4>.’**

- [NLP Aug.](https://github.com/makcedward/nlpaug)

  - This python package does these NLP algorithm automatically.
  - NLPAug offers three types of augmentation:
    - Character level augmentation
    - Word level augmentation 
    - Sentence level augmentation
    - NLPAug provides all the methods discussed in the previous sections such as:
      - random deletion, 
      - random insertion, 
      - shuffling, 
      - synonym replacement, 
      - etc.

# StanfordSentimentAnalysis Dataset Solution - High Level Sentiment Analysis - Two class

Here sentiment sentences been marked to Positive or negative depending on my cutoff value of 0.5. So any sentence having sentiment value <= 0.5 being marked as Negative. Otherwise it is being marked as Positive.

## Network hyperparameters and Architecture

```
classifier(
  (embedding): Embedding(16524, 100, padding_idx=1)
  (encoder): LSTM(100, 256, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (fc): Linear(in_features=256, out_features=2, bias=True)
)
The model has 5,540,018 trainable parameters
```

```
# Define hyperparameters
size_of_vocab = len(Sentence.vocab)
embedding_dim = 100
num_hidden_nodes = 256
num_output_nodes = 2
num_layers = 3
dropout = 0.2
bidirectional = True
PAD_IDX = Sentence.vocab.stoi[Sentence.pad_token]
```

## Network Performance

- Embedding layer is being pretrained from Gloves 100d
- Embedding layer being kept as Frozen during Training
- **Val Accuracy achieved after 15 epoch is 74.17%** 

```
	Train Loss: 0.693 | Train Acc: 56.20%
	 Val. Loss: 0.685 |  Val. Acc: 55.59% 

	Train Loss: 0.639 | Train Acc: 64.48%
	 Val. Loss: 0.595 |  Val. Acc: 69.64% 

	Train Loss: 0.582 | Train Acc: 70.92%
	 Val. Loss: 0.577 |  Val. Acc: 71.93% 

	Train Loss: 0.558 | Train Acc: 74.02%
	 Val. Loss: 0.571 |  Val. Acc: 72.59% 

	Train Loss: 0.528 | Train Acc: 77.23%
	 Val. Loss: 0.554 |  Val. Acc: 73.79% 

	Train Loss: 0.503 | Train Acc: 79.84%
	 Val. Loss: 0.554 |  Val. Acc: 73.88% 

	Train Loss: 0.480 | Train Acc: 82.30%
	 Val. Loss: 0.578 |  Val. Acc: 72.08% 

	Train Loss: 0.465 | Train Acc: 84.17%
	 Val. Loss: 0.557 |  Val. Acc: 73.98% 

	Train Loss: 0.444 | Train Acc: 86.16%
	 Val. Loss: 0.549 |  Val. Acc: 75.74% 

	Train Loss: 0.430 | Train Acc: 87.78%
	 Val. Loss: 0.565 |  Val. Acc: 74.20% 

	Train Loss: 0.422 | Train Acc: 88.66%
	 Val. Loss: 0.557 |  Val. Acc: 74.83% 

	Train Loss: 0.415 | Train Acc: 89.31%
	 Val. Loss: 0.560 |  Val. Acc: 74.79% 

	Train Loss: 0.411 | Train Acc: 89.63%
	 Val. Loss: 0.565 |  Val. Acc: 73.88% 

	Train Loss: 0.406 | Train Acc: 89.93%
	 Val. Loss: 0.568 |  Val. Acc: 73.92% 

	Train Loss: 0.403 | Train Acc: 90.63%
	 Val. Loss: 0.565 |  Val. Acc: 74.17% 
```

```

```



# StanfordSentimentAnalysis Dataset Solution - Three class

Three Level of sentiments such as positive, negative, and neutral. 

- 0 to 0.4 - Negative
- 0.4 to 0.6 - Neutral
- 0.6 to 1 - Positive

## Network hyperparameters and Architecture

```
classifier(
  (embedding): Embedding(16524, 100, padding_idx=1)
  (encoder): LSTM(100, 256, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (fc): Linear(in_features=256, out_features=2, bias=True)
)
The model has 5,540,018 trainable parameters
```

```
# Define hyperparameters
size_of_vocab = len(Sentence.vocab)
embedding_dim = 100
num_hidden_nodes = 256
num_output_nodes = 2
num_layers = 3
dropout = 0.2
bidirectional = True
PAD_IDX = Sentence.vocab.stoi[Sentence.pad_token]
```

## Network Performance

- Embedding layer is being pretrained from Gloves 100d
- Embedding layer being kept as Frozen during Training
- **Val Accuracy achieved after 15 epoch is 61.59%** 

```
	Train Loss: 1.041 | Train Acc: 47.25%
	 Val. Loss: 1.104 |  Val. Acc: 43.18% 

	Train Loss: 0.985 | Train Acc: 55.55%
	 Val. Loss: 0.980 |  Val. Acc: 55.11% 

	Train Loss: 0.937 | Train Acc: 60.28%
	 Val. Loss: 0.981 |  Val. Acc: 55.59% 

	Train Loss: 0.918 | Train Acc: 62.42%
	 Val. Loss: 0.933 |  Val. Acc: 60.87% 

	Train Loss: 0.901 | Train Acc: 64.52%
	 Val. Loss: 0.926 |  Val. Acc: 61.91% 

	Train Loss: 0.880 | Train Acc: 66.56%
	 Val. Loss: 0.916 |  Val. Acc: 62.42% 

	Train Loss: 0.857 | Train Acc: 69.15%
	 Val. Loss: 0.928 |  Val. Acc: 60.63% 

	Train Loss: 0.839 | Train Acc: 71.20%
	 Val. Loss: 0.921 |  Val. Acc: 61.99% 

	Train Loss: 0.828 | Train Acc: 72.34%
	 Val. Loss: 0.924 |  Val. Acc: 61.69% 

	Train Loss: 0.810 | Train Acc: 74.05%
	 Val. Loss: 0.939 |  Val. Acc: 60.34% 

	Train Loss: 0.798 | Train Acc: 75.68%
	 Val. Loss: 0.919 |  Val. Acc: 62.44% 

	Train Loss: 0.786 | Train Acc: 76.72%
	 Val. Loss: 0.920 |  Val. Acc: 61.59% 

	Train Loss: 0.778 | Train Acc: 77.46%
	 Val. Loss: 0.930 |  Val. Acc: 61.34% 

	Train Loss: 0.772 | Train Acc: 78.00%
	 Val. Loss: 0.921 |  Val. Acc: 61.67% 

	Train Loss: 0.758 | Train Acc: 79.78%
	 Val. Loss: 0.923 |  Val. Acc: 61.59% 
```



# StanfordSentimentAnalysis Dataset Solution - Three class - With Data Augmentation

Three Level of sentiments such as positive, negative, and neutral. 

- 0 to 0.4 - Negative
- 0.4 to 0.6 - Neutral
- 0.6 to 1 - Positive

## Network hyperparameters and Architecture

- Used following data augmentation for training data
  - Substitute word by WordNet's synonym
  - Swap word randomly
  - Delete word randomly augmentation

```
classifier(
  (embedding): Embedding(16524, 100, padding_idx=1)
  (encoder): LSTM(100, 256, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (fc): Linear(in_features=256, out_features=2, bias=True)
)
The model has 5,540,018 trainable parameters
```

```
# Define hyperparameters
size_of_vocab = len(Sentence.vocab)
embedding_dim = 100
num_hidden_nodes = 256
num_output_nodes = 2
num_layers = 3
dropout = 0.2
bidirectional = True
PAD_IDX = Sentence.vocab.stoi[Sentence.pad_token]
```

## Network Performance

- Embedding layer is being pretrained from Gloves 100d
- Embedding layer being kept as Frozen during Training
- **Highest Val Accuracy achieved after 15 epoch is 63.01%** 
- Starting accuracy of Validation data is 60.83% which is quite good compare to earlier initial starting accuracy 43.18%

```
	Train Loss: 0.992 | Train Acc: 54.29%
	 Val. Loss: 0.933 |  Val. Acc: 60.83% 

	Train Loss: 0.919 | Train Acc: 62.74%
	 Val. Loss: 0.939 |  Val. Acc: 60.30% 

	Train Loss: 0.864 | Train Acc: 68.61%
	 Val. Loss: 0.927 |  Val. Acc: 61.78% 

	Train Loss: 0.825 | Train Acc: 72.79%
	 Val. Loss: 0.918 |  Val. Acc: 63.01% 

	Train Loss: 0.794 | Train Acc: 75.70%
	 Val. Loss: 0.925 |  Val. Acc: 61.12% 

	Train Loss: 0.766 | Train Acc: 78.52%
	 Val. Loss: 0.917 |  Val. Acc: 62.54% 

	Train Loss: 0.745 | Train Acc: 80.85%
	 Val. Loss: 0.918 |  Val. Acc: 62.35% 

	Train Loss: 0.731 | Train Acc: 82.18%
	 Val. Loss: 0.912 |  Val. Acc: 63.24% 

	Train Loss: 0.712 | Train Acc: 84.14%
	 Val. Loss: 0.934 |  Val. Acc: 60.85% 

	Train Loss: 0.702 | Train Acc: 85.24%
	 Val. Loss: 0.938 |  Val. Acc: 60.55% 

	Train Loss: 0.695 | Train Acc: 85.89%
	 Val. Loss: 0.933 |  Val. Acc: 61.12% 

	Train Loss: 0.692 | Train Acc: 86.22%
	 Val. Loss: 0.931 |  Val. Acc: 61.21% 

	Train Loss: 0.679 | Train Acc: 87.37%
	 Val. Loss: 0.938 |  Val. Acc: 60.21% 

	Train Loss: 0.676 | Train Acc: 87.81%
	 Val. Loss: 0.923 |  Val. Acc: 62.05% 

	Train Loss: 0.673 | Train Acc: 88.17%
	 Val. Loss: 0.923 |  Val. Acc: 62.31% 
```



# StanfordSentimentAnalysis Dataset Solution - Fine Grained Analysis

## Preprocessing

## Network Architecture Analysis and Findings

### Multilayer LSTM without Bidirectional

**Network hyperparameters**

```
# Define hyperparameters
size_of_vocab = len(Sentence.vocab)
embedding_dim = 300
num_hidden_nodes = 100
num_output_nodes = 5
num_layers = 2
dropout = 0.2

# define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()
```

**Network architecture is**

```
classifier(
  (embedding): Embedding(16524, 300)
  (encoder): LSTM(300, 100, num_layers=2, batch_first=True, dropout=0.2)
  (fc): Linear(in_features=100, out_features=5, bias=True)
)
```

**Network Performance**

- Network has quite a lot overfitting

```
	Train Loss: 1.436 | Train Acc: 46.26%
	 Val. Loss: 1.552 |  Val. Acc: 32.71% 

	Train Loss: 1.352 | Train Acc: 54.78%
	 Val. Loss: 1.555 |  Val. Acc: 33.33% 

	Train Loss: 1.274 | Train Acc: 63.03%
	 Val. Loss: 1.573 |  Val. Acc: 31.12% 

	Train Loss: 1.195 | Train Acc: 71.44%
	 Val. Loss: 1.559 |  Val. Acc: 32.33% 

	Train Loss: 1.130 | Train Acc: 78.20%
	 Val. Loss: 1.578 |  Val. Acc: 30.47% 

	Train Loss: 1.076 | Train Acc: 83.39%
	 Val. Loss: 1.578 |  Val. Acc: 30.68% 

	Train Loss: 1.040 | Train Acc: 86.96%
	 Val. Loss: 1.598 |  Val. Acc: 27.80% 

	Train Loss: 1.019 | Train Acc: 89.10%
	 Val. Loss: 1.578 |  Val. Acc: 30.21% 

	Train Loss: 1.005 | Train Acc: 90.30%
	 Val. Loss: 1.567 |  Val. Acc: 31.91% 

	Train Loss: 0.987 | Train Acc: 92.09%
	 Val. Loss: 1.571 |  Val. Acc: 31.72% 
```

### Multilayer LSTM with Bidirectional

#### Without dropout in FC layer

**Network Architecture**

```
classifier(
  (embedding): Embedding(16524, 300)
  (encoder): LSTM(300, 100, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
  (fc): Linear(in_features=100, out_features=5, bias=True)
)
The model has 5,762,505 trainable parameters
```

**Network hyperparameter**

```
# Define hyperparameters
size_of_vocab = len(Sentence.vocab)
embedding_dim = 300
num_hidden_nodes = 100
num_output_nodes = 5
num_layers = 2
dropout = 0.2

# define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()
```

**Network Performance**

- Slight increase in Train accuracy w.r.t “Multilayer LSTM without Bidirectional”
- 2% increase in validation accuracy
- Network has quite a lot overfitting

```
     Train Loss: 1.569 | Train Acc: 30.03%
	 Val. Loss: 1.559 |  Val. Acc: 32.88% 

	Train Loss: 1.493 | Train Acc: 39.81%
	 Val. Loss: 1.527 |  Val. Acc: 34.47% 

	Train Loss: 1.370 | Train Acc: 53.84%
	 Val. Loss: 1.526 |  Val. Acc: 34.79% 

	Train Loss: 1.246 | Train Acc: 66.45%
	 Val. Loss: 1.543 |  Val. Acc: 33.75% 

	Train Loss: 1.154 | Train Acc: 76.01%
	 Val. Loss: 1.545 |  Val. Acc: 33.24% 

	Train Loss: 1.084 | Train Acc: 83.22%
	 Val. Loss: 1.552 |  Val. Acc: 32.90% 

	Train Loss: 1.041 | Train Acc: 87.08%
	 Val. Loss: 1.554 |  Val. Acc: 32.90% 

	Train Loss: 1.012 | Train Acc: 89.77%
	 Val. Loss: 1.546 |  Val. Acc: 34.19% 

	Train Loss: 0.995 | Train Acc: 91.37%
	 Val. Loss: 1.555 |  Val. Acc: 32.14% 

	Train Loss: 0.985 | Train Acc: 92.14%
	 Val. Loss: 1.551 |  Val. Acc: 33.33% 

```

#### With dropout

**Network Architecture**

```
classifier(
  (embedding): Embedding(16524, 300)
  (encoder): LSTM(300, 100, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (fc): Linear(in_features=100, out_features=5, bias=True)
)
The model has 5,762,505 trainable parameters
```

**Network Performance**

- 10% reduction in Train accuracy
- 1% reduction in validation accuracy
- Network has still quite a lot overfitting although due to dropout it has reduced a bit now but validation accuracy did not increase

```
	Train Loss: 1.573 | Train Acc: 29.20%
	 Val. Loss: 1.559 |  Val. Acc: 32.67% 

	Train Loss: 1.504 | Train Acc: 39.47%
	 Val. Loss: 1.532 |  Val. Acc: 35.28% 

	Train Loss: 1.414 | Train Acc: 49.27%
	 Val. Loss: 1.541 |  Val. Acc: 33.47% 

	Train Loss: 1.329 | Train Acc: 58.44%
	 Val. Loss: 1.556 |  Val. Acc: 31.40% 

	Train Loss: 1.250 | Train Acc: 66.58%
	 Val. Loss: 1.560 |  Val. Acc: 32.22% 

	Train Loss: 1.190 | Train Acc: 73.12%
	 Val. Loss: 1.562 |  Val. Acc: 31.61% 

	Train Loss: 1.148 | Train Acc: 77.27%
	 Val. Loss: 1.557 |  Val. Acc: 32.92% 

	Train Loss: 1.112 | Train Acc: 80.31%
	 Val. Loss: 1.564 |  Val. Acc: 32.46% 

	Train Loss: 1.087 | Train Acc: 82.91%
	 Val. Loss: 1.559 |  Val. Acc: 32.86% 

	Train Loss: 1.068 | Train Acc: 84.42%
	 Val. Loss: 1.579 |  Val. Acc: 30.87% 
```

#### Concatination of hidden layer 1 and 2

**Network hyperparameter**

- Here have added 2nd and 1st hidden layer output

```
        # Hidden = [batch size, hid dim * num directions]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        dense_outputs = self.dropout(self.fc(hidden))  
        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        # Final activation function softmax
        # output = F.softmax(dense_outputs[0], dim=1)
        output = F.softmax(dense_outputs, dim=1)
```

```
classifier(
  (embedding): Embedding(16524, 300)
  (encoder): LSTM(300, 100, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (fc): Linear(in_features=200, out_features=5, bias=True)
)
The model has 5,763,005 trainable parameters
```

**Network Performance**

- 28% reduction in Train accuracy
- 1% increase in validation accuracy
- Network has still quite a lot overfitting although due to dropout validation accuracy increased a bit but does not look worthy of this architecture

```
	Train Loss: 1.577 | Train Acc: 28.86%
	 Val. Loss: 1.552 |  Val. Acc: 31.97% 

	Train Loss: 1.507 | Train Acc: 38.72%
	 Val. Loss: 1.561 |  Val. Acc: 32.39% 

	Train Loss: 1.447 | Train Acc: 45.12%
	 Val. Loss: 1.554 |  Val. Acc: 32.77% 

	Train Loss: 1.399 | Train Acc: 50.40%
	 Val. Loss: 1.551 |  Val. Acc: 32.99% 

	Train Loss: 1.365 | Train Acc: 54.27%
	 Val. Loss: 1.554 |  Val. Acc: 33.37% 

	Train Loss: 1.322 | Train Acc: 58.59%
	 Val. Loss: 1.561 |  Val. Acc: 32.95% 

	Train Loss: 1.288 | Train Acc: 62.18%
	 Val. Loss: 1.584 |  Val. Acc: 30.40% 

	Train Loss: 1.255 | Train Acc: 65.40%
	 Val. Loss: 1.564 |  Val. Acc: 32.44% 

	Train Loss: 1.236 | Train Acc: 67.35%
	 Val. Loss: 1.574 |  Val. Acc: 31.27% 

	Train Loss: 1.214 | Train Acc: 69.68%
	 Val. Loss: 1.575 |  Val. Acc: 31.78% 
```

#### With Pretrained Gloves 100d embedding

**Network hyperparameter**

- embedding_dim = 100 has been set
- Gloves Embedding has been used
- num_hidden_nodes = 256 has been set

```
MAX_VOCAB_SIZE = 25_000

Sentence.build_vocab(train, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)
```

**Network Performance**

- 10% reduction in Train accuracy w.r.t “Without dropout in FC layer”
- Validation Accuracy almost similar

```
	Train Loss: 1.572 | Train Acc: 29.14%
	 Val. Loss: 1.533 |  Val. Acc: 34.89% 

	Train Loss: 1.506 | Train Acc: 39.19%
	 Val. Loss: 1.517 |  Val. Acc: 37.27% 

	Train Loss: 1.444 | Train Acc: 45.85%
	 Val. Loss: 1.528 |  Val. Acc: 37.05% 

	Train Loss: 1.394 | Train Acc: 50.99%
	 Val. Loss: 1.532 |  Val. Acc: 34.34% 

	Train Loss: 1.338 | Train Acc: 57.23%
	 Val. Loss: 1.561 |  Val. Acc: 33.03% 

	Train Loss: 1.288 | Train Acc: 61.94%
	 Val. Loss: 1.562 |  Val. Acc: 32.46% 

	Train Loss: 1.246 | Train Acc: 66.55%
	 Val. Loss: 1.555 |  Val. Acc: 33.75% 

	Train Loss: 1.215 | Train Acc: 69.73%
	 Val. Loss: 1.558 |  Val. Acc: 33.16% 

	Train Loss: 1.184 | Train Acc: 73.00%
	 Val. Loss: 1.554 |  Val. Acc: 33.81% 

	Train Loss: 1.158 | Train Acc: 75.23%
	 Val. Loss: 1.575 |  Val. Acc: 31.84% 

	Train Loss: 1.138 | Train Acc: 77.40%
	 Val. Loss: 1.556 |  Val. Acc: 33.60% 

	Train Loss: 1.127 | Train Acc: 78.09%
	 Val. Loss: 1.542 |  Val. Acc: 35.23% 

	Train Loss: 1.113 | Train Acc: 79.51%
	 Val. Loss: 1.556 |  Val. Acc: 33.75% 

	Train Loss: 1.102 | Train Acc: 80.61%
	 Val. Loss: 1.553 |  Val. Acc: 33.94% 

	Train Loss: 1.093 | Train Acc: 81.70%
	 Val. Loss: 1.555 |  Val. Acc: 33.79% 
```

#### With Pretrained Gloves 100d embedding and Freezing/Unfreezing Embedded

**Network Hyperparameter**

- Exactly same network as “With Pretrained Gloves 100d embedding”

**Network Performance**

- Made embedding frozen for 15 epoch and then unfreezed embedding layer to train for 15 more epoch
- Validation Accuracy increased by 6% w.r.t to “With Pretrained Gloves 100d embedding”
- Training Accuracy increased by 3% w.r.t to “With Pretrained Gloves 100d embedding”

```
	Train Loss: 1.293 | Train Acc: 61.33%
	 Val. Loss: 1.487 |  Val. Acc: 40.76% 

	Train Loss: 1.252 | Train Acc: 65.49%
	 Val. Loss: 1.498 |  Val. Acc: 39.56% 

	Train Loss: 1.211 | Train Acc: 69.94%
	 Val. Loss: 1.497 |  Val. Acc: 39.26% 

	Train Loss: 1.182 | Train Acc: 72.76%
	 Val. Loss: 1.494 |  Val. Acc: 39.43% 

	Train Loss: 1.162 | Train Acc: 75.27%
	 Val. Loss: 1.502 |  Val. Acc: 38.71% 

	Train Loss: 1.140 | Train Acc: 77.16%
	 Val. Loss: 1.518 |  Val. Acc: 36.70% 

	Train Loss: 1.132 | Train Acc: 77.76%
	 Val. Loss: 1.528 |  Val. Acc: 35.44% 

	Train Loss: 1.110 | Train Acc: 79.83%
	 Val. Loss: 1.511 |  Val. Acc: 38.33% 

	Train Loss: 1.100 | Train Acc: 81.13%
	 Val. Loss: 1.523 |  Val. Acc: 36.72% 

	Train Loss: 1.094 | Train Acc: 81.78%
	 Val. Loss: 1.517 |  Val. Acc: 37.52% 

	Train Loss: 1.084 | Train Acc: 82.57%
	 Val. Loss: 1.512 |  Val. Acc: 37.92% 

	Train Loss: 1.079 | Train Acc: 83.33%
	 Val. Loss: 1.518 |  Val. Acc: 37.52% 

	Train Loss: 1.076 | Train Acc: 83.21%
	 Val. Loss: 1.511 |  Val. Acc: 38.30% 

	Train Loss: 1.076 | Train Acc: 83.57%
	 Val. Loss: 1.509 |  Val. Acc: 38.37% 

	Train Loss: 1.067 | Train Acc: 84.31%
	 Val. Loss: 1.509 |  Val. Acc: 39.38% 
```

#### Data Augmentation and Gloves 200D

**Network Hyperparameter**

- Gloves 200d has been used

- Embedded layer has been frozen

  ```
  classifier(
    (embedding): Embedding(17052, 200, padding_idx=1)
    (encoder): LSTM(200, 256, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
    (dropout): Dropout(p=0.2, inplace=False)
    (fc): Linear(in_features=256, out_features=5, bias=True)
  )
  The model has 7,503,589 trainable parameters
  ```

  

**Network Performance**

- Highest validation accuracy is **51.38% which is almost 12%** higher compare to previous one

```
	Train Loss: 1.476 | Train Acc: 42.70%
	 Val. Loss: 1.408 |  Val. Acc: 47.80% 

	Train Loss: 1.398 | Train Acc: 51.32%
	 Val. Loss: 1.402 |  Val. Acc: 48.90% 

	Train Loss: 1.339 | Train Acc: 57.50%
	 Val. Loss: 1.410 |  Val. Acc: 48.37% 

	Train Loss: 1.279 | Train Acc: 64.02%
	 Val. Loss: 1.387 |  Val. Acc: 51.38% 

	Train Loss: 1.239 | Train Acc: 67.94%
	 Val. Loss: 1.390 |  Val. Acc: 50.68% 

	Train Loss: 1.208 | Train Acc: 71.08%
	 Val. Loss: 1.391 |  Val. Acc: 50.21% 

	Train Loss: 1.186 | Train Acc: 73.49%
	 Val. Loss: 1.390 |  Val. Acc: 51.00% 

	Train Loss: 1.166 | Train Acc: 75.43%
	 Val. Loss: 1.415 |  Val. Acc: 48.28% 

	Train Loss: 1.152 | Train Acc: 76.78%
	 Val. Loss: 1.418 |  Val. Acc: 48.12% 

	Train Loss: 1.139 | Train Acc: 78.30%
	 Val. Loss: 1.417 |  Val. Acc: 48.37% 

	Train Loss: 1.130 | Train Acc: 79.20%
	 Val. Loss: 1.409 |  Val. Acc: 48.88% 

	Train Loss: 1.120 | Train Acc: 80.00%
	 Val. Loss: 1.399 |  Val. Acc: 50.04% 

	Train Loss: 1.114 | Train Acc: 80.68%
	 Val. Loss: 1.403 |  Val. Acc: 49.89% 

	Train Loss: 1.105 | Train Acc: 81.79%
	 Val. Loss: 1.400 |  Val. Acc: 50.15% 

	Train Loss: 1.105 | Train Acc: 81.58%
	 Val. Loss: 1.410 |  Val. Acc: 49.13% 
```



### GRU

### Multilayer with Bidirectional and Data Augmentation

#### Network Architecture

```
classifier(
  (embedding): Embedding(17033, 200, padding_idx=1)
  (encoder): GRU(200, 256, num_layers=5, batch_first=True, dropout=0.4, bidirectional=True)
  (dropout): Dropout(p=0.4, inplace=False)
  (fc): Linear(in_features=256, out_features=5, bias=True)
)
The model has 8,842,253 trainable parameters
```

```
# Define hyperparameters
size_of_vocab = len(Sentence.vocab)
embedding_dim = 200
num_hidden_nodes = 256
num_output_nodes = 5
num_layers = 5
dropout = 0.4
bidirectional = True
PAD_IDX = Sentence.vocab.stoi[Sentence.pad_token]

# Instantiate the model
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, dropout, bidirectional,PAD_IDX)
```

#### Network Performance

```
	Train Loss: 1.536 | Train Acc: 35.11%
	 Val. Loss: 1.492 |  Val. Acc: 39.38% 

	Train Loss: 1.448 | Train Acc: 45.68%
	 Val. Loss: 1.496 |  Val. Acc: 39.58% 

	Train Loss: 1.372 | Train Acc: 53.58%
	 Val. Loss: 1.507 |  Val. Acc: 37.69% 

	Train Loss: 1.331 | Train Acc: 58.04%
	 Val. Loss: 1.506 |  Val. Acc: 37.54% 

	Train Loss: 1.299 | Train Acc: 61.12%
	 Val. Loss: 1.506 |  Val. Acc: 38.31% 

	Train Loss: 1.279 | Train Acc: 63.08%
	 Val. Loss: 1.508 |  Val. Acc: 38.83% 

	Train Loss: 1.265 | Train Acc: 64.72%
	 Val. Loss: 1.518 |  Val. Acc: 38.54% 

	Train Loss: 1.258 | Train Acc: 65.57%
	 Val. Loss: 1.505 |  Val. Acc: 39.09% 

	Train Loss: 1.250 | Train Acc: 66.24%
	 Val. Loss: 1.510 |  Val. Acc: 38.73% 

	Train Loss: 1.249 | Train Acc: 66.04%
	 Val. Loss: 1.512 |  Val. Acc: 37.92% 

	Train Loss: 1.242 | Train Acc: 66.82%
	 Val. Loss: 1.508 |  Val. Acc: 38.67% 

	Train Loss: 1.242 | Train Acc: 66.92%
	 Val. Loss: 1.499 |  Val. Acc: 40.25% 

	Train Loss: 1.239 | Train Acc: 67.09%
	 Val. Loss: 1.535 |  Val. Acc: 36.04% 

	Train Loss: 1.241 | Train Acc: 66.95%
	 Val. Loss: 1.529 |  Val. Acc: 36.27% 

	Train Loss: 1.239 | Train Acc: 66.97%
	 Val. Loss: 1.503 |  Val. Acc: 39.77% 

```

