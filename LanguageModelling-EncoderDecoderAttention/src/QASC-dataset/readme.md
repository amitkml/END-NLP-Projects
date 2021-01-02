# QASC: A Dataset for Question Answering via Sentence Composition
QASC is a question-answering dataset with a focus on sentence composition. It consists of 9,980 8-way multiple-choice questions about grade school science (8,134 train, 926 dev, 920 test), and comes with a corpus of 17M sentences. This repository shows how to download the QASC dataset and corpus. Note that the test set does not have the answer key or fact annotations. To evaluate your model on the test set, please submit your predictions (limited to once/week to prevent over-fitting) to the QASC leaderboard in the CSV format described here. We also provide two sample baseline models that can be used to produce the predictions in this CSV format.

# Key Links
- QASC Dataset: http://data.allenai.org/downloads/qasc/qasc_dataset.tar.gz
- QASC Corpus: http://data.allenai.org/downloads/qasc/qasc_corpus.tar.gz
- Leaderboard: https://leaderboard.allenai.org/qasc
- Paper: [QASC: A Dataset for Question Answering via Sentence Composition](https://arxiv.org/abs/1910.11473)

# Downloading Data

Download and unzip the dataset into the data/QASC_Dataset folder:

```
mkdir -p data
wget http://data.allenai.org/downloads/qasc/qasc_dataset.tar.gz
tar xvfz qasc_dataset.tar.gz  -C data/
rm qasc_dataset.tar.gz
```

# Download

- For the full documentation of the dataset and its format please refer to our [Github repository](https://github.com/allenai/Break).
- Click here to [download Break](https://github.com/allenai/Break/raw/master/break_dataset/Break-dataset.zip).

