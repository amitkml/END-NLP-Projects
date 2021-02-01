# What is this all about?

This session is all about implementation of Multi Head attention network and its implementation.

## What is attention network?
![image](https://lilianweng.github.io/lil-log/assets/images/encoder-decoder-attention.png)
- The encoder is a bidirectional RNN (or other recurrent network setting of your choice) with a forward hidden state h→i and a backward one h←i. 
- A simple concatenation of two represents the encoder state. 
- The decoder network has hidden state st=f(st−1,yt−1,ct) for the output word at position t, t=1,…,m, where the context vector ct is a sum of hidden states of the input sequence, weighted by alignment scores:

The matrix of alignment scores is a nice byproduct to explicitly show the correlation between source and target words.
![im](https://lilianweng.github.io/lil-log/assets/images/bahdanau-fig3.png)

