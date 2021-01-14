[TOC]

# Language Translation Network design from Scratch

It is essential for us to understand tensor size/shape during network architecture analysis and this will help us to decide how to merge/permute/concat tensors.

This assignments is aimed towards writing the code from scratch and understanding of network tensors and how they are being updated through different layers.

Have tried through following network for this English to German translation.

- Translation through Encoder & Decoder Network
- Translation through Encoder & Decoder with combination of Attention mechanism into decoder
- Translation through Encoder & Decoder with combination of Attention mechanism into decoder with Padding and masking

## Analysis Summary

| Parameter        | Encoder - Decoder | Encoder Decoder with Attention | Encoder Decoder with Attention and Padding, Masking |
| ---------------- | ----------------- | ------------------------------ | --------------------------------------------------- |
| Final PPL        | 43.400            | 21.779                         | 23.087                                              |
| Epoch Time       | 0m 47s            | 1m 26s                         | 0m 35s                                              |
| No of Parameters | 18,101,509        | 20,518,917                     | 20,518,917                                          |
|                  |                   |                                |                                                     |

## Encoder & Decoder Network

![Image](https://miro.medium.com/max/875/0*BpmYIit1tmLKlpDm.png)

![im](http://www.wildml.com/wp-content/uploads/2016/04/nct-seq2seq.png) 

The encoder simply takes the input data, and train on it then it passes the last state of its recurrent layer as an initial state to the first recurrent layer of the decoder part.

The decoder takes the last state of encoder’s last recurrent layer and uses it as an initial state to its first recurrent layer , the input of the decoder is the sequences that we want to get ( in our case French sentences).

![im](https://miro.medium.com/proxy/1*3lj8AGqfwEE5KCTJ-dXTvg.png)

### Encoder

1. Input Layer : Takes the English sentence and pass it to the embedding layer.
2. Embedding Layer : Takes the English sentence and convert each word to fixed size vector
3. First LSTM Layer : Every time step, it takes a vector that represents a word and pass its output to the next layer, We used CuDNNLSTM layer instead of LSTM because it’s much much faster.
4. Second LSTM Layer : It does the same thing as the previous layer, but instead of passing its output, it passes its states to the first LSTM layer of the decoder .

![im](https://miro.medium.com/max/875/1*3pH2NH_8i7QMxpV0TFOdxw.jpeg)

#### Building the Encoder

```
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers      
        self.embedding = nn.Embedding(input_dim, emb_dim)        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)       
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        ## SHAPE: [23, 128] (Length varrying here but our batch size is fixed and remains as 128)
        embedded = self.dropout(self.embedding(src))        
        #embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs = [src len, batch size, hid dim * n directions]

        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer        
        return hidden, cell
```

  ### Decoder

1. Input Layer : Takes the French sentence and pass it to the embedding layer.

2. Embedding Layer : Takes the French sentence and convert each word to fixed size vector

3. First LSTM Layer : Every time step, it takes a vector that represents a word and pass its output to the next layer, but here in the decoder, we initialize the state of this layer to be the last state of the last LSTM layer from the decoder .

4. Second LSTM Layer : Processing the output from the previous layer and passes its output to a dense layer .

5. Dense Layer (Output Layer) : Takes the output from the previous layer and outputs a one hot vector representing the target French word

   **Note :**

   We have to know that we don’t convert each English sentence into French in one time step, we do that in a number of time steps that equals the number of words that the longest English sentence has.

   so if the longest English sentence has 10 words, we have to take 10 time steps to get its French translation.

   ![im](https://miro.medium.com/max/875/1*sDlV9_-PXBlt8jol-7Xjhg.jpeg)

#### Building the Encoder

```
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]   

        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))

return prediction, hidden, cell
```

### Encoder and decoder together into single model

```
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
```

### Network Performance

- Best PPL value on validation data is **42.474 after 20 epoch**
- Network took approximately 47s

```
Epoch: 01 | Time: 0m 47s
	Train Loss: 5.105 | Train PPL: 164.846
	 Val. Loss: 4.798 |  Val. PPL: 121.270
Epoch: 02 | Time: 0m 46s
	Train Loss: 4.566 | Train PPL:  96.195
	 Val. Loss: 4.702 |  Val. PPL: 110.175
Epoch: 03 | Time: 0m 46s
	Train Loss: 4.181 | Train PPL:  65.441
	 Val. Loss: 4.528 |  Val. PPL:  92.582
Epoch: 04 | Time: 0m 46s
	Train Loss: 3.992 | Train PPL:  54.173
	 Val. Loss: 4.427 |  Val. PPL:  83.713
Epoch: 05 | Time: 0m 46s
	Train Loss: 3.852 | Train PPL:  47.096
	 Val. Loss: 4.380 |  Val. PPL:  79.812
Epoch: 06 | Time: 0m 47s
	Train Loss: 3.732 | Train PPL:  41.764
	 Val. Loss: 4.266 |  Val. PPL:  71.219
Epoch: 07 | Time: 0m 46s
	Train Loss: 3.619 | Train PPL:  37.292
	 Val. Loss: 4.424 |  Val. PPL:  83.413
Epoch: 08 | Time: 0m 46s
	Train Loss: 3.499 | Train PPL:  33.095
	 Val. Loss: 4.197 |  Val. PPL:  66.508
Epoch: 09 | Time: 0m 47s
	Train Loss: 3.380 | Train PPL:  29.367
	 Val. Loss: 4.100 |  Val. PPL:  60.361
Epoch: 10 | Time: 0m 46s
	Train Loss: 3.301 | Train PPL:  27.138
	 Val. Loss: 4.063 |  Val. PPL:  58.126
Epoch: 11 | Time: 0m 47s
	Train Loss: 3.185 | Train PPL:  24.168
	 Val. Loss: 3.983 |  Val. PPL:  53.663
Epoch: 12 | Time: 0m 46s
	Train Loss: 3.095 | Train PPL:  22.076
	 Val. Loss: 3.952 |  Val. PPL:  52.062
Epoch: 13 | Time: 0m 46s
	Train Loss: 3.001 | Train PPL:  20.102
	 Val. Loss: 3.829 |  Val. PPL:  46.012
Epoch: 14 | Time: 0m 46s
	Train Loss: 2.908 | Train PPL:  18.317
	 Val. Loss: 3.860 |  Val. PPL:  47.455
Epoch: 15 | Time: 0m 46s
	Train Loss: 2.861 | Train PPL:  17.471
	 Val. Loss: 3.810 |  Val. PPL:  45.149
Epoch: 16 | Time: 0m 47s
	Train Loss: 2.771 | Train PPL:  15.971
	 Val. Loss: 3.787 |  Val. PPL:  44.109
Epoch: 17 | Time: 0m 47s
	Train Loss: 2.693 | Train PPL:  14.777
	 Val. Loss: 3.757 |  Val. PPL:  42.841
Epoch: 18 | Time: 0m 46s
	Train Loss: 2.635 | Train PPL:  13.946
	 Val. Loss: 3.771 |  Val. PPL:  43.425
Epoch: 19 | Time: 0m 47s
	Train Loss: 2.559 | Train PPL:  12.929
	 Val. Loss: 3.749 |  Val. PPL:  42.474
Epoch: 20 | Time: 0m 46s
	Train Loss: 2.478 | Train PPL:  11.912
	 Val. Loss: 3.768 |  Val. PPL:  43.285
```

### Generating output

- Extract sample data
- Tokenize and push to GPU
- model is being being put into eval mode
- Pass on source and target to model
- Take the output (wherever prob value is max) and then map with TRG vocab to find out word from id

```
## Extracting the sample data
example_idx = 0
example = train_data.examples[example_idx]
print('source sentence: ', ' '.join(example.src))
print('target sentence: ', ' '.join(example.trg))

## Process the source and target data
src_tensor = SRC.process([example.src]).to(device)
trg_tensor = TRG.process([example.trg]).to(device)
print(trg_tensor.shape)

model.eval()
with torch.no_grad():
    outputs = model(src_tensor, trg_tensor, teacher_forcing_ratio=0)

outputs.shape

output_idx = outputs[1:].squeeze(1).argmax(1)
' '.join([TRG.vocab.itos[idx] for idx in output_idx])
```

**Here is how output is**

| Source Sentence                                              | Ground Truth                                           | Predicted Text                                            |
| ------------------------------------------------------------ | ------------------------------------------------------ | --------------------------------------------------------- |
| . büsche vieler nähe der in freien im sind männer weiße junge zwei | two young , white males are outside near many bushes . | two young boys are in in the background with their arms . |

### Limitation of Above network

- We have not used Batching Sequence data with bucketing. A robust optimization requires to work on large batches of utterances and training time as well as recognition performance can vary strongly depending on the choice of how batches were put together.The main reason is that combining utterances of different lengths in a mini-batch requires to extend the length of each utterance to that of the longest utterance within the batch, usually by appending zeros. T**hese zero frames are ignored later on when gradients are computed but the forward-propagation of zeros through the RNN is a waste of computing power**
- Learning rate is being fixed and we should vary the LR

## How Bucketing prevents waste computation power?

- The idea is to perform a bucketing of the training corpus, where each bucket represents a range of utterance lengths and each training sample is assigned to the bucket that corresponds to its length.
- buckets too large will increase training time due to irrelevant computations on zero padded frames.

![im](https://miro.medium.com/max/770/0*S1Fv9pK1qvw8biOn.png)

![im](https://miro.medium.com/max/770/1*hcGuja_d5Z_rFcgwe9dPow.png)

 ## Encoder & Decoder with combination of Attention mechanism into decoder

> A potential issue with this encoder–decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector. This may make it difficult for the neural network to cope with long sentences, especially those that are longer than the sentences in the training corpus.

— [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), 2015.

Now, we can enhance the above Seq2Seq network with Attention mechanism. Below is a picture of this approach taken from the paper. **Note the dotted lines explictly showing the use of the decoders attended hidden state output (ht) providing input to the decoder on the next timestep**.



![image](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Feeding-Hidden-State-as-Input-to-Decoder.png)

Instead of encoding the input sequence into a single fixed context vector, the attention model develops a context vector that is filtered specifically for each output time step.

![im](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Example-of-Attention.png)

# Encoder-Decoder Papers

- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), 2014.
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), 2014.

### Attention Papers

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), 2015.
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044), 2015.
- [Hierarchical Attention Networks for Document Classification](https://www.microsoft.com/en-us/research/publication/hierarchical-attention-networks-document-classification/), 2016.
- [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://aclweb.org/anthology/P/P16/P16-2034.pdf), 2016
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025), 2015.

### More on Attention

- [Attention in Long Short-Term Memory Recurrent Neural Networks](http://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)
- [Lecture 10: Neural Machine Translation and Models with Attention](https://www.youtube.com/watch?v=IxQtK2SjWWM), Stanford, 2017
- [Lecture 8 – Generating Language with Attention](https://www.youtube.com/watch?v=ah7_mfl7LD0), Oxford.