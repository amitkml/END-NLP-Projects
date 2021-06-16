
# Model NLP with Python

## First RNN for NLP

**Sequnce to Sequence Network**
This network has two components and they are Encoder and Decoder. Pictorically it can be represented as ![sequence to sequence](https://miro.medium.com/max/3170/1*sO-SP58T4brE9EHazHSeGA.png).

- It contains an encoder RNN (LSTM) and a decoder rnn. 
- One to ‘understand’ the input sequence and the decoder to ‘decode’ the ‘thought vector’ and construct an output sequence.
- Words are embededed into Vectprs
- Each new state into Encoder is computed from previous cell output and next word
- Final state of the Encoder fed into Decoder
- 
