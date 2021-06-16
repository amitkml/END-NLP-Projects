
# Model NLP with Python

## First RNN for NLP

### Sequnce to Sequence Network
This network has two components and they are Encoder and Decoder. Pictorically it can be represented as ![sequence to sequence](https://miro.medium.com/max/3170/1*sO-SP58T4brE9EHazHSeGA.png).

- It contains an encoder RNN (LSTM) and a decoder rnn. 
- One to ‘understand’ the input sequence and the decoder to ‘decode’ the ‘thought vector’ and construct an output sequence.
- Words are embededed into Vectprs
- Each new state into Encoder is computed from previous cell output and next word
- Final state of the Encoder fed into Decoder
- Information from initial state gets faded slowly

### Attention
- We can explain the relationship between words in one sentence or close context. When we see “eating”, we expect to encounter a food word very soon. The color term describes the food, but probably not so much with “eating” directly. ![im](https://lilianweng.github.io/lil-log/assets/images/sentence-example-attention.png)
- Global context vector
- Ues all input sequence and last state from encoder
- Attention in deep learning can be broadly interpreted as a vector of **importance weights**: in order to predict or infer one element, such as a pixel in an image or a word in a sentence, we estimate using the attention vector how strongly it is correlated with (or “attends to” as you may have read in many papers) other elements and take the sum of their values weighted by the attention vector as the approximation of the target.
- The mechanism allows the model to focus and place more “Attention” on the relevant parts of the input sequence as needed. ![im](https://blog.floydhub.com/content/images/2019/09/Slide41-1.JPG)

### Transformer - Intution
- Transformer only uses Attention Network and does not use RNN
- The encoding component is a stack of encoders (the paper stacks six of them on top of each other – there’s nothing magical about the number six, one can definitely experiment with other arrangements). The decoding component is a stack of decoders of the same number. ![im](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)
- The encoders are all identical in structure (yet they do not share weights). Each one is broken down into two sub-layers.
- The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position.
- The encoder’s inputs first flow through a self-attention layer – a layer that helps the encoder look at other words in the input sentence as it encodes a specific word. We’ll look closer at self-attention later in the post. ![im](https://jalammar.github.io/images/t/Transformer_encoder.png)
- The decoder has both those layers, but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence. ![im](https://jalammar.github.io/images/t/Transformer_decoder.png)
