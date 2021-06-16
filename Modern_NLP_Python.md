
# Model NLP with Python

## First RNN for NLP

**Sequnce to Sequence Network**
This network has two components and they are Encoder and Decoder. Pictorically it can be represented as ![sequence to sequence](https://miro.medium.com/max/3170/1*sO-SP58T4brE9EHazHSeGA.png).

- It contains an encoder RNN (LSTM) and a decoder rnn. 
- One to ‘understand’ the input sequence and the decoder to ‘decode’ the ‘thought vector’ and construct an output sequence.
- Words are embededed into Vectprs
- Each new state into Encoder is computed from previous cell output and next word
- Final state of the Encoder fed into Decoder
- Information from initial state gets faded slowly

**Attention**
- We can explain the relationship between words in one sentence or close context. When we see “eating”, we expect to encounter a food word very soon. The color term describes the food, but probably not so much with “eating” directly. ![im](https://lilianweng.github.io/lil-log/assets/images/sentence-example-attention.png)
- Global context vector
- Ues all input sequence and last state from encoder
- Attention in deep learning can be broadly interpreted as a vector of **importance weights**: in order to predict or infer one element, such as a pixel in an image or a word in a sentence, we estimate using the attention vector how strongly it is correlated with (or “attends to” as you may have read in many papers) other elements and take the sum of their values weighted by the attention vector as the approximation of the target.
- The mechanism allows the model to focus and place more “Attention” on the relevant parts of the input sequence as needed. ![im](https://blog.floydhub.com/content/images/2019/09/Slide41-1.JPG)
