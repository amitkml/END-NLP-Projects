# What is this all about?

This session is all about implementation of Multi Head attention network and its implementation.

### Understanding why self-attention

Let’s say you run a movie rental business and you have some movies, and some users, and you would like to recommend movies to your users that they are likely to enjoy.

One way to go about this, is to create manual features for your movies, such as how much romance there is in the movie, and how much action, and then to design corresponding features for your users: how much they enjoy romantic movies and how much they enjoy action-based movies. If you did this, the dot product between the two feature vectors would give you a score for how well the attributes of the movie match what the user enjoys.

![o](http://peterbloem.nl/files/transformers/dot-product.svg)

If the signs of a feature match for the user and the movie—the movie is romantic and the user loves romance or the movie is *unromantic* and the user hates romance—**then the resulting dot product gets a positive term for that feature**. If the signs don’t match—the movie is romantic and the user hates romance or vice versa—**the corresponding term is negative**

Furthermore, the ***magnitudes*** of the features indicate how much the feature should contribute to the total score: a movie may be a little romantic, but not in a noticeable way, or a user may simply prefer no romance, but be largely ambivalent.

This is the basic principle at work in the self-attention.  Let’s say we are faced with a sequence of words. To apply self-attention, we simply assign each word t in our vocabulary an *embedding vector* 𝐯t (the values of which we’ll learn). This is what’s known as an *embedding layer* in sequence modeling. It turns the word sequence **the,cat,walks,on,the,street** into the vector sequence 𝐯(the),𝐯(cat),𝐯(walks),𝐯(on),𝐯(the),𝐯(street)

If we feed this sequence into a self-attention layer, the output is another sequence of vectors *𝐲(the),𝐲(cat),𝐲(walks),𝐲(on),𝐲(the),𝐲(street)* where *𝐲(cat)* is a weighted sum over all the embedding vectors in the first sequence, weighted by their (normalized) dot-product with 𝐯cat.

Since we are *learning* what the values in 𝐯t should be, how "related" two words are is entirely determined by the task. In most cases, the definite article the is not very relevant to the interpretation of the other words in the sentence; therefore, we will likely end up with an embedding 𝐯(the) that has a low or negative dot product with all other words. **On the other hand, to interpret what walks means in this sentence, it's very helpful to work out *who* is doing the walking. This is likely expressed by a noun, so for nouns like cat and verbs like walks**, we will likely learn embeddings 𝐯(cat) and 𝐯(walks) that have a high, positive dot product together.

This is the basic intuition behind self-attention. The *dot product expresses how related two vectors in the input sequence are, with “related” defined by the learning task, and the output vectors are weighted sums over the whole input sequence, with the weights determined by these dot products*

## What is attention network?

![image](https://lilianweng.github.io/lil-log/assets/images/encoder-decoder-attention.png)
- The encoder is a bidirectional RNN (or other recurrent network setting of your choice) with a forward hidden state h→i and a backward one h←i. 

- A simple concatenation of two represents the encoder state. 

- The decoder network has hidden state 
  $$
  st=f(st−1,yt−1,ct)
  $$
  for the output word at position t, t=1,…,m, where the context vector ct is a sum of hidden states of the input sequence, weighted by alignment scores:

The matrix of alignment scores is a nice byproduct to explicitly show the correlation between source and target words.
![im](https://lilianweng.github.io/lil-log/assets/images/bahdanau-fig3.png)

