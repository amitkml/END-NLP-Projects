[TOC]

# Explained CNN for Text Architecture

The notebook https://github.com/amitkml/END-NLP-Projects/blob/main/Convolutional%20Seq2Seq/END_NLP_Assignment_11_Convolutional_Seq2S.ipynb has comments been added for every blocks



## How CNN works for Text?

A PyTorch CNN for classifying the sentiment of movie reviews, based on the paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) by Yoon Kim (2014).

![im](https://raw.githubusercontent.com/cezannec/CNN_Text_Classification/master/notebook_ims/complete_embedding_CNN.png)

*Image from the original paper, Convolutional Neural Networks for Sentence Classification.*

# Convolutional Seq2Seq



## Representation of Text for Convolution 1D

Starting point of 1D Conv is that we have a text input and the text is represented with a set of vector. The following example shows that each word of the sentence is being represented by corresponding dense vector of 4d(example). So the input is same as image vector value and only difference here is that it is 1d.

In the case of text classification, a **convolutional kernel will still  be a sliding window**, only its job is to **look at embeddings** for multiple  words, rather than small areas of pixels in an image. The dimensions of  the convolutional kernel will also have to change, according to this  task. To look at sequences of word embeddings, we want a window to look  at multiple word embeddings in a sequence. The kernels will no longer be  square, instead, they will be a wide rectangle with dimensions like  3x300 or 5x300 (assuming an embedding length of 300).

![image](https://github.com/amitkml/END-NLP-Projects/blob/main/Convolutional%20Seq2Seq/src/Conv1d_Text.JPG?raw=true)

We, then apply a Filter and our traditional filter size is 3x3. Let’s take the following filter example ![im](https://github.com/amitkml/END-NLP-Projects/blob/main/Convolutional%20Seq2Seq/src/filter.JPG?raw=true)
So, the filter is going to take 3 words at a time and across 4 dimension (referred as channel).

- The height of the kernel will be the number of embeddings it will see at once, similar to representing an [n-gram](https://en.wikipedia.org/wiki/N-gram) in a word model.
- The width of the kernel should span the length of an entire word embedding.

![im](https://cezannec.github.io/assets/cnn_text/conv_dimensions.png)

## Convolution over Word Sequences

Below, we have a toy example that shows each word encoded as an embedding with 3 values, usually, this will contain many more values, but this is for ease of visualization. To look at two words in this example sequence, we can use a 2x3 convolutional kernel. The kernel weights are placed on top of two word 
embeddings; in this case, the downwards-direction represents time, so, the word “movie” comes right after “good” in this short sequence. The kernel weights and embedding values are multiplied in pairs and then 
summed to get a **single output value** of 0.54.

![im](https://cezannec.github.io/assets/cnn_text/conv_kernel_operation.gif)

In this way, the convolution operation can be viewed as **window-based feature extraction**,
where the features are patterns in sequential word groupings that indicate traits like the sentiment of a text, the grammatical function of different words, and so on.

**It is also worth noting that the number of input channels for this convolutional layer—a value that is typically 1 for grayscale images or 3 for RGB images; will be 1 in this case, because a single, input text** **source will be encoded as *one* list of word embeddings.**

In the following example of 1d CONV, we see that our feature output is being shrink and it has come to down from 7 to 5.

![im](https://github.com/amitkml/END-NLP-Projects/blob/main/Convolutional%20Seq2Seq/src/conv_filter_output.JPG?raw=true)

So, in order to avoid this, we add padding for text data also as similar to convolution as shared below. So here my output is same size as input vector.	

![im](https://github.com/amitkml/END-NLP-Projects/blob/main/Convolutional%20Seq2Seq/src/conv_filter_output_padding.JPG?raw=true)

## Recognizing General Patterns

There is another nice property of this convolutional operation.  Recall that similar words will have similar embeddings and a convolution  operation is just a linear operation on these vectors. So, when a  convolutional kernel is applied to different sets of similar words, it  will produce a similar output value!

In the below example, **you can see that the convolutional output value  for the input 2-grams “good movie” and “fantastic song” are about the  same because the word embeddings for those pairs of words are also very  similar**. 

![im](https://cezannec.github.io/assets/cnn_text/similar_phrases_conv_out.png)

In this example, the convolutional kernel has learned to capture a more general feature; not just a good movie or song, but a *positive * thing, generally. Recognizing these kinds of high-level features can be
especially useful in text classification tasks, which often rely on general grouping

## 1D Convolutions

A single kernel will move one-by-one down a list of input embeddings, looking at the first word embedding (and a small window of next-word embeddings) then the next word embedding, and the next, and so on. The resultant output will be a feature vector that contains about as many values as there were input embeddings, so the input sequence size does matter.

![im](https://cezannec.github.io/assets/cnn_text/conv_1D_time.gif)



## Maxpooling over Time

 One thing to think about is how a feature vector might look when applied to an important phrase in a text source. If we are trying to classify movie reviews, and we see the phrase, “great plot,” it doesn’t 
matter *where* this appears in a review; it is a good indicator that this is a positive review, no matter its location in the source text.

**The following output shows the key word coming out after max pooling is amazing and hence it can indicate sentence key word/pattern.** 

![im](https://cezannec.github.io/assets/cnn_text/maxpooling_over_time.png)

Since this operation is looking at a sequence of local feature values, it is often called **maxpooling over time**

Sometime, we also do **k-max pooling** whereby we take k no of max values column wise. Following example shows that we are taking here 2-max pooling.

![im](https://github.com/amitkml/END-NLP-Projects/blob/main/Convolutional%20Seq2Seq/src/con1d_kmp.JPG?raw=true)





## Convolution1d with Pytorch

Here is a sample code which explains how we can use Pytorch for conv1d. Key point here to note here is that input no of channel is word_embedding_size.

![im](https://github.com/amitkml/END-NLP-Projects/blob/main/Convolutional%20Seq2Seq/src/conv1d_pytorch.JPG?raw=true)

















