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

**It is also worth noting that the number of input channels for this convolutional layer—a value that is typically 1 for grayscale images or 3 for RGB images; will be 1 in this case, because a single, input text** **source will be encoded as *one* list of word embeddings.**![im]()











