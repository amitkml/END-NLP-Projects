# Convolutional Seq2Seq



## Representation of Text for Convolution 1D

Starting point of 1D Conv is that we have a text input and the text is represented with a set of vector. The following example shows that each word of the sentence is being represented by corresponding dense vector of 4d(example). So the input is same as image vector value and only difference here is that it is 1d.

![image](https://github.com/amitkml/END-NLP-Projects/blob/main/Convolutional%20Seq2Seq/src/Conv1d_Text.JPG?raw=true)

We, then apply a Filter and our traditional filter size is 3x3.