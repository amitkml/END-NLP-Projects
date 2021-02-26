# Transformer Network
The Transformer architecture featuting a two-layer Encoder / Decoder. The Encoder processes all three elements of the input sequence (w1, w2, and w3) in parallel, whereas the Decoder generates each element sequentially (only timesteps 0 and 1, where the output sequence elements v1 and v2 are generated, are depicted). Output token generation continues until an end of the sentence token <EOS> appears.
- The inputs to the Decoder come in two varieties: the hidden states that are outputs of the Encoder (these are used for the Encoder-Decoder Attention within each Decoder layer) and the previously generated tokens of the output sequence (for the Decoder Self-Attention, also computed at each Decoder layer). 
 - Since during the training phase, the output sequences are already available, one can perform all the different timesteps of the Decoding process in parallel by masking (replacing with zeroes) the appropriate parts of the "previously generated" output sequences. 
 - This masking results in the Decoder Self-Attention being uni-directional, as opposed to the Encoder one. Finally, at inference time, the output elements are generated one by one in a sequential manner.  
 
  
![image](https://blog.scaleway.com/content/images/2019/08/transformer2.jpg)

# Assignment

This assignment includes developing a chatbot depending on transformer network and using loss function.
- Take the last code 
- Convert that code to make it into a chatbot. Refer to this Pytorch tutorial: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html (Links to an external site.) for integrating the dataset. 
- Achieve a loss of less than 2.4 in any epoch. 
- You MUST use the same Transformer we used (free to change a number of layers) in this code and not the GRU based encoder-decoder architecture mentioned in the PyTorch code. 
- You MUST use the same loss function/methodology as used in the PyTorch sample code, else your evaluation would not be comparable and 2.4 doesn't make sense 
- Submit solution to Assignment-Solution 13

### Machine Translation on Chatbot Cornell Dialog Dataset using Transformer
Session 13 Assignment for training Cornell Dialog Dataset using Transformer. We used the following

Glove Embedding
Dropout
Transformer
Use same loss function/methodology as used in the https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
Got best loss of 1.6520626544952393
Some of training logs are as below:

Iteration: 1; Percent complete: 0.0%; Average loss: 8.5598 Iteration: 2; Percent complete: 0.0%; Average loss: 8.5534 Iteration: 3; Percent complete: 0.1%; Average loss: 8.5476 Iteration: 4; Percent complete: 0.1%; Average loss: 8.5416 Iteration: 5; Percent complete: 0.1%; Average loss: 8.5354

Iteration: 2904; Percent complete: 72.6%; Average loss: 2.1554 Iteration: 2905; Percent complete: 72.6%; Average loss: 2.0213 Iteration: 2906; Percent complete: 72.6%; Average loss: 2.2699 Iteration: 2907; Percent complete: 72.7%; Average loss: 2.0068 Iteration: 2908; Percent complete: 72.7%; Average loss: 1.6521 Iteration: 2909; Percent complete: 72.7%; Average loss: 1.9076 Iteration: 2910; Percent complete: 72.7%; Average loss: 2.0834 Iteration: 2911; Percent complete: 72.8%; Average loss: 2.0729 Iteration: 2912; Percent complete: 72.8%; Average loss: 2.1877 Iteration: 2913; Percent complete: 72.8%; Average loss: 2.0870

Iteration: 4036; Percent complete: 100.9%; Average loss: 2.1200 Iteration: 4037; Percent complete: 100.9%; Average loss: 2.1006

Iteration: 4038; Percent complete: 100.9%; Average loss: 1.8822 Iteration: 4039; Percent complete: 101.0%; Average loss: 2.1544 Iteration: 4040; Percent complete: 101.0%; Average loss: 2.1531 Iteration: 4041; Percent complete: 101.0%; Average loss: 2.0272 Iteration: 4042; Percent complete: 101.0%; Average loss: 1.9674 Iteration: 4043; Percent complete: 101.0%; Average loss: 2.2645 Iteration: 4044; Percent complete: 101.1%; Average loss: 2.1373 Iteration: 4045; Percent complete: 101.1%; Average loss: 1.9795 Iteration: 4046; Percent complete: 101.1%; Average loss: 2.0344 Iteration: 4047; Percent complete: 101.2%; Average loss: 2.1357 Iteration: 4048; Percent complete: 101.2%; Average loss: 1.7522

Best Loss Value: 1.6520626544952393 At Iteration: 2908

Validation Loss: 2.066

Test Loss: 2.090
