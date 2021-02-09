# Transformer Network
The Transformer architecture featuting a two-layer Encoder / Decoder. The Encoder processes all three elements of the input sequence (w1, w2, and w3) in parallel, whereas the Decoder generates each element sequentially (only timesteps 0 and 1, where the output sequence elements v1 and v2 are generated, are depicted). Output token generation continues until an end of the sentence token <EOS> appears.
  
![image](https://blog.scaleway.com/content/images/2019/08/transformer2.jpg)

# Assignment

This assignment includes developing a chatbot depending on transformer network and using loss function.
- Take the last code 
- Convert that code to make it into a chatbot. Refer to this Pytorch tutorial: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html (Links to an external site.) for integrating the dataset. 
- Achieve a loss of less than 2.4 in any epoch. 
- You MUST use the same Transformer we used (free to change a number of layers) in this code and not the GRU based encoder-decoder architecture mentioned in the PyTorch code. 
- You MUST use the same loss function/methodology as used in the PyTorch sample code, else your evaluation would not be comparable and 2.4 doesn't make sense 
- Submit solution to Assignment-Solution 13. 
