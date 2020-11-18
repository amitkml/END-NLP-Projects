# Session 4 Assignment Instructions
Use this as a [reference](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb.)

## Tasks
Change this code in such a way that:
- it has 3 LSTM layers
- it has used a for loop to do so in the forward function
- the dropout value used is 0.2
- trained on the text that is reversed (for example "my name is Rohan" becomes "Rohan is name my"
- achieves 87% or more accuracy
## Proposed Network Design
- Have done the training data reverse after spliting of training and test data
- Have used nn.ModuleList to store multiple LSTM Layers with layer no as 1 and bidirectional set as False
- Added additional layer of FC Layer
- Have stored 1st and 2nd layer hidden cell value and concatenated the same to pass on to FC Layer
- Have increased no of hidden dimension from 256 to 512
- Have freezed Embedding layers after 10 epoch when validation accuracy was flattening
- Have then made the training data to original and trained further and accuracy on test data did not improve further
- Final test data accuracy stands at 86.40%
