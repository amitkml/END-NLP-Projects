# Understanding self-attention

To understand what self-attention is and how it works, you need to know only the following terms:

- dot product
- matrix multiplication
- softmax

in nlp, each sentence is represented by a bunch of tokens.

- each token maps to some number
- each number is represented by a vector (embedding)
- embedding carries the meaning of the token 2/

In the example (image):
- Which words are “closer” to the word “xgboost”
- Which words are “far” from the word “xgboost”?

![im](https://pbs.twimg.com/media/FGp8JXsXoAASmjB?format=png&name=small)

- If we take dot product of the embedding for the word “xgboost” with embeddings for all other words, we have this information.
- “xgboost” will be closer to itself and some other words and far from the rest of the words.

- Closer => dot product is large
- Far => dot product is small
- Is the word “it” closer to “problem” or is it closer to “xgboost” ?

If there are N tokens and we take the dot product of each token with itself and the rest of the tokens, we have an NxN matrix.

- Let’s call this matrix score matrix
- If we take row-wise softmax of this matrix, we will end up with another matrix in which all values are between 0 and 1 (for each row)

![im](https://pbs.twimg.com/media/FGp9OOYXMAUsWLO?format=jpg&name=small)

If we had 8 tokens, and each had an embedding size of 512,we will end up with an 8x8 matrix. After softmax, we have a matrix that tells us which token “focuses” on which other tokens. In other words, which token is contextually similar to other tokens in the context (sentence)

![im](https://pbs.twimg.com/media/FGp9NHaWYAgCNvy?format=jpg&name=small)

- Now we use this matrix to give weights to the original embeddings. 
- For each token give high weights to the tokens it focuses more on.
- This weight matrix is obtained using row-wise softmax explained previously 
- weight matrix: 8x8
- original matrix: 8x512
- multiply the original matrix by weight. Resulting matrix: 8x512 (same as original) 
- Each embedding value is multiplied by the corresponding token’s weights 
- Now, in the resulting matrix, we now have weighted embeddings.
- The weighted embeddings are “contextually influenced” and no longer depend only on the individual tokens.
- And this is how self-attention works!

