# Summary of the tokenizers

The following are will be covered:

- Byte Pair Encoding (BPE)
- WordPiece
- Unigram Language Model
- SentencePiece



On this page, we will have a closer look at tokenization. As we saw in [the preprocessing tutorial](https://huggingface.co/transformers/preprocessing.html), tokenizing a text is splitting it into words or subwords, which then are converted to ids through a look-up table. Converting words or subwords to ids is straightforward, so in this summary, we will focus on splitting a text into words or subwords (i.e. tokenizing a text). More specifically, we will look at the three main types of tokenizers used in 🤗 Transformers: [Byte-Pair Encoding (BPE)](https://huggingface.co/transformers/tokenizer_summary.html#byte-pair-encoding), [WordPiece](https://huggingface.co/transformers/tokenizer_summary.html#wordpiece), and [SentencePiece](https://huggingface.co/transformers/tokenizer_summary.html#sentencepiece), and show examples of which tokenizer type is used by which model.

Note that on each model page, you can look at the documentation of the associated tokenizer to know which tokenizer type was used by the pretrained model. For instance, if we look at [`BertTokenizer`](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertTokenizer), we can see that the model uses [WordPiece](https://huggingface.co/transformers/tokenizer_summary.html#wordpiece).

## Byte Pair Encoding (BPE)

### Algorithm

1. Prepare a large enough training data (i.e. corpus)
2. Define a desired subword vocabulary size
3. Split word to sequence of characters and appending suffix “</w>” to end of word with word frequency. So the basic unit is character in this stage. For example, the frequency of “low” is 5, then we rephrase it to “l o w </w>”: 5
4. Generating a new subword according to the high frequency occurrence.
5. Repeating step 4 until reaching subword vocabulary size which is defined in step 2 or the next highest frequency pair is 1.

### Example

- Taking “low: 5”, “lower: 2”, “newest: 6” and “widest: 3” as an example, the highest frequency subword pair is `e` and `s`. It is because we get 6 count from `newest` and 3 count from `widest`. Then new subword (`es`) is formed and it will become a candidate in next iteration.

- In the second iteration, the next high frequency subword pair is `es` (generated from previous iteration )and `t`. It is because we get 6count from `newest` and 3 count from `widest`.

- Keep iterate until built a desire size of vocabulary size or the next highest frequency pair is 1.

## WordPiece

WordPiece is another word segmentation algorithm and it is similar with BPE. Basically, WordPiece is similar with BPE and the difference part is forming a new subword by likelihood but not the next highest frequency pair.

### Algorithm

1. Prepare a large enough training data (i.e. corpus)
2. Define a desired subword vocabulary size
3. Split word to sequence of characters
4. Build a languages model based on step 3 data
5. Choose the new word unit out of all the possible ones that increases the likelihood on the training data the most when added to the model.
6. Repeating step 5until reaching subword vocabulary size which is defined in step 2 or the likelihood increase falls below a certain threshold.

**Now, if we tokenize the sentence using the WordPiece tokenizer, then we obtain the tokens**
**as shown here:**

tokens = [let, us, start, pre, ##train, ##ing, the, model]

**We can observe that while tokenizing the sentence using the WordPiece tokenizer, the**
**word pertaining is split into the following subwords** – pre, ##train, ##ing.