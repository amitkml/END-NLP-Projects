# A Comprehensive Introduction to Torchtext (Practical Torchtext part 1)

![img](https://i2.wp.com/mlexplained.com/wp-content/uploads/2018/02/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88-2018-02-07-10.32.59.png?fit=1200%2C563&ssl=1)

If you've ever worked on a project for deep learning for NLP, you'll know how painful and tedious all the preprocessing is. Before you start training your model, you have to:

1. Read the data from disk
2. Tokenize the text
3. Create a mapping from word to a unique integer
4. Convert the text into lists of integers
5. Load the data in whatever format your deep learning framework requires
6. Pad the text so that all the sequences are the same length, so you can process them in batch

[Torchtext](https://github.com/pytorch/text) is a library that makes all the above processing much easier. Though still relatively new, its convenient functionality - particularly around batching and loading - make it a library worth learning and using.

In this post, I'll demonstrate how torchtext can be used to build and train a text classifier from scratch.To make this tutorial realistic, I'm going to use a small sample of data from [this Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The data and code are available in [my GitHub repo](https://github.com/keitakurita/practical-torchtext), so feel free to clone it and follow along. Or, if you just want to see the minimal working example, feel free to skip the rest of this tutorial and just read the [notebook](https://github.com/keitakurita/practical-torchtext/blob/master/Lesson%201:%20intro%20to%20torchtext%20with%20text%20classification.ipynb).



Update: The current pip release of torchtext contains bugs that will cause the notebook to work incorrectly. I've fixed these bugs on the master branch of the [GitHub repo](https://github.com/pytorch/text), so I highly recommend you install torchtext with the following command:

```
$ pip install --upgrade git+https://github.com/pytorch/text
```

## 1. The Overview

Torchtext follows the following basic formula for transforming data into working input for your neural network:

![img](https://i2.wp.com/mlexplained.com/wp-content/uploads/2018/02/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88-2018-02-07-10.32.59.png?resize=840%2C395)

Torchtext takes in raw data in the form of text files, csv/tsv files, json files, and directories (as of now) and converts them to Datasets. Datasets are simply preprocessed blocks of data read into memory with various fields. They are a canonical form of processed data that other data structures can use.

Torchtext then passes the Dataset to an Iterator. Iterators handle numericalizing, batching, packaging, and moving the data to the GPU. Basically, it does all the heavy lifting necessary to pass the data to a neural network.



In the following sections, we'll see how each of these processes plays out in an actual working example.

## 2. Declaring the Fields

Torchtext takes a declarative approach to loading its data: you tell torchtext how you want the data to look like, and torchtext handles it for you.

The way you do this is by declaring a Field. The Field specifies how you want a certain (you guessed it) field to be processed. Let's look at an example:

```
from torchtext.data import Field
tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)

LABEL = Field(sequential=False, use_vocab=False)
```

In the toxic comment classification dataset, there are two kinds of fields: the comment text and the labels (toxic, severe toxic, obscene, threat, insult, and identity hate).

![img](https://i1.wp.com/mlexplained.com/wp-content/uploads/2018/02/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88-2018-02-07-12.19.35.png?resize=653%2C135)

Let's look at the LABEL field first, since it's simpler. All fields, by default, expect a sequence of words to come in, and they expect to build a mapping from the words to integers later on (this mapping is called the vocab, and we will see how it is created later). If you are passing a field that is already numericalized by default and is not sequential, you should pass use_vocab=False and sequential=False.

For the comment text, we pass in the preprocessing we want the field to do as keyword arguments. We give it the tokenizer we want the field to use, tell it to convert the input to lowercase, and also tell it the input is sequential.



In addition to the keyword arguments mentioned above, the Field class also allows the user to specify special tokens (the `unk_token` for out-of-vocabulary words, the `pad_token` for padding, the `eos_token` for the end of a sentence, and an optional `init_token` for the start of the sentence), choose whether to make the first dimension the batch or the sequence (the first dimension is the sequence by default), and choose whether to allow the sequence lengths to be decided at runtime or decided in advance. Fortunately, [the docstrings](https://github.com/pytorch/text/blob/c839a7934930819be7e240ea972e4d600966afdc/torchtext/data/field.py#L61) for the Field class are relatively well written, so if you need some advanced preprocessing you should refer to them for more information.



The field class is at the center of torchtext and is what makes preprocessing such an ease. Aside from the standard field class, here's a list of the fields that are currently available (along with their use cases):

| Name              | Description                                                  | Use Case                                                     |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Field             | A regular field that defines preprocessing and postprocessing | Non-text fields and text fields where you don't need to map integers back to words |
| ReversibleField   | An extension of the field that allows reverse mapping of word ids to words | Text fields if you want to map the integers back to natural language (such as in the case of language modeling) |
| NestedField       | A field that takes processes non-tokenized text into a set of smaller fields | Char-based models                                            |
| LabelField (New!) | A regular field with sequential=False and no <unk> token. Newly added on the master branch of the torchtext github repo, not yet available for release. | Label fields in text classification.                         |

###  

## 3. Constructing the Dataset

The fields know what to do when given raw data. Now, we need to tell the fields what data it should work on. This is where we use Datasets.

There are various built-in Datasets in torchtext that handle common data formats. For csv/tsv files, the TabularDataset class is convenient. Here's how we would read data from a csv file using the TabularDataset:

```
from torchtext.data import TabularDataset

tv_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("comment_text", TEXT), ("toxic", LABEL),
                 ("severe_toxic", LABEL), ("threat", LABEL),
                 ("obscene", LABEL), ("insult", LABEL),
                 ("identity_hate", LABEL)]
trn, vld = TabularDataset.splits(
               path="data", # the root directory where the data lies
               train='train.csv', validation="valid.csv",
               format='csv',
               skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
               fields=tv_datafields)

tst_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
                  ("comment_text", TEXT)]
tst = TabularDataset(
           path="data/test.csv", # the file path
           format='csv',
           skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
           fields=tst_datafields)
```

For the TabularDataset, we pass in a list of (name, field) pairs as the fields argument. The fields we pass in must be in the same order as the columns. For the columns we don't use, we pass in a tuple where the field element is None. [1](https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/#easy-footnote-bottom-1-310)

The splits method creates a dataset for the train and validation data by applying the same processing. It can also handle the test data, but since out test data has a different format from the train and validation data, we create a different dataset.



Datasets can mostly be treated in the same way as lists. To understand this, it's instructive to take a look inside our Dataset. Datasets can be indexed and iterated over like normal lists, so let's see what the first element looks like:

```
>>> trn[0]
torchtext.data.example.Example at 0x10d3ed3c8
>>> trn[0].__dict__.keys()
dict_keys(['comment_text', 'toxic', 'severe_toxic', 'threat', 'obscene', 'insult', 'identity_hate'])
>>> trn[0].comment_text[:3]
['explanation', 'why', 'the']
```

we get an Example object. The Example object bundles the attributes of a single data point together. We also see that the text has already been tokenized for us, but has not yet been converted to integers. This makes sense since we have not yet constructed the mapping from words to ids. Constructing this mapping is our next step.

Torchtext handles mapping words to integers, but it has to be told the full range of words it should handle. In our case, we probably want to build the vocabulary on the training set only, so we run the following code:

```
TEXT.build_vocab(trn)
```

This makes torchtext go through all the elements in the training set, check the contents corresponding to the `TEXT` field, and register the words in its vocabulary. Torchtext has its own class called Vocab for handling the vocabulary. The Vocab class holds a mapping from word to id in its `stoi` attribute and a reverse mapping in its `itos` attribute. In addition to this, it can automatically build an embedding matrix for you using various pretrained embeddings like word2vec (more on this in [another tutorial](http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/)). The Vocab class can also take options like `max_size` and `min_freq` that dictate how many words are in the vocabulary or how many times a word has to appear to be registered in the vocabulary. Words that are not included in the vocabulary will be converted into <unk>, a token standing for "unknown".

Here is a list of the currently available set of datasets and the format of data they take in:

| Name                    | Description                                                  | Use Case                                                     |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| TabularDataset          | Takes paths to csv/tsv files and json files or Python dictionaries as inputs. | Any problem that involves a label (or labels) for each piece of text |
| LanguageModelingDataset | Takes the path to a text file as input.                      | Language modeling                                            |
| TranslationDataset      | Takes a path and extensions to a file for each language. e.g. If the files are English: "hoge.en", French: "hoge.fr", path="hoge", exts=("en","fr") | Translation                                                  |
| SequenceTaggingDataset  | Takes a path to a file with the input sequence and output sequence separated by tabs. [2](https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/#easy-footnote-bottom-2-310) | Sequence tagging                                             |

Now that we have our data formatted and read into memory, we turn to the next step: creating an Iterator to pass the data to our model.

## 4. Constructing the Iterator

In torchvision and PyTorch, the processing and batching of data is handled by DataLoaders. For some reason, torchtext has renamed the objects that do the exact same thing to Iterators. The basic functionality is the same, but Iterators, as we will see, have some convenient functionality that is unique to NLP.

Below is code for how you would initialize the Iterators for the train, validation, and test data.

```
from torchtext.data import Iterator, BucketIterator

train_iter, val_iter = BucketIterator.splits(
 (trn, vld), # we pass in the datasets we want the iterator to draw data from
 batch_sizes=(64, 64),
 device=-1, # if you want to use the GPU, specify the GPU number here
 sort_key=lambda x: len(x.comment_text), # the BucketIterator needs to be told what function it should use to group the data.
 sort_within_batch=False,
 repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
)
test_iter = Iterator(tst, batch_size=64, device=-1, sort=False, sort_within_batch=False, repeat=False)
```

Update: The `sort_within_batch` argument, when set to True, sorts the data within each minibatch in decreasing order according to the `sort_key`. This is necessary when you want to use `pack_padded_sequence` with the padded sequence data and convert the padded sequence tensor to a `PackedSequence` object.

The BucketIterator is one of the most powerful features of torchtext. It automatically shuffles and buckets the input sequences into sequences of similar length.



The reason this is powerful is that - as I mentioned earlier - we need to pad the input sequences to be of the same length to enable batch processing. For instance, the sequences

```
[ [3, 15, 2, 7],
  [4, 1],
  [5, 5, 6, 8, 1] ]
```

would need to be padded to become

```
[ [3, 15, 2, 7, 0],
  [4, 1, 0, 0, 0],
  [5, 5, 6, 8, 1] ]
```

As you can see, the amount of padding necessary is determined by the longest sequence in the batch. Therefore, padding is most efficient when the sequences are of similar lengths. The BucketIterator does all this behind the scenes. As a word of caution, you need to tell the BucketIterator what attribute you want to bucket the data on. In our case, we want to bucket based on the lengths of the comment_text field, so we pass that in as a keyword argument. See the code above for details on the other arguments.

For the test data, we don't want to shuffle the data since we'll be outputting the predictions at the end of training. This is why we use a standard iterator.

Here's a list of the Iterators that torchtext currently implements:

| Name           | Description                                                  | Use Case                                                     |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Iterator       | Iterates over the data in the order of the dataset.          | Test data, or any other data where the order is important.   |
| BucketIterator | Buckets sequences of similar lengths together.               | Text classification, sequence tagging, etc. (use cases where the input is of variable length) |
| BPTTIterator   | An iterator built especially for language modeling that also generates the input sequence delayed by one timestep. It also varies the BPTT (backpropagation through time) length. This iterator [deserves its own post](http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/), so I'll omit the details here. | Language modeling                                            |

## 5. Wrapping the Iterator

Currently, the iterator returns a custom datatype called torchtext.data.Batch. The Batch class has a similar API to the Example type, with a batch of data from each field as attributes. Unfortunately, this custom datatype makes code reuse difficult (since each time the column names change, we need to modify the code), and makes torchtext hard to use with other libraries for some use cases (like torchsample and fastai).

I hope this will be dealt with in the future (I'm considering filing a PR if I can decide what the API should look like), but in the meantime, we'll hack on a simple wrapper to make the batches easy to use.

Concretely, we'll convert the batch to a tuple in the form (x, y) where x is the independent variable (the input to the model) and y is the dependent variable (the supervision data). Here's the code:

```
class BatchWrapper:
      def __init__(self, dl, x_var, y_vars):
            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x 

      def __iter__(self):
            for batch in self.dl:
                  x = getattr(batch, self.x_var) # we assume only one input in this wrapper

                  if self.y_vars is None: # we will concatenate y into a single tensor
                        y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
                  else:
                        y = torch.zeros((1))

                  yield (x, y)

      def __len__(self):
            return len(self.dl)

train_dl = BatchWrapper(train_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
valid_dl = BatchWrapper(val_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
test_dl = BatchWrapper(test_iter, "comment_text", None)
```

All we're doing here is converting the batch object to a tuple of inputs and outputs.

```
&amp;gt;&amp;gt;&amp;gt; next(train_dl.__iter__())
(Variable containing:
   606   354   334  ...     63    15    15
   693    63    55  ...      4   601    29
   584     4   520  ...    664   242    21
        ...          ⋱          ...
     1     1     1  ...      1     1    84
     1     1     1  ...      1     1   118
     1     1     1  ...      1     1    15
 [torch.LongTensor of size 494x25], Variable containing:
     0     0     0     0     0     0
     1     1     0     1     1     0
     1     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     1     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
     0     0     0     0     0     0
 [torch.FloatTensor of size 25x6])
```

Nothing fancy here. Now, we're finally ready to start training a text classifier.

## 6. Training the Model

We'll use a simple LSTM to demonstrate how to train the text classifier on the data we've built:

```
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class SimpleLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300, num_linear=1):
        super().__init__() # don't forget to call this!
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 6)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
          feature = layer(feature)
          preds = self.predictor(feature)
        return preds

em_sz = 100
nh = 500
nl = 3
model = SimpleBiLSTMBaseline(nh, emb_dim=em_sz)
```

Now, we'll write the training loop. Thanks to all our preprocessing, this is very simple. We can iterate using our wrapped Iterator, and the data will automatically be passed to us after being moved to the GPU and numericalized appropriately.

```
import tqdm

opt = optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.BCEWithLogitsLoss()

epochs = 2

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train() # turn on training mode
    for x, y in tqdm.tqdm(train_dl): # thanks to our wrapper, we can intuitively iterate over our data!
        opt.zero_grad()

        preds = model(x)
        loss = loss_func(y, preds)
        loss.backward()
        opt.step()

        running_loss += loss.data[0] * x.size(0)

    epoch_loss = running_loss / len(trn)

    # calculate the validation loss for this epoch
    val_loss = 0.0
    model.eval() # turn on evaluation mode
    for x, y in valid_dl:
        preds = model(x)
        loss = loss_func(y, preds)
        val_loss += loss.data[0] * x.size(0)

    val_loss /= len(vld)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
```

There's not much to explain here: this is just a standard training loop. Now, let's generate our predictions

```
test_preds = []
for x, y in tqdm.tqdm(test_dl):
    preds = model(x)
    preds = preds.data.numpy()
    # the actual outputs of the model are logits, so we need to pass these values to the sigmoid function
    preds = 1 / (1 + np.exp(-preds))
    test_preds.append(preds)
    test_preds = np.hstack(test_preds)
```

Finally, we can write our predictions to a csv file.

```
import pandas as pd
df = pd.read_csv("data/test.csv")
for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
    df[col] = test_preds[:, i]

df.drop("comment_text", axis=1).to_csv("submission.csv", index=False)
```

 

And we're done! We can submit this file to Kaggle, try refining our model, changing the tokenizer, or whatever we feel like, and it will only take a few changes in the code above.

##  7. Conclusion and Further Readings

I hope this tutorial has provided insight into how torchtext can be used, and how useful it is. Though the library is still new and there are many rough edges, I believe that torchtext is a great first step towards standardized text preprocessing that will improve the productivity of people working in NLP all throughout the world.

If you want to see torchtext used for language modeling, I've uploaded [another tutorial](http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/) detailing language modeling and the BPTT iterator. If you have any further questions, feel free to ask me in the comments!



1. The next release of torchtext (and the current version on GitHub) will be able to take a dictionary mapping each column by name to its corresponding field instead of a list.
2. The API is a subset of the API of TabularDataset for tsvs, so this might be deprecated in the future. 

# Language modeling tutorial in torchtext (Practical Torchtext part 2)

In a [previous article](http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/), I wrote an introductory tutorial to torchtext using text classification as an example.

In this post, I will outline how to use torchtext for training a language model. We'll also take a look at some more practical features of torchtext that you might want to use when training your own practical models. Specifically, we'll cover

- Using a built-in dataset
- Using a custom tokenizer
- Using pretrained word embeddings



The full code is available [here](https://github.com/keitakurita/practical-torchtext/blob/master/Lesson%202:%20torchtext%20for%20language%20modeling.ipynb). As a word of caution, if you're running the code in this tutorial, I assume that you have access to a GPU for the sake of training speed. If you don't have a GPU, you can still follow along, but the training will be very slow.

## 1. What is Language Modeling?

Language modeling is a task where we build a model that can take a sequence of words as input and determine how likely that sequence is to be actual human language. For instance, we would want our model to predict "This is a sentence" to be a likely sequence and "cold his book her" to be unlikely.

Though language models may seem uninteresting on their own, they can be used as an unsupervised pretraining method or the basis of other tasks like chat generation. In any case, language modeling is one of the most basic tasks in deep learning for NLP, so it's a good idea to learn language modeling as the basis of other, more complicated tasks (like machine translation).



The way we generally train language models is by training them to predict the next word given all previous words in a sentence or multiple sentences. Therefore, all we need to do language modeling is a large amount of language data. In this tutorial, we'll be using the famous WikiText2 dataset, which is a built-in dataset provided by torchtext.

 

## 2. Preparing the Data

To use the WikiText2 dataset, we'll need to prepare the field that handles the tokenization and numericalization of the text. This time, we'll try using our own custom tokenizer: the spacy tokenizer. [Spacy](https://spacy.io/) is a framework that handles many natural language processing tasks, and torchtext is designed to work closely with it. Using the tokenizer is easy with torchtext: all we have to do is pass in the tokenizer function!

```
`import` `torchtext``from` `torchtext ``import` `data``import` `spacy` `from` `spacy.symbols ``import` `ORTH``my_tok ``=` `spacy.load(``'en'``)` `def` `spacy_tok(x):``    ``return` `[tok.text ``for` `tok ``in` `my_tok.tokenizer(x)]` `TEXT ``=` `data.Field(lower``=``True``, tokenize``=``spacy_tok)`
```

 

`add_special_case` simply tells the tokenizer to parse a certain string in a certain way. The list after the special case string represents how we want the string to be tokenized.

If we wanted to tokenize "don't" into "do" and "'nt", then we would write

```
`my_tok.tokenizer.add_special_case(``"don't"``, [{ORTH: ``"do"``}, {ORTH: ``"n't"``}])`
```

Now, we're ready to load the WikiText2 dataset. There are two effective ways of using these datasets: one is loading as a Dataset split into the train, validation, and test sets, and the other is loading as an Iterator. The dataset offers more flexibility, so we'll use that approach here.

```
`from` `torchtext.datasets ``import` `WikiText2` `train, valid, test ``=` `WikiText2.splits(TEXT) ``# loading custom datasets requires passing in the field, but nothing else.`
```

Let's take a quick look inside. Remember, datasets behave largely like normal lists, so we can measure the length using the `len` function.

```
`>>> ``len``(train)``1`
```

Only one training example?! Did we do something wrong? Turns out not. It's just that the entire corpus of the dataset is contained within a single example. We'll see how this example gets batched and processed later.

Now that we have our data, let's build the vocabulary. This time, let's try using precomputed word embeddings. We'll use GloVe vectors with 200 dimensions this time. There are various other precomputed word embeddings in torchtext (including GloVe vectors with 100 and 300 dimensions) as well which can be loaded in mostly the same way.

```
`TEXT.build_vocab(train, vectors``=``"glove.6B.200d"``)`
```

Great! We've prepared our dataset in only 3 lines of code (excluding imports and the tokenizer). Now we move on to building the Iterator, which will handle batching and moving the data to GPU for us.

This is the climax of our tutorial and shows why torchtext is so handy for language modeling.  It turns out that torchtext has a very handy iterator that does most of the heavy lifting for us. It's called the `BPTTIterator`. The `BPTTIterator` does the following for us:

- Divide the corpus into batches of sequence length `bptt`

For instance, suppose we have the following corpus:

*"Machine learning is a field of computer science that gives computers the ability to learn without being explicitly programmed."*

Though this sentence is short, the actual corpus is thousands of words long, so we can't possibly feed it in all at once. We'll want to divide the corpus into sequences of a shorter length. In the above example, if we wanted to divide the corpus into batches of sequence length 5, we would get the following sequences:

["*Machine*", "*learning*", "*is*", "*a*", "*field*"],

["*of*", "*computer*", "*science*", "*that*", "*gives*"],

["*computers*", "*the*", "*ability*", "*to*", "*learn*"],

["*without*", "*being*", "*explicitly*", "*programmed*", EOS]

 

- Generate batches that are the input sequences offset by one

In language modeling, the supervision data is the next word in a sequence of words. We, therefore, want to generate the sequences that are the input sequences offset by one. In the above example, we would get the following sequence that we train the model to predict:

["*learning*", "*is*", "*a*", "*field*", "*of*"],

["*computer*", "*science*", "*that*", "*gives*", "*computers*"],

["*the*", "*ability*", "*to*", "*learn*", "*without*"],

["*being*", "*explicitly*", "*programmed*", EOS, EOS]

Here's the code for creating the iterator:

```
`train_iter, valid_iter, test_iter ``=` `data.BPTTIterator.splits(``    ``(train, valid, test),``    ``batch_size``=``32``,``    ``bptt_len``=``30``, ``# this is where we specify the sequence length``    ``device``=``0``,``    ``repeat``=``False``)`
```

As always, it's a good idea to take a look into what is actually happening behind the scenes.

```
`>>> b ``=` `next``(``iter``(train_iter)); ``vars``(b).keys()``dict_keys([``'batch_size'``, ``'dataset'``, ``'train'``, ``'text'``, ``'target'``])`
```

 

We see that we have an attribute we never explicitly asked for: target. Let's hope it's the target sequence.

```
`>>> b.text[:``5``, :``3``]``Variable containing:``     ``9`    `953`      `0``    ``10`    `324`   `5909``     ``9`     `11`  `20014``    ``12`   `5906`     `27``  ``3872`  `10434`      `2` `>>> b.target[:``5``, :``3``]``Variable containing:``    ``10`    `324`   `5909``     ``9`     `11`  `20014``    ``12`   `5906`     `27``  ``3872`  `10434`      `2``  ``3892`      `3`  `10780`
```

Be careful, the first dimension of the text and target is the sequence, and the next is the batch. We see that the target is indeed the original text offset by 1 (shifted downwards by 1). Which means we have all we need to start training a language model!

## 3. Training the Language Model

With the above iterators, training the language model is easy.

First, we need to prepare the model. We'll be borrowing and customizing the model from the [examples](https://github.com/pytorch/examples/tree/master/word_language_model) in PyTorch.

```
`import` `torch``import` `torch.nn as nn``import` `torch.nn.functional as F``import` `torch.optim as optim``from` `torch.autograd ``import` `Variable as V` `class` `RNNModel(nn.Module):``    ``def` `__init__(``self``, ntoken, ninp,``                 ``nhid, nlayers, bsz,``                 ``dropout``=``0.5``, tie_weights``=``True``):``        ``super``(RNNModel, ``self``).__init__()``        ``self``.nhid, ``self``.nlayers, ``self``.bsz ``=` `nhid, nlayers, bsz``        ``self``.drop ``=` `nn.Dropout(dropout)``        ``self``.encoder ``=` `nn.Embedding(ntoken, ninp)``        ``self``.rnn ``=` `nn.LSTM(ninp, nhid, nlayers, dropout``=``dropout)``        ``self``.decoder ``=` `nn.Linear(nhid, ntoken)``        ``self``.init_weights()``        ``self``.hidden ``=` `self``.init_hidden(bsz) ``# the input is a batched consecutive corpus``                                            ``# therefore, we retain the hidden state across batches` `    ``def` `init_weights(``self``):``        ``initrange ``=` `0.1``        ``self``.encoder.weight.data.uniform_(``-``initrange, initrange)``        ``self``.decoder.bias.data.fill_(``0``)``        ``self``.decoder.weight.data.uniform_(``-``initrange, initrange)` `    ``def` `forward(``self``, ``input``):``        ``emb ``=` `self``.drop(``self``.encoder(``input``))``        ``output, ``self``.hidden ``=` `self``.rnn(emb, ``self``.hidden)``        ``output ``=` `self``.drop(output)``        ``decoded ``=` `self``.decoder(output.view(output.size(``0``)``*``output.size(``1``), output.size(``2``)))``        ``return` `decoded.view(output.size(``0``), output.size(``1``), decoded.size(``1``))` `    ``def` `init_hidden(``self``, bsz):``        ``weight ``=` `next``(``self``.parameters()).data``        ``return` `(V(weight.new(``self``.nlayers, bsz, ``self``.nhid).zero_().cuda()),``                ``V(weight.new(``self``.nlayers, bsz, ``self``.nhid).zero_()).cuda())`` ` `    ``def` `reset_history(``self``):``        ``self``.hidden ``=` `tuple``(V(v.data) ``for` `v ``in` `self``.hidden)`
```

The language model itself is simple: it takes a sequence of word tokens, embeds them, puts them through an LSTM, then emits a probability distribution over the next word for each input word. We've made slight modifications like saving the hidden state in the model object and adding a reset history method. The reason we need to retain the history is because the entire dataset is a continuous corpus, meaning we want to retain the hidden state between sequences within a batch. Of course, we can't possibly retain the entire history (it will be too costly), so we'll periodically reset the history during training.

To use the precomputed word embeddings, we'll need to pass the initial weights of the embedding matrix explicitly. The weights are contained in the vectors attribute of the vocabulary.

```
`weight_matrix ``=` `TEXT.vocab.vectors``model ``=` `RNNModel(weight_matrix.size(``0``),`` ``weight_matrix.size(``1``), ``200``, ``1``, BATCH_SIZE)` `model.encoder.weight.data.copy_(weight_matrix)``model.cuda()`
```

Now we can begin training the language model. We'll use the Adam optimizer here. For the loss, we'll use the `nn.CrossEntropyLoss` function. This loss takes the index of the correct class as the ground truth instead of a one-hot vector. Unfortunately, it only takes tensors of dimension 2 or 4, so we'll need to do a bit of reshaping.

```
`criterion ``=` `nn.CrossEntropyLoss()``optimizer ``=` `optim.Adam(model.parameters(), lr``=``1e``-``3``, betas``=``(``0.7``, ``0.99``))``n_tokens ``=` `weight_matrix.size(``0``)`
```

We'll write the training loop

```
`from` `tqdm ``import` `tqdm ``def` `train_epoch(epoch):``"""One epoch of a training loop"""``    ``epoch_loss ``=` `0``    ``for` `batch ``in` `tqdm(train_iter):``    ``# reset the hidden state or else the model will try to backpropagate to the``    ``# beginning of the dataset, requiring lots of time and a lot of memory``         ``model.reset_history()` `    ``optimizer.zero_grad()` `    ``text, targets ``=` `batch.text, batch.target``    ``prediction ``=` `model(text)``    ``# pytorch currently only supports cross entropy loss for inputs of 2 or 4 dimensions.``    ``# we therefore flatten the predictions out across the batch axis so that it becomes``    ``# shape (batch_size * sequence_length, n_tokens)``    ``# in accordance to this, we reshape the targets to be``    ``# shape (batch_size * sequence_length)``    ``loss ``=` `criterion(prediction.view(``-``1``, n_tokens), targets.view(``-``1``))``    ``loss.backward()` `    ``optimizer.step()` `    ``epoch_loss ``+``=` `loss.data[``0``] ``*` `prediction.size(``0``) ``*` `prediction.size(``1``)` `    ``epoch_loss ``/``=` `len``(train.examples[``0``].text)` `    ``# monitor the loss``    ``val_loss ``=` `0``    ``model.``eval``()``    ``for` `batch ``in` `valid_iter:``        ``model.reset_history()``        ``text, targets ``=` `batch.text, batch.target``        ``prediction ``=` `model(text)``        ``loss ``=` `criterion(prediction.view(``-``1``, n_tokens), targets.view(``-``1``))``        ``val_loss ``+``=` `loss.data[``0``] ``*` `text.size(``0``)``    ``val_loss ``/``=` `len``(valid.examples[``0``].text)` `    ``print``(``'Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'``.``format``(epoch, epoch_loss, val_loss))&lt;``/``pre>`
```

and we're set to go!

```
`n_epochs ``=` `2``for` `epoch ``in` `range``(``1``, n_epochs ``+` `1``):``    ``train_epoch(epoch)`
```

Understanding the correspondence between the loss and the quality of the language model is very difficult, so it's a good idea to check the outputs of the language model periodically. This can be done by writing a bit of custom code to map integers back into words based on the vocab:

```
`def` `word_ids_to_sentence(id_tensor, vocab, join``=``None``):``    ``"""Converts a sequence of word ids to a sentence"""``    ``if` `isinstance``(id_tensor, torch.LongTensor):``        ``ids ``=` `id_tensor.transpose(``0``, ``1``).contiguous().view(``-``1``)``    ``elif` `isinstance``(id_tensor, np.ndarray):``        ``ids ``=` `id_tensor.transpose().reshape(``-``1``)``    ``batch ``=` `[vocab.itos[ind] ``for` `ind ``in` `ids] ``# denumericalize``    ``if` `join ``is` `None``:``        ``return` `batch``    ``else``:``        ``return` `join.join(batch)`
```

which can be run like this:

```
`arrs ``=` `model(b.text).cpu().data.numpy()``word_ids_to_sentence(np.argmax(arrs, axis``=``2``), TEXT.vocab, join``=``' '``)`
```

Limiting the results to the first few words, we get results like the following:

```
'<unk>   <eos> = = ( <eos>   <eos>   = = ( <unk> as the <unk> @-@ ( <unk> species , <unk> a <unk> of the <unk> ( the <eos> was <unk> <unk> <unk> to the the a of the first " , the , <eos>   <eos> reviewers were t'
```

It's hard to assess the quality, but it seems pretty obvious that we'll be needing to do more work or training to get the language model working.

 

## 4. Conclusion

Hopefully, this tutorial provided basic insight into how to use torchtext for language modeling, as well as some of the more advanced features of torchtext like built-in datasets, custom tokenizers, and pretrained word embeddings.

In this tutorial, we used a very basic language model, but there are many best practices that can improve performance significantly. In a future post, I'll discuss best practices for language modeling along with implementations.

# References

- https://mlexplained.com/

