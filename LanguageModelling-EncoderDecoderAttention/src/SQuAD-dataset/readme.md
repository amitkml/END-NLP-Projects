# SQuAD
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

# Downloading Dataset
Download a copy of the dataset (distributed under the CC BY-SA 4.0 license):
- [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
Following command can be used from colab to download dataset

!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# What makes SQuAD so good? 

Of course, with a task like question answering, there are tons of datasets out there. When comparing SQuAD with other datasets, there are a few primary differences:
- SQuAD is big. Other reading comprehension datasets such as MCTest and Deep Read are too small to support intensive and complex models. MCTest only has a total of 2,640 questions, and Deep Read only has a total of 600 questions. SQuAD has these datasets dominated with a whopping 100,000+ questions.
- SQuAD is challenging. In other document-based question answering datasets that focus on answer extraction, the answer to a given question occurs in multiple documents. In SQuAD, however, the model only has access to a single passage, presenting a much more difficult task since it isn’t as forgiving to miss the answer.
- SQuAD requires reasoning. A popular type of dataset is the cloze dataset, which asks a model to predict a missing word in a passage. These datasets are large, and they present a somewhat-similar task as SQuAD. The key improvement that SQuAD makes on this aspect is that its answers are more complex and thus require more-intensive reasoning, thus making SQuAD better for evaluating model understanding and capabilities.
- Concluding thoughts. SQuAD is probably one of the most popular question answering datasets (it’s been cited over 2,000 times) because it’s well-created and improves on many aspects that other datasets fail to address. I’d highly recommend anyone that wants to evaluate an NLP model to test it on SQuAD, as it’s a great dataset for testing model understanding of language and even just performance in general.

# Downloading dataset
