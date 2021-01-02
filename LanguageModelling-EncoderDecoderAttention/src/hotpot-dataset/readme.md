# HotpotQA
HotpotQA is a set of question response data that includes natural multi-skip questions, with a strong emphasis on supporting facts to allow for more explicit question answering systems. The data set consists of 113,000 Wikipedia-based QA pairs.

# Data Download and Preprocessing
There are three HotpotQA files:
- Training set http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
- Dev set in the distractor setting http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
- Dev set in the fullwiki setting http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json This is just hotpot_dev_distractor_v1.json without the gold paragraphs, but instead with the top 10 paragraphs obtained using our retrieval system. If you want to use your own IR system (which is encouraged!), you can replace the paragraphs in this json with your own retrieval results. Please note that the gold paragraphs might or might not be in this json because our IR system is pretty basic.
- Test set in the fullwiki setting http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json Because in the fullwiki setting, you only need to submit your prediction to our evaluation server without the code, we publish the test set without the answers and supporting facts. The context in the file is paragraphs obtained using our retrieval system, which might or might not contain the gold paragraphs. Again you are encouraged to use your own IR system in this setting --- simply replace the paragraphs in this json with your own retrieval result

# JSON Format
The top level structure of each JSON file is a list, where each entry represents a question-answer data point. Each data point is a dict with the following keys:

- _id: a unique id for this question-answer data point. This is useful for evaluation.
- question: a string.
- answer: a string. The test set does not have this key.
- supporting_facts: a list. Each entry in the list is a list with two elements [title, sent_id], where title denotes the title of the paragraph, and sent_id denotes the supporting fact's id (0-based) in this paragraph. The test set does not have this key.
- context: a list. Each entry is a paragraph, which is represented as a list with two elements [title, sentences] and sentences is a list of strings.

There are other keys that are not used in our code, but might be used for other purposes (note that these keys are not present in the test sets, and your model should not rely on these two keys for making preditions on the test sets):

- type: either comparison or bridge, indicating the question type. (See our paper for more details).
- level: one of easy, medium, and hard. (See our paper for more details).
