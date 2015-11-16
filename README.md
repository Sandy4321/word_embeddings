Evaluation of word embeddings
=============================

Original post: [Making sense of word2vec](http://radimrehurek.com/2014/12/making-sense-of-word2vec/).

Code for the blog post evaluating python implementations of [word2vec](https://github.com/piskvorky/gensim), [GloVe](https://github.com/maciejkula/glove-python) and GloVe implementation in [text2vec](https://github.com/dselivanov/text2vec).

Run `run_all.sh` to run all experiments. Logs with results will be stored in the data directory.

To replicate my results from the blog article, download and preprocess Wikipedia using [this code](https://github.com/piskvorky/sim-shootout).
You can use your own corpus though (the corpus path is a parameter to `run_all.sh`).
