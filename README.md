Evaluation of word embeddings
=============================

Code for the [blog post](http://dsnotes.com/blog/text2vec/2015/12/01/glove-enwiki/) evaluating implementations of [word2vec](https://github.com/piskvorky/gensim) in *gensim* and [GloVe](https://github.com/dselivanov/text2vec) in *text2vec*.

## Running

1. This will create vocabulary, train *text2vec* GloVe and evaluate it:
    `bash -i -c "./memusg Rscript ./run_glove_text2vec.R ~/Downloads/datasets/enwiki_splits/ ~/Downloads/datasets/questions-words.txt  ./enwiki_dim=600_vocab=30k/"  > ./enwiki_dim=600_vocab=30k/glove.log 2>&1 &`
2. This will train *gensim* and evaluate its accuracy: 
`bash -i -c "./memusg python ./run_word2vec.py ~/Downloads/datasets/title_tokens.txt.gz ~/Downloads/datasets/questions-words.txt ./enwiki_dim=600_vocab=30k" > ./enwiki_dim=600_vocab=30k/word2vec.log 2>&1 &`

To replicate my results from the blog article, download and preprocess Wikipedia using [this code](https://github.com/piskvorky/sim-shootout).
