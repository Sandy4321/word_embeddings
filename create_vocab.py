#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s title_tokens.txt.gz OUTPUT_DIR

Example: python ./create_vocab.py /Users/dmitryselivanov/Downloads/datasets/title_tokens.txt.gz ./wiki_experiments

"""


import os
import sys
import logging
import csv
import itertools
from collections import defaultdict

import gensim
from gensim import utils, matutils

DOC_LIMIT = None
TOKEN_LIMIT = 30000
PRUNE_AT = None

logger = logging.getLogger("create_vocab")

if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
	logger.info("running %s" % " ".join(sys.argv))

	# check and process cmdline input
	program = os.path.basename(sys.argv[0])
	if len(sys.argv) < 3:
	    print(globals()['__doc__'] % locals())
	    sys.exit(1)
	in_file = gensim.models.word2vec.LineSentence(sys.argv[1])
	outf = lambda prefix: os.path.join(sys.argv[2], prefix)
	logger.info("output file template will be %s" % outf('PREFIX'))

	sentences = lambda: itertools.islice(in_file, DOC_LIMIT)

	if os.path.exists(outf('word2id')):
	    logger.info("dictionary already exists")
	else:
	    logger.info("dictionary not found, creating")
	    id2word = gensim.corpora.Dictionary(sentences(), prune_at=PRUNE_AT)
	    
	    id2word.save_as_text(outf("full_vocab.txt"))

	    utils.pickle(id2word, outf('id2word'))

	    id2word.filter_extremes(keep_n=TOKEN_LIMIT)  # filter out too freq/infreq words
	    word2id = dict((v, k) for k, v in id2word.iteritems())
	    
	    w = csv.writer(open(outf("TOKEN_LIMIT_vocab.txt"), "w"))
	    for key, val in word2id.items():
	        w.writerow([key.encode('utf-8').lower(), val])

	    utils.pickle(word2id, outf('word2id'))
	id2word = gensim.utils.revdict(word2id)

	logger.info("finished running %s" % program)