library(text2vec)
library(readr)
library(stringr)
library(Matrix)

WINDOW = 10L
DIM = 600L
X_MAX = 100L
WORKERS = 4L
NUM_ITERS = 20L
TOKEN_LIMIT = 30000L
LEARNING_RATE = 0.15
# filter out tokens which are represented at least in 30% of documents
TOKEN_DOC_PROPORTION = 0.3
CONVERGENCE_THRESHOLD = 0.005

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  print("USAGE: %(program)s INPUT_DIR QUESTIONS OUTPUT_DIR")
  print('Example: Rscript ./run_glove_text2vec.R ~/Downloads/datasets/enwiki_splits/ 
        ~/Downloads/datasets/questions-words.txt 
        ./results_dim300_vocab30k')
}
  
input_dir <- args[[1]]
questions_file <- args[[2]]
output_dir <- args[[3]]

if (!dir.exists(output_dir))
  dir.create(output_dir)

# read only body of the article
preprocess_fun <- function(x) { x %>%
    str_split(pattern = fixed('\t')) %>%
    sapply(FUN = .subset2, 2)
}

get_vocabulary <- function(output_dir) {
  print(paste(Sys.time(), 'looking for existing vocabulary'))
  
  vocab_rds_path <- paste0(output_dir, '/', 'pruned_vocab_text2vec.rds')
  if ( file.exists(vocab_rds_path) ) {
    print(paste(Sys.time(), "vocabulary found, reading it..."))
    pruned_vocab <- readRDS(vocab_rds_path)
  } else {
    print(paste(Sys.time(), "vocabulary not found,  creating it..."))
    it1 <- idir(path = input_dir)
    it2 <- itoken(it1, preprocess_function = preprocess_fun, 
                  tokenizer = function(x) str_split(x, pattern = fixed(" ")) )
    vocab <- vocabulary(src = it2)
    print(paste(Sys.time(), "saving full vocabulary..."))
    saveRDS(vocab, file = paste0(output_dir, '/', 'full_vocab_text2vec.rds'), compress = FALSE)
    
    print(paste(Sys.time(), "pruning vocabulary..."))
    pruned_vocab <- prune_vocabulary(vocabulary = vocab, 
                                     doc_proportion_max = TOKEN_DOC_PROPORTION, 
                                     max_number_of_terms = TOKEN_LIMIT)
    print(paste(Sys.time(), "saving pruned vocabulary..."))
    saveRDS(pruned_vocab, file = vocab_rds_path, compress = FALSE)
    
    print(paste(Sys.time(), "saving pruned vocabulary in csv..."))
    write.table(x = data.frame("word" = pruned_vocab$vocab$terms, "id" = 0:(TOKEN_LIMIT - 1)), 
                file = paste0(output_dir, "/", "pruned_vocab.csv"),
                quote = F, sep = ',', row.names = F, col.names = F)
  }
  pruned_vocab
}

print(paste(Sys.time(), "looking for existing corpus ..."))
tcm_path <- paste0(output_dir, '/', 'tcm.rds')
if ( file.exists(tcm_path) ) {
  print(paste(Sys.time(), "tcm found, reading it..."))
  tcm <- readRDS(file = tcm_path)
} else {
  vocab <- get_vocabulary(output_dir)
  it1 <- idir(path = input_dir)
  it2 <- itoken(it1, 
                preprocess_function = preprocess_fun, 
                tokenizer = function(x) str_split(x, pattern = fixed(" ")) )
  
  print(paste(Sys.time(), "creating tcm ..."))
  corpus <- create_vocab_corpus(iterator = it2, vocabulary = vocab, grow_dtm = F, 
                                skip_grams_window = WINDOW)
  
  # get upper-triangular tcm matrix
  print(paste(Sys.time(), "converting tcm from map to sparse matrix ..."))
  tcm <- get_tcm(corpus)
  
  # remove corpus to reduce memory consumption
  print(paste(Sys.time(), "cleaning a little bit ..."))
  rm(corpus)
  gc()
  
  print(paste(Sys.time(), "saving tcm ..."))
  saveRDS(tcm, file = tcm_path, compress = FALSE)
}

print(paste(Sys.time(), "training glove ..."))
RcppParallel::setThreadOptions(numThreads = WORKERS)

fit <- glove(tcm = tcm, 
             word_vectors_size = DIM, 
             num_iters = NUM_ITERS,
             learning_rate = LEARNING_RATE,
             x_max = X_MAX, 
             shuffle_seed = 42L, 
             max_cost = 10,
             # we will stop if global cost will be reduced less then 1% then previous SGD iteration
             convergence_threshold = CONVERGENCE_THRESHOLD)

print(paste(Sys.time(), "saving glove model ..."))
saveRDS(fit, file = paste0( output_dir, '/glove_fit.rds'), compress = FALSE)

words <- rownames(tcm)
m <- fit$word_vectors$w_i + fit$word_vectors$w_j
rownames(m) <-  words

print(paste(Sys.time(), "reading questions ..."))
qlst <- prepare_analogue_questions(questions_file, rownames(m))

print(paste(Sys.time(), "checking accuracy ..."))
res <- check_analogue_accuracy(questions_lst = qlst, m_word_vectors = m)
