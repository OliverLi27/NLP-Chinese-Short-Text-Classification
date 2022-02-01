'''
Created on Jan 25, 2022
@author: Xingchen Li
'''
import os
import pickle as pkl

import numpy as np

# Remove redundant features from raw data and convert them into (sequence, label) form

UNK, PAD = "<UNK>", "<PAD>"
MAX_VOCAB_SIZE = 10000
# The dictionary is constructed from the training data, and the sequence and label of each line of the file are separated by \t
def build_vocab(path, tokenizer, min_word_freq, max_vocab_size):
    vocab = {}
    # Iterate through the training data to generate a word list
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            content = line.strip().split("\t")[0]
            seq = tokenizer(content)
            for word in seq:
                vocab[word] = vocab.get(word, 0) + 1
    # Sort the dictionary by the number of occurrences of words and truncate it according to min_word_freq and max_VOCab_size
    vocab_list = sorted([item for item in vocab.items() if item[1] > min_word_freq], 
                key= lambda x: x[1], reverse=True)[:max_vocab_size]
    vocab = {item[0]:idx for idx, item in enumerate(vocab_list)}
    vocab.update({UNK:len(vocab), PAD:len(vocab)+1})
    return vocab




if __name__ == "__main__":
    train_path = "./data/News/train.txt"
    vocab_path = "./data/News/vocab.pkl"
    pretrained_path = "./data/News/sgns.sogounews.bigram-char"
    tabel_path = "./data/News/sougouNews"
    # tokenizer = lambda seq: [ch for ch in seq.strip()]
    tokenizer = lambda seq: seq.strip().split(" ")
    if os.path.exists(vocab_path):
        vocab = pkl.load(open(vocab_path, "rb"))
    else:
        vocab = build_vocab(train_path, tokenizer, 1,MAX_VOCAB_SIZE)
        pkl.dump(vocab, open(vocab_path, "wb"))

    # Randomly initializes the word vector
    embeddings = np.random.rand(len(vocab), 300)
    cnt = 0
    with open(pretrained_path, mode="r", encoding="utf-8") as f:
        for embed in f:
            # if i == 0:  
            # The first line is the title. Skip it
            # continue
            seq = embed.strip().split(" ")
            if seq[0] in vocab:
                cnt += 1
                embeddings[vocab[seq[0]]] = np.array([float(x) for x in seq[1:]])
    np.savez_compressed(tabel_path, embedding = embeddings)

