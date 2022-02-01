# coding: UTF-8
'''
Created on Jan 25, 2022
@author: Xingchen Li
'''
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000  # Word length limits
UNK, PAD = '<UNK>', '<PAD>'  # Unknown characters and padding symbols
# Build dictionaries from training data
def build_vocab(file_path, tokenizer, max_size, min_freq):
    stopwords = None
    with open("./News/data/stopwords.txt") as f:
        stopwords = set(word.strip() for word in f.readlines())
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0] # Take out the text and build the word list without categorizing labels
            # Walk through all the words in the text and count the frequency of the words
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # Remove words from the word list that are less than the lowest frequency
        # Then sort the word list in reverse order according to word frequency, taking the first K items (k is the set size of the word list)
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[0] not in stopwords and _[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[:max_size]
        # Transform a word list into a word - to - index mapping
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # Finally, UNK and PAD characters are added
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # Separated by Spaces，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for i, line in enumerate(tqdm(f)):
                lin = line.strip()
                if not lin:
                    continue
                if len(lin.split('\t')) == 1:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token) # Sentence length (number of word/char contained), need to return
                if pad_size:
                    # padding,In case the length of the sentence does not change
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                # The format of a record is (sentence, label, seq_len) and setence is the index sequence of the word vector
                contents.append((words_line, int(label), seq_len))
        # [([id1, id2, ...], label, seq_len), 
        #  ([id1, id2, ...], label, seq_len),
        #  ...]
        return contents 
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # Record batch whether the number is an integer
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # Pad length (if larger than pad_size, set it to pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            # The training sample in the last batch was insufficient batCH_size and was not supplemented
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """Gets the used time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''Extract pre-trained word vector'''
    # The following directories and file names are changed as needed.
    train_dir = "./News/data/train.txt"
    vocab_dir = "./News/data/vocab.pkl"
    # The pre-trained word vectors are trained in char and word forms
    pretrain_dir = "./News/data/sgns.sogounews.bigram-char"
    emb_dim = 300
    filename_trimmed_dir = "./News/data/embedding_SougouNews"
    word = True
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        if word:
            # Build word lists by words (separated by Spaces between words in the dataset)
            tokenizer = lambda x: x.split(' ')  
        else:
            # Build a word list in terms of words
            tokenizer = lambda x: [y for y in x]
        # Build the dictionary from the training set  
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    # The dimensions of the pre-trained word vector are 365114x300 tasks
    # The dictionary size of the training set is 3791 when built word by word and 10002 when built word by word
    # Embedding dimension as / 3791 x 300 | 10002 x 300] [
    # In the word vector of training set, the number and proportion of pre-trained word vector are [3693/97%] | [8771/87%]  
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')    
    # The pre-trained word vector is stored as a string,
    cnt = 0
    for i, line in enumerate(f.readlines()):
        if i == 0:  # If the first line is a heading, skip it
            continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            cnt += 1
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    print("[", cnt, "/", len(word_to_id), ", ", cnt / len(word_to_id) * 100, "%]", sep="")
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
