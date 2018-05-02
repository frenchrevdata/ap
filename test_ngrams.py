#!/usr/bin/env python
# -*- coding=utf-8 -*-

import nltk
import collections
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
import pickle

def compute_bigrams(input):
    token = word_tokenize(input)
    bigrams = ngrams(token, 2)
    return bigrams

def compute_trigrams(input):
    token = word_tokenize(input)
    trigrams = ngrams(token, 3)
    return trigrams

def compute_fourgrams(input):
    token = word_tokenize(input)
    fourgrams = ngrams(token, 4)
    return fourgrams


if __name__ == '__main__':
    import sys
    
    filename = sys.argv[1]

    with open(filename, 'rb') as handle:
        dictionary = pickle.load(handle)

    text = dictionary['M. Paul Nairac']
    text = text.replace(",", "").replace(".", "")

    new_bigrams = compute_bigrams(text)
    bigram_count = Counter(new_bigrams)
    print("BIGRAMS")
    print(bigram_count.most_common(10))
    print("\n")

    new_trigrams = compute_trigrams(text)
    trigram_count = Counter(new_trigrams)
    print("TRIGRAMS")
    print(trigram_count.most_common(10))
    print("\n")

    new_fourgrams = compute_fourgrams(text)
    fourgram_count = Counter(new_fourgrams)
    print("FOURGRAMS")
    print(fourgram_count.most_common(10))
    print("\n")

    with open('Nairac_bigram.pickle', 'wb') as handle:
        pickle.dump(bigram_count, handle, protocol = 0)

    with open('Nairac_trigram.pickle', 'wb') as handle:
        pickle.dump(trigram_count, handle, protocol = 0)

    with open('Nairac_fourgram.pickle', 'wb') as handle:
        pickle.dump(fourgram_count, handle, protocol = 0)