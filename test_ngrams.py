#!/usr/bin/env python
# -*- coding=utf-8 -*-

import nltk
import collections
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
import pickle
import unicodedata

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

def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

def remove_stopwords(input):
    french_stopwords = stopwords.words('french')
    cleaned_stopwords = []
    cleaned_stopwords.append("les")
    for stopword in french_stopwords:
        cleaned_stopwords.append(remove_diacritic(stopword))
    filtered_text = ""
    for word in input.split():
        if word not in french_stopwords:
            if word != "les" and word != "a":
                filtered_text = filtered_text + " " + word
    return filtered_text

if __name__ == '__main__':
    import sys
    
    filename = sys.argv[1]

    with open(filename, 'rb') as handle:
        dictionary = pickle.load(handle)

    """text = ""
    for entry in dictionary:
        text = text + dictionary[entry]
    text = text.replace(",", "").replace(".", "").replace(":","")
    clean_text = remove_stopwords(text)"""

    text = dictionary['M. Paul Nairac'].replace(";","").replace(",", "").replace(":","").replace(".","").replace("(","").replace(")","")
    clean_text = remove_stopwords(text)

    new_bigrams = compute_bigrams(clean_text)
    bigram_count = Counter(new_bigrams)
    print("BIGRAMS")
    print(bigram_count.most_common(10))
    print("\n")

    new_trigrams = compute_trigrams(clean_text)
    trigram_count = Counter(new_trigrams)
    print("TRIGRAMS")
    print(trigram_count.most_common(10))
    print("\n")

    new_fourgrams = compute_fourgrams(clean_text)
    fourgram_count = Counter(new_fourgrams)
    print("FOURGRAMS")
    print(fourgram_count.most_common(10))
    print("\n")