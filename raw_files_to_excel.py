#!/usr/bin/env python
# -*- coding=utf-8 -*-

from bs4 import BeautifulSoup
import unicodedata
import csv
import pickle
import regex as re
import pandas as pd
from pandas import *
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
import collections
from collections import Counter
import os
import math
from collections import defaultdict

if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))

    df = pd.DataFrame.from_dict(raw_speeches, orient = "index")
    """#df.columns = ["speechid", "speech"]
    filename = "raw_speeches.xlsx"
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, 'Sheet1')
    writer.save()"""

    df2 = pd.DataFrame.from_dict(speechid_to_speaker, orient = "index")
    """#df2.columns = ["speechid", "speaker"]
    filename2 = "speechid_to_speaker.xlsx"
    writer2 = pd.ExcelWriter(filename2)
    df2.to_excel(writer2, 'Sheet1')
    writer2.save()"""

    #joined = df.set_index("speechid").join(df2.set_index("speechid"))
    joined = pd.concat([df,df2], axis = 1)
    filename3 = "speeches_and_speakers.xlsx"
    writer3 = pd.ExcelWriter(filename3)
    joined.to_excel(writer3, 'Sheet1')
    writer3.save()

