#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
This file creates Excel files with the raw speeches and speaker names for ease of import
into R to group to data based on various criteria (e.g. month, date, speaker)
"""

import pickle
import pandas as pd
from pandas import *

if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))

    # Write just the raw speeches to Excel
    df = pd.DataFrame.from_dict(raw_speeches, orient = "index")
    filename = "raw_speeches.xlsx"
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, 'Sheet1')
    writer.save()

    # Write just the speaker names to Excel
    df2 = pd.DataFrame.from_dict(speechid_to_speaker, orient = "index")
    filename2 = "speechid_to_speaker.xlsx"
    writer2 = pd.ExcelWriter(filename2)
    df2.to_excel(writer2, 'Sheet1')
    writer2.save()

    # Concatenante the speeches with the speaker names to have all data in one Excel file
    joined = pd.concat([df,df2], axis = 1)
    filename3 = "speeches_and_speakers.xlsx"
    writer3 = pd.ExcelWriter(filename3)
    joined.to_excel(writer3, 'Sheet1')
    writer3.save()

