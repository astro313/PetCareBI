# NLP
import re
import string
import pandas as pd
from collections import defaultdict
import spacy
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

from gensim.summarization import summarize, keywords
from pprint import pprint


def summarize_this_review(review):

    summary = summarize(review, ratio=0.2)  # word_count=20)    # max words
    kw = keywords(review)
    # Important keywords from the paragraph
    print("important keywords: \n", )
    return summary, kw


def clean_special_char(df, text_field):
    df[text_field] = df[text_field].str.replace(r"\n", " ")
    return df

