# NLPf
import re
import string
import pandas as pd
from collections import defaultdict
import spacy
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

from gensim.summarization import summarize, keywords
from pprint import pprint


def extractive_sum(review, ratio=0.2):
    try:
        summary = summarize(review, ratio=ratio)  # word_count=20)    # max words
        summary = summary.replace("\n", " ")
    except ValueError:
        # too few sentences
        summary = review
    # finally:
    #     # Important keywords from the paragraph
    #     kw = keywords(review).replace('\n', '/')
    # print("important keywords: \n", )
    return summary   # , kw


def clean_special_char(df, text_field):
    df[text_field] = df[text_field].str.replace(r"\n", " ")
    return df


def abstractive_sum(review, num_beams=4, no_repeat_ngram_size=2,
                    min_length=30,
                    max_length=200,
                    early_stopping=True):
    import torch
    import json
    from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')

    preprocess_text = review.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    print("original text preprocessed: \n", preprocess_text)

    tokenized_text = tokenizer.encode(t5_prepared_Text,
                                      return_tensors="pt").to(device)

    # summmarize
    summary_ids = model.generate(tokenized_text,
                                 num_beams=num_beams,
                                 no_repeat_ngram_size=no_repeat_ngram_size,
                                 min_length=min_length,
                                 max_length=max_length,
                                 early_stopping=early_stopping)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # print ("\n\nSummarized text: \n",output)
    return output
