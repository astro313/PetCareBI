# NLP
import gensim
from gensim import models
from gensim.models import Word2Vec
from gensim import corpora

import re
import string
import pandas as pd
from collections import defaultdict
import spacy
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# use spaCy for lemmatization
import en_core_web_sm
# don't need, disable to speed up
nlp = en_core_web_sm.load(disable=['ner', 'parser'])


def standardize_text(df, text_field):
    # followed by one or more non-whitespaces, for the domain name
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    # df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"&", "and")
    df[text_field] = df[text_field].str.replace(r"#", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at ")
    df[text_field] = df[text_field].str.replace(
        r"[^A-Za-z0-9(),!?@\''\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r'\d+', ' ')
    # but want to keep e.g., 18 years old, how would this info be captured?

    df[text_field] = df[text_field].str.replace(r"\'", " ")
    df[text_field] = df[text_field].str.replace(r"\"", " ")
    df[text_field] = df[text_field].str.replace(r"\n", " ")
    df[text_field] = df[text_field].str.replace(r"\r", " ")
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].str.replace(r"bc", "because")
    df[text_field] = df[text_field].str.replace(r"b c", "because")
    df[text_field] = df[text_field].str.replace(r"b/c", "because")
    df[text_field] = df[text_field].str.replace(r"dr\.", "dr")
    df[text_field] = df[text_field].str.replace(r"d\.r\.", "dr")
    df[text_field] = df[text_field].str.replace(r"sf", "")
    df[text_field] = df[text_field].str.replace(r"ny", "")
    df[text_field] = df[text_field].str.replace(r"nyc", "")
    df[text_field] = df[text_field].str.replace(r"tbh", "to be honest")
    return df


from contractions import CONTRACTION_MAP


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
            if contraction_mapping.get(match)\
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    # text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"hr", "hour", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    # text = re.sub('\W', ' ', text)     # any non-alphanumeric character. Equivalent to [^a-zA-Z0-9_]
    # text = re.sub('\s+', ' ', text)    # all whitespaces
    text = text.strip(' ')
    return text


def clean_text_2(text):
    '''Eemove text in square brackets, remove punctuation, remove words containing numbers.'''
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove a sentence if it is only one word long
    if len(text) > 2:
        return ' '.join(word for word in text.split() if word not in STOPWORDS)


def clean_text_3(text):
    """ Reomve Yelp"""
    text = re.sub(r'yelp(?i)', ' ', text)
    text = re.sub(r"(mon|tues|wednes|thurs|fri|satur|sun)day", " ", text)
    return text


def remove_accented_chars(text):
    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def lemmatizer(text, tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    Keep noun and adjectivess
    """
    sent = []
    doc = nlp(text)
    output = [
        token.lemma_ if token.lemma_ not in '-PRON-' else '' for token in doc if token.pos_ in tags]
    return " ".join(output)


def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


def remove_stopwords(tokenized_list):
    """
    Done above when cleaning, but just to make sure
    """
    text = [word for word in tokenized_list if word not in STOPWORDS]
    return text

# remove less freq words (only occured once ever)
def remove_single_time_words(list_of_tokenized_texts):
    frequency = defaultdict(int)
    for text in list_of_tokenized_texts:
        for token in text:
             frequency[token] += 1
    list_of_tokenized_texts = [[token for token in text if frequency[token] > 1] for text in list_of_tokenized_texts]
    return list_of_tokenized_texts



def apply_NLP_cleaning(df_new):
    df_new = standardize_text(df_new, 'review_text')
    df_new['review_text'] = df_new['review_text'].map(
        lambda x: expand_contractions(x))
    df_new['review_text'] = df_new['review_text'].map(lambda x: clean_text(x))
    df_new['review_text'] = df_new[
        'review_text'].apply(lambda x: clean_text_2(x))
    df_new['review_text'] = df_new[
        'review_text'].apply(lambda x: clean_text_3(x))
    df_new['review_text'] = df_new['review_text'].apply(
        lambda x: remove_accented_chars(x))
    df_new['review_text_lem_cleaned'] = df_new[
        'review_text'].apply(lambda x: lemmatizer(x))
    df_new['review_text_lem_cleaned_tokenized'] = df_new[
        'review_text_lem_cleaned'].apply(lambda x: tokenize(x.lower()))
    df_new['review_text_lem_cleaned_tokenized_nostop'] = df_new['review_text_lem_cleaned_tokenized'].apply(lambda x: remove_stopwords(x))
    df_new['review_text_lem_cleaned_tokenized_nostop'] = remove_single_time_words(df_new['review_text_lem_cleaned_tokenized_nostop'])
    return df_new