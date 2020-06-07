# NLP
from contractions import CONTRACTION_MAP
import re
import string
from collections import defaultdict
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# use spaCy for lemmatization
import en_core_web_sm
nlp = en_core_web_sm.load(disable=["ner", "parser"])

import spacy
from spacy import displacy
from collections import Counter


def correct_spelling(text):
    from textblob import TextBlob
    return TextBlob(text).correct()


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
    text = re.sub(r'yelp', ' ', text)
    text = re.sub(r"(mon|tues|wednes|thurs|fri|satur|sun)day", " ", text)
    return text


def remove_accented_chars(text):
    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def lemmatizer(text, tags=['NOUN']): # , 'ADJ', 'VERB', 'ADV']):
    """
    Keep noun
    """
    sent = []
    doc = nlp(text)
    output = [
        token.lemma_ if token.lemma_ not in '-PRON-' else '' for token in doc if token.pos_ in tags]
    return " ".join(output)


def display_NER(text):
    doc = nlp(lemmatizer(text))
    displacy.render(doc, jupyter=False, style='ent')


def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


def remove_stopwords(tokenized_list):
    """
    Done when standardizing, but to make sure and also include more stopwords
    """
    text = [word for word in tokenized_list if word not in STOPWORDS]
    return text


def remove_customized_stopwords(tokenized_list):
    import os
    if not os.path.isfile("stopwords-en.json"):
        os.system("wget https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.json")

    import json
    with open('stopwords-en.json', encoding='utf-8') as fopen:
        stopwords = json.load(fopen)
    text = [word for word in tokenized_list if word not in stopwords]
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
    df_new['review_text'] = df_new['review_text'].map(lambda x: correct_spelling(x))
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
    df_new['remove_customized_stopwords'] = df_new['remove_customized_stopwords'].apply(lambda x: remove_customized_stopwords(x))
    df_new['review_text_lem_cleaned_tokenized_nostop'] = remove_single_time_words(df_new['review_text_lem_cleaned_tokenized_nostop'])
    return df_new



# function to plot most frequent terms
def freq_words(x, terms=30):
    """

    x: df['review_text']
    """

    from nltk import FreqDist
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    import seaborn as sns
    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms)
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()