import spacy  # for our NLP processing
import nltk  # to use the stopwords library
import string  # for a list of all punctuation
from nltk.corpus import stopwords  # for a list of stopwords
import gensim
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from gensim import models
from gensim.models import Word2Vec
from gensim import corpora
# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim

# Now we can load and use spacy to analyse our complaint
import en_core_web_sm
# don't need, disable to speed up
nlp = en_core_web_sm.load(disable=['ner', 'parser'])


def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each
        # document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(
                        prop_topic, 4), topic_keywords]),
                    ignore_index=True,
                )
            else:
                break
    sent_topics_df.columns = ["Dominant_Topic",
                              "Perc_Contribution", "Topic_Keywords"]

    # Add original text to the end of the output
    contents = pd.Series(texts)
    contents = contents.reset_index()
    sent_topics_df = sent_topics_df.reset_index()
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


def topics_per_document(model, corpus, start=0, end=1):
    # most discussed topics in the reviews?
    # Sentence Coloring of N Sentences
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key=lambda x: x[
                                1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)


def get_dict_and_corpus(list_of_tokenized_text):
    """
    list_of_tokenized_text: all reviews
    """
    dictionary = gensim.corpora.Dictionary(list_of_tokenized_text)
    dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=100000)

    bow_corpus = [dictionary.doc2bow(doc) for doc in list_of_tokenized_text]
    return dictionary, bow_corpus



def lda_analysis(list_of_tokenized_text, num_topics=5):
    """
    list_of_tokenized_text: all reviews
    """
    dictionary, bow_corpus = get_dict_and_corpus(list_of_tokenized_text)
    lda_model = gensim.models.LdaModel(bow_corpus,
                                       num_topics=num_topics,
                                       id2word=dictionary,
                                       passes=10
                                       )

    # Visualize the topics
    # vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
    # pyLDAvis.display(vis)

    # dom_topc, percent, review_texts
    df_topic_sents_keywords = format_topics_sentences(
        ldamodel=lda_model, corpus=bow_corpus, texts=list_of_tokenized_text)

    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = [
        "Document_No",
        "Dominant_Topic",
        "Topic_Perc_Contrib",
        "Keywords",
        "Text",
    ]

    topic_counts, topic_contribution = get_fractional_con_per_topic(
        df_topic_sents_keywords)

    return lda_model, df_dominant_topic, topic_contribution


def get_fractional_con_per_topic(df_topic_sents_keywords):
    # Number of Documents for Each Topic
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts / topic_counts.sum(), 4)
    return topic_counts, topic_contribution


def get_topics_distributions_per_biz(list_of_tokenized_text, lda_model, dictionary=None):
    """
    list_of_tokenized_text here is for one single business
    """

    if dictionary is None:
        dictionary = gensim.corpora.Dictionary(list_of_tokenized_text)
        dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in list_of_tokenized_text]

    # # of reviews in each dominant_topic
    dominant_topics, topic_percentages = topics_per_document(model=lda_model,
                                                             corpus=bow_corpus)#,
                                                             # end=-1)
    df_123 = pd.DataFrame(dominant_topics, columns=[
                          'Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df_123.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(
        name='count').reset_index()
    return df_dominant_topic_in_each_doc


def plot_topics_distribution_per_biz(df):
    # df_dominant_topic_in_each_doc = get_topics_distributions_per_biz()

    # Topic Distribution by Dominant Topics
    import matplotlib.pyplot as plt
    plt.bar('Dominant_Topic', 'count', data=df, width=.5, color='firebrick')
    # plt.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    # tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
    # plt.xaxis.set_major_formatter(tick_formatter)
    plt.title('Number of Documents by Dominant Topic',
                  fontdict=dict(size=10))
    plt.ylabel('Number of Reviews')
#    plt.ylim(0, 40)
    plt.show()


def topic_vector(dictionary, topic_model: gensim.models.LdaModel, single_list_of_tokens: str):
    fingerprint = [0] * topic_model.num_topics
    for topic, prob in topic_model[dictionary.doc2bow(single_list_of_tokens)]:
        fingerprint[topic] = prob
    return fingerprint


def show_fingerprint(num_topics: int, topic_model, text: str):
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    VECTOR_SIZE = num_topics

    print(text)
    vector = topic_vector(topic_model, text)

    plt.figure(figsize=(8, 1))
    ax = plt.bar(range(len(vector)),
                 vector,
                 0.25,
                 linewidth=1)

    plt.ylim(top=0.4)
    plt.tick_params(axis='both',
                    which='both',
                    left=False,
                    bottom=False,
                    top=False,
                    labelleft=True,
                    labelbottom=True)
    plt.grid(False)
    plt.show()


def tsne_analysis(ldamodel, corpus):
    topic_weights = []
    for i, row_list in enumerate(ldamodel[corpus]):
        topic_weights.append([w for i, w in row_list])

    # Array of topic weights
    df_topics = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    # arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_nums = np.argmax(df_topics, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1,
                      random_state=0, angle=0.99, init="pca")
    tsne_lda = tsne_model.fit_transform(df_topics)

    return (topic_nums, tsne_lda)
