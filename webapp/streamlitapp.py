import streamlit as st
import numpy as np
import pandas as pd
import NLP_cleaning
import NLP_summarization
import ldacomplaints
import altair as alt
from gensim import models

DATA_PATH = '../data_model/cleaned_tokenized_df-2020-06-07.csv'


def get_unique_biz_names(df):
    biz = df.biz_name.unique().tolist()
    biz.sort()
    return biz


def get_review_single_biz(df, biz_name):
    if biz_name is not None:
        df = df[df.biz_name.str.lower() == biz_name.lower()]
    return df

def preprocess_loaded_reviews(df):
    df = NLP_cleaning.apply_NLP_cleaning(df)
    return df

def load_pretrain_ldamodel(fname):
    lda_model = models.LdaModel.load(fname)
    return lda_model


def extract_topics_given_biz(df_new, lda_model, dictionary=None, rating='all'):
    if rating.lower() != 'all':
        df_new = df_new[df_new['review_rating'] == float(rating)]

    tokenized_text = df_new['review_text_lem_cleaned_tokenized_nostop']
    # if len(tokenized_text) == 1:
    #     tokenized_text = [NLP_cleaning.prep_lda_input(tokenized_text.tolist()[0])]

    df_dominant_topic_in_each_doc = ldacomplaints.get_topics_distributions_per_biz(tokenized_text, lda_model,
                                         dictionary)
    return df_dominant_topic_in_each_doc


def plot_topic_distribution(df_dominant_topic_in_each_doc):
    ldacomplaints.plot_topics_distribution_per_biz(df_dominant_topic_in_each_doc)
    return None

def text_summarization(df, review_rating='all'):
    if type(review_rating) != str:
        df = df[df['review_rating'] == float(review_rating)]

    # text summarization
    df_0 = NLP_summarization.clean_special_char(df, 'review_text')
    # summary, kw = NLP_summarization.extractive_sum(df_0['review_text'])
    df_0['summary'], df_0['kw'] = df_0['review_text'].apply(lambda x: NLP_summarization.extractive_sum(x))
    return df_0


@st.cache(allow_output_mutation=True)
def load_data(DATA_PATH=DATA_PATH):
    df_new = pd.read_csv(DATA_PATH)
    return df_new

@st.cache(allow_output_mutation=True)
def load_LDA_model(fname=r'/Users/dleung/Hack/pethotel/data_model/lda_hypertuned_nysf_reviews-2020-06-08-09-47.model'):
    lda_model = load_pretrain_ldamodel(fname=fname)
    return lda_model




# ------ streamlit ----------
st.title('PetCare Business Intelligence')
st.markdown('It’s all about the experience! Good experience inspires pet owners to also generate referrals. \n  PetCare is a BI dashboard built to help business owner in pet service industry stand out from competitors and improve customer retention by understanding customers\' feedback on services quickly. \n\n We understand that customer reviews are often wordy with important information buried in unstructued text, written in different styles, and cover a few different aspects in a single review. \n\n Use PetCare to gain insights into your strengths and weaknesses!')

df_new = load_data()
biz_name_option = get_unique_biz_names(df_new)
biz_name_option.insert(0, None)
lda_model = load_LDA_model()
try:
    dictionary, _ = ldacomplaints.get_dict_and_corpus(df_new['review_text_lem_cleaned_tokenized_nostop'])
except TypeError:
    df_new['review_text_lem_cleaned_tokenized_nostop'] = df_new['review_text_lem_cleaned_tokenized_nostop'].apply(lambda x: NLP_cleaning.prep_lda_input(x)).tolist()
    dictionary, _ = ldacomplaints.get_dict_and_corpus(df_new['review_text_lem_cleaned_tokenized_nostop'])

if st.checkbox("Show all reviews", False):
    st.subheader("")
    st.write(df_new[['biz_name', 'review_date', 'review_rating', 'review_text']])

st.header("Select your business: ")
biz_name = st.selectbox("Pick business name", biz_name_option)
df_new = get_review_single_biz(df_new, biz_name)

# # plot review distribution
review_pd = df_new.groupby('review_rating').count()['review_text'].reset_index()
review_pd.columns=['review_rating', 'count']
bars = alt.Chart(review_pd, width=500, height=400,
                 title='Review Distribution').mark_bar(clip=True,
                 color='firebrick', opacity=0.7, size=70).encode(x='review_rating',
                 y='count')
st.altair_chart(bars)


if biz_name is not None and len(df_new[df_new['review_rating'] <= 3]) > 0:
    st.write('Oh no! There are some bad reviews.')

if st.checkbox("Show reviews", False):
    st.subheader("Reviews for {}".format(biz_name))
    tmp = df_new[['review_date', 'review_rating', 'review_text']].reset_index(drop=True)
    st.write(tmp)

review_rating = st.selectbox("Select review rating:",
                             ['All', '1', '2', '3', '4', '5']
                             )
df_dominant_topic_in_each_doc = extract_topics_given_biz(df_new,
                                                         lda_model,
                                                         dictionary,
                                                         review_rating)
if len(df_dominant_topic_in_each_doc) > 0:
    bars = alt.Chart(df_dominant_topic_in_each_doc, width=500, height=400, title='Dominant topics across reviews').mark_bar(clip=True, color='firebrick', opacity=0.7, size=20).encode(x='Dominant_Topic', y='count')
    st.altair_chart(bars)
else:
    st.write("No reviews with {} stars on Yelp.".format(review_rating))



    #     df_0 = text_summarization(df_new, review_rating)  # not tested


# ------- end --------

st.markdown("## Party time!")
st.write("Yay! You've read all the reviews! Click below to celebrate.")
btn = st.button("Celebrate!")
if btn:
    st.balloons()