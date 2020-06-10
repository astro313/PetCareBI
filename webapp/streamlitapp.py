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
    return df.biz_name.unique().tolist()


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
    if rating != 'all':
        print(len(df_new))
        df_new = df_new[df_new['review_rating'] == float(rating)]
        print(len(df_new))

    tokenized_text = df_new['review_text_lem_cleaned_tokenized_nostop']

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
st.markdown(r'It’s all about the experience! Good experience inspire pet owners to also generate referrals. But currently, there are pet owners hestiate to use these services because there aren\'t any comprehensive, summarization on the experience of other pet owners. They don\'t know where to start.')

df_new = load_data()
biz_name_option = get_unique_biz_names(df_new)
biz_name_option.insert(0, None)

if st.checkbox("Show all reviews", False):
    st.subheader("")
    st.write(df_new[['biz_name', 'review_date', 'review_rating', 'review_text']])

st.header("Select your businesss: ")
biz_name = st.selectbox("Pick business name", biz_name_option)
df_new = get_review_single_biz(df_new, biz_name)
if st.checkbox("Show reviews", False):
    st.subheader("Reviews for {}".format(biz_name))
    tmp = df_new[['biz_name', 'review_date', 'review_rating', 'review_text']].reset_index(drop=True)
    st.write(tmp)


lda_model = load_LDA_model()
try:
    dictionary, _ = ldacomplaints.get_dict_and_corpus(df_new['review_text_lem_cleaned_tokenized_nostop'])
except TypeError:
    df_new['review_text_lem_cleaned_tokenized_nostop'] = df_new['review_text_lem_cleaned_tokenized_nostop'].apply(lambda x: NLP_cleaning.prep_lda_input(x)).tolist()
    dictionary, _ = ldacomplaints.get_dict_and_corpus(df_new['review_text_lem_cleaned_tokenized_nostop'])


review_rating = st.selectbox("Select review rating:", ['1', '2', '3', '4', '5', 'all'])
df_dominant_topic_in_each_doc = extract_topics_given_biz(df_new,
                                                         lda_model,
                                                         dictionary,
                                                         review_rating)
# st.plot(df_new.query("review_rating == @review_rating")[['','']].dropna(how='any'))
bars = alt.Chart(df_dominant_topic_in_each_doc, width=500, height=400, title='Dominant topics across reviews').mark_bar(clip=True, color='firebrick', opacity=0.7).encode(x='Dominant_Topic', y='count')
st.altair_chart(bars)




# plot_topic_distribution(df_dominant_topic_in_each_doc
    # alt.Chart('Dominant_Topic', 'count', data=df, width=.5, color='firebrick')
    # # plt.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    # # tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
    # # plt.xaxis.set_major_formatter(tick_formatter)
    # plt.title('Number of Documents by Dominant Topic',
    #               fontdict=dict(size=10))
    # plt.ylabel('Number of Reviews')
# st.pyplot()

    #     df_0 = text_summarization(df_new, review_rating)  # not tested


# ------- end --------

st.markdown("## Party time!")
st.write("Yay! You're done with this tutorial of Streamlit. Click below to celebrate.")
btn = st.button("Celebrate!")
if btn:
    st.balloons()