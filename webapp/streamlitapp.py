import streamlit as st
import numpy as np
import pandas as pd
import NLP_cleaning
import NLP_summarization
import ldacomplaints
import altair as alt
from gensim import models
import sys
sys.path.append('/Users/dleung/Hack/pethotel/src')


DATA_PATH = '../data_model/cleaned_tokenized_df-2020-06-10.csv'
MODEL_PATH = r'/Users/dleung/Hack/pethotel/data_model/lda_hypertuned_nysf_reviews-2020-06-10-19-30.model'


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


def extract_topics_given_biz(df_new, lda_model, dictionary=None):
    tokenized_text = df_new['review_text_lem_cleaned_tokenized_nostop']
    # if len(tokenized_text) == 1:
    #     tokenized_text = [NLP_cleaning.prep_lda_input(tokenized_text.tolist()[0])]

    df_dominant_topic_in_each_doc = ldacomplaints.get_topics_distributions_per_biz(tokenized_text, lda_model, dictionary)
    return df_dominant_topic_in_each_doc


def plot_topic_distribution(df_dominant_topic_in_each_doc):
    ldacomplaints.plot_topics_distribution_per_biz(
        df_dominant_topic_in_each_doc)
    return None


def text_summarization(df, mode):
    if mode.lower() == 'extractive':
        df_0 = NLP_summarization.clean_special_char(df, 'review_text_raw')
        df_0['ex_summary'] = df_0['review_text_raw'].apply(
            lambda x: NLP_summarization.extractive_sum(x))

    elif mode.lower() == 'abstractive':
        df_0['ab_summary'] = df['review_text_raw'].apply(
            lambda x: NLP_summarization.abstractive_sum(x))

    return df_0


def render_summary(df, ii, summary_options):
    if summary_options.lower() == 'extractive':
        summary_field = 'ex_summary'
    elif summary_options.lower() == 'abstractive':
        summary_field = 'ab_summary'
    review_raw = df.iloc[ii]['review_text_raw']
    review_sum = df.iloc[ii][summary_field]
    st.text(review_raw)
    st.text(review_sum)


@st.cache(allow_output_mutation=True)
def load_data(DATA_PATH):
    df_new = pd.read_csv(DATA_PATH)
    return df_new


@st.cache(allow_output_mutation=True)
def load_LDA_model(fname):
    lda_model = load_pretrain_ldamodel(fname=fname)
    return lda_model


@st.cache(allow_output_mutation=True)
def get_dictionary_from_df(df_new):
    try:
        dictionary, _ = ldacomplaints.get_dict_and_corpus(
            df_new['review_text_lem_cleaned_tokenized_nostop'])
    except TypeError:
        df_new['review_text_lem_cleaned_tokenized_nostop'] = df_new[
            'review_text_lem_cleaned_tokenized_nostop'].apply(lambda x: NLP_cleaning.prep_lda_input(x)).tolist()
        dictionary, _ = ldacomplaints.get_dict_and_corpus(
            df_new['review_text_lem_cleaned_tokenized_nostop'])
    return dictionary, df_new


def main(DATA_PATH=None):

    st.sidebar.subheader("About App")
    st.sidebar.text("PetCare BI with Streamlit")

    st.sidebar.subheader("By")
    st.sidebar.text("T. K. Daisy Leung")
    st.sidebar.text("tkdaisyleung@gmail")

    st.title('PetCare Business Intelligence')
    st.markdown('It’s all about personalized experience! Good experience inspires pet owners to also generate referrals. \n  PetCare is a BI dashboard built to help business owner in pet service industry stand out from competitors and improve customer retention by understanding customers\' feedback on services quickly. \n\n We understand that customer reviews are often wordy with important information buried in unstructued text, written in different styles, and cover a few different aspects in a single review. \n\n Use PetCare to gain insights into your **strengths** and **weaknesses**!')

    # 1. fetch reviews regularly
    # 1. scrape_biz_link_multipage.py
    # 2. scrape_reviews.py
    # 3. zipcode_based_drop_dupl_biz.py
    # 3b. df_NY = load_yelp_files('ny')
    #  df_SF = load_yelp_files('sf')
    # df = pd.concat([df_NY, df_SF])
    # 4. df_new = NLP_cleaning.apply_NLP_cleaning(df)
    # 5. df_new.to_csv('data_model/cleaned_tokenized_df-2020-....csv',
    # index=False)

    # offline mode -- not updating database, use existing cleaning dataframe
    # stored
    df_new = load_data(DATA_PATH)
    biz_name_option = get_unique_biz_names(df_new)
    biz_name_option.insert(0, None)
    lda_model = load_LDA_model(MODEL_PATH)
    dictionary, df_new = get_dictionary_from_df(df_new)

    if st.checkbox("Show all reviews", False):
        st.subheader("")
        st.write(df_new[['biz_name', 'review_date',
                         'review_rating', 'review_text_raw']])

    st.header("Select your business: ")
    biz_name = st.selectbox("Pick business name", biz_name_option)
    df_new = get_review_single_biz(df_new, biz_name)

    # # plot review distribution
    review_pd = df_new.groupby('review_rating').count()[
        'review_text'].reset_index()
    review_pd.columns = ['review_rating', 'count']
    bars = alt.Chart(review_pd, width=500, height=400,
                     title='Review Distribution').mark_bar(clip=True,
                                                           color='firebrick', opacity=0.7, size=70).encode(x='review_rating', y='count')
    st.altair_chart(bars)

    if biz_name is not None and len(df_new[df_new['review_rating'] <= 3]) > 0:
        st.write('Oh no! There are some bad reviews.')

    review_rating_option = df_new['review_rating'].unique()
    review_rating_option.sort()
    review_rating_option = [str(int(i)) for i in review_rating_option]
    review_rating_option.insert(0, 'All')
    review_rating = st.selectbox("Select review rating:",
                                 review_rating_option,
                                 )

    if review_rating.lower() != 'all':
        df_new = df_new[df_new['review_rating'] == float(review_rating)]
    if len(df_new) < 0:
        st.write('No reviews ')

    df_dominant_topic_in_each_doc = extract_topics_given_biz(df_new,
                                                             lda_model,
                                                             dictionary)

    if len(df_dominant_topic_in_each_doc) > 0:
        # bars = alt.Chart(df_dominant_topic_in_each_doc, width=500, height=400, title='Dominant topics across reviews').mark_bar(clip=True, color='firebrick', opacity=0.7, size=20).encode(x='Dominant_Topic', y='count')
        # st.altair_chart(bars)

        # pie chart
        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots()
        ax1.pie(df_dominant_topic_in_each_doc['count'], labels=df_dominant_topic_in_each_doc['Dominant_Topic'], autopct='%1.1f%%',
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.axis('equal')
        st.write(fig1)

        if st.checkbox("Plot topic trend over the years (given selected review rating)"):
            st.write("Showing the distribution of topics covered by customer reviews over the years. \n\n (Normalized by the number of review per year shown.)")

            df_new['year'] = pd.to_datetime(df_new.review_date).dt.year
            min_yr = df_new['year'].min()
            max_yr = df_new['year'].max()
            nbin = max(1, max_yr - min_yr)
            bu = pd.cut(df_new.year, bins=nbin)
            gp = df_new.groupby(bu).count()['review_text']
            gp = gp.reset_index()
            gp.columns = ['year', 'review_count']
            gp_review = df_new.groupby(
                bu)['review_text_lem_cleaned_tokenized_nostop'].agg(list)
            gp_review = gp_review.reset_index()

            counting_dict = {}
            for xx in range(len(gp_review)):
                # loop through each year.
                year = int(round(gp_review.year.loc[xx].left))
                df_dominant_topic_in_each_doc = extract_topics_given_biz(gp_review.iloc[xx], lda_model, dictionary)
                counting_dict[str(year)] = df_dominant_topic_in_each_doc
            for jj in counting_dict.keys():
                if jj == list(counting_dict.keys())[0]:
                    holder = counting_dict[jj]
                try:
                    holder = pd.merge(holder, counting_dict[str(int(jj) + 1)],
                                      on='Dominant_Topic', how='outer')
                except:
                    pass
            holder.index = holder['Dominant_Topic']
            holder.drop(columns=['Dominant_Topic'], inplace=True)
            holder.fillna(0, inplace=True)   # replace Nan with 0
            # normalize # using total count in each year
            holder.columns = list(counting_dict.keys())
            holder = holder.T
            tmp = holder.div(holder.sum(axis=1), axis=0)
            tmp.plot(kind='bar')
            st.pyplot()

        if st.checkbox("Show executive summary on reviews (given selected review rating)"):
            # st.subheader("Summarize Your Text")
            reviewsNum = st.number_input(
                label="Pick number of most recent reviews", min_value=1, max_value=len(df_new))
            summary_options = st.selectbox("Choose Summarizer Mode", [
                                           'Extractive', 'Abstractive'])
            # sort the review
            df_tmp = df_new.sort_values(by='review_date')[::-1].iloc[:reviewsNum]
            df_tmp = df_tmp[['review_name', 'review_date', 'review_text_raw', 'review_rating']]
            if st.button("Summarize"):
                df_0 = text_summarization(
                    df_tmp, summary_options)

                for ii in range(len(df_0)):
                    render_summary(df_0, ii, summary_options)
                # st.success(df_0)
    else:
        st.write("No reviews with {} stars on Yelp.".format(review_rating))

    st.markdown("## Party time!")
    st.write("Yay! You've read all the reviews! Click below to celebrate.")
    btn = st.button("Celebrate!")
    if btn:
        st.balloons()


if __name__ == '__main__':
    main(DATA_PATH)
