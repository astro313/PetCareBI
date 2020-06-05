import flask
import pandas as pd
import load_SF_reviews
import NLP_cleaning
import NLP_summarization
import glob
import ldacomplaints
from gensim import models

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        biz_name = flask.request.form['biz_name']
        region = flask.request.form['region']
        reivew_rating = flask.request.form['review_rating']
        input_variables = pd.DataFrame([[biz_name, region, reivew_rating]],
                                       columns=['biz_name', 'region', 'review_rating'])
        df_new = load_yelp_files(region)
        df_new = preprocess_loaded_reviews(df_new)

        lda_model = load_pretrain_ldamodel(fname='data_model/lda_bl_allservice_NYSF_reviews-2020-06-05-10:45.model')

        df_dominant_topic_in_each_doc = extract_topics_given_biz(df_new, lda_model, biz_name, review_rating)
        # plot_topic_distribution(df_dominant_topic_in_each_doc)

        summary, kw = text_summarization(df_new, biz_name,
                                         review_rating)


        return flask.render_template('main.html',
                                     original_input={'biz_name':biz_name,
                                                     'region': region,
                                                     'review_rating': review_rating
                                                     },
                                     result=summary,
                                    )


def load_yelp_files(region):
    region = region.lower()
    if region == 'ny':
        review_files = glob.glob('../scraped_data_nodupl_biz/' + '1????*csv')
    elif region == 'sf':
        review_files = glob.glob('../scraped_data_nodupl_biz/' + '9????*csv')
    df_new = load_SF_reviews.read_all_reviews_in_area(review_files)
    df_new.drop_duplicates(inplace=True, keep='first')
    return df_new


def preprocess_loaded_reviews(df):
    df = NLP_cleaning.apply_NLP_cleaning(df)
    return df


def build_ldamodel(df_new):
    # if build model
    df_new = NLP_cleaning.apply_NLP_cleaning(df_new)
    lda_model, df_dominant_topic, topic_contribution = ldacomplaints.lda_analysis(df_new['review_text_lem_cleaned_tokenized_nostop'])
    return lda_model, df_dominant_topic


def load_pretrain_ldamodel(fname):
    lda_model = models.load(fname)
    return lda_model


def extract_topics_given_biz(df_new, lda_model, biz_name=None, rating='all'):
    # select biz from drop downs
    if biz_name is not None:
        df_new = df_new[(df_new['biz_name' == biz_name])]

    if rating.lower() != 'all':
        df_new = df_new[df_new[review_rating] == float(rating)]

    tokenized_text = df_new['review_text_lem_cleaned_tokenized_nostop']

    df_dominant_topic_in_each_doc = ldacomplaints.get_topics_distributions_per_biz(tokenized_text, lda_model)
    return df_dominant_topic_in_each_doc


def plot_topic_distribution(df_dominant_topic_in_each_doc):
    ldacomplaints.plot_topics_distribution_per_biz(df_dominant_topic_in_each_doc)
    return None

def text_summarization(df, biz_name=None, review_rating='all'):
    if biz_name is not None:
        df = df[df['biz_name' == biz_name]]

    if review_rating.lower() != 'all':
        df = df[df['review_rating' == float(review_rating)]]

    # text summarization
    df_0 = NLP_summarization.clean_special_char(df, 'review_text')
    summary, kw = NLP_summarization.summarize_this_review(df_0['review_text'])
    return summary, kw



if __name__ == '__main__':
    app.run()



