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
        biz_name_option = get_unique_biz_names()
        biz_name = flask.request.form['biz_name']   # ideally, dropdown menu

        region = flask.request.form['region']
        review_rating = flask.request.form['review_rating']

        input_variables = pd.DataFrame([[biz_name, region, review_rating]],
                                       columns=['biz_name', 'region', 'review_rating'])
        df_new = load_yelp_files(region)
        df_new = get_review_single_biz(df_new, biz_name)
        df_new = preprocess_loaded_reviews(df_new)

        lda_model = load_pretrain_ldamodel(fname=r'data_model/lda_bl_allservice_NYSF_reviews-2020-06-05-10\:45.model')

        df_dominant_topic_in_each_doc = extract_topics_given_biz(df_new, lda_model, review_rating)
        # plot_topic_distribution(df_dominant_topic_in_each_doc)

        df_0 = text_summarization(df_new, review_rating)


        return flask.render_template('main.html',
                                     original_input={'biz_name':biz_name,
                                                     'region': region,
                                                     'review_rating': review_rating
                                                     },
                                     result=df_0['summary'],
                                    )

def get_NY_SF_combined_fnames():
    review_files1 = glob.glob('../scraped_data_nodupl_biz/' + '1????*csv')
    review_files2 = glob.glob('../scraped_data_nodupl_biz/' + '2????*csv')
    review_files1.extend(review_files2)
    return review_files1


def get_unique_biz_names():
    review_files1 = get_NY_SF_combined_fnames()
    for iii, fff in enumerate(review_files1):
        df = pd.read_csv(fff,
                         usecols=['biz_name']
                        )  # nrows=5, chunksize=1000000,
                                    # dtype={"country":"category", "beer_servings":"float64"},
                                    # na_values=0.0
                                    # index_col=0, skiprows=1      # header
        # assert(len(df[df.duplicated()]) == 0)
        # drop dupl
        df.drop_duplicates(inplace=True, keep='first')

        if iii ==0:
            df_all = df
        else:
            df_all= pd.concat([df_all, df])
    return df_all.biz_name.unique().tolist()


def load_yelp_files(region):
    region = region.lower()
    if region == 'ny':
        review_files = glob.glob('../scraped_data_nodupl_biz/' + '1????*csv')
    elif region == 'sf':
        review_files = glob.glob('../scraped_data_nodupl_biz/' + '9????*csv')
    df_new = load_SF_reviews.read_all_reviews_in_area(review_files)
    df_new.drop_duplicates(inplace=True, keep='first')
    return df_new



def get_review_single_biz(df, biz_name):
    df = df[df.biz_name.str.lower() == biz_name]

def preprocess_loaded_reviews(df):
    df = NLP_cleaning.apply_NLP_cleaning(df)
    return df

def load_pretrain_ldamodel(fname):
    lda_model = models.LdaModel.load(fname)
    return lda_model


def extract_topics_given_biz(df_new, lda_model, rating='all'):
    if type(rating) != str:
        df_new = df_new[df_new['review_rating'] == float(rating)]

    tokenized_text = df_new['review_text_lem_cleaned_tokenized_nostop']

    df_dominant_topic_in_each_doc = ldacomplaints.get_topics_distributions_per_biz(tokenized_text, lda_model)
    return df_dominant_topic_in_each_doc


def plot_topic_distribution(df_dominant_topic_in_each_doc):
    ldacomplaints.plot_topics_distribution_per_biz(df_dominant_topic_in_each_doc)
    return None

def text_summarization(df, review_rating='all'):
    if type(review_rating) != str:
        df = df[df['review_rating'] == float(review_rating)]

    # text summarization
    df_0 = NLP_summarization.clean_special_char(df, 'review_text')
    # summary, kw = NLP_summarization.summarize_this_review(df_0['review_text'])
    df_0['summary'], df_0['kw'] = df_0['review_text'].apply(lambda x: NLP_summarization.summarize_this_review(x))
    return df_0



if __name__ == '__main__':
    app.run()



