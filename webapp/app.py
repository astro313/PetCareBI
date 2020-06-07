import flask
import pandas as pd
import NLP_cleaning
import NLP_summarization
import ldacomplaints
from gensim import models

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':

        df_new = pd.read_csv('data_model/......csv')
        biz_name_option = get_unique_biz_names(df_new)

        biz_name = flask.request.form['biz_name']   # ideally, dropdown menu
        review_rating = flask.request.form['review_rating']

        input_variables = pd.DataFrame([[biz_name, review_rating]],
                                       columns=['biz_name', 'review_rating'])

        df_new = get_review_single_biz(df_new, biz_name)
        lda_model = load_pretrain_ldamodel(fname=r'data_model/lda_bl_allservice_NYSF_reviews-2020-06-05-10\:45.model')

        df_dominant_topic_in_each_doc = extract_topics_given_biz(df_new, lda_model, review_rating)
        # plot_topic_distribution(df_dominant_topic_in_each_doc)

        df_0 = text_summarization(df_new, review_rating)


        return flask.render_template('main.html',
                                     original_input={'biz_name':biz_name,
                                                     'review_rating': review_rating
                                                     },
                                     result=df_0['summary'],
                                    )


def get_unique_biz_names(df):
    return df.biz_name.unique().tolist()


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
    app.run(host='0.0.0.0', port=5000, debug=True)



