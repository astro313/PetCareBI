import numpy as np
import scipy as sp
import pandas as pd
import glob

def read_all_reviews_in_area(list_of_files):
    for iii, fff in enumerate(list_of_files):
        df = pd.read_csv(fff,
                         usecols=['review_name', 'review_rating', 'review_date', 'review_text',
        'biz_url', 'biz_rating', 'biz_name']
                        )  # nrows=5, chunksize=1000000,
                                    # dtype={"country":"category", "beer_servings":"float64"},
                                    # na_values=0.0
                                    # index_col=0, skiprows=1      # header
        assert(len(df[df.duplicated()]) == 0)
        if iii ==0:
            df_all = df
        else:
            df_all= pd.concat([df_all, df])
    return df_all


def remove_irrelevant_reviews(df):
    import re
    long_con = (df.review_text.str.contains(re.compile(r'kennel(?i)')) & ~(df.review_text.str.contains(re.compile(r'kennel cough(?i)'))))

    condition = ((df.review_text.str.contains(re.compile(r'board(?i)')))
                 | (df.review_text.str.contains(re.compile(r'cation(?i)')))
                 | (df.review_text.str.contains(re.compile(r'hotel(?i)')))
                 | (df.review_text.str.contains(re.compile(r'travel(?i)')))
                 | (df.review_text.str.contains(re.compile(r'sitter(?i)')))
                 | (df.review_text.str.contains(re.compile(r'sitting(?i)')))
                 | (df.review_text.str.contains(re.compile(r'day care(?i)')))
                 | long_con)
    df = df[condition]
    return df


def biz_most_lowest_ratings(df):
    return df[df.review_rating == 1.0].biz_name.value_counts()

