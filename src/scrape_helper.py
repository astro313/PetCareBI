"""

quick count on number of reviews scraped so far.

"""

import os, glob
import subprocess

def file_len(fname):
    """

    calc the number of rows in fnamme

    Parameters
    ----------
    fname: str

    Returns
    -------
    int

    """
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def standardize_colnames(df):
    df.columns = df.columns.str.replace("_", "-").str.lower()
    return df


def check_number_of_nans(fname):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(fname)
    df = standardize_colnames(df)
    if len(df) > 0:
        print("===")
        print("Calculate the % of missing values in each row")
        print(df.isna().mean())
        print("Min # of words in a review wrote for this biz: ", df['review-text'].str.len().min())
        if df['review-text'].str.len().min() == 0 or np.isnan(df['review-text'].str.len().min()):
            import pdb; pdb.set_trace()
            print("*** There are empty reviews in this file: ", fname)
            print("***")
            return fname
        else:
            return ''
    # else:
        # remove the .csv file?


if __name__ == '__main__':
    # review_files = glob.glob('../data/raw/*_2020*.csv')
#    review_files = glob.glob('../data/interim/9*.csv')
#    review_files = glob.glob('../data/interim/1*.csv')

    x = 0
    weird_review_file = []
    for filename in review_files:
        x += file_len(filename)
        fname = check_number_of_nans(filename)
        weird_review_file.append(fname)

    print("totel reviews scraped, some may be duplicate: ", x)

