"""

Read .csv files with reviews from same zipcode, and reomve duplicated bizname across the pet services categories.


This step should actually be done before I scrape the reviews, and after scraping their urls. To be changed later.....

"""

import pandas as pd
import glob
import numpy as np
import os


def remove_dupl_biz(cities, out_path, columns, catnum=5):
    for zipc in cities:
        zipc = str(zipc)
        # find files for reviews of a given zip code
        fff = glob.glob('scraped_data/' + zipc + '_pet_*' + '_2020*.csv')
        num_df_to_create = len(fff)

        if num_df_to_create == catnum:
            # check if already done
            if not os.path.isfile(out_path + zipc + '_reviews.csv'):
                for num in range(num_df_to_create):
                    df = pd.read_csv(fff[num], header=0, names=columns)
                    if num == 0:
                        df_all = df
                    else:
                        df_all = pd.concat([df_all, df])

                # drop the duplicated business and then save as one .csv file zipcode
                before_len = len(df_all)
                # print(df_all.duplicated(subset=['biz_name', 'review_text']).sum())
                df_all.drop_duplicates(subset=['biz_name', 'review_text'], keep='first', inplace=True)
                print("from {} total rows to {} rows for all biz in this zipcode.".format(before_len, len(df_all)))
                df_all.to_csv(out_path + zipc + '_reviews.csv', \
                              index=False, index_label=False)
            else:
                print("zipcode done: {}".format(zipc))
        else:
            print("to run scripts again to deal with: ",  zipc)
            # import pdb; pdb.set_trace()
            # if not all categories extracted: pass now and we will run this script again to deal w/ those later


if __name__ == '__main__':
    NYC_review_files = glob.glob('scraped_data/' + '1????_pet_*' + '_2020*.csv')
    SF_review_files = glob.glob('scraped_data/' + '9????_pet_*' + '_2020*.csv')

    zipcodes_SF = np.loadtxt('zip_code_SF.txt', delimiter=',')
    cities_csv_SF = list(map(int, zipcodes_SF))

    zipcodes_NYC = np.loadtxt('zip_code.txt', delimiter=',')
    cities_csv_NY = list(map(int, zipcodes_NYC))

    columns = [
        'Ind', 'review_name', 'review_rating', 'review_date', 'review_text',
        'biz_url', 'biz_rating', 'biz_name', 'biz_phone', 'mon', 'tues', 'wed',
        'thurs', 'fri', 'sat', 'sun'
    ]

    out_path = 'scraped_data_nodupl_biz/'

    remove_dupl_biz(cities_csv_NY, out_path, columns, catnum=4)
    remove_dupl_biz(cities_csv_SF, out_path, columns)



