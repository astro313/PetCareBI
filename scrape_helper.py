""" 

quick count on number of reviews scraped so far.

"""

import os, glob
import subprocess

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

if __name__ == '__main__':
    review_files = glob.glob('scraped_data/*_2020*.csv')

    x = 0
    for filename in review_files:
        x += file_len(filename)

    print("totel reviews scraped, some may be duplicate: ", x)


