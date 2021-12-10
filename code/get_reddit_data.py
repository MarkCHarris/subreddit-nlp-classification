"""
This script collects data from the Futurology and Science subreddits for NLP.
"""

############# DEPENDENCIES ##############

import requests
import pandas as pd
import time
import datetime

#########################################

# Save start time so time elapsed can be printed later.
t0 = time.time()

# General URL stub for the pushshift reddit API to retrieve comments.
url = 'https://api.pushshift.io/reddit/search/comment'

# Choose Futurology subreddit and maximum number of comments.
params = {
    'subreddit' : 'Futurology',
    'size' : 100
}
# Collect 100 comments and extract the 'data' portion.
fut_posts = requests.get(url, params).json()['data']
# Save the post time of the last comment collected so the next request can get the next 100 comments.
last_fut = fut_posts[-1]['created_utc']

# Repeat the above for the Science subreddit.
params = {
    'subreddit' : 'science',
    'size' : 100
}
sci_posts = requests.get(url, params).json()['data']
last_sci = sci_posts[-1]['created_utc']

"""
Below, j sets the number of files of comments to collect.

NOTES:
- I set j to 200, but this was much more than I needed.
- Collection was interrupted overnight after 17 files were collected.
- I adjusted j to range from 17 to 200 and manually set last_fut and last_sci
to the last posted comment to continue collection.
- This issue could have been avoided with try/except, but I already had plenty of data.
- I stopped collection after 52 files, which was already more than enough for this project.

- At one point, I accidentally over-wrote the first file of science comments.
- I ran a modified version of this script to reclaim the data from the same start time.
- Alignment with the original data wasn't perfect, but sufficient to have no likely impact on the analysis.
- The range of post times matched within minutes.
- This issue could be avoided in the future using Git.
"""
for j in range(200):
    
    # Number of times to collect 100 comments.  This setting gives me 10,000 comments per file.
    if j == 0:
        count = 99
    else:
        count = 100
    
    for i in range(count):

        # Slow request rate to avoid overburdening the server.
        time.sleep(2)

        # Collect comments as above, updating the start time so each 100 comments starts were the last left off.
        params = {
            'subreddit' : 'Futurology',
            'size' : 100,
            'before' : last_fut
        }
        fut_posts += requests.get(url, params).json()['data']
        last_fut = fut_posts[-1]['created_utc']

        time.sleep(2)

        params = {
            'subreddit' : 'science',
            'size' : 100,
            'before' : last_sci
        }
        sci_posts += requests.get(url, params).json()['data']
        last_sci = sci_posts[-1]['created_utc']

        # Print an update to the user after every 1000 comments are collected from each subreddit.
        if (i+1) % 10 == 0:
            # Referenced this resource to convert time from seconds: https://stackoverflow.com/questions/775049/how-do-i-convert-seconds-to-hours-minutes-and-seconds
            print(f'Collected {(i+1)*100} comments for file {j+1} out of 200 files. {datetime.timedelta(seconds=time.time()-t0)} elapsed.')
    
    # Save comments to a csv after each pass through the outer loop.
    fut_df = pd.DataFrame(fut_posts)
    sci_df = pd.DataFrame(sci_posts)
    fut_df.to_csv(f'../data/future_{j}.csv')
    sci_df.to_csv(f'../data/sci_{j}.csv')
    fut_posts = []
    sci_posts = []