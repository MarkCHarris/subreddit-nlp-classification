### Problem Statement ###

As science and technology continue to advance at an accelerating pace, we face an ever-deepening problem of how this advancement impacts society and how we can ensure that our progress does as much good and as little harm as possible.  One piece of this puzzle is a disconnect between how scientific progress is achieved and how it is portrayed, viewed, and understood by non-scientists.  This project aims to shed light on the differences between scientific discourse and discourse among broader audiences with demonstrated enthusiasm for human progress.  To achieve this, comments are collected from two very popular subreddits, Science and Futurology, using [Pushshift's API](https://github.com/pushshift/api).  After removal of deleted comments and posts by moderator bots, natural language processing methods are used to examine top words and trends in each subreddit.  Finally, multiple supervised classification methods are used to identify posts from each subreddit, testing how well different machine learning algorithms are able to distinguish between posts from these two subreddits.  The models are evaluated using precision, recall, F1 score, and ROC AUC.  This combination of metrics ensures that the best-scoring algorithms are equally successful when labelling posts as either Science or Futurology, while imbalanced algorithms are rejected.  The baseline score is 50%, since the models are fit on a 50/50 split of comments from each subreddit.  Words rated as most important by the most successful algorithms are analyzed for further insights.

### Structure of this Repo ###

README.md

Presentation Slides

models_pickled.txt: A text file that summarizes the parameters and abbreviated results of each model run during the final stage of the project.

pickles folder: Contains a copy of each model fit during the final stage of the project.  These copies are loaded into Code Notebook Part 3 for analysis.
- Note than one pickle file was far too large for storage on GitHub and has been excluded.

code folder:
- get_reddit_data.py: Script used to collect data from Reddit.
- Part 1: Initial cleaning and EDA of the data, including removal of deleted and moderator posts.
- Part 2: NLP EDA: Investigation of word counts, top words, and trends in the two subreddits.
- run_models.py : Script used to fit and store copies of different classification models.
- Part 3: Classification: Application of classification methods to choose stop words and evaluation of models fit in run_models.py.  Conclusions and Recommendations.
- cleantools.py: A script that contains a fuction used during the cleaning process in Part 1.
- metricstools.py: A script that contains the function used in Part 3 to check the metrics of each model after fitting.

data folder:
- corpus.csv: A copy of the data saved after cleaning and preprocessing in Part 1.
- The original data obtained from the API is not included in this repo due to its large size.

### Dictionary of Data Included in the Analysis ###

|Feature|Type|Description|
|---|---|---|
|subreddit|str|Name of the subreddit the post is from|
|body|str|Text of the comment|
|score|int|Score of the comment based on user upvotes and downvotes|
|author|str|Username of comment author|
|year|int|Year comment was posted|
|month|int|Month comment was posted|
|day|int|Day comment was posted|

### Data Exploration ###

Data exploration includes the following steps:
- Check for words that are very frequent in the corpus.
- Create a stop words list of words that are among the most frequent in both subreddits.
- Examine the most popular words in each subreddit.
- Look for trends in post length, post frequency by author, post scores, and sentiment of posts.

### Classificiation ###

Random forest and logistic regression classifiers are trained.  Words being given the most weight by these models are examined to identify stop words.  Ultimately, adding these stop words turns out to rarely be beneficial to the predictive power of the model and is not done in the final models.

The following classification models are used, and Gridsearch is applied to test multiple hyperparameter values of both the models and the TfidfVectorizer.
- Naive Bayes
- Logistic regression
- Random Forest
- Adaboost
- Support Vector Classifier

### Conclusions and Recommendations ###

Each of the models perform similarly in terms of F1 score, with different hyperparameter settings rarely affecting F1 score by more than a few percentage points. The highest F1 scores achieved are 83.5%, which is not bad considering that this includes some very short comments.  Models achieving this score include naive Bayes, logistic regression, and support vector classifier.  I consider naive Bayes to be the most successful, however, due to its achieving this score with a relatively short training time, with little dependence on hyperparameter settings and only moderate overfitting when compared to some of the other high-scoring models.

Observations during EDA, especially the most frequent twenty words in each subreddit, are suggestive of substantive differences between the two subreddits.  Futurology posts apper to be more focused on specific topics having to do with future hopes and goals, while Science posts are marked by references to sources and current problems and issues.

There are a few possible future directions for this work:

- Broaden the timeframe of the posts: These posts came from a timeframe of only one or two weeks. Sampling a wider time frame could reduce the reliance of the models on terms related to current events, possibly reducing both performance and overfitting. On the other hand, posts from a wider time frame may improve performance. With sufficient time or computing power, many more posts could be included. It would be interesting to see how this affects the model.
- Consider adding in the score or sentiment features to weight posts. Although these did not look like they would make a large difference when examined in the EDA section, they may make some difference if handled properly.
- Continue grid search options, especially for models that were overfit. Although model quality appears to have reached a maximum for these models and this data, there is more hyperparameter tuning that could be done with additional time or computing power.