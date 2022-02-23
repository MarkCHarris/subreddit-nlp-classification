"""
This script runs GridsearchCV on multiple models and generates a pickle of each model.
For each model pickled, filename and model parameters are printed to a text file.
This record makes it easy to select and load pickles for further analysis.
"""

############# DEPENDENCIES ##############

import pandas as pd

import time
import datetime

import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


###### SHARED PARAMETERS AND DATA PREP ######

# Used as baseline to output overall script runtime.
overall_t0 = time.time()

# Custom stop word list.
stop_words = ['the', 'to', 'and', 'of', 'it', 'that', 'is', 'in', 'you', 'for', 'are', 'be', 'not', 'this', 'but',
             'we', 'they', 'on', 'have', 'with', 'can', 'a', 'if', 'or', 'just', 'people', 'would', 'so', 'like',
             'more', 'all', 'there', 'at', 'what', 'from', 'about', 'do', 'an', 'wa', 'by', 'don', 'one', 'get',
             'how', 'their', 'no', 'than', 'your', 're', 'ha', 'think', 'out', 'because', 'thing', 'even', 'my',
             'will', 'year', 'up', 'make', 'need', 'them', 'when', 'could', 'some', 'only', 'much', 'gt', 'which',
             'way', 'also', 'then', 'other', 'now', 'who', 'http', 'being', 'know', 'why', 'good', 'most', 'any',
             'still', 'see', 'really', 'should', 'me', 'these', 'he', 'time', 'u', 'our', 'into', 'going', 'go',
             'want', 'work', 'use', 'been', 'well', 'those', 'take', 'where', 'point', 'mean', 'very', 'lot',
             'problem', 'over', 'here', 'say', 'something', 'right', 'doesn', 'many', 'same', 'isn',
             'too', 'were', 've', 'actually', 'every', 'le', 'had', 'used', 'www', 'sure']

# Random seed to be used for all fits for reproducibility.
shared_seed = 2904

# Import data and train/test/split.
corpus = pd.read_csv('../data/corpus.csv')
X = corpus['body']
y = corpus['subreddit'].map({'Futurology':0, 'science':1})
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.7, random_state=4835)

# Custom tokenizer to be passed to all vectorizers.
def token_lem(in_str):
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    tokenized = tokenizer.tokenize(in_str)
    lemmatized = [lemmatizer.lemmatize(token.lower()) for token in tokenized if len(token) > 1]
    return lemmatized

#### FUNCTION THAT FITS EACH MODEL AND OUTPUTS INFO TO A FILE ####

def fit_and_save(X_data, y_data, model_name, pipeline, pipeline_params):
    
    t0 = time.time()

    # The actual Gridsearch happens here.
    gs = GridSearchCV(pipeline, pipeline_params, scoring='f1', cv=5)
    gs.fit(X_data, y_data)

    # Save the model to a pickle.
    pickle.dump(gs, open(f'../pickles/{model_name}.p', 'wb'))

    # How long this Gridsearch took.
    elapsed = datetime.timedelta(seconds=time.time()-t0)

    # Print the pickle name, parameters, and basic score info to 'models_pickled.txt'
    f = open('../models_pickled.txt', 'a')
    f.write('\n\n**********\n')
    f.write(f'Model name: {model_name}\n')
    f.write(f'Best score: {gs.best_score_}\n')
    f.write(f'Time to process: {elapsed}\n')
    f.write(f'\nParameters tested:\n')
    for param in pipeline_params:
        f.write(f'{param}: {pipeline_params[param]}\n')
    f.write(f'\nBest parameters:\n')
    for param in gs.best_params_:
        f.write(f'{param}: {gs.best_params_[param]}\n')
    f.write('**********')
    f.close()

    # Update the user after each Gridsearch completes.
    print(f'Model {model_name} pickled!  Time elapsed: {elapsed}')


############ MODELS TO BE GRIDSEARCHED ############

name = 'bayes_2'

pipe = Pipeline([
    ('tf_vec', TfidfVectorizer(tokenizer=token_lem, min_df=5)),
    ('bayes', MultinomialNB())
])

pipe_params = {
    'tf_vec__ngram_range' : [(1,1)],
    'tf_vec__max_features' : [1_000, 3_000, 5_000],
    'tf_vec__stop_words' : [None, stop_words],
    'bayes__alpha' : [0.5],
    'bayes__fit_prior' : [True]
}

fit_and_save(X_train, y_train, name, pipe, pipe_params)

#####

name = 'forest_2'

pipe = Pipeline([
    ('tf_vec', TfidfVectorizer(max_df=0.2, stop_words=stop_words)),
    ('forest', RandomForestClassifier(random_state=shared_seed, n_jobs=-1))
])

pipe_params = {
    'tf_vec__ngram_range' : [(1,1)],
    'tf_vec__max_features' : [1_000, 5_000],
    'tf_vec__stop_words' : [None, stop_words],
    'forest__ccp_alpha' : [0.0],
    'forest__max_depth' : [10],
    'forest__n_estimators' : [500]
}

fit_and_save(X_train, y_train, name, pipe, pipe_params)

#####

name = 'ada_2'

pipe = Pipeline([
    ('tf_vec', TfidfVectorizer(max_df=0.2, stop_words=stop_words)),
    ('ada', AdaBoostClassifier(random_state=shared_seed))
])

pipe_params = {
    'tf_vec__ngram_range' : [(1,2)],
    'tf_vec__max_features' : [1_000, 5_000],
    'tf_vec__stop_words' : [None, stop_words],
    'ada__n_estimators' : [600, 800],
    'ada__learning_rate' : [.5]
}

fit_and_save(X_train, y_train, name, pipe, pipe_params)

#####

name = 'log_2'

pipe = Pipeline([
    ('tf_vec', TfidfVectorizer(max_df=0.2, stop_words=stop_words)),
    ('logreg', LogisticRegression(random_state=shared_seed, n_jobs=-1, solver='saga', max_iter=10_000))
])

pipe_params = {
    'tf_vec__ngram_range' : [(1,1)],
    'tf_vec__max_features' : [1_000, 5_000],
    'tf_vec__stop_words' : [None, stop_words],
    'logreg__C' : [1],
    'logreg__penalty' : ['l2']
}

fit_and_save(X_train, y_train, name, pipe, pipe_params)

#####

name = 'svc_2'

pipe = Pipeline([
    ('tf_vec', TfidfVectorizer(max_df=0.2, stop_words=stop_words)),
    ('svc', SVC(random_state=shared_seed))
])

pipe_params = {
    'tf_vec__ngram_range' : [(1,1)],
    'tf_vec__max_features' : [1_000, 5_000],
    'tf_vec__stop_words' : [None, stop_words],
    'svc__C' : [1],
    'svc__kernel' : ['poly'],
    'svc__degree' : [2],
    'svc__gamma' : ['scale']
}

fit_and_save(X_train, y_train, name, pipe, pipe_params)

##########################################################

# Let the user know all the models are fit and the total time it took.
print(f'All pickling complete!  Total time elapsed: {datetime.timedelta(seconds=time.time()-overall_t0)}')