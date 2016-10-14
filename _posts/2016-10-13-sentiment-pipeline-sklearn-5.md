---
layout: post
title: "Building a Sentiment Analysis Pipeline in scikit-learn Part 5: Parameter Search With Pipelines"
author: Ryan Cranfill
tags:
    - python
    - notebook
--- 
We have all these delicious preprocessing steps, feature extraction, and a neato classifier in our pipeline. Now it's time to tune this pipeline.

*This is Part 5 of 5 in a series on building a sentiment analysis pipeline using
scikit-learn. You can find the introduction [here](../sentiment-pipeline-
sklearn-1).*

*Jump to:*

* ***[Part 1 - Introduction and requirements](../sentiment-pipeline-
sklearn-1)***
* ***[Part 2 - Building a basic pipeline](../sentiment-pipeline-
sklearn-2)***
* ***[Part 3 - Adding a custom function to a pipeline](../sentiment-pipeline-
sklearn-3)***
* ***[Part 4 - Adding a custom feature to a pipeline with FeatureUnion
](../sentiment-pipeline-sklearn-4)***

# Part 5 - Hyperparameter tuning in pipelines with GridSearchCV

We've come so far together. Give yourself a hand.

Alright, ready to finish things up? Let's move on to the final entry in this series. We're going to do a parameter search to try to make this pipeline the best it can be.

# Setup 

**In [1]:**

{% highlight python %}
%%time
from fetch_twitter_data import fetch_the_data
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = fetch_the_data()
X, y = df.text, df.sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y)

tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
count_vect = CountVectorizer(tokenizer=tokenizer.tokenize) 
classifier = LogisticRegression()
{% endhighlight %}

    got 92 posts from page 1...
    got 88 posts from page 2...
    got 88 posts from page 3...
    got 91 posts from page 4...
    got 87 posts from page 5...
    got 89 posts from page 6...
    got 95 posts from page 7...
    got 93 posts from page 8...
    got 86 posts from page 9...
    got 90 posts from page 10...
    got all pages - 899 posts in total
    CPU times: user 853 ms, sys: 348 ms, total: 1.2 s
    Wall time: 7.7 s

 
# Construct the pipeline

This is the same pipeline that we ended up with in [part 4](../sentiment-
pipeline-sklearn-4). 

**In [2]:**

{% highlight python %}
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn_helpers import pipelinize, pipelinize_feature, get_tweet_length, genericize_mentions

sentiment_pipeline = Pipeline([
        ('genericize_mentions', pipelinize(genericize_mentions, active=True)),
        ('features', FeatureUnion([
                    ('vectorizer', count_vect),
                    ('post_length', pipelinize_feature(get_tweet_length, active=True))
                ])),
        ('classifier', classifier)
    ])
{% endhighlight %}
 
# Searching for golden hyperparameters 
 
One really sweet thing that scikit-learn has is a nice built-in parameter search
class called [GridSearchCV](http://scikit-learn.org/stable/modules/generated/skl
earn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV). It
plays nicely with the pipelines too. First we'll construct our parameter grid
and instantiate our `GridSearchCV`. 

**In [3]:**

{% highlight python %}
from sklearn.model_selection import GridSearchCV
from sklearn_helpers import train_test_and_evaluate
import numpy as np
import json

tokenizer_lowercase = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=False)
tokenizer_lowercase_reduce_len = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
tokenizer_uppercase = nltk.casual.TweetTokenizer(preserve_case=True, reduce_len=False)
tokenizer_uppercase_reduce_len = nltk.casual.TweetTokenizer(preserve_case=True, reduce_len=True)

# Our parameter dictionary
# You access parameters by giving the dictionary keys of <featurename>__<parameter>
# The values of each keys are a list of values that you want to test

parameters = {
    'genericize_mentions__kw_args': [{'active':False}, {'active':True}], # genericizing mentions on/off
    'features__post_length__kw_args': [{'active':False}, {'active':True}], # adding post length feature on/off
    'features__vectorizer__ngram_range': [(1,1), (1,2), (1,3)], # ngram range of tokenizer
    'features__vectorizer__tokenizer': [tokenizer_lowercase.tokenize, # differing parameters for the TweetTokenizer
                                        tokenizer_lowercase_reduce_len.tokenize,
                                        tokenizer_uppercase.tokenize,
                                        tokenizer_uppercase_reduce_len.tokenize,
                                        None], # None will use the default tokenizer
    'features__vectorizer__max_df': [0.25, 0.5], # maximum document frequency for the CountVectorizer
    'classifier__C': np.logspace(-2, 0, 3) # C value for the LogisticRegression
}

grid = GridSearchCV(sentiment_pipeline, parameters, verbose=1)
{% endhighlight %}
 
Now we're ready to perform that grid search. It's going to take a while (~3
minutes on my laptop) so kick back and relax. Or pace around and be tense. I'm
not going to police the way you spend your downtime. 

**In [4]:**

{% highlight python %}
grid, confusion_matrix = train_test_and_evaluate(grid, X_train, y_train, X_test, y_test)
{% endhighlight %}

    Fitting 3 folds for each of 360 candidates, totalling 1080 fits
    null accuracy: 40.44%
    accuracy score: 64.00%
    model is 23.56% more accurate than null accuracy
    ---------------------------------------------------------------------------
    Confusion Matrix
    
    Predicted  negative  neutral  positive  __all__
    Actual                                         
    negative         29       15        20       64
    neutral           8       43        19       70
    positive          7       12        72       91
    __all__          44       70       111      225
    ---------------------------------------------------------------------------
    Classification Report
    
                    precision    recall  F1_score support
    Classes                                              
    negative         0.659091  0.453125  0.537037      64
    neutral          0.614286  0.614286  0.614286      70
    positive         0.648649  0.791209  0.712871      91
    __avg / total__  0.640928      0.64  0.632185     225


    [Parallel(n_jobs=1)]: Done 1080 out of 1080 | elapsed:  2.9min finished

 
And now we'll print out what hyperparameters the search found made the best
model: 

**In [5]:**

{% highlight python %}
def print_best_params_dict(param_grid):
    used_cv = param_grid['features__vectorizer__tokenizer']
    if used_cv is None:
        params_to_print = grid.best_params_
        print 'used default CountVectorizer tokenizer'
    else:
        params_to_print = {i:grid.best_params_[i] for i in grid.best_params_ if i!='features__vectorizer__tokenizer'}
        print 'used CasualTokenizer with settings:'
        print '\tpreserve case: %s' % grid.best_params_['features__vectorizer__tokenizer'].im_self.preserve_case
        print '\treduce length: %s' % grid.best_params_['features__vectorizer__tokenizer'].im_self.reduce_len
    print 'best parameters: %s' % json.dumps(params_to_print, indent=4)
    
print_best_params_dict(grid.best_params_)
{% endhighlight %}

    used CasualTokenizer with settings:
    	preserve case: False
    	reduce length: False
    best parameters: {
        "features__vectorizer__ngram_range": [
            1, 
            1
        ], 
        "features__vectorizer__max_df": 0.25, 
        "classifier__C": 0.10000000000000001, 
        "features__post_length__kw_args": {
            "active": true
        }, 
        "genericize_mentions__kw_args": {
            "active": true
        }
    }

 
Thanks, `GridSearchCV`! You can also build your own custom scorers for use in
parameter grid searches in case you wanted to optimize for a particular metric
(such as negative recall), but that's a subject for another time.

# Now to summarize what we learned

Well, we're finally at the end of the series! You've learned so much - just take
a look at this list:

1. How to build a basic data pipeline
1. How to add text preprocessing inside a pipeline via FunctionTransformers
1. How to add new feature columns using FeatureUnion and some funky
FunctionTransformer stuff
1. How to run a cross-validated parameter grid search on the pipeline

So there you have it. A functional, living, breathing scikit-learn pipeline to
analyze sentiment. Keep building on to it, adding preprocessing steps, new
metafeatures, and tweaking hyperparameters.

Hope this was helpful, and thanks for reading.

Until next time,

#### *Ryan Cranfill*

# Thanks to
Dylan Lingelbach, Gordon Towne, Nathaniel Meierpolys, and the rest of the crew
at Earshot for all the help along the way.

*This is Part 5 of 5 in a series on building a sentiment analysis pipeline using
scikit-learn. You can find the introduction [here](../sentiment-pipeline-
sklearn-1). You can't find more parts because they don't exist.*

*Jump to:*

* ***[Part 1 - Introduction and requirements](../sentiment-pipeline-
sklearn-1)***
* ***[Part 2 - Building a basic pipeline](../sentiment-pipeline-
sklearn-2)***
* ***[Part 3 - Adding a custom function to a pipeline](../sentiment-pipeline-
sklearn-3)***
* ***[Part 4 - Adding a custom feature to a pipeline with FeatureUnion
](../sentiment-pipeline-sklearn-4)*** 
