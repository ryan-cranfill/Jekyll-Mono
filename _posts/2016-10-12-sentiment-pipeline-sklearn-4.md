---
layout: post
title: "Building a Sentiment Analysis Pipeline in scikit-learn Part 4: Adding Custom Feature Extraction Functions With FeatureUnion"
author: Ryan Cranfill
tags:
    - python
    - notebook
--- 
Adding functions that execute in series with a pipeline is useful, but what if you want to create a new feature with a function?

*This is Part 4 of 5 in a series on building a sentiment analysis pipeline using
scikit-learn. You can find Part 5 [here](../sentiment-pipeline-sklearn-5),
and the introduction [here](../sentiment-pipeline-sklearn-1).*

*Jump to:*

* ***[Part 1 - Introduction and requirements](../sentiment-pipeline-
sklearn-1)***
* ***[Part 2 - Building a basic pipeline](../sentiment-pipeline-
sklearn-2)***
* ***[Part 3 - Adding a custom function to a pipeline](../sentiment-pipeline-
sklearn-3)***
* ***[Part 5 - Hyperparameter tuning in pipelines with GridSearchCV
](../sentiment-pipeline-sklearn-5)***

# Part 4 - Adding a custom feature to a pipeline with FeatureUnion

Let's learn how to add the *output* of a function as an additional feature for our classifier.

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
    CPU times: user 710 ms, sys: 193 ms, total: 903 ms
    Wall time: 6.24 s

 
# The function

For this example, we'll append the length of the post to the output of the count
vectorizer, the thinking being that longer posts could be more likely to be
polarized (such as someone going on a rant). 

**In [2]:**

{% highlight python %}
def get_tweet_length(text):
    return len(text)
{% endhighlight %}
 
# Adding new features
scikit-learn has a nice [FeatureUnion class](http://scikit-
learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) that
enables you to essentially concatenate more feature columns to the output of the
count vectorizer. This is useful for adding "meta" features.

It's pretty silly, but to add a feature in a `FeatureUnion`, it has to come back
as a numpy array of `dim(rows, num_cols)`. For our purposes in this example,
we're only bringing back a single column, so we have to reshape the output to
`dim(rows, 1)`. Gotta love it. So first, we'll define a method to reshape the
output of a function into something acceptable for `FeatureUnion`. After that,
we'll build our function that will wrap a function to be easily pipeline-able. 

**In [3]:**

{% highlight python %}
import numpy as np

def reshape_a_feature_column(series):
    return np.reshape(np.asarray(series), (len(series), 1))

def pipelinize_feature(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            processed = [function(i) for i in list_or_series]
            processed = reshape_a_feature_column(processed)
            return processed
#         This is incredibly stupid and hacky, but we need it to do a grid search with activation/deactivation.
#         If a feature is deactivated, we're going to just return a column of zeroes.
#         Zeroes shouldn't affect the regression, but other values may.
#         If you really want brownie points, consider pulling out that feature column later in the pipeline.
        else:
            return reshape_a_feature_column(np.zeros(len(list_or_series)))
{% endhighlight %}
 
# Adding the function and testing 

**In [4]:**

{% highlight python %}
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn_helpers import pipelinize, genericize_mentions, train_test_and_evaluate


sentiment_pipeline = Pipeline([
        ('genericize_mentions', pipelinize(genericize_mentions, active=True)),
        ('features', FeatureUnion([
                    ('vectorizer', count_vect),
                    ('post_length', pipelinize_feature(get_tweet_length, active=True))
                ])),
        ('classifier', classifier)
    ])

sentiment_pipeline, confusion_matrix = train_test_and_evaluate(sentiment_pipeline, X_train, y_train, X_test, y_test)
{% endhighlight %}

    null accuracy: 38.67%
    accuracy score: 62.22%
    model is 23.56% more accurate than null accuracy
    ---------------------------------------------------------------------------
    Confusion Matrix
    
    Predicted  negative  neutral  positive  __all__
    Actual                                         
    negative         28       19        17       64
    neutral          10       44        20       74
    positive          5       14        68       87
    __all__          43       77       105      225
    ---------------------------------------------------------------------------
    Classification Report
    
                    precision    recall  F1_score support
    Classes                                              
    negative         0.651163    0.4375  0.523364      64
    neutral          0.571429  0.594595  0.582781      74
    positive         0.647619  0.781609  0.708333      87
    __avg / total__  0.623569  0.622222  0.614427     225

 
# Almost done
Even though we did it in kind of a weird way, we are now able to add arbitrary
functions as new feature columns!

![](http://gifrific.com/wp-content/uploads/2012/09/Are-We-Having-Fun-Yet-Party-
Down.gif?style=centerme "Awwwww yeah")

We're now ready for the last part of the series - doing a parameter grid search
on the pipeline. [Come on, let's do it!](../sentiment-pipeline-sklearn-5)

*This is Part 4 of 5 in a series on building a sentiment analysis pipeline using
scikit-learn. You can find Part 5 [here](../sentiment-pipeline-sklearn-5),
and the introduction [here](../sentiment-pipeline-sklearn-1).*

*Jump to:*

* ***[Part 1 - Introduction and requirements](../sentiment-pipeline-
sklearn-1)***
* ***[Part 2 - Building a basic pipeline](../sentiment-pipeline-
sklearn-2)***
* ***[Part 3 - Adding a custom function to a pipeline](../sentiment-pipeline-
sklearn-3)***
* ***[Part 5 - Hyperparameter tuning in pipelines with GridSearchCV
](../sentiment-pipeline-sklearn-5)*** 
