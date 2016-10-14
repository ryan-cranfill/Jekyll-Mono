---
layout: post
title: "Building a Sentiment Analysis Pipeline in scikit-learn Part 3: Adding a Custom Function for Preprocessing Text"
author: Ryan Cranfill
tags:
    - python
    - notebook
--- 
This time we're going learn how to add a step in a pipeline that will preprocess the text - in this case by genericizing @ mentions.

*This is Part 3 of 5 in a series on building a sentiment analysis pipeline using
scikit-learn. You can find Part 4 [here](../sentiment-pipeline-sklearn-4),
and the introduction [here](../sentiment-pipeline-sklearn-1).*

*Jump to:*

* ***[Part 1 - Introduction and requirements](../sentiment-pipeline-
sklearn-1)***
* ***[Part 2 - Building a basic pipeline](../sentiment-pipeline-
sklearn-2)***
* ***[Part 4 - Adding a custom feature to a pipeline with FeatureUnion
](../sentiment-pipeline-sklearn-4)***
* ***[Part 5 - Hyperparameter tuning in pipelines with GridSearchCV
](../sentiment-pipeline-sklearn-5)***

# Part 3 - Adding a custom function to a pipeline

Now that we know how to build a basic scikit-learn pipeline, let's take it to
the next level. Text data often rewards feature engineering and preprocessing,
but there aren't a ton of built-in ways to do so. We're going to have to do some
weird stuff to be able to add in our own functions, but once we do so, we'll be
able to include any arbitrary function (to a certain extent, of course) in a
pipeline.

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
    CPU times: user 1.1 s, sys: 256 ms, total: 1.36 s
    Wall time: 6.61 s

 
# The function

What happens when we replace all @ mentions with a generic token? 

**In [2]:**

{% highlight python %}
import re

def genericize_mentions(text):
    return re.sub(r'@[\w_-]+', 'thisisanatmention', text)
{% endhighlight %}
 
# Preparing a function for a scikit-learn pipeline

scikit-learn's pipelines are dope, but every step has to look like a sklearn
transformer. Basically this means that everything that goes into a pipeline has
to implement `fit()` and `transform()` methods. The built-in [FunctionTranformer
](scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransf
ormer.html) does this handily, but items in a pipeline get passed as a full
array/series/list to each step, not individual items. So, we're going to wrap
our custom functions in a function that creates a list comprehension that
applies our custom function to the series passed in, then wraps that in a
FunctionTransformer. [Cue Inception horn](http://inception.davepedu.com/). 

**In [3]:**

{% highlight python %}
from sklearn.preprocessing import FunctionTransformer

def pipelinize(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            return [function(i) for i in list_or_series]
        else: # if it's not active, just pass it right back
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})
{% endhighlight %}
 
# Adding a function to a scikit-learn pipeline

Okay, so now that we have our function to wrap our function, we're going to
insert it into our pipeline and train and test. 

**In [4]:**

{% highlight python %}
from sklearn.pipeline import Pipeline
from sklearn_helpers import train_test_and_evaluate

sentiment_pipeline = Pipeline([
        ('genericize_mentions', pipelinize(genericize_mentions)),
        ('vectorizer', count_vect),
        ('classifier', classifier)
    ])

sentiment_pipeline, confusion_matrix = train_test_and_evaluate(sentiment_pipeline, X_train, y_train, X_test, y_test)
{% endhighlight %}

    null accuracy: 45.33%
    accuracy score: 65.78%
    model is 20.44% more accurate than null accuracy
    ---------------------------------------------------------------------------
    Confusion Matrix
    
    Predicted  negative  neutral  positive  __all__
    Actual                                         
    negative         28        9        12       49
    neutral          15       46        13       74
    positive         12       16        74      102
    __all__          55       71        99      225
    ---------------------------------------------------------------------------
    Classification Report
    
                    precision    recall  F1_score support
    Classes                                              
    negative         0.509091  0.571429  0.538462      49
    neutral          0.647887  0.621622  0.634483      74
    positive         0.747475   0.72549  0.736318     102
    __avg / total__  0.662807  0.657778  0.659737     225

 
Look at you, so accomplished! Now you can define whatever kind of function you
like and include it in a pipeline.

# What's next?

We're going to do \*nearly\* the same thing we just did, but instead of using
the output of a step as the input for the next step, we're going to take the
output of a step and use it as a new feature. Click on over to [part four
](../sentiment-pipeline-sklearn-4). You know you want to do it.

*This is Part 3 of 5 in a series on building a sentiment analysis pipeline using
scikit-learn. You can find Part 4 [here](../sentiment-pipeline-sklearn-4),
and the introduction [here](../sentiment-pipeline-sklearn-1).*

*Jump to:*

* ***[Part 1 - Introduction and requirements](../sentiment-pipeline-
sklearn-1)***
* ***[Part 2 - Building a basic pipeline](../sentiment-pipeline-
sklearn-2)***
* ***[Part 4 - Adding a custom feature to a pipeline with FeatureUnion
](../sentiment-pipeline-sklearn-4)***
* ***[Part 5 - Hyperparameter tuning in pipelines with GridSearchCV
](../sentiment-pipeline-sklearn-5)*** 
