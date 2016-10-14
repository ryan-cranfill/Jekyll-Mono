---
layout: post
title: "Building a Sentiment Analysis Pipeline in scikit-learn Part 2: Building a Basic Pipeline"
author: Ryan Cranfill
tags:
    - python
    - notebook
--- 
Ready to build an extremely basic pipeline? Good, lets do it!

*This is Part 2 of 5 in a series on building a sentiment analysis pipeline using
scikit-learn. You can find Part 3 [here](../sentiment-pipeline-sklearn-3),
and the introduction [here](../sentiment-pipeline-sklearn-1).*

*Jump to:*

* ***[Part 1 - Introduction and requirements](../sentiment-pipeline-
sklearn-1)***
* ***[Part 3 - Adding a custom function to a pipeline](../sentiment-pipeline-
sklearn-3)***
* ***[Part 4 - Adding a custom feature to a pipeline with FeatureUnion
](../sentiment-pipeline-sklearn-4)***
* ***[Part 5 - Hyperparameter tuning in pipelines with GridSearchCV
](../sentiment-pipeline-sklearn-5)***

# Part 2 - Building a basic scikit-learn pipeline

In this post, we're going to build a very simple pipeline, consisting of a count
vectorizer for feature extraction and a logistic regression for classification.

# Get The Data

First, we're going to read in a CSV that has the ids of 1000 tweets and the
posts' labeled sentiments, then we're going to fetch the contents of those
tweets via Twython and return a dataframe. This is defined in a function in
`fetch_twitter_data.py`. You'll need to populate the variables for Twitter API
app key and secret, plus a user token and secret at the top of that file in
order to fetch from Twitter. You can create an app
[here](https://apps.twitter.com/app/new). 

**In [1]:**

{% highlight python %}
%%time
from fetch_twitter_data import fetch_the_data
df = fetch_the_data()
{% endhighlight %}

    got 92 posts from page 1...
    got 90 posts from page 2...
    got 89 posts from page 3...
    got 91 posts from page 4...
    got 87 posts from page 5...
    got 88 posts from page 6...
    got 95 posts from page 7...
    got 93 posts from page 8...
    got 86 posts from page 9...
    got 90 posts from page 10...
    got all pages - 901 posts in total
    CPU times: user 412 ms, sys: 193 ms, total: 605 ms
    Wall time: 6.17 s

 
If you're following along at home, a nonzero amount of those 1000 tweets will be
unable to be fetched. That happens when the posts are deleted by the users. All
we can do is fondly remember them and carry on. ¯\\_(ツ)_/¯

We're going to take a quick peek at the head of the data and move on to building
the model, since the emphasis of this post is on building a robust classifier
pipeline. You're a great data scientist, though, so you already know the
importance of getting really friendly with the contents of the data, right? 

**In [2]:**

{% highlight python %}
df.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>sentiment</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>750068048607010816</td>
      <td>negative</td>
      <td>When you're drinking a beer and you get to tha...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>773620706441474048</td>
      <td>neutral</td>
      <td>Yupppp https://t.co/mpzJ6yGI0r</td>
    </tr>
    <tr>
      <th>2</th>
      <td>778301474816294912</td>
      <td>negative</td>
      <td>@JetBlue three places just took off! Let us go...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>750079494560550912</td>
      <td>positive</td>
      <td>Enjoying a cold #beer @gbbrewingco @ATLairport...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>776834202205552640</td>
      <td>neutral</td>
      <td>@SuperNovs1 @zjlaing not sure how airlines han...</td>
    </tr>
  </tbody>
</table>
</div>


 
# Building a basic pipeline 
 
A quick note before we begin: I'm taking liberties with assessing the
performance of the model, and we're in danger of overfitting the model to the
testing data. We're looking at the results of a particular set to show how the
model changes with new features, preprocessing, and hyperparameters. This is for
illustratory purposes only. Always remember to practice safe model evaluation,
using proper cross-validation.

Okay, time for some fun - we're going to make our first pipeline that we'll
continue to build upon.

## Train-test split

First, we'll split up the data between training and testing sets. 

**In [3]:**

{% highlight python %}
from sklearn.model_selection import train_test_split

X, y = df.text, df.sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
{% endhighlight %}
 
## Tokenizer

We're going to override sklearn's default tokenizer with NLTK's TweetTokenizer.
This has the benefit of tokenizing hashtags and emoticons correctly, and
shortening repeated characters (e.g., "stop ittttttttttt" -> "stop ittt") 

**In [4]:**

{% highlight python %}
import nltk
tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True) # Your milage may vary on these arguments
{% endhighlight %}
 
## Pipeline steps
Now let's initialize our [Count Vectorizer](http://scikit-learn.org/stable/modul
es/generated/sklearn.feature_extraction.text.CountVectorizer.html) and our
[Logistic Regression](http://scikit-
learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
for classification. Many models tend to use Naive Bayes approaches for text
classification problems. However, I'm not going to for this example for a few
reasons:

1. Later on, we're going to be adding continuous features to the pipeline, which
is difficult to do with scikit-learn's implementation of NB. For example,
Gaussian NB (the flavor which produces best results most of the time from
continuous variables) requires dense matrices, but the output of a
`CountVectorizer` is sparse. It's more work, and from my tests LR will still
outperform even on this small of a dataset.
1. We're thinking big, creating a pipeline that will still be useful on a much
bigger dataset than we have in front of us right now. While Naive Bayes tends to
converge quicker (albeit with a higher error rate than Logistic Regression) on
smaller datasets, LR should outperform NB in the long run with more data to
learn from.
1. Finally, and probably most importantly, I've simply observed LR working
better than NB on this kind of text data in actual production use cases. Data
sparsity is likely our enemy here. Twitter data is short-form, so each example
will have few "rows" to look up from our vectorized vocabulary, and our model
may not see the same unigram/phrase enough times to really get its head around
the true class probabilities for every word. sklearn's Logistic Regression's
built-in regularization will handle these kinds of cases better than NB (again,
based on my experience in this specific domain of social media). The experiences
I've seen with LR vs. NB on social media posts probably warrant their own post.

For more information, here's a nice book excerpt on Naive Bayes vs. Logistic
Regression [here](http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf), a quick
article on how to pick the right classifier
[here](http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/),
and a great walkthrough of how logistic regression works [here](http://nbviewer.
jupyter.org/github/justmarkham/DAT8/blob/master/notebooks/12_logistic_regression.ipynb
). 

**In [5]:**

{% highlight python %}
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

count_vect = CountVectorizer(tokenizer=tokenizer.tokenize) 
classifier = LogisticRegression()
{% endhighlight %}
 
And now to put them together in a pipeline... 

**In [6]:**

{% highlight python %}
sentiment_pipeline = Pipeline([
        ('vectorizer', count_vect),
        ('classifier', classifier)
    ])
{% endhighlight %}
 
Alright, we've got a pipeline. Let's unleash this beast all over the training
data!

We'll import a helper function to make training and testing a more pleasant
experience. This function trains, predicts on test data, checks accuracy score
against null accuracy, and displays a confusion matrix and classification
report. 

**In [7]:**

{% highlight python %}
from sklearn_helpers import train_test_and_evaluate

sentiment_pipeline, confusion_matrix = train_test_and_evaluate(sentiment_pipeline, X_train, y_train, X_test, y_test)
{% endhighlight %}

    null accuracy: 47.35%
    accuracy score: 61.95%
    model is 14.60% more accurate than null accuracy
    ---------------------------------------------------------------------------
    Confusion Matrix
    
    Predicted  negative  neutral  positive  __all__
    Actual                                         
    negative         23       19        15       57
    neutral           6       45        11       62
    positive         10       25        72      107
    __all__          39       89        98      226
    ---------------------------------------------------------------------------
    Classification Report
    
                    precision    recall  F1_score support
    Classes                                              
    negative         0.589744  0.403509  0.479167      57
    neutral          0.505618  0.725806  0.596026      62
    positive         0.734694  0.672897  0.702439     107
    __avg / total__  0.635292  0.619469  0.616934     226

 
# All done

That wasn't so bad, was it? Pipelines are great, as you can easily keep adding
steps and tweaking the ones you have. In the [next lesson](../sentiment-pipeline-
sklearn-3) we're going to define a custom preprocessing function and add
it as a step in the model. [Hit it!](../sentiment-pipeline-sklearn-3)

*This is Part 2 of 5 in a series on building a sentiment analysis pipeline using
scikit-learn. You can find Part 3 [here](../sentiment-pipeline-sklearn-3),
and the introduction [here](../sentiment-pipeline-sklearn-1).*

*Jump to:*

* ***[Part 1 - Introduction and requirements](../sentiment-pipeline-
sklearn-1)***
* ***[Part 3 - Adding a custom function to a pipeline](../sentiment-pipeline-
sklearn-3)***
* ***[Part 4 - Adding a custom feature to a pipeline with FeatureUnion
](../sentiment-pipeline-sklearn-4)***
* ***[Part 5 - Hyperparameter tuning in pipelines with GridSearchCV
](../sentiment-pipeline-sklearn-5)*** 
