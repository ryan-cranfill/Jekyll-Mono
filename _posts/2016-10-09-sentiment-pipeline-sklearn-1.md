---
layout: post
title: "Building a Sentiment Analysis Pipeline in scikit-learn Part 1: Introduction and Requirements"
author: Ryan Cranfill
tags:
    - python
    - notebook
--- 
scikit-learn pipelines have been enormously helpful to me over the course of building a new sentiment analysis engine for [Earshot](http://www.earshotinc.com), so it's time to spread the good news. Read on for more information on what we'll cover in this series, what requirements you'll need, and more badly executed attempts to use the rule of three.

*This is Part 1 of 5 in a series on building a sentiment analysis pipeline using
scikit-learn. You can find Part 2 [here](../sentiment-pipeline-sklearn-2).*

*Jump to:*

* ***[Part 2 - Building a basic pipeline](../sentiment-pipeline-
sklearn-2)***
* ***[Part 3 - Adding a custom function to a pipeline](../sentiment-pipeline-
sklearn-3)***
* ***[Part 4 - Adding a custom feature to a pipeline with FeatureUnion](../sentiment-pipeline-sklearn-4)***
* ***[Part 5 - Hyperparameter tuning in pipelines with GridSearchCV](../sentiment-pipeline-sklearn-5)***

# Part 1 - Introduction and requirements

As a data scientist at [Earshot](http://www.earshotinc.com), I'm always looking
for ways to cut through the noise of social media and get to posts that are
interesting for our clients. One such signal that we use to determine relevance
is the sentiment of a post - is this user expressing positive or negative
feelings? If an influential person is talking unfavorably on social media about
you, you want to get out ahead of it quickly.

Previously, we were using various off-the-shelf packages and APIs to get
sentiment on our posts. We found their results to be somewhat dissatisfactory.
These third party sentiment solutions did not perform well on social media
posts. As we dug in to figure out why, some were trained on longer texts - Yelp
reviews, IMDB reviews, Amazon reviews, etc., some did not handle emojis at all,
and some were based on predefined dictionaries of specific adjectives that
couldn't take into account the way slang and the English language changes.
Because of this, we thought "maybe we can do better".

We could, and we did.

By creating a sentiment analysis engine that is attuned to the "unique" way
users express themselves on social media, we've created a solution that works
better for our customers.

How did we do it? With blood, sweat, and tears – and a little TLC. More
specifically, we used the magic of [scikit-learn pipelines](http://scikit-
learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to help
rapidly build, iterate, and productionize our model. They're unimpeachably
awesome, and if you're using Python for machine learning and not using these
things, your life is about to get much easier. Come with me as I show you how to
build a pipeline and add all sorts of fun* steps to it!

<sub>*Fun not guaranteed</sub>

# What are we going to cover?

* Basic scikit-learn pipeline building
* Adding custom functions as pipeline steps for text preprocessing
* Adding custom functions as additional features
* Using `GridSearchCV` to search for optimal parameters for each step


# What are we not going to cover?

* Machine learning basics
* Machine learning with text basics
* What is sentiment analysis?
* How to scrape Twitter data (though the code is available in the repo)
* Where babies come from

# Requirements

First things first - we have to make sure you have everything you need to do
this thing. You'll need the following packages for maximum enjoyment:

* [pandas](https://github.com/pydata/pandas)
* [NumPy](http://www.numpy.org/)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn) (>= 0.18)
* [Twython](https://github.com/ryanmcgrath/twython) (for getting Twitter data)
* [NTLK](http://www.nltk.org/)
* [pandas_confusion](https://github.com/pandas-ml/pandas_confusion) (for neato
confusion matrices)
* [Jupyter](http://jupyter.org/) (if you want to run the notebooks yourself)

# Code

There are several helper functions that we will refer to in `sklearn_helpers.py`
and `fetch_twitter_data.py`. If you want to follow along at home, clone the repo
from [GitHub](https://github.com/ryan-cranfill/sentiment-pipeline-sklearn) and
run the Jupyter notebooks found therein. Note - you will need to modify
`fetch_twitter_data.py` with your Twitter credentials in order to download the
data.

# Ready?
[Let's go to part 2!](../sentiment-pipeline-sklearn-2)

*This is Part 1 of 5 in a series on building a sentiment analysis pipeline using
scikit-learn. You can find Part 2 [here](../sentiment-pipeline-sklearn-2).*

*Jump to:*

* ***[Part 2 - Building a basic pipeline](../sentiment-pipeline-
sklearn-2)***
* ***[Part 3 - Adding a custom function to a pipeline](../sentiment-pipeline-
sklearn-3)***
* ***[Part 4 - Adding a custom feature to a pipeline with FeatureUnion](../sentiment-pipeline-sklearn-4)***
* ***[Part 5 - Hyperparameter tuning in pipelines with GridSearchCV](../sentiment-pipeline-sklearn-5)***
