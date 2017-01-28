---
layout: post
title: "Serverless Word2vec Models On AWS Lambda With Gensim"
author: Ryan Cranfill
tags:
    - python
    - notebook
--- 
Does the idea of extracting document vectors for 55 million documents per month for less than $25 sound appealing to you? Me too. AWS Lambda is pretty radical. It uses real live magic to handle DevOps for people who don't want to handle DevOps. At [Earshot](www.earshotinc.com) we've been working with Lambda to productionize a number of models, most recently a sentiment analysis model using word vectors and a neural net. Today we're going to learn how to deploy your very own Gensim model on Lambda.

The instructions are pretty similar to [Ryan Brown's post here](https://serverlesscode.com/post/deploy-scikitlearn-on-lamba/). Many thanks to him for putting that out there (and also, nice name!).

# Set up an EC2 instance

First, launch an EC2 instance using the Amazon Linux AMI (we use `Amazon Linux AMI 2015.09.1 x86_64 HVM GP2`. Others may work but we haven't tested them). We use a c4.4xlarge instance since we're going to be compiling some stuff, but you can use whatever you want as long as it has more than 1GB of memory or has a swapfile that will get it there.

# Install system dependencies For Numpy/Gensim

SSH into that bad boy and install the dependencies:

`sudo yum install -y git python-pip python-devel python-nose python-setuptools gcc gcc-gfortran gcc-c++ blas-devel lapack-devel atlas-devel`

# Create virtual environment

Make the virtualenv that we're going to install the python packages into:

`virtualenv virtualenv --python="/usr/bin/python" --always-copy --no-site-packages`

And activate that virtualenv:

`source virtualenv/bin/activate`

# Upgrade Pip and Wheel

To smooth over any potential bumps we might get installing our dependencies, upgrade Pip and Wheel:

`pip install --upgrade pip wheel`

# Install dependencies:

Now we install the packages that we need:

`pip install --use-wheel numpy==1.6.1`

`pip install --use-wheel scipy==0.10.1`

`pip install --use-wheel gensim==0.13.3`

`pip install --use-wheel smart-open==1.3.5`

# Strip down binaries to save package weight

This will get us down to our goal package size, as Lambda is touchy about that kind of thing: 

`find virtualenv/ -name \"*.so\" | xargs strip` 

# Copy libraries to lib directory

We're going to make a directory where we copy the needed libraries for packaging everything up:

`mkdir build`

`cd build`

`mkdir lib`

`find /usr/lib64 -name "libblas.*" -exec cp -P {} lib/ \;`

`find /usr/lib64 -name "libgfortran.*" -exec cp -P {} lib/ \;`

`find /usr/lib64 -name "liblapack.*" -exec cp -P {} lib/ \;`

`find /usr/lib64 -name "libopenblas.*" -exec cp -P {} lib/ \;`

`find /usr/lib64 -name "libquadmath.*" -exec cp -P {} lib/ \;`

`find /usr/lib64 -name "libf77blas.*" -exec cp -P {} lib/ \;`

`find /usr/lib64 -name "libcblas.*" -exec cp -P {} lib/ \;`

`find /usr/lib64 -name "libatlas.*" -exec cp -P {} lib/ \;`

# Copy python depencies to build directory

We're going to copy the python packages from the site-packages folders where they live in our virtualenv to our build directory so they're included in our distribution. Starting with the 32-bit packages:

`cp -r ~/virtualenv/lib/python2.7/site-packages/boto .`

`cp -r ~/virtualenv/lib/python2.7/site-packages/requests .`

`cp -r ~/virtualenv/lib/python2.7/site-packages/smart_open .`

And the 64-bit packages:

`cp -r ~/virtualenv/lib64/python2.7/site-packages/* .`

# Add your handler code

I'll leave it up to you how you want to write the handler code. However, there's a couple things we have to do in order to make Gensim load a model properly. First, create your `handler.py` (or whatever you want to call it) in the build directory.

Next, inside the handler we're going to load the system libraries that we put into the `lib` directory before importing Gensim since those libraries are not available in the default `LD_LIBRARY_PATH`. This code is pretty much verbatim from [Ryan Brown's post](https://serverlesscode.com/post/deploy-scikitlearn-on-lamba/):

{% highlight python %}
import ctypes
import os

for d, _, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a') or f.endswith('.settings'):
            continue
        print('loading %s...' % f)
        ctypes.cdll.LoadLibrary(os.path.join(d, f))

import gensim
{% endhighlight %}

That gets us to the point where we can import Gensim. Now we have to give it a model. Unfortunately Lambda has a limit on how much space can be used inside the container, so for models that will be more than a few hundred MB (most of them, probably), downloading them to a temp directory is out of the question. Fortunately, the creators of Gensim are also the creators of the awesome Smart Open package which makes streaming a model from S3 a breeze. We'll pass the load command a Boto S3 key that Gensim will load.

Before you load the model, you'll need to put it on S3. With your model loaded, use `mymodel.save_word2vec_format('filename.bin', binary=True)` to ensure it's the correct file type. Copy it to the S3 bucket of your choice, and then add to `handler.py` below `import gensim`:

{% highlight python %}
import boto
from boto.s3.connection import OrdinaryCallingFormat

AWS_ACCESS_KEY_ID = 'my aws access key'
AWS_SECRET_ACCESS_KEY = 'my aws secret access key'
S3_BUCKET = 'my model S3 bucket'
S3_KEY = 'my model S3 key'

def download_gensim_model():
    s3_conn = boto.connect_s3(aws_access_key_id=AWS_ACCESS_KEY_ID,
                              aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                              calling_format=OrdinaryCallingFormat())
    bucket = s3_conn.get_bucket(S3_BUCKET)
    key = bucket.get_key(S3_KEY)
    model = gensim.models.Word2Vec.load_word2vec_format(key, binary=True)
    return model

model = download_gensim_model()
{% endhighlight %}

Et voila, you have a working model. Now just define your handler code to give back whatever you want. For our example here, we'll just send back a list of word vectors for each word in the request's text:

{% highlight python %}
def handler(event, context):
    text = event['text']
    split_text = text.split()
    vectors = [model[word].tolist() for word in split_text]

    return vectors
{% endhighlight %}

# Zip it and rip it

Now zip it up:

`zip -r -q gensim_dist.zip ./*`

Upload it to S3:

`aws s3 cp gensim_dist.zip s3://lambda-dist.mycompany.com/gensim_dist.zip`

And go into the Lambda page on the AWS Console. Create a new function (be sure it has enough memory - for our purposes we used 832MB) and use a link to the zip file you uploaded as the code entry type. Configure the test function the way you want, and test it. The first initialization may take a couple dozen seconds, but after that it should fly and only take a few milliseconds per execution.

# Conclusion

So there you have it - Gensim working on AWS Lambda. Congratulate yourself for not having to manage scaling your model, and go worry about making the model better.

# Thanks to

Gordon Towne, Dylan Lingelbach, Nathaniel Meierpolys, and the rest of the crew
at Earshot helping keep me unstuck.

