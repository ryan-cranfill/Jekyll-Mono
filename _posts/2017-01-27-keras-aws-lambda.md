---
layout: post
title: "Serving Deep Learning Models On AWS Lambda With Keras"
author: Ryan Cranfill
tags:
    - python
    - notebook
--- 
Truly, we are living in the future. At [Earshot](www.earshotinc.com) we've been recently developing Deep Learning models using [Keras](https://keras.io/), which has an awesome high-level API that sits on top of Tensorflow or Theano to enable rapid model development. As espoused in [my previous post](gensim-aws-lambda), we're fans of AWS Lambda as a way to serve up machine learning models. So let's make a good thing better and serve a Keras model on Lambda. 

The instructions are going to be pretty similar to [Ryan Brown's post here](https://serverlesscode.com/post/deploy-scikitlearn-on-lamba/) and [my previous post on productionizing Gensim](gensim-aws-lambda). Again, thanks to Ryan helping to blaze the trail here.

# Set up an EC2 instance

First, launch an EC2 instance using the Amazon Linux AMI (we use `Amazon Linux AMI 2015.09.1 x86_64 HVM GP2`. Others may work but we haven't tested them). We use a c4.4xlarge instance since we're going to be compiling some stuff, but you can use whatever you want as long as it has more than 1GB of memory or has a swapfile that will get it there.

# Install system dependencies For Numpy/Keras

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

`pip install --use-wheel Theano==0.8.2`

`pip install --use-wheel Keras==1.2.0`

`pip install --use-wheel h5py==2.6.0`

# Strip down binaries to save package weight

This will get us down to our goal package size, as Lambda is touchy about that kind of thing: 

`find virtualenv/ -name "*.so" | xargs strip` 

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

`cp -r ~/virtualenv/lib/python2.7/site-packages/theano .`

`cp -r ~/virtualenv/lib/python2.7/site-packages/pkg_resources .`

`cp -r ~/virtualenv/lib/python2.7/site-packages/keras .`

And the 64-bit packages:

`cp -r ~/virtualenv/lib64/python2.7/site-packages/* .`

# Add your handler code

I'll leave it up to you what you want to do inside the handler code. However, there's a couple things we have to do in order to make Keras load a model properly. First, create your `handler.py` (or whatever you want to call it) in the build directory.

Next, inside the handler we're going to load the system libraries that we put into the `lib` directory before importing Keras since those libraries are not available in the default `LD_LIBRARY_PATH`. This code is pretty much verbatim from [Ryan Brown's post](https://serverlesscode.com/post/deploy-scikitlearn-on-lamba/):

{% highlight python %}
import ctypes
import os

for d, _, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a') or f.endswith('.settings'):
            continue
        print('loading %s...' % f)
        ctypes.cdll.LoadLibrary(os.path.join(d, f))

import keras
{% endhighlight %}

That gets us to the point where we can import Keras. Unlike Gensim, Keras models are typically small enough that they should be able to be distributed inside the zip holding your Lambda code. Save your model that's been trained using `model.save('mymodel.h5')`, then copy that file into a directory inside your `build` directory. For this example, we're going with `models/model.h5`

Now add the code to your handler below importing keras:

{% highlight python %}
from numpy import array
model = keras.models.load_model('models/model.h5')
{% endhighlight %}

And now you have a working model! Now just define your handler code to give back whatever you want. For our example here, we're expecting a feature vector in the form of a list. Keras doesn't always play nicely with lists, so we'll convert it to an array before predicting:

{% highlight python %}
def handler(event, context):
    data = event['features']
    data = array([data])
    prediction = model.predict_classes(data)[0]
    return {'predicted_class': prediction}
{% endhighlight %}

# Zip it and rip it

Now zip it up:

`zip -r -q keras_dist.zip ./*`

Upload it to S3:

`aws s3 cp keras_dist.zip s3://lambda-dist.mycompany.com/keras_dist.zip`

And go into the Lambda page on the AWS Console. Create a new function (this one will need not a ton of memory - the default 128MB was more than enough for us) and use a link to the zip file you uploaded as the code entry type.

### IMPORTANT!

There are a couple of environment variables that you must set in order for Keras to run. First, set `THEANO_FLAGS` as `base_compiledir=/tmp/.theano`. The default value will make it want to compile in a directory that doesn't exist in the Lambda container, and as such it will be cranky.

The second will be to set `KERAS_BACKEND` as `theano`. The default there is to go with TensorFlow, but since we're using Theano since it is slimmer, we need to tell Keras to use it.

Configure the test function the way you want, and test it. The first initialization may take a couple seconds, but after that it should be much faster per prediction.

# Conclusion

And there it is - Keras working on AWS Lambda. Now go build and scale some more models!

# Thanks to

Gordon Towne, Dylan Lingelbach, Nathaniel Meierpolys, and the rest of the crew
at Earshot.

