---
layout: page
title: About Me
permalink: /about/
---
Hello and welcome! I'm Ryan Cranfill, a Chicago-based data person interested in data mining, exploration, analysis, visualization, and modeling. I've worked with a lot of different flavors of data across a lot of different industries, most currently dealing with social media data and NLP. I'm always looking for a new challenge, and love to learn.

# **Work Experience**

---

### Data Scientist at [Earshot](http://www.earshotinc.com)

#### *2016 - Present*

I currently work as a Data Scientist for Earshot, a real-time social media marketing startup in Chicago. Our software enables brands to have positive conversations with the right consumers.

As a Data Scientist, I'm responsible for identifying opportunities to use the vast amounts of data we have in impactful ways, then implement solutions. My work is end-to-end, including: defining requirements, exploring prior art, gathering data and analyzing it, building and evaluating a model, and then putting that model in production.

#### Accomplishments include:

- Developed a new sentiment analysis engine tailored for interpreting social media sentiment.
    - Uses [Stanford GLoVE](http://nlp.stanford.edu/projects/glove/) word vectors in conjunction with word vectors trained from 12M+ social media posts. Word vectors are  combined with metafeatures such as ratio of capitalization and number of repeated characters. Prediction is handled by a neural network with 2 hidden layers.
    - Built using Gensim and Keras and productionized as AWS Lambda functions ([one for feature extraction](../gensim-aws-lambda), [one for prediction](../keras-aws-lambda)). The function evaluates the sentiment for 60M+ posts monthly costing less than $50 per month.
    - Evaluated against a collection of posts that mentioned airlines (one of our important verticals), the new Earshot model had an overall recall of 77% – better than IBM Watson AlchemyLanguage (69%) and TextBlob (53%) on the same dataset. Recall on negative posts was 91% and precision on positive posts was 94%, with overall precision over 12% higher than the closest off-the-shelf competitor.

- Created a pure Python implementation of a probabilistic hashtag segmentation model. 
    - The model uses unigrams and bigrams to calculate the most probable combination of words within a string - so "#CoolStuff2016" turns into `['cool', 'stuff', '2016']`, or "I love #coolstuff" becomes "I love cool stuff". 
    - Evaluated against 750 of the most frequent hashtags that come into Earshot, the model outputs a correct split over 90% of the time.

- Made improvements to Earshot's individual vs. non-individual model.
    - This model analyzes the content of a post as well as the content of the author's bio and other metadata and predicts if it is from an individual person or a non-individual (such as a restaurant, brand, or event).
    - The model is built with Python, using the scikit-learn and NLTK packages.
    - Improved existing model overall accuracy by 1.5% up to 95%, with gains in non-individual precision of 12% and recall of 8%. Crucially, these gains did not come at the expense of performance on individuals, and in fact individual performance gained modestly as well.
    - Gains were found mostly via feature engineering and classifier hyperparameter tuning.

### Product Data Analyst at [Earshot](http://www.earshotinc.com)

#### *2015 - 2016*

Prior to being promoted to Data Scientist, I was Earshot's Product Data Analyst. This was a wide-ranging role in which I was responsible for setting up new campaigns in our Django admin backend, doing data pulls and reports as needed, performing analyses for customers and internal stakeholders, and myriad other ad hoc tasks as necessary.

Day to day, I got my hands dirty with Django shells and management commands, queries to Redshift and Elasticsearch, and data exploration and visualization in Jupyter notebooks.

#### Accomplishments include: 

- Developed a report in Python and Django to retrieve and analyze global campaign volume to give us an idea at a glance where the biggest amounts of posts are coming from. The report runs hourly on a cron job and is hosted on an AWS EC2 server.
- Quality-of-life enhancements for myself and other users of the Django admin interface, such as commands to copy entire campaigns, data upload via CSV, and autocomplete on fields with many values instead of dropdown lists. These enhancements reduced the amount of time spent creating and managing a typical campaign by 80% or more.
- Provided UX advice and thoroughly beta tested the lovely client-facing campaign creation UI implemented by Earshot's talented engineers. Being in the trenches with campaign setup allowed me to strongly speak to the needs and wants of our customers.

### Data Analyst (Contract) at [BeerMenus](http://www.beermenus.com)

#### *2015*

In between MarkITx and Earshot, I had the pleasure of working with the fine people at BeerMenus to analyze and provide insights into user behavior on their newly launched iOS app. I helped to visualize growth, look at the distribution of ratings, and try to understand where the ratings were coming from.

To perform the analysis, I used PostgreSQL, the scientific Python stack (NumPy, Pandas, Matplotlib, Seaborn, scikit-learn), Jupyter notebooks, and good old-fashioned elbow grease.

### Pricing Data Analyst at [MarkITx](http://www.markitx.com)

#### *2014 - 2015*

Cleaned, analyzed, and priced lists of used and new enterprise IT hardware using publicly available and proprietary data analysis tools. Developed new and augmented existing tools to enhance and streamline pricing process. Created Market Intelligence Reports for customers to provide in-depth insights about the value of their IT assets. Resolved discrepancies between customer-sourced lists and audited reports. Created and administered Block Trades to trade equipment in competitive auctions that provided 3x-5x value uplift over traditional disposition methods.

#### Accomplishments include:

- Created infrastructure using Python to sync pricing data across Redshift, Google Sheets, S3, and ElasticSearch for large partnership initiative with industry-leading VAR; data pipeline was fully operational within 3 weeks.
- Assisted with conception and creation of Market Intelligence Reports, creating a consistent framework with which to report pricing analysis and adding a valuable tool to bolster company's brand.


### Trade Development Analyst at [Breakthru Beverage Group (née Wirtz Beverage Inc.)](http://www.breakthrubev.com/)

#### *2013 - 2014*

Created and updated sales intelligence reports to assist the Regal Division's Trade Development team. Distributed reports and analyses to sales team in a timely manner to assist in achieving long and short term goals. Developed Excel VBA macros to expedite analysis procedures. 

#### Accomplishments include:

- Developed several Excel VBA macros to automate process of updating frequently revised reports, resulting in a greater than 75% time reduction for these tasks.
- Provided analytical support to augment sales teams' efforts on division's largest ever product launch.

### Surface Transportation Consultant at [Harris Miller Miller & Hanson](http://www.hmmh.com)

#### *2012 - 2013*

Performed transportation noise analyses for government and private clients. I was primarily focused on highway noise. I participated in field measurement programs, modeling, impact and barrier analysis, and reporting.

#### Accomplishments include:

- Joined the company with the complex I-64 Hampton Roads Bridge Tunnel Project in progress. Quickly learned use of ArcGIS, TNM, and analysis spreadsheets to assist in supplying a quality noise impact assessment to the client despite an aggressive delivery schedule.
- Took on all aspects of modeling and analysis for Odd Fellows Road Interchange Project as well as providing reporting support to ensure a superior product was given to the client despite challenging deadlines.
- Created a number of tools over the course of project work to streamline analysis procedures. Examples include automatic processing of ENTRADA traffic sheets for loudest hour calculation and TNM Automated Model Builder (TAMB), a program to automatically build Traffic Noise Model runs from objects in Excel files.

# **Education**

---

### Bachelor of Science, Acoustics from [Columbia College Chicago](http://www.colum.edu)

#### *2008 - 2012*

I studied Acoustics (the science of sound, not acoustic guitars) and math at Columbia College Chicago. 

My junior and senior years I was a Teaching and Research assistant, where I provided demonstrations and instruction on the use of our [Sound Transmission Class](https://en.wikipedia.org/wiki/Sound_transmission_class) chamber. I programmed an Excel macro that worked with SpectraPLUS FFT to automagically perform tests that measure how well a given building material reduces sound passing through it.

Among other projects, I worked with another student to complete an acoustical assessment of a noisy theater on campus. We took measurements of the theater, created a 3D model to perform physical acoustics calculations, and provided recommendations for enhancing the intelligibility of performances.

# **Awards**

---

### Founders' Award for Excellence from [Harris Miller Miller & Hanson](http://www.hmmh.com)

#### *2013*

"In recognition of outstanding performance on a project that was uniquely challenging, technically innovated, and resulted in proven client satisfaction."

Won for project "[On-Board Sound Intensity](http://www.hmmh.com/on-board-sound-intensity.html) Measurements to Evaluate the Noise Reduction of Pavement Grinding, I-195, Providence, RI". 

### [Robert Bradford Newman Student Medal](http://www.newmanfund.org/) for Merit in Architectural Acoustics

#### *2012*

Won this award for the STC chamber work mentioned above.

# **Skills and Software I Use**

---

These days my weapon of choice is Python for doing data analysis, exploration, visualization, and modeling. I'm a big fan of its huge open source community, [ease of use](https://xkcd.com/353/), and ubiquity in the data science world. Some packages that I use frequently:

<style type="text/css">
.table {
    border-bottom:0px !important;
}
.table th, .table td {
    border: 1px !important;
    text-align:center;
    vertical-align:top;
}
.fixed-table-container {
    border:0px !important;
}
.table th {
    font-weight: bold;
    text-decoration: underline;
}
</style>

<table class="table">
  <tr>
    <th>Analysis/Modeling</th>
    <th>Plotting/Visualization</th>
    <th>NLP</th>
    <th>Plumbing</th>
    <th>Other Stuff</th>
  </tr>
  <tr>
    <td>Pandas</td>
    <td>Matplotlib</td>
    <td>NLTK</td>
    <td>Dask</td>
    <td>Jupyter</td>
  </tr>
  <tr>
    <td>NumPy</td>
    <td>Seaborn</td>
    <td>Textblob</td>
    <td>psycopg2</td>
    <td>Django</td>
  </tr>
  <tr>
    <td>scikit-learn</td>
    <td>Plot.ly</td>
    <td>Pattern</td>
    <td>Requests</td>
    <td>Beautifulsoup</td>
  </tr>
  <tr>
    <td>Gensim</td>
    <td>Folium</td>
    <td>spaCy</td>
    <td>Boto</td>
    <td>twython</td>
  </tr>
  <tr>
    <td>Keras</td>
    <td>Shapely</td>
    <td></td>
    <td>Elasticsearch</td>
    <td></td>
  </tr>
</table>


Other software that I use frequently:

- Git and Github
- ElasticSearch
- Postgres/SQL
- Ansible
- Bash

AWS services:

- S3
- EC2
- Lambda
- Redshift
- DynamoDB

# **Personal**

---

When I'm not science-ing data, you can find me cooking, brewing, playing frisbee, watching movies, reading books, and playing music (regardless of how bad it may sound).

---

### Contact me

[ryan.c.cranfill@gmail.com](mailto:ryan.c.cranfill@gmail.com)

