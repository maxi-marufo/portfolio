# **Data Science Project: Bank dataset**

In this dataset, we have data on 10.000 (fictitious) customers of a
bank, and want to use the insights to improve the customer retention,
and identify customers at risk of leaving the bank. Finally, we want
to predict which of these customers we will be able to retain over the
next 12 months.

## **Business Questions**
1. Which of the variables have more importance for customer retention?
2. How are these variables related to each other?
3. Which are the top k customers at highest risk of leaving the bank?

## Blog

You can find a post in Medium describing the results of this project [here](https://medium.com/@maximiliano.marufo/this-is-what-a-data-scientist-can-do-with-your-data-588207deb0db)

## Project Overview

The project consist of a [data.csv](https://github.com/maxi-marufo/challenges/blob/master/Bank_Data/data.csv)
file, which is our dataset, and a [notebook](https://github.com/maxi-marufo/challenges/blob/master/Bank_Data/notebook.ipynb) where we can run the code and see the results.

If you don't want to see the results but run it as a script, you can use
the [notebook.py](https://github.com/maxi-marufo/challenges/blob/master/Bank_Data/notebook.py)
script.

The libraries dependencies are listed in [requirements.txt](https://github.com/maxi-marufo/challenges/blob/master/Bank_Data/requirements.txt)

You can also run the notebook inside Docker. To build the image, run:

`docker build --tag=bank_data .`

And then run the container:

`docker run -p 8989:8989 bank_data`
