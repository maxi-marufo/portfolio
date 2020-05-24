# Portfolio

This is a repository with some of the challenges, mini-projects, course
homeworks, tutorials, etc, that I've done.

Each one is contained in a sub-directory, and has a README file describing a bit
more about it.

## Projects

* [Sentiment Analysis Web App](https://github.com/maxi-marufo/portfolio/tree/master/Sentiment_Analysis_Web_App)
  This project uses nltk and PyTorch to build a Sentiment Analysis model.
  It trains the model locally, but then deploys it to SageMaker. It also
  creates a Web App, which uses AWS API Gateway to receive the POST
  requests, and AWS Lambda to do the NLP pre-processing.

* [Plagiarism Detection](https://github.com/maxi-marufo/portfolio/tree/master/Plagiarism_Detection)
  This project contains code and associated files for deploying a
  plagiarism detector using AWS SageMaker. It examines a text file and
  performs binary classification; labeling that file as either
  plagiarized or not, depending on how similar that text file is to a
  provided source text. The project is broken down into three main
  notebooks: Data Exploration, Feature Engineering and Train and Deploy
  in SageMaker

* [Movie Recommendations Engine](https://github.com/maxi-marufo/portfolio/blob/master/Movie_Recommendations_Engine)
  This project builds Knowledge Based, Content Based and Collaborative
  Filtering Based (both Neighborhood Based and Model Based)
  Recommendations Engines using the [MovieTweetings](http://crowdrec2013.noahlab.com.hk/papers/crowdrec2013_Dooms.pdf) dataset.

* [Disaster Response Pipeline Project](https://github.com/maxi-marufo/portfolio/tree/master/Disaster_Response_Pipeline_Project)
  In this project, we will analyze messages sent during disasters to
  build a model for an API that classifies disaster messages. It creates
  both a ETL (for NLP processing) and a ML pipeline (using scikit
  Pipelines, GridSearch, etc.) and then hosts the model in a Flask web
  app.

* [Bank Data](https://github.com/maxi-marufo/portfolio/tree/master/Bank_Data)
  In this project, we have data on 10.000 (fictitious) customers of a bank,
  and want to use the insights to improve the customer retention, and
  identify customers at risk of leaving the bank. Finally, we want to
  predict which of these customers we will be able to retain over the
  next 12 months.

* [AWS ETL and ML Pipeline](https://github.com/maxi-marufo/portfolio/tree/master/AWS_DS_ML_Pipeline)
  Notebooks for Data preparation, model development, ETLs, model training,
  inference pipelines and batch transformations, using AWS Glue, Athena
  and SageMaker servicies.

* [Fashion MNIST](https://github.com/maxi-marufo/portfolio/blob/master/Fashion_MNIST/notebook.ipynb)
  Uses the Fashion MNIST dataset from Zalando to build a CNN using Keras.

* [Kaggle Reuters](https://github.com/maxi-marufo/portfolio/blob/master/Kaggle_Reuters/Kaggle_Reuters_Challenge.ipynb)
  Uses NLP and ML to classify texts into categories. The domain is Topic
  modeling and focused on newspaper articles. It also does EDA.