# Disaster Response Pipeline Project

## Project Motivation

In this project, we will analyze disaster data from [Figure Eight](https://www.linkedin.com/company/figureeight/) to build a model for an API that classifies disaster messages.

In `data/`, you'll find a data set containing real messages that were
sent during disaster events. We will be creating a machine learning
pipeline to categorize these events so that we can send the messages to
an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new
message and get classification results in several categories. The web
app also displays visualizations of the data.

## File Description

    .
    ├── app     
    │   ├── run.py                           # Flask file that runs app
    │   └── templates   
    │       ├── go.html                      # Classification result page of web app
    │       └── master.html                  # Main page of web app    
    ├── data                   
    │   ├── disaster_categories.csv          # Dataset including all the categories  
    │   ├── disaster_messages.csv            # Dataset including all the messages
    │   └── process_data.py                  # Data cleaning
    ├── models
    │   └── train_classifier.py              # Train ML model
    ├── notebooks
    │   └── ETL_Pipeline_Preparation.ipynb   # Jupyter notebook to develop the ETL Pipeline
    │   └── ML_Pipeline_Preparation.ipynb    # Jupyter notebook to develop the ML Pipeline     
    └── README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![ScreenShot](img.png)

## Example
Type in: We are more than 50 people sleeping on the street. Please help us find tent, food.

![Example](Example.png)