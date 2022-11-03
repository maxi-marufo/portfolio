# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import joblib

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    '''
    Loads the database from the given filepath and process
    them as X, y and category_names.
    input:
        database_filepath: File path where sql database was saved.
    output:
        X: Training message List.
        Y: Training target.
        category_names: Categorical name for labeling.
    '''

    # Read the table as pandas dataframe
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('FigureEight', engine)

    # Split the dataframe into x and y
    X = df.message.values
    Y = df[df.columns[4:]].values

    # Get the label names
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenizes and and lemmatizes the text messages.
    input:
        text: Message data for tokenization.
    output:
        clean_tokens: Result list after tokenization.
    '''

    # Tokenize the string text and initiate the lemmatizer
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize each word in tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip(),
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''
    Builds a model, create pipeline, hypertuning as well as gridsearchcv.
    input:
        None
    output:
        cv: GridSearch model result.
    '''

    # Create a pipeline
    #pipeline = Pipeline([
    #    ('vect', CountVectorizer(tokenizer=tokenize)),
    #    ('tfidf', TfidfTransformer()),
    #    ('clf', MultiOutputClassifier(
    #        RandomForestClassifier(class_weight='balanced', random_state=0)
    #        ))
    #])
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            OneVsRestClassifier(LinearSVC(random_state=0))
            ))
    ])

    # Find the optimal model using GridSearchCV
    #parameters = {
    #            'tfidf__smooth_idf': [True, False],
    #            'clf__estimator__n_estimators': [20, 100],
    #            'clf__estimator__max_depth': [2, 10]
    #         }
    parameters = {
                'tfidf__smooth_idf':[True, False],
                'clf__estimator__estimator__C': [1, 2, 5]
             }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates a model.
    input:
        model: GridSearch model
        X_test: Test set inputs 
        Y_test: Test set outputs
        category_names: Names from the different categories / targets
    output:
        None
    '''
    # Predict the given X_test
    Y_pred = model.predict(X_test)

    # Create the report based on the Y_pred
    for idx, category in enumerate(category_names):
        print(category)
        print(classification_report(Y_test[:,idx], Y_pred[:,idx]))
    print('---------------------------------')
    for i in range(Y_test.shape[1]):
        print(
            '%25s accuracy : %.2f' %(
                category_names[i], accuracy_score(Y_test[:,i], Y_pred[:,i])
                )
                )


def save_model(model, model_filepath):
    '''
    Saves the model.
    input:
        model: GridSearch model
        model_filepath: Path to save the model file
    output:
        None
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()