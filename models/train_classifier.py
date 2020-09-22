# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

# import ssl

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# download nltk packages

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# import sklearn packages

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import warnings
warnings.simplefilter('ignore')


def load_data(database_filepath):
    """
    Load messages and categories data
    
    Arguments:
    database_filepath: path of the SQLite database
    
    Return:
    X: dataframe for the feature dataset
    Y: dataframe for the label dataset
    category_names: string list of category names 
    
    """
    
    # load data from SQL database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    
    # create features and labels data
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    # save category names
    category_names = list(Y.columns.values)
    
    return X, Y, category_names


def tokenize(text):
    
    """
    Tokenize the text function
    
    Arguments: 
    text: message which needs to be tokenized
    
    return:
    clean_tokens: tokens extracted from the provided text/message
    """
    
    # the word tokens from the provided message
    tokens = word_tokenize(text)

    # Lemmanitizer
    lemmatizer = WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = []
    for tok in tokens:

        # lemmatizer and lower
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()

        # append the cleaned tokens to the list
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    """
    Description: Build the machine learning model using TF-IDF pipeline
    
    Arguments: None
    
    Return: the ML GridSearchCV model which does the message classification
    """

    #model = Pipeline([
        #('features', FeatureUnion([

            #('text_pipeline', Pipeline([
                #('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                #('tfidf_transformer', TfidfTransformer())
            #])),

        #])),

        #('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    #])

    # the TF-IDF pipeline
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    
    # parameter set for searchCV
    parameters = {'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                  'clf__estimator__n_estimators': [50, 100],
                 }
    
    # Gridsearch parameters
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluate the ML pipeline accuracy
    
    Arguments:
    model: the ML GridSearchCV pipeline model
    X_test: Test features set
    Y_test: Test labels set
    category_names: label names
    
    """
    # predict on test data
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=Y_test.keys()))


def save_model(model, model_filepath):

    '''
    Save model as a pickle file 
    
    Arguments:
    model: Model to be saved
    model_filepath: path of the output pick file
    
    Return: none, but a pickle file can be saved for the model
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():

    """
    Machine Learning classifier main function. 
    The main function applies the Machine Learning Pipeline:
    - Load data from SQLite db
    - Train ML model on training dataset
    - Evaluate model performance on testing dataset
    - Save the model as pickle file
    """



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