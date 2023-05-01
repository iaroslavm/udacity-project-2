# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])

import re
import pandas as pd
from string import punctuation as punkt

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """Load messages from SQL database,
    separate data into dependent and explanatory variables

    Parameters:
        database_filepath (str): path to SQL database

    Returns:
        X: explanatory variable
        Y: dependent variables
        Y_columns: names of dependent variables
    """
    # load data from database
    db_connection = ''.join(['sqlite:///', database_filepath])
    engine = create_engine(db_connection)
    table_name = 'DisasterResponseTable'
    df = pd.read_sql_table(table_name, engine)

    # prepare model data
    X = df.message.values
    Y_columns = df.columns[4:]
    Y = df[Y_columns].values
    return X, Y, Y_columns


def tokenize(text):
    """Tokenize function

    Parameters:
        text (str): text message to tokenize

    Returns:
        clean_tokens: clean tokens derived from the text message
    """
    # prepare url check
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # prepare check for stop words and punctuation
    stop_words = stopwords.words('english')
    punctuation = list(punkt)

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok \
                    in tokens if tok not in stop_words and tok not in punctuation]

    return clean_tokens


def build_model():
    """Build classification pipeline

    Returns:
        pipeline: model object
    """
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def use_grid_search(initial_pipiline):
    """Initialize grid search

    Returns:
        cv: grid search object
    """
    parameters = {
        'clf__estimator__n_estimators': list(10 * np.array([2, 3])),
        'clf__estimator__max_depth': list(10 * np.array([2, 3]))
    }

    cv = GridSearchCV(initial_pipiline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Use test data to estimate model

    Parameters:
        model (model object): classification pipeline
        X_test: test explanatory variable
        Y_test: test dependent variables
        category_names: names of message categories

    Returns:
        classification_reports: classification reports for each category
    """
    # predict on test data
    Y_pred = model.predict(X_test)

    classification_reports = []

    for i in range(0, len(category_names)):
        classification_reports.append(classification_report(Y_test[:, i], Y_pred[:, i]))

    return classification_reports


def save_model(model, model_filepath):
    """Saving estimated model into pickle

    Parameters:
        model (model object): estimate model
        model_filepath: destination of the pickle file

    """
    # export model with decision tree classifier as a pickle
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


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

        print('Initiating Grid Search to improve parameters...')
        cv = use_grid_search(model)

        print('Training new model...')
        cv.fit(X_train, Y_train)

        print('Evaluating grid search results...')
        evaluate_model(cv, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: ' \
              'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl')


if __name__ == '__main__':
    main()