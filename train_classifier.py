# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from string import punctuation

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
from sklearn.tree import DecisionTreeClassifier

def load_data(database_filepath):
    # load data from database
    db_connection = ''.join(['sqlite:///', database_filepath])
    engine = create_engine(db_connection)
    table_name = database_filepath.split('.')[0]
    df = pd.read_sql_table(table_name, engine)

    # prepare model data
    X = df.message.values
    Y_columns = list(set(df.columns) - {'id', 'message', 'original', 'genre'})
    Y = df[Y_columns].values
    return X, Y, Y_columns


def tokenize(text):
    # prepare url check
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_match = re.compile(url_regex).match
    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # prepare check for stop words and punctuation
    stop_words = stopwords.words('english')
    punctuation = list(punctuation)

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok \
                    in tokens if tok not in stop_words and tok not in punctuation]

    return clean_tokens


def build_model():
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    Y_pred = model.predict(X_test)

    classification_reports = []

    for i in range(0, len(category_names)):
        classification_reports.append(classification_report(Y_test[:, i], Y_pred[:, i]))

    return classification_reports


def save_model(model, model_filepath):
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

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()