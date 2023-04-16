import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # calculate the frequency of the categories
    categories_frequency = sum(df[df.columns[5:]].values)/len(df)*100
    category_frequency_dict = dict(zip(df.columns[5:], categories_frequency))
    categories_list = list(category_frequency_dict.keys())
    freq_list = list(category_frequency_dict.values())
    most_frequent_category = list(category_frequency_dict.keys())[0]

    # calculate what categories correlate the most with most frequent category
    corr_matrix = df[df.columns[6:]].corr()
    # remove columns that are not correlated with others
    corr_matrix = corr_matrix[corr_matrix.sum() != 0][corr_matrix.columns[corr_matrix.sum() != 0]]
    corr_matrix = corr_matrix[most_frequent_category]
    corr_matrix = corr_matrix.drop(most_frequent_category)
    correlated_categories = list(corr_matrix.index)
    correlations = list(corr_matrix.values)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_list,
                    y=freq_list
                )
            ],

            'layout': {
                'title': 'Frequency of Categories, percent',
                'yaxis': {
                    'title': "Percent"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=correlated_categories,
                    y=correlations
                )
            ],

            'layout': {
                'title': ': '.join(['Category correlations with most frequent category', most_frequent_category]),
                'yaxis': {
                    'title': "Percent"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()