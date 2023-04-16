import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath).drop_duplicates()
    categories = pd.read_csv(categories_filepath).drop_duplicates()

    # merge datasets
    df = messages.merge(categories, how='inner', on='id').drop_duplicates()
    return df


def clean_data(df):

    # find names of categories
    categories_list = [cat.split('-')[0] for cat in df['categories'].values[0].split(';')]
    # expand categories into separate columns
    df[categories_list] = df['categories'].str.split(';', expand=True)
    # keep only boolean values in categories
    for column in categories_list:
        # set each value to be the last character of the string
        df[column] = df[column].str.split('-', expand=True).iloc[:, 1]

        # convert column from string to numeric
        df[column] = df[column].astype(int)

    # drop initial categories column and remove duplicates
    df = df.drop(['categories'], axis=1).drop_duplicates()
    return df


def save_data(df, database_filename):
    # create sql engine
    db_connection = ''.join(['sqlite:///', database_filename])
    table_name = database_filename.split('.')[0]
    engine = create_engine(db_connection)
    if not engine.has_table(table_name):
        df.to_sql(table_name, engine, index=False)
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()