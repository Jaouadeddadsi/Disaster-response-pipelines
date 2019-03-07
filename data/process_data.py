# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Function to load data from csv files

    Args:
        messages_filepath (str) messages csv file path
        categories_filepath (str) categories csv file path

    return:
        df (pandas Dataframe) Dataframe contains messages and categories data
    """

    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on=['id'])

    return df


def clean_data(df):
    """Function to clean df data

    Args:
        df (pandas Dataframe)

    return:
        df (pandas Dataframe)
    """

    # split categories into separate category columns
    categories = df.categories.str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column],
                                           downcast='integer')
    # replace categories column in df
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Function to save df in sqlite database

    Args:
        df (pandas Dataframe)
        database_filename (str) database name

    return:
        None
    """

    engine = create_engine('sqlite:///'+database_filename)
    # Split df to multiple dataframe
    list_df = []
    i = 0
    while i < df.shape[0]:
        list_df.append(df.iloc[i:i+1000, :])
        i += 1000
    for data in list_df:
        data.to_sql(name='Messages', con=engine, index=False,
                    if_exists='append')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath,\
            database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
