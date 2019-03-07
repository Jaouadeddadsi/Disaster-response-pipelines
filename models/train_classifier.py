import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re

# sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.externals import joblib

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet', 'stopwords',
               'averaged_perceptron_tagger'])


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        if len(sentence_list) != 0:
            for sentence in sentence_list:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                if len(pos_tags) != 0:
                    first_word, first_tag = pos_tags[0]
                    if first_tag in ['VB', 'VBP', 'VBG'] or first_word == 'RT':
                        return True
            return False
        else:
            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


class LenMessageExtractor(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return len(tokenize(X))


def load_data(database_filepath):
    """Function to load data from database

    Args:
        database_filepath (str) database file path

    return:
        X (numpy array) feature variables
        Y (numpy array) target variables
        category_names (list) category names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('Messages', con=engine)
    X = df.message.values
    df.drop(labels=['id', 'message', 'original', 'genre'], axis=1,
            inplace=True)
    category_names = df.columns
    Y = df.values

    return X, Y, category_names


def tokenize(text):
    """Function to

    Args:


    return:

    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words]

    return tokens


def build_model():
    """Function to

    Args:


    return:

    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor()),

            ('length_text', StartingVerbExtractor())

        ])),

        ('clf', MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=10, random_state=42)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Function to

    Args:


    return:

    """

    # predict
    Y_pred = model.predict(X_test)
    # dict
    dic_metrics = {'precision': [], 'recall': [], 'f1-score': []}
    for column in range(Y_pred.shape[1]):
        metrics = classification_report(Y_test[:, column], Y_pred[:, column],
                                        output_dict=True)['micro avg']
        dic_metrics['precision'].append(metrics['precision'])
        dic_metrics['recall'].append(metrics['recall'])
        dic_metrics['f1-score'].append(metrics['f1-score'])

    # Stock metrics in a Dataframe
    df_metrics = pd.DataFrame(dic_metrics, index=category_names)

    print(df_metrics.mean())


def save_model(model, model_filepath):
    """ Function to

    Args:


    return:

    """

    _ = joblib.dump(model, model_filepath, compress=9)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
