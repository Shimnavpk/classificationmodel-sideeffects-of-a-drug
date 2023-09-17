import os
import sys
import joblib
import warnings
warnings.filterwarnings( 'ignore' )
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Need to install wget, nltk for word embeddings

class NaNFiller(BaseEstimator, TransformerMixin):
    def __init__(self, features='Drug', substitute=''):
        self.features = features
        self.substitute = substitute

    def transform(self, X):
        X[self.features] = X[self.features].fillna(self.substitute)

        return X

class StandardScaler_(BaseEstimator, TransformerMixin):
    def __init__(self, features=[
                'EaseofUse', 'Satisfaction',
                #'day', 'month', 'year' # TODO: not required by UI 
                ]):
        self.features = features
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        self.scaler.fit(X[self.features])
        return self

    def transform(self, X):
        X[self.features] = self.scaler.transform(X[self.features])

        return X

class DatePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        X['Date'] = pd.to_datetime(X['Date'], format='%m/%d/%Y')
        X['day'] = X['Date'].dt.day
        X['month'] = X['Date'].dt.month
        X['year'] = X['Date'].dt.year
        X['is_weekend'] = X['Date'].dt.dayofweek.isin([5, 6])
        X.drop('Date', inplace=True, axis=1)

        return X

class FeatureHashPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=40,
                features=[
                    'Condition',
                    #'Drug', 'DrugId' # TODO: not required by UI 
                ]):
        self.fh = FeatureHasher(n_features=n_features, input_type='string')
        self.features = features

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.reset_index()
        for feature in self.features:
            hashed = self.fh.transform(X[feature].apply(str))
            hashed_df = pd.DataFrame(hashed.toarray(), columns=[f'{feature.lower()}_{i}' for i in range(40)])
            X = pd.concat((X, hashed_df), axis=1)
            X.drop(feature, inplace=True, axis=1)

        return X

class CategoryPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['Age', 'Sex']): # TODO: Age should be ordinal encoded.
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.columns = columns
    
    def fit(self, X, y):
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X):
        ohe = self.encoder.transform(X[self.columns]).toarray()
        X[[f'ohe_{i}' for i in range(15)]] = ohe
        X.drop(self.columns, inplace=True, axis=1)

        return X
    
if __name__ == "__main__":
    df_drug = pd.read_csv('webmd.csv')

    X = df_drug.drop(['Effectiveness'], axis=1)
    y = df_drug['Effectiveness']

    ignore_text = True # Set to False to embed words

    if ignore_text:
        X.drop(['Sides', 'Reviews'], axis=1, inplace=True)
    else:
        try:
            import wget
            import zipfile
            import nltk
            nltk.download('punkt')

            from nltk.tokenize import word_tokenize

            def bar_progress(current, total, width=80):
                progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
                sys.stdout.write("\r" + progress_message)
                sys.stdout.flush()

            save_path = "./glove"
            if not os.path.exists(f"{save_path}.zip"):
                print("Fetching GloVe model...")
                wget.download("https://nlp.stanford.edu/data/glove.6B.zip", f"{save_path}.zip", bar=bar_progress)
                print("\nGloVe model downloaded successfully!")

                with zipfile.ZipFile(f"{save_path}.zip", 'r') as zip_ref:
                    zip_ref.extractall(save_path)
            else:
                print("GloVe model already exists.")
            
            embeddings_dict = {}
            with open("./glove/glove.6B.50d.txt", 'r', encoding="utf8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word] = vector

            mx_tokens = 10

            def filter_words(sentence):
                embedding = []
                try:
                    tokens = word_tokenize(sentence)
                except: # nan
                    return np.zeros(50)
                for token in tokens:
                    if token.lower() in embeddings_dict:
                        embedding.append(embeddings_dict[token.lower()])
                if embedding:
                    return np.array(embedding).mean(axis=0)
                return np.zeros(50)

            print('Creating word embeddings. This might take a while.')
            sides_embed = X['Sides'].apply(filter_words).apply(pd.Series)
            sides_embed.columns = [f'sides_embed_{i}' for i in sides_embed.columns]
            reviews_embed = X['Reviews'].apply(filter_words).apply(pd.Series)
            reviews_embed.columns = [f'reviews_embed_{i}' for i in reviews_embed.columns]

            X = pd.concat((X, sides_embed), axis=1)
            X = pd.concat((X, reviews_embed), axis=1)
        except ImportError:
            pass

        X.drop(['Sides', 'Reviews'], axis=1, inplace=True)

    numerical_columns = ['EaseofUse', 'Satisfaction']
    required_columns = ['EaseofUse', 'Satisfaction', 'Age', 'Condition', 'Sex']

    X = X[required_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    model_exists = False

    try:
        pipeline = joblib.load("pipeline.pkl")
        model_exists = True
    except FileNotFoundError:
        model = GradientBoostingClassifier()
        category_processor = CategoryPreprocessor()
        scaler = StandardScaler_()

        pipeline = Pipeline([
            # ('nan_filler', NaNFiller()), # TODO: ideally we need this
            # ('date_preprocessor', DatePreprocessor()), # # TODO: not required by UI 
            ('feature_hash_preprocessor', FeatureHashPreprocessor()),
            ('category_preprocessor', category_processor),
            ('standard_scaler', scaler),
            ('classifier', model)
        ])

    if not model_exists:
        print("Started training.")
        pipeline.fit(X_train, y_train)
        print("Finished training.")
        
        joblib.dump(Pipeline, open('pipeline.pkl'))

    y_pred = pipeline.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred, average=None)}")
    print(f"Precision: {precision_score(y_test, y_pred, average=None)}")
    print(f"F-1 Score: {f1_score(y_test, y_pred, average=None)}")
