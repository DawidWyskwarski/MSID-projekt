import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def remove_nans(df):
    df['artist_name'] = df['artist_name'].replace('empty_field', np.nan)
    df['tempo'] = df['tempo'].replace('?', np.nan)

    df.dropna(inplace=True)

    df['duration_ms'] = df['duration_ms'].replace(-1, np.nan)

    imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)

    df['duration_ms'] = imputer.fit_transform(df[['duration_ms']])

def format_df(df):
    #get rid of nans
    remove_nans(df)

    #Don't need it
    df.drop('obtained_date', inplace=True, axis=1)
    df.drop('instance_id', inplace=True, axis=1)

    #It was a string smh
    df['tempo'] = df['tempo'].apply(lambda x: float(x))

    #Encoding
    le_keys = LabelEncoder()
    df['key'] = le_keys.fit_transform(df['key'])

    mode_encoding = {
        'Minor': 0,
        'Major': 1
    }
    df['mode'] = df['mode'].map(mode_encoding)

    ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
    encoded_genres = ohe.fit_transform(df[['music_genre']])

    df = pd.concat([df, encoded_genres], axis=1).drop(columns=['music_genre'])

    return df

def music_genre_clean_up(df):
    df.drop('instance_id', inplace=True, axis=1)
    df.drop('artist_name', inplace=True, axis=1)
    df.drop('track_name', inplace=True, axis=1)
    df.drop('obtained_date', inplace=True, axis=1)

    df.dropna(inplace=True)

    df['tempo'] = df['tempo'].replace('?', np.nan)
    df['duration_ms'] = df['duration_ms'].replace(-1, np.nan)
    df['tempo'] = df['tempo'].apply(lambda x: float(x))