import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from utils import remove_nans, format_df

TARGET = 'music_genre'

def summarize_numerical(numerical):
    num_summary: pd.DataFrame = numerical.describe(percentiles=[0.05, 0.95]).loc[
        ['mean', '50%', 'min', 'max', 'std', '5%', '95%']
    ]
    num_summary.loc['missing'] = numerical.isna().sum()
    num_summary = num_summary.reset_index().rename(columns={'index': 'statistic'})

    return num_summary

def summarize_categorical(categorical):
    cat_summary = pd.DataFrame(columns=categorical.columns)
    cat_summary.loc['missing'] = categorical.isna().sum()

    cat_summary.loc['unique classes'] = categorical.nunique()

    cat_summary = cat_summary.reset_index().rename(columns={'index': 'statistic'})

    #For the love of God don't uncomment it
    #Too many categories :O

    #for col in categorical.columns:
    #    proportions = categorical[col].value_counts(normalize=True)
    #    for idx, value in proportions.items():
    #        cat_summary.loc[f'{idx}_proportion', col] = value

    return cat_summary

def box_plots(ys:pd.DataFrame ,x, data_frame):
    for y in ys:
        plt.figure(figsize=(16, 5))
        sns.boxplot(x=x, y=y, data=data_frame)
        plt.title(f"{y} x {x}")

        plt.savefig(f"plots/boxplots/{y} x {x}.png")

        plt.show()

def violin_plots(ys:pd.DataFrame ,x, data_frame):
    for y in ys:
        plt.figure(figsize=(16, 5))
        sns.violinplot(x=x, y=y, data=data_frame)
        plt.title(f"{y} x {x}")

        plt.savefig(f"plots/violinplots/{y} x {x}.png")

        plt.show()

def error_bars(numerical:pd.DataFrame ,x, data_frame, eb_type):
    for col in numerical:
        plt.figure(figsize=(12, 4))
        plt.title(f"{eb_type} {col} x {x}")

        sns.pointplot(x=x, y=col, data=data_frame, errorbar=eb_type, linestyles='none')

        plt.savefig(f"plots/error_bars/{eb_type} {col} x {x}.png")

        plt.show()

def histograms(df: pd.DataFrame):
    df.hist(figsize=(12, 12), bins=25)
    plt.savefig(f"plots/histograms/numerical_histograms")

    plt.show()

    genres = df['music_genre'].unique()

    mode_palette = {
        'Minor': 'blue',
        'Major': 'orange'
    }

    keys_order = df['key'].unique()
    keys_order.sort()

    for genre in genres:
        subset = df[df['music_genre'] == genre]

        plt.figure(figsize=(12, 4))
        sns.countplot(data=subset, x='key', hue='mode', palette=mode_palette, order=keys_order)
        plt.title(genre)

        plt.savefig(f"plots/histograms/{genre} keys x modes")

        plt.show()


    plt.figure(figsize=(12, 4))
    sns.countplot(data=df, x='key', hue='mode', palette=mode_palette, order=keys_order)
    plt.title("Keys-mode hist")

    plt.savefig(f"plots/histograms/all keys x modes")

    plt.show()

def correlation_heatmap(df,name):
    plt.figure(figsize=(15, 15))
    sns.heatmap(df.corr(), annot=True, cmap="Greens", fmt=".2f")

    plt.savefig(f"plots/heatmap/{name} correlation.png")
    plt.show()

def reg_lines(numerical: pd.DataFrame):
    for col in numerical:
        for col1 in numerical:
            if col is col1:
                continue

            if abs(numerical[col].corr(numerical[col1])) >= 0.2:
                plt.figure(figsize=(10, 6))
                sns.regplot(x=col, y=col1, data=numerical, line_kws={"color": "Black"})
                plt.title(f"{col} x {col1}")

                plt.savefig(f"plots/regline/{col} x {col1}.png")

                plt.show()

def analysis(file_name):
    df = pd.read_csv(file_name)

    remove_nans(df)

    ### Divide into numerical and categorical
    numerical: pd.DataFrame = df.select_dtypes(exclude=object)
    categorical = df.select_dtypes(include=object)

    ### Save the numerical and categorical summary to csv
    summarize_numerical(numerical).to_csv("summaries/numerical_summary.csv", index=False)
    summarize_categorical(categorical).to_csv("summaries/categorical_summary.csv", index=False)

    ### Save the box plots of the numerical x music_genre
    box_plots(numerical,TARGET,df)
    violin_plots(numerical,TARGET,df)

    ### Save error bar plots of the numerical x music_genre
    error_bars(numerical,TARGET,df, "sd")
    error_bars(numerical,TARGET,df, "se")

    ### Save numerical histograms
    histograms(df)

    df = format_df(df)
    ### Save the correlation heatmap
    correlation_heatmap(df.select_dtypes(exclude=object), TARGET)

    ### Save the regression lines for the features with correlation >= 0.2
    reg_lines(df.select_dtypes(exclude=object))

def create_dirs():
    directories = ["plots", "plots/boxplots", "plots/error_bars", "plots/heatmap", "plots/histograms", "plots/regline",
                   "plots/violinplots", "summaries"]

    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    create_dirs()
    analysis("data/music_genre.csv")