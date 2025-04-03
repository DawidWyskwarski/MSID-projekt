import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def summarize_numerical(numerical):
    num_summary: pd.DataFrame = numerical.describe(percentiles=[0.05, 0.95]).loc[
        ['mean', '50%', 'min', 'max', 'std', '5%', '95%']]
    num_summary.loc['missing'] = numerical.isna().sum()

    return num_summary

def summarize_categorical(categorical):
    cat_summary = pd.DataFrame(columns=categorical.columns)
    cat_summary.loc['missing'] = categorical.isna().sum()

    cat_summary.loc['unique classes'] = categorical.nunique()

    for col in categorical.columns:
        proportions = categorical[col].value_counts(normalize=True)
        for idx, value in proportions.items():
            cat_summary.loc[f'{idx}_proportion', col] = value

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

        sns.pointplot(x='NObeyesdad', y=col, data=data_frame, errorbar=eb_type, linestyles='none')

        plt.savefig(f"plots/error_bars/{eb_type} {col} x {x}.png")

        plt.show()

def histograms(numerical: pd.DataFrame, data_frame=None, hue=None):
    for col in numerical:
        sns.histplot(data=data_frame if data_frame is not None else numerical, x=col,hue=hue)

        plt.title(f"{col} {f"with {hue} " if hue else ""}histogram")
        plt.savefig(f"plots/histograms/{col} {f"with {hue} " if hue else ""}histogram.png")

        plt.show()

def correlation_heatmap(df,name):
    sns.heatmap(df.corr(), annot=True, cmap="Greens")

    plt.savefig(f"plots/heatmap/{name} correlation.png")
    plt.show()


def analysis(file_name):
    df = pd.read_csv(file_name)

    #divide into numerical and categorical
    numerical: pd.DataFrame = df.select_dtypes(exclude=object)
    categorical = df.select_dtypes(include=object)

    #save the numerical and categorical summary to csv

    summarize_numerical(numerical).to_csv("summaries/numerical_summary.csv", index=False)
    summarize_categorical(categorical).to_csv("summaries/categorical_summary.csv", index=False)

    #save the box plots of the numerical x NObeysesdad

    box_plots(numerical,"NObeyesdad",df)
    violin_plots(numerical,"NObeyesdad",df)

    #save error bar plots of the numerical x NObeysesdad

    error_bars(numerical,"NObeyesdad",df, "sd")
    error_bars(numerical,"NObeyesdad",df, "se")

    #save numerical histograms
    histograms(numerical)

    #save parametric histograms
    histograms(numerical=numerical,data_frame=df,hue='Gender')
    histograms(numerical=numerical,data_frame=df,hue='NObeyesdad')

    #save the correlation heatmap
    correlation_heatmap(numerical,"obesity")

if __name__ == "__main__":
    analysis("data/ObesityDataSet_raw_and_data_sinthetic.csv")