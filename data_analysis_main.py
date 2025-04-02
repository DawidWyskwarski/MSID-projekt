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

        plt.savefig(f"plots/boxplots/{y} x {x}.png")

        plt.show()

def violin_plots(ys:pd.DataFrame ,x, data_frame):
    for y in ys:
        plt.figure(figsize=(16, 5))
        sns.violinplot(x=x, y=y, data=data_frame)

        plt.savefig(f"plots/violinplots/{y} x {x}.png")

        plt.show()

def analysis(file_name):
    df = pd.read_csv(file_name)

    #divide into numerical and categorical
    numerical: pd.DataFrame = df.select_dtypes(exclude=object)
    categorical = df.select_dtypes(include=object)

    #save the numerical and categorical summary to csv
    summarize_numerical(numerical).to_csv("numerical_summary.csv", index=False)
    summarize_categorical(categorical).to_csv("categorical_summary.csv", index=False)

    #save the box plots of the numerical x NObeysesdad
    box_plots(numerical,"NObeyesdad",df)
    violin_plots(numerical,"NObeyesdad",df)


if __name__ == "__main__":
    analysis("data/ObesityDataSet_raw_and_data_sinthetic.csv")