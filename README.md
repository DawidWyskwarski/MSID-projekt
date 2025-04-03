# MSiD-projekt 

# Obesity Data Analysis

## Overview

This project provides a comprehensive analysis of the "ObesityDataSet_raw_and_data_sinthetic.csv" dataset. The analysis includes numerical and categorical summaries, various plots (boxplots, violin plots, error bars, histograms, heatmaps, and regression lines), and correlation analysis. The results are saved as CSV files and visualizations in the `plots/` directory.

## Features

- **Numerical Summary:** Provides statistical summaries such as mean, median, min, max, standard deviation, and missing values.
- **Categorical Summary:** Computes missing values, unique classes, and class proportions.
- **Box Plots & Violin Plots:** Visualizes numerical features against the "NObeyesdad" category.
- **Error Bars:** Displays standard deviation and standard error for numerical features against "NObeyesdad."
- **Histograms:** Generates histograms for numerical features with and without hue.
- **Correlation Heatmap:** Displays feature correlations in a heatmap.
- **Regression Lines:** Plots regression lines for feature pairs with correlation >= 0.2.


## Usage

### Prerequisites

Ensure you have Python installed. Install the required libraries using:

```sh
pip install -r requirements.txt
```

### Running the Analysis

1. Place the dataset inside the `data/` folder.
2. Run the script:

```sh
python data_analysis_main.py
```

3. The script will create necessary directories, generate summaries, and save plots.

## Outputs

- **Summaries:** `summaries/numerical_summary.csv`, `summaries/categorical_summary.csv`
- **Plots:** Various visualization outputs stored under `plots/`



