# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 21:18:12 2025

@author: Shantanu
"""

"""visualization_utils.py
This script provides utility functions for data visualization in data science and automation tasks. It includes functions for creating bar plots, line plots, scatter plots, histograms, heatmaps, and interactive plots using Matplotlib, Seaborn, and Plotly, designed to work with datasets like `sales.csv` and `hr_data.csv` in the `data/` directory.

Prerequisites:
- Python 3.9+
- Libraries: pandas, matplotlib, seaborn, plotly
- Datasets: `sales.csv`, `hr_data.csv` in the `data/` directory
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

# Ensure output directory exists
output_dir = '../data/output'
os.makedirs(output_dir, exist_ok=True)

def load_dataset(file_name):
    """Load a CSV file from the data directory.
    
    Args:
        file_name (str): Name of the CSV file (e.g., 'sales.csv').
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    file_path = os.path.join('../data', file_name)
    return pd.read_csv(file_path)

def save_plot(fig, file_name, plotly=False):
    """Save a plot to the output directory.
    
    Args:
        fig: Matplotlib figure or Plotly figure object.
        file_name (str): Name of the output file (e.g., 'plot.png').
        plotly (bool): Whether the figure is a Plotly figure.
    """
    file_path = os.path.join(output_dir, file_name)
    if plotly:
        fig.write_html(file_path.replace('.png', '.html'))
    else:
        fig.savefig(file_path, bbox_inches='tight')
        plt.close()
    print(f'Saved plot to {file_path}')

def create_bar_plot(df, x_col, y_col, title='Bar Plot', hue=None):
    """Create a bar plot using Seaborn.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
        title (str): Plot title.
        hue (str, optional): Column for color grouping.
        
    Returns:
        matplotlib.figure.Figure: Bar plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=45)
    return fig

def create_line_plot(df, x_col, y_col, title='Line Plot', hue=None):
    """Create a line plot using Seaborn.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
        title (str): Plot title.
        hue (str, optional): Column for color grouping.
        
    Returns:
        matplotlib.figure.Figure: Line plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return fig

def create_scatter_plot(df, x_col, y_col, title='Scatter Plot', hue=None, size=None):
    """Create a scatter plot using Seaborn.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
        title (str): Plot title.
        hue (str, optional): Column for color grouping.
        size (str, optional): Column for size variation.
        
    Returns:
        matplotlib.figure.Figure: Scatter plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, size=size, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return fig

def create_histogram(df, column, title='Histogram', bins=20):
    """Create a histogram using Seaborn.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column for histogram.
        title (str): Plot title.
        bins (int): Number of bins.
        
    Returns:
        matplotlib.figure.Figure: Histogram figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=column, bins=20, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column)
    return fig

def create_heatmap(df, title='Correlation Heatmap'):
    """Create a correlation heatmap using Seaborn.
    
    Args:
        df (pd.DataFrame): Input DataFrame with numerical columns.
        title (str): Plot title.
        
    Returns:
        matplotlib.figure.Figure: Heatmap figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(title)
    return fig

def create_box_plot(df, x_col, y_col, title='Box Plot', hue=None):
    """Create a box plot using Seaborn.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
        title (str): Plot title.
        hue (str, optional): Column for color grouping.
        
    Returns:
        matplotlib.figure.Figure: Box plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=45)
    return fig

def create_pie_chart(df, column, title='Pie Chart'):
    """Create a pie chart using Matplotlib.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column for pie chart categories.
        title (str): Plot title.
        
    Returns:
        matplotlib.figure.Figure: Pie chart figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    counts = df[column].value_counts()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title(title)
    return fig

def create_interactive_scatter(df, x_col, y_col, color=None, size=None, title='Interactive Scatter Plot'):
    """Create an interactive scatter plot using Plotly.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
        color (str, optional): Column for color grouping.
        size (str, optional): Column for size variation.
        title (str): Plot title.
        
    Returns:
        plotly.graph_objects.Figure: Interactive scatter plot.
    """
    fig = px.scatter(df, x=x_col, y=y_col, color=color, size=size, title=title)
    return fig

def create_interactive_line(df, x_col, y_col, color=None, title='Interactive Line Plot'):
    """Create an interactive line plot using Plotly.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
        color (str, optional): Column for color grouping.
        title (str): Plot title.
        
    Returns:
        plotly.graph_objects.Figure: Interactive line plot.
    """
    fig = px.line(df, x=x_col, y=y_col, color=color, title=title)
    return fig

def create_interactive_histogram(df, column, title='Interactive Histogram', bins=20):
    """Create an interactive histogram using Plotly.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column for histogram.
        title (str): Plot title.
        bins (int): Number of bins.
        
    Returns:
        plotly.graph_objects.Figure: Interactive histogram.
    """
    fig = px.histogram(df, x=column, nbins=bins, title=title)
    return fig

def create_pair_plot(df, columns, title='Pair Plot'):
    """Create a pair plot using Seaborn.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to include.
        title (str): Plot title.
        
    Returns:
        matplotlib.figure.Figure: Pair plot figure.
    """
    pair_plot = sns.pairplot(df[columns])
    pair_plot.fig.suptitle(title, y=1.02)
    return pair_plot.figure

def create_violin_plot(df, x_col, y_col, title='Violin Plot', hue=None):
    """Create a violin plot using Seaborn.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
        title (str): Plot title.
        hue (str, optional): Column for color grouping.
        
    Returns:
        matplotlib.figure.Figure: Violin plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=45)
    return fig

def create_time_series_plot(df, date_col, y_col, title='Time Series Plot'):
    """Create a time series line plot using Seaborn.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Column for date/time (must be datetime type).
        y_col (str): Column for y-axis.
        title (str): Plot title.
        
    Returns:
        matplotlib.figure.Figure: Time series plot figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x=date_col, y=y_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(y_col)
    plt.xticks(rotation=45)
    return fig

def create_confusion_matrix_heatmap(y_true, y_pred, title='Confusion Matrix'):
    """Create a confusion matrix heatmap using Seaborn.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        title (str): Plot title.
        
    Returns:
        matplotlib.figure.Figure: Confusion matrix heatmap.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    return fig

def create_faceted_grid(df, x_col, y_col, col=None, row=None, title='Faceted Grid Plot'):
    """Create a faceted grid plot using Seaborn.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
        col (str, optional): Column for column-wise faceting.
        row (str, optional): Column for row-wise faceting.
        title (str): Plot title.
        
    Returns:
        seaborn.FacetGrid: Faceted grid plot.
    """
    g = sns.FacetGrid(df, col=col, row=row, margin_titles=True)
    g.map(sns.scatterplot, x_col, y_col)
    g.fig.suptitle(title, y=1.02)
    return g.figure

# Example Usage
if __name__ == "__main__":
    # Load datasets
    sales = load_dataset('sales.csv')
    hr = load_dataset('hr_data.csv')
    
    # Example: Bar plot
    sales_agg = sales.groupby('store_size')['sales_amount'].mean().reset_index()
    bar_fig = create_bar_plot(sales_agg, 'store_size', 'sales_amount', 'Average Sales by Store Size')
    save_plot(bar_fig, 'sales_bar.png')
    
    # Example: Scatter plot
    scatter_fig = create_scatter_plot(sales, 'marketing_spend', 'sales_amount', 'Sales vs Marketing Spend')
    save_plot(scatter_fig, 'sales_scatter.png')
    
    # Example: Histogram
    hist_fig = create_histogram(sales, 'sales_amount', 'Sales Amount Distribution')
    save_plot(hist_fig, 'sales_histogram.png')
    
    # Example: Heatmap
    heatmap_fig = create_heatmap(sales, 'Sales Correlation Heatmap')
    save_plot(heatmap_fig, 'sales_heatmap.png')
    
    # Example: Interactive scatter plot
    interactive_fig = create_interactive_scatter(sales, 'marketing_spend', 'sales_amount', title='Interactive Sales vs Marketing Spend')
    save_plot(interactive_fig, 'sales_interactive_scatter', plotly=True)