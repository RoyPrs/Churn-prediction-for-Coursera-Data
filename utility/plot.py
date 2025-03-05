# Standard Python packages
from math import sqrt

# Data packages
import pandas as pd
import numpy as np

# Visualization Packages
from matplotlib import pyplot as plt
import seaborn as sns


def grouping(df, columns):
    if len(columns) == 1:
        g = (df[columns[0]].value_counts(normalize=True) * 100).round(2)
        return g.sort_values(ascending=False)
    else:
        g = df.groupby(columns, observed=True).size().unstack()
        g = g.fillna(0)
        total = g.sum(axis=1)
        for col in g.columns:
            col_str = f'{col}%'
            g[col_str] = ((g[col] / total) * 100).round(2)
        by = g.columns[-1]
        return g.sort_values(by, ascending=False)


def paing(df, column, lbl=[], size=(3,3), colors = ['#36ad82','#b52b5e']):
    d=df[column].value_counts()
    if not lbl:
        lbl = list(d.index)
    l = list(zip(lbl, d.values))
    labels =[f'{i}: {j}' for i,j in l]
    fig = plt.figure(figsize=(3,3))
    plt.pie(d, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title(column);
	

def baring(df, columns, size=(8,4), annotate=True, colors = ['#b52b5e', '#36ad82'], grid=True, stacked=False):
    if len(columns) > 1:
        g = grouping(df, columns).iloc[:,-2:]
        title = f"{columns[0]} vs {columns[1]}"
    else:
        g = grouping(df, columns)
        title = f"{columns[0]}"
        colors = [colors[1]]
        
    ax = g.plot(kind='barh', color=colors, figsize=size, rot=0, width=0.85, stacked=stacked,
                                        xlabel='Percenrage', ylabel=f"{columns[0]}", title=title)

    if annotate:
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            if width == 0:
                continue
            ax.text(x+width/2, y+height/2, '{:.0f} %'.format(width), 
                    horizontalalignment='left', verticalalignment='center')
					
					
					
def histing(df, columns, size=(5,3)):
    plt.figure(figsize=size)
    if type(df) == pd.core.series.Series:
        h = sns.histplot(x=df)
        median = df.median()
    else:
        h = sns.histplot(x=df[columns])
        median = df[columns].median()
    plt.axvline(median, color='red', linestyle='--')
    heights = [i.get_height() for i in h.patches]
    x = median + 50
    y = max(heights) * 0.66
    print(x,y)
    plt.text(x, y, f'median={median}', color='red')
    plt.title(columns);
	

def scattering(df, columns, size=(5,3)):
    plt.figure(figsize=size)
    sns.scatterplot(data=df, x=columns[0], y=columns[1])
    plt.title(f"{columns[0]} vs {columns[1]}")
	

def boxing(df,columns, size=(5,1)):
    plt.figure(figsize=size)
    sns.boxplot(x=df[columns], fliersize=1)
    plt.title(columns);
	
