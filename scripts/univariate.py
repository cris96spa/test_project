import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import polars as pl
import seaborn as sns

def plot_hist(df, col, threshold=1e-2, max_it=10, n_bins=20):
    if len(df[col].unique())==1:
        return
    balanced = False
    it = 0
    series = df[col]
    min_val = series.min()
    max_val = series.max()

    while(not balanced and it < max_it):
        hist = np.histogram(series, bins=n_bins,density=False)
        densities = hist[0]/len(series)
        bins = hist[1]
        sums_left = densities.cumsum()
        sums_right = densities[::-1].cumsum()[::-1]
        min_lim = np.where(sums_left<threshold)[0]
        max_lim = np.where(sums_right<threshold)[0]
        if len(min_lim):
            min_val = bins[min_lim[-1]]
        if len(max_lim):
            max_val = bins[max_lim[0]]
        
        balanced = len(min_lim) + len(max_lim) < 2
        df = df.filter((pl.col(col)>=min_val).and_(pl.col(col)<=max_val))
        series = df[col]
        it += 1

    if (max_val - min_val) <= n_bins and series.dtype in [pl.Int32, pl.Int64]:
        bins = np.arange(min_val, max_val + 2) - 0.5
        ticks = np.arange(min_val, max_val + 1)
    else:
        bins = n_bins
        ticks = None
    fig = plt.figure(figsize=(10,6)) 
    plt.hist(series, density=True, bins=bins, alpha=0.6, edgecolor='black', linewidth=.3)
    if len(series.unique())>n_bins:
        sns.kdeplot(data=series, clip=(min_val, max_val))
    plt.xlim()
    plt.xticks(ticks=ticks)
    plt.ylabel('Density')
    plt.title(col)
    plt.grid()
    plt.show()
    return fig


def group_hist(df, col, threshold=0.01):
    series = df[col]
    counts = series.value_counts().sort(by='count', descending=True)
    counts=counts.with_columns(
        pl.col('count').cast(pl.Int32),
        (pl.col('count')/len(df)).alias('freq')
    )
    valid_counts = counts.filter(pl.col('freq')>threshold)
    other_counts = counts.filter(pl.col('freq')<=threshold)
    sum_others = other_counts.select(pl.col('count').sum().cast(pl.Int32)).item()
    valid_counts = valid_counts.vstack(pl.DataFrame({col: "others", "count": sum_others, "freq": sum_others/len(df)}, schema={col:pl.String, "count":pl.Int32, "freq": pl.Float64}))
    return valid_counts, other_counts    



def group_others(other, col, n_els=6):
    L = int((other['count']/other['freq']).head(1).item())
    other = other.sort(by='count', descending=True)
    top_d2 = other.head(n_els)
    bottom_d2 = other.tail(-n_els)
    sum_bottom = bottom_d2.select(pl.col('count').sum().cast(pl.Int32)).item()
    other_final = top_d2.vstack(pl.DataFrame({col: "others*", "count": sum_bottom, "freq": sum_bottom/L}, schema={col:pl.String, "count":pl.Int32, "freq": pl.Float64}))
    return other_final


def single_cat_plot(ax, counts, col, how='hist', title='', color_last=True):
    if how == 'pie':
        ax.pie(counts['count'], labels=counts[col])
    else:
        barlist = ax.bar(x=counts[col], height=counts["freq"], )
        if color_last:
            barlist[-1].set_color('r')
    ax.set_title(title)
    ax.set_ylabel('frequency')
    ax.tick_params(axis='x', labelrotation=30)
    return ax


def plot_categorical(df, col, threshold=0.02, n_others=6, how='hist'):
    valid, others = group_hist(df, col, threshold)
    if len(others)>1:
        others =  group_others(others, col, n_els=n_others)
        fig, ax = plt.subplots(ncols=2, figsize=(12,5), squeeze=True)
        single_cat_plot(ax[0], valid, col, how, 'major occurrences')
        single_cat_plot(ax[1], others, col, how, 'minor occurrences')
    else:
        if len(others)==1:
            other_name = others.select(pl.col(col)).unique().head().item()
            valid=valid.with_columns(pl.col(col).replace("others", other_name))
        fig, ax = plt.subplots(ncols=1, figsize=(6,5), squeeze=True)
        valid=valid.filter(pl.col(col)!='others')
        single_cat_plot(ax, valid, col, how, '', color_last=False)
    fig.suptitle(col)
    return fig


def plot_feature(df, col, save_path=None):
    if df[col].dtype == pl.String:
        fig = plot_categorical(df, col)
    else:
        plot_hist(df,col)
    if save_path is not None:
        plt.savefig(osp.join(save_path, f'{col}.png'))
