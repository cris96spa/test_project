import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd
import polars as pl
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform as sf
from scipy.stats import pearsonr, spearmanr, pointbiserialr, chi2_contingency
from sklearn.preprocessing import LabelEncoder

from typing import Union

def correlation_ratio(categories: Union[pd.Series, pl.Series], 
                      values: Union[pd.Series, pl.Series]) -> float:
    '''
    Function to calculate correlation ratio (η) for numerical-categorical variables
    Parameters:
        - categories: polars.Series or pandas.Series
        - values: polars.Series or pandas.Series
    Returns:
        - correlation_ratio (float within [0,1])
    '''
    if isinstance(categories, pl.Series):
        categories = categories.to_pandas()
    if isinstance(values, pl.Series):
        values = values.to_pandas()
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    return np.sqrt(numerator / denominator)
    

def rearrange_cols(matrix: pl.DataFrame, new_order: list) -> pl.DataFrame:
    '''
    Arrange the order of the matrix columns/rows according to a list with the new feature order
    Parameters:
        - matrix: polars.DataFrame to be reordered
        - new_oder: list of features in required order
    Returns
        - new_mat: ordered polars.DataFrame
    '''
    
    old_cols = matrix.columns
    if 'feature' in old_cols:
        old_cols.remove('feature')
    row_orders = [new_order.index(elem) for elem in old_cols if elem in new_order]
    matrix = matrix.filter(pl.col('feature').is_in(new_order)).select(pl.exclude('feature'))
    new_mat = matrix[new_order]
    new_mat = new_mat.with_columns(
        pl.Series(row_orders).alias('row')
    ).sort(by='row').select(pl.exclude('row')).with_columns(
        pl.Series(new_mat.columns).alias('feature')
    )
    return new_mat


class Correlator:
    '''
    Class defined to compute the correlation among features
    Requires:
        - df: features dataset
        - save_path (str, optional): path to store the results
        - exclude_cols (list, optional): list of features to be removed
    '''
    def __init__(
        self, df: pl.DataFrame,
        save_path: str | None = None,
        exclude_cols: list | None = None
    ) -> None:
        self.df = df
        self.columns = df.columns
        if exclude_cols is not None:
            self.columns = set(self.columns)-set(exclude_cols)
        self.save_path = save_path
        self.numeric = [
            col for col in self.columns if (
                (
                    df[col].dtype != pl.String and 'Date' not in col
                ) and (
                    len(df[col].unique())>1
                )
            )
        ]
        self.categoric = [
            col for col in self.columns if 
            (
                df[col].dtype == pl.String and len(df[col].unique())>1 and 'Date' not in col
            ) or (
                len(df[col].unique())>1 and np.all(np.isin(df[col].unique(), np.array([0,1])))
            )
        ]
        self.corr = None
        self.cramer = None
        self.mix_corrs = None

    def fix_matrix(self, matrix: pl.DataFrame, title: str | None = None) -> pl.DataFrame:
        '''
        Arrange the order of the matrix columns/rowst to cluset highly correlated features
        Parameters:
            - matrix: polars.DataFrame to be reordered
            - title (str, optional): plot title 
        Returns
            - new_mat: ordered polars.DataFrame
        '''
        
        new_order = self.get_order(matrix, title=title)
        return rearrange_cols(matrix, new_order)
    
    def make_num_analysis(
        self,
        numeric: list | None = None,
        title: str | None = "Correlations between Numerical Features"
    ) -> None:
        '''
        Perform correlation analysis among numeric features.

        Parameters:
            numeric (list, optional): list of numerical features to consider
            title (str, optional): title for the saved figure
        '''
        self.get_correlations(numeric)
        self.corr = self.fix_matrix(self.corr, title='Correlations hierarchies')
        self.plot_heatmap(self.corr, title)

    def make_cat_analysis(
        self,
        categoric: list | None = None,
        title: str | None = "Correlations between Categorical Features"
    ) -> None:
        '''
        Perform Cramer's v correlation analysis among categoric features.

        Parameters:
            categoric (list, optional): list of categoric features to consider
            title (str, optional): title for the saved figure
        '''
        self.get_cramer_correlations(categoric)
        self.cramer = self.fix_matrix(self.cramer, title='Cramer hierarchies')
        self.plot_heatmap(self.cramer, title)  

    def make_cross_analysis(
        self,
        categoric: list | None = None,
        numeric: list | None= None,
        title: str | None = "Correlation ratios (Num vs Cat)"
    ) -> None:
        '''
        Perform crossed correlation analysis between numeric and categoric features.

        Parameters:
            categoric (list, optional): list of categoric features to consider
            numeric (list, optional): list of numeric features to consider
            title (str, optional): title for the saved figure
        '''
        categoric = self.categoric if categoric is None else categoric
        old_numeric = self.numeric if numeric is None else numeric
        numeric = old_numeric.copy()
        for col in old_numeric:
            if len(self.df[col].unique())<10:
                numeric.remove(col)
                categoric+=[col]
        self.get_mixed_correlations(numeric, categoric)
        self.plot_asymmetric_heatmap(self.mix_corrs, title)

    def make_full_analysis(
        self,
        categoric: list | None = None,
        numeric: list | None= None,
    ) -> None:
        '''
        Perform all types of correlation analysis:
            i) between numeric  features (Pearson's).
            ii) between categoric features (Cramer V's).
            iii) between numeric and categoric features (correlation ratios).

        Parameters:
            categoric (list, optional): list of categoric features to consider
            numeric (list, optional): list of numeric features to consider
        '''
        self.make_num_analysis(numeric)
        self.make_cat_analysis(categoric)
        self.make_cross_analysis(categoric, numeric)
    
    def get_correlations(self, numeric: list | None = None) -> None:
        '''
            retrieve Pearson's correlation index among numeric columns
        Parameters:
            numeric (list, optional): list of numeric features to consider
        '''
        if numeric is not None:
            cols = numeric
        else:
            cols = self.numeric
        self.corr = self.df[cols].corr().with_columns(
            pl.Series(cols).alias('feature')
        )


    def plot_heatmap(self, matrix: pl.DataFrame, title: str = "") -> None:
        '''
        Heatmap plot of a symmetric matrix
        Parameters:
            - matrix (polars.DataFrame): matrix to be represented
            - title (str): figure title
        The resulting figure is shown and saved in self.save_path, if available
        '''
        if  'feature' in matrix.columns:
            matrix = matrix.select(pl.exclude('feature'))
        col_list = matrix.columns
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(matrix, ax=ax)
        ax.set_xticklabels(col_list, rotation=90)
        ax.set_yticklabels(col_list, rotation=0)
        fig.suptitle(title)
        if self.save_path is not None:
            plt.savefig(osp.join(self.save_path, f'{title}.png'))

    def plot_asymmetric_heatmap(self, matrix: pl.DataFrame, title : str = "") -> None:
        '''
        Heatmap plot of an asymmetric matrix
        Parameters:
            - matrix (polars.DataFrame): matrix to be represented
            - title (str): figure title
        The resulting figure is shown and saved in self.save_path, if available
        '''
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(matrix, ax=ax)
        fig.suptitle(title)
        if self.save_path is not None:
            plt.savefig(osp.join(self.save_path, f'{title}.png'))

    def single_cramer(self, col1: str, col2: str) -> float:
        '''
        Computation of Cramer V correlation coefficients among two categoric features:

        Parameters:
            - col1 (str): first categorical feature  
            - col2 (str): second categorical feature  

        Returns:
            - cram_V (float): cramer V correlation coefficient in [0,1]
        '''
        pivot = self.df.pivot(
            values=col1, index=col2, on=col1, aggregate_function="len"
        ).fill_null(0).select(pl.exclude(col2))
    
        # comment to perform the computations below
        pivot = np.array(pivot)
        stats = chi2_contingency(pivot)[0]
        cram_V = stats / (np.sum(pivot) * (min(pivot.shape) - 1))
    
        """
        Other method for computation of the chi2 coefficient.
        It is possible (?) to modify the following rows for multiple column computation
        
        # joint frequencies computation
        freq = pivot/len(df)
    
        # marginal frequencies
        vy = freq.sum_horizontal()
        vx = freq.sum()
    
        # statistical coefficients
        diffs = pivot-vx*vy*len(data)
        sqs = diffs.select([pl.col(col)**2 for col in diffs.columns])
        stats = (sqs/(vx*vy*len(data))).sum_horizontal().sum()
        cram_V = stats / (pivot.sum_horizontal().sum() * (min(pivot.shape) - 1))
        """
        
        return cram_V

    def get_cramer_correlations(self, categoric: list | None = None) -> None:
        '''
        Computation of Cramer V correlation coefficients among a list categoric features:

        Parameters:
            - categoric (list, optional): list of strings representing categoric features 
        '''
        if categoric is not None:
            cols = [feat for feat in categoric if feat in self.categoric]
        else:
            cols = self.categoric
        cramer = np.empty((len(cols),len(cols)))
        for i, col1 in enumerate(cols):
            for j,col2 in enumerate(cols):
                cramer[i,j] = self.single_cramer(col1, col2)
                self.cramer = pl.from_numpy(cramer, schema = cols).with_columns(
                    pl.Series(cols).alias('feature')
                )


    def get_order(self, matrix: pl.DataFrame, title: str | None = None) -> list:
        '''
        order the correlation matrix according to a dendrogram

        Parameters:
            - matrix: polars.DataFrame to be evaluated
            - title (str, optional): figure title
        if self.save_path is defined, to resulting image is saved within the folder
        '''
        # Compute the pairwise distances using correlation
        matrix = matrix.select(pl.exclude('feature'))
        distances = 1 - np.abs(matrix.to_numpy())
        # Perform hierarchical clustering
        linkage = sch.linkage(distances, method='complete')
        fig, ax=plt.subplots(figsize=(12, 6))
        dendro = sch.dendrogram(
            linkage, 
            labels=matrix.columns, 
            leaf_rotation=70, 
            ax=ax, 
            no_plot=self.save_path is None
        )
        fig.suptitle(title)
        if self.save_path is not None:
            plt.savefig(osp.join(self.save_path, f'{title}.png'))
        plt.close()
        return dendro['ivl']

    def get_mixed_correlations(self, numeric: list | None = None, categoric: list | None = None):
        '''
        Compute correlations between numeric and categorical features.
        If the categorical features have only 2 classes, Point biserial correlation is used.
        Otherwise, the correlation_ration is retrieved.

        Parameters:
            numeric (list, optional): list of numeric features
            categoric (list, optional): list of categoric features
        '''
        numeric = self.numeric if numeric is None else numeric
        categoric = self.categoric if categoric is None else categoric
        mix_correlations = np.empty((len(numeric), len(categoric)))
        for n_num, num_col in enumerate(numeric):
                for n_cat, cat_col in enumerate(categoric):
                    if len(self.df[cat_col].unique()) == 2:  # Binary categorical
                        le = LabelEncoder()
                        encoded_cat = le.fit_transform(self.df[cat_col])
                        mix_correlations[n_num, n_cat] = pointbiserialr(self.df[num_col], encoded_cat)[0]
                    else:  # Multiclass categorical
                        mix_correlations[n_num, n_cat] = correlation_ratio(self.df[cat_col],self. df[num_col])
        self.mix_corrs = pd.DataFrame(mix_correlations, index=numeric, columns=categoric)
    