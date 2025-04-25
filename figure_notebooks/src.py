# Helper functions for IsletsOmics analysis

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from biomart import BiomartServer
from scipy.stats import false_discovery_control, pearsonr, ttest_ind, gaussian_kde
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
import logging
import matplotlib.patheffects as pe
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score
from matplotlib.patches import Patch, Rectangle
from itertools import cycle
from typing import Union, Tuple
import anndata as ad
import seaborn as sns

class Utils():
    def __init__(self):
        pass

    @staticmethod
    def map_ensembl_to_uniprot(
        organism: str = "hsapiens",
        coalesce: bool = True,
    ):
        """Get Ensembl to Uniprot mapping for a given organism.

        Keeping it simple in this function: enter organism name and
        whether to fill in missing uniprot ids with ensembl ids.

        Parameters
        ----------
        organism : str
            Name of the organism to get the mapping for.

        coalesce : bool
            Whether to fill in missing uniprot ids with ensembl ids.

        Returns
        -------
        dict
            Dictionary with Ensembl gene ids as keys and Uniprot ids as values.

        """

        # connect to Ensembl Biomart server
        server = BiomartServer("http://www.ensembl.org/biomart")

        # Select corresponding database
        ensembl = server.datasets[f"{organism}_gene_ensembl"]

        # Define attributes to retrieve, for now set to ensembl gene id and uniprot id
        attributes = [
            "ensembl_gene_id",
            "uniprotswissprot",
            "uniprot_gn_id",
        ]

        # send query to server
        response = ensembl.search(
            {
                "filters": {},
                "attributes": attributes,
            }
        )

        # parse response: response is a generator object,
        # exhaust it and store the mapping in a dictionary
        mappings = {}
        for line in response.iter_lines():
            fields = line.decode("utf-8").split("\t")
            ensembl_id = fields[0]
            uniprot_candidates = fields[1:]
            for uid in uniprot_candidates:
                if uid:
                    uid = uid.split("-")[0]
                    mappings[ensembl_id] = uid
                    break

        # coalesce the mappings: if no uniprot id is found, use the ensemble id
        if coalesce:
            for e in list(mappings.keys()):
                if not mappings[e]:
                    mappings[e] = e

        return mappings
    
    @staticmethod
    def map_pg_to_id(
        pg_series: pd.Series,
        id_series: pd.Series,
    ):
        # first, check that each PG maps to a single ID, i.e. no PG maps to multiple IDs
        pgs = pg_series.dropna().values
        ids = id_series.dropna().values

        # ineffient nested loops but works for now
        found_status = []
        for t in ids:
            found_counter = 0
            for p in pgs:
                if t in p.split(";"):
                    found_counter += 1
            found_status.append(found_counter)

        if max(found_status) > 1:
            raise ValueError("PGs map to multiple IDs, cannot proceed with mapping.")

        id_to_group = {}
        for group in pgs:
            for id in ids:
                if id in group.split(";"):
                    id_to_group[id] = group

        return id_to_group
    
    @staticmethod
    def deduplicate_alphanumeric_dataframe(
        df: pd.DataFrame, aggregation_method: str = "mean", axis: int = 0
    ):
        """Deduplicate a dataframe.

        What it says, if a datframe has duplicate indices and
        consists of numeric and string columns, it is difficult
        to deduplicated. We add an aggregation method for the
        numeric columns, and paste the string values with a
        semicolon separator.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to deduplicate.
        aggregation_method : str
            Aggregation method for numeric columns. May be one of
            'mean', 'sum', 'first'
        axis : int
            Axis to deduplicate along. Default is 0 (rows).

        Returns
        -------
        pd.DataFrame
            Deduplicated dataframe.

        """

        if aggregation_method == "mean":
            agg_number = "mean"
        elif aggregation_method == "sum":
            agg_number = "sum"
        elif aggregation_method == "median":
            agg_number = "median"

        shape_before = df.shape

        # function to aggregate pandas series into string
        def agg_string(x):
            x = x.astype(str)
            return ";".join(x)

        # check if dataframe has duplicate indices
        if df.index.duplicated().any():
            # deduplicate via dict comprehension
            deduplicated_df = df.groupby(level=0).agg(
                {
                    **{
                        col: agg_number
                        for col in df.select_dtypes(include=[np.number]).columns
                    },
                    **{
                        col: agg_string
                        for col in df.select_dtypes(include=[object]).columns
                    },
                }
            )

            shape_after = deduplicated_df.shape
            print(
                f"Shape before deduplication: {shape_before}, shape after deduplication: {shape_after}, aggregated {shape_before[0] - shape_after[0]} rows."
            )

            return deduplicated_df
        else:
            print("No duplicate indices found, returning original dataframe.")

            return df
        
    @staticmethod
    def nan_safe_df_log(  # explicitly unit tested in test_nan_safe_df_log
        X: pd.DataFrame,
        log: int = 2,
    ):
        """Wrapper for nplog() functions.

        Changes data inplace. Replace zeros and negatives with
        nan and return either log2 or log10 transformed values.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame, may contain negatives and/or zeros
        log : int
            Log-level, allowed values are 2 and 10.

        Returns
        -------
        pd.DataFrame:
            dataframe with log-transformed original values. Zeros
            and negative values are replaced by np.nan.

        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if log != 2 and log != 10:
            raise ValueError("'log' must be either 2 or 10")

        # always copy for now, implement inplace later if needed
        X = X.copy()

        X[X == 0] = np.nan
        X[X < 0] = np.nan

        if log == 2:
            return np.log2(X)
        elif log == 10:
            return np.log10(X)

    @staticmethod
    def nan_safe_ttest_ind(  # implicitly tested via unit-test group_ratios_ttest_ind
        a: pd.Series,
        b: pd.Series,
        **kwargs,
    ):
        if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
            warnings.warn(
                " --> nan_safe_ttest_ind warning: Input must be a pandas Series. Converting to series...",
                stacklevel=2,
            )
            a = pd.Series(a)
            b = pd.Series(b)

        if a.count() < 2 or b.count() < 2:
            return (np.nan, np.nan)
        else:
            return ttest_ind(a, b, **kwargs)
        
    @staticmethod
    def parse_label(
        label: str,
        lookup_dict: dict = None,
        general_regex = None,
        round_digits: int = 4,
    ):
        """Function to parse labels for plotting

        Frequently, plot labels have to be more succinct than the actual data labels. Adjusting
        the labels may either involve replacing an existing label, or performing a regex modification
        of an existing label. These functionalities are summarized in parse_labels(). Importantly, the
        general_regex function is only applied to those labels that do not have a direct modifier in the
        lookup_dict.

        Parameters
        ----------
        label : str
            label to parse

        lookup_dict : dict
            dictionary to lookup labels, contains either direct replacements or functions for more complex
            modifications. Functions should be defined outside of the lookup_dict and called in the dictionary.

        general_regex : function
            general regex function to apply to all labels

        round_digits : int
            number of digits to round numerical labels to, set to 4 by default

        Returns
        -------
        str
            parsed label

        Example
        -------

        def title_parse_fct(label):
            label = label.replace("_", " ")
            label = label.replace("this", "that")
            return label

        lookup_dict = {
            'label1': 'The Label 1',
            'title_for_this_plot': title_parse_fct()
        }

        # plotting function call
        # ...
        l1 = Plots.parse_labels('label1', lookup_dict)
        l2 = Plots.parse_labels('title_for_this_plot', lookup_dict)

        # result
        # l1 = 'The Label 1'
        # l2 = 'Title for that plot'

        """

        # check if label is in lookup_dict, apply general regex if not
        if lookup_dict is None:
            lookup_dict = {}

        if isinstance(label, (int, float)):
            label = round(label, round_digits)

        if label in lookup_dict:
            lookup = lookup_dict[label]
            newlabel = lookup(label) if callable(lookup) else lookup
        else:
            newlabel = general_regex(label) if general_regex is not None else label

        return newlabel
    
    @staticmethod
    def parse_labels(
        labels: list,
        lookup_dict: dict = None,
        general_regex = None,
    ):
        """Convenience wrapper for parse_label() to parse multiple labels at once

        Example:

        def genrex(label):
            nl = label.upper()
            nl = nl.replace('_', ' ')
            return nl

        def lab2mod(label):
            nl = label.replace('_','-')
            return nl

        Plots.parse_labels(
            labels = ['lab_1', 'lab_2', 'title_3', 'lab4'],
            lookup_dict = {'lab_1' : 'special_lab_1', 'lab_2' : lab2mod},
            general_regex = genrex,
        )

        # result
        # ['special_lab_1', 'lab-2', 'TITLE 3', 'LAB4']

        """

        return [
            Utils.parse_label(label, lookup_dict, general_regex) for label in labels
        ]
    
    @staticmethod
    def gaussian_density(
        x: np.array,
        y: np.array,
    ):
        # create correct shape for gaussian density
        xz = np.vstack([x, y])
        # estimate density
        pdf = gaussian_kde(xz)
        # get density values
        z = pdf(xz)

        return z
    
    @staticmethod
    def regression(
        x: np.array,
        y: np.array,
        return_x: bool = False,
        custom_range: tuple = None,
    ):
        """Perform a linear regression and return a linspace of the regression

        Parameters
        ----------
        x : np.array
            x values
        y : np.array
            y values
        return_y_pred : bool
            return y_pred values or x linspace for plotting

        Returns
        -------
        tuple
            x, y_pred, r2, a, b


        """

        # remove na values
        x_nan_mask = ~np.isnan(x)
        y_nan_mask = ~np.isnan(y)
        nan_mask = np.logical_and(x_nan_mask, y_nan_mask)
        x = x[nan_mask]
        y = y[nan_mask]

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        reg = LinearRegression().fit(x, y)

        # get r2 score
        r2 = r2_score(y, reg.predict(x))

        # get coefficients
        a = reg.coef_[0][0]
        b = reg.intercept_[0]

        # return actual values or linspace for line plotting
        if return_x:
            y_pred = reg.predict(x)
            return x, y_pred, r2, a, b
        else:
            if custom_range is not None:
                x = np.linspace(custom_range[0], custom_range[1], 100).reshape(-1, 1)
            else:
                x = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            y_pred = reg.predict(x)
            return x, y_pred, r2, a, b
        
    @staticmethod
    def impute_gaussian(
        X: pd.DataFrame,
        std_offset: float = 3,
        std_factor: float = 0.3,
        random_state: int = 42,
    ):
        """Impute missing values in each column by sampling from
        a gaussian distribution. The distribution is centered at
        std_offset standard deviations below the mean of the feature
        and has a standard deviation of std_factor times the feature's.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with features as columns and samples as rows.
        std_offset : float
            Number of standard deviations below the mean to center the
            gaussian distribution.
        std_factor : float
            Factor to multiply the feature's standard deviation with to
            get the standard deviation of the gaussian distribution.

        Returns
        -------
        pd.DataFrame
            Dataframe with missing values imputed.

        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # All columns must be either int or float
        if not all([X[col].dtype in [int, float] for col in X.columns]):
            raise ValueError("All columns must be either int or float.")

        # set random seed
        np.random.seed(random_state)

        # always copy for now, implement inplace later if needed
        idxs = X.index
        cols = X.columns
        X = X.values.copy()

        # check for missing values
        na_col_idxs = np.where(np.isnan(X).sum(axis=0) > 0)[0]

        # generate corresponding downshifted features
        stds = np.nanstd(X, axis=0)
        means = np.nanmean(X, axis=0)
        shifted_means = means - std_offset * stds
        shifted_stds = stds * std_factor

        # iterate over na-containing columns and impute from corresponding gaussian
        for i in na_col_idxs:
            na_idx = np.where(np.isnan(X[:, i]))[0]
            X[na_idx, i] = np.random.normal(
                shifted_means[i], shifted_stds[i], len(na_idx)
            )

        return pd.DataFrame(X, index=idxs, columns=cols)
    
    @staticmethod
    def scale_and_center(  # explicitly unit tested in test_scale_and_center()
        X: Union[pd.DataFrame, np.ndarray],
        center: bool = True,
        scale: bool = True,
        scaler: str = "standard",  # standard or robust
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Scale and/or center data.

        Either use standard or outlier-robust scaler. robust scaler
        relies on interquartile range and is more resistant to
        outliers. Scaling operates on COLUMNS, i.e. features.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data
        center : bool
            Whether to center features (mean for standard, median for
            robust scaler).
        scale : bool
            Whether to scale features to constant std (standard scaler)
            or IQR (robust scaler)
        scaler : str
            Sklearn scaler to use. Available scalers are 'standard' or
            'robust'.

        Returns
        -------
        pd.DataFrame or np.ndarray
            Data with optionally centered / scaled features

        """

        logging.info(
            f"|-> Scaling and centering data: applying Scikit-Learn {scaler} Scaler ..."
        )

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("Input data must be pd.DataFrame or np.ndarray")

        # always copy for now, implement inplace later if needed
        X = X.copy()

        # RobustScaler is used to scale and center data, as it is less sensitive to outliers
        if scaler == "standard":
            scaler = StandardScaler(with_mean=center, with_std=scale)
        elif scaler == "robust":
            scaler = RobustScaler(
                with_centering=center, with_scaling=scale, quantile_range=(25.0, 75.0)
            )
        else:
            raise ValueError(
                "Unsupported scaler, must be either 'standard' or 'robust'."
            )

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                scaler.fit_transform(X), index=X.index, columns=X.columns
            )
        elif isinstance(X, np.ndarray):
            return scaler.fit_transform(X)
        
    @staticmethod
    def multilegend_patch_generator(
        color_dict: dict,
        pad_max_length: bool = False,
        linewidth: float = 0.5,
        single_column: bool = True,
    ) -> tuple:
        """Make a patch list and a title list from a color-dictionary.

        The point of this function is to take a color reference dictionary
        and generate a list of patches along with a list of titles to be used
        in a single legend call.

        Parameters:
            color_dict (dict) : Dictionary of dictionaries where each top
            level key is a variable and the second level keys are its unique
            values and the values are corresponding colors to be shown in the
            legend.
            pad_max_length (bool) : This parameter is used when the legend is
            to be shown in multiple columns, to avoid breaking headings into
            separate columns.
            linewidth (float) : Width of the line around the patches.
            single_column (bool) : Whether to show the legend in a single column,
            which means that sublegend titles have to be added in the texts with
            invisible patches. If this is set to False, a title will be returned
            along with patches and legend labels.

        Returns:
            tuple : A tuple containing a list of matplotlib patches and a list
            of titles for these patches. Whenever a new variable is encountered,
            the corresponding patch is empty and the title is the variable name.

        Example:
            Input : {"var1" : {"high" : "red", "low" : "blue"}, "var2" : {"high" : "green", "low" : "yellow"}}
            returns : ([Rectangle((0,0), 0, 0, color='w'), Patch(facecolor = 'red'), Patch(facecolor = 'blue'),
                        Rectangle((0,0), 0, 0, color='w'), Patch(facecolor = 'green'), Patch(facecolor = 'yellow'],
                        ['var1', 'high', 'low',
                            'var2', 'high', 'low'])

        """

        longest_column = max([len(color_dict[k]) for k in color_dict])

        patches = []
        titles = []

        for var, values in color_dict.items():
            titles.append(var + ":")
            patches.append(Patch(color="w"))
            for v, c in values.items():
                titles.append(v)
                patches.append(Patch(facecolor=c, edgecolor="black", lw=linewidth))

            if pad_max_length:
                while len(titles) < longest_column + 1:
                    titles.append("")
                    patches.append(Rectangle((0, 0), 0, 0, color="w"))

        if single_column:
            return patches, titles
        else:
            # remove first patch and title
            patches.pop(0)
            title = titles.pop(0)
            return patches, titles, title
        
    @staticmethod
    def nan_safe_bh_correction(
        pvals: np.array,
    ):
        """Scipy.stats.false_discovery_control with nan-safe handling

        Scipy.stats.false_discovery_control is not nan-safe, we need to delete nans, apply correction, then re-insert nans.
        This method adds a unique index to the input array, drops nans, applies correction, then merges the outcome back to
        the original array indices.

        Parameters:
            pvals (np.array): Array of p-values.
        """

        # convert array to dataframe with distinct index
        pval_df = pd.DataFrame({"pvals": pvals})

        initial_index = range(len(pval_df))
        pval_df.index = initial_index

        pval_df_no_nans = pval_df.copy().dropna()
        pval_df_no_nans["pvals_corrected"] = false_discovery_control(
            pval_df_no_nans["pvals"], method="bh"
        )

        # merge back to original index
        pval_df = pval_df.join(pval_df_no_nans["pvals_corrected"], how="left")

        # verify that the original index is preserved
        if not all(pval_df.index == initial_index):
            raise ValueError("Index mismatch in nan_safe_bh_correction.")

        return pval_df["pvals_corrected"].values

    @staticmethod
    def pairwise_correlation(
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
    ):
        """Pairwise pearson correlation between two datasets

        Calculate pairwise pearson correlation with p-values between
        matching columns of two datasets. This method is useful for
        assessing the correlation between two datasets.

        Parameters
        ----------
        data_a : pd.DataFrame
            Dataframe with features as columns and samples as rows.

        data_b : pd.DataFrame
            Dataframe with features as columns and samples as rows.

        Returns
        -------
        pd.DataFrame
            p x 2 dataframe with pairwise pearson correlations between data_a and data_b and p-values,
            where p is the number of shared column names between data_a and data_b.

        """

        # align datasets on sample dimension if necessary
        if data_a.shape[0] != data_b.shape[0]:
            raise ValueError("Dataframes have different number of samples, aligning...")

        # data must be aligned on samples to perform correlation analysis
        data_a, data_b = data_a.align(data_b, join="inner", axis=0)

        a = data_a.values
        b = data_b.values

        # calculate correlation by iteration, since the time-complexity is only O(p)
        output_df = pd.DataFrame(
            index=data_a.columns,
            columns=["r", "p"],
        )

        for i, col in enumerate(output_df.index):
            # get aligned columns
            x = a[:, i]
            y = b[:, i]

            # remove missing values. Note that x and y are aligned
            na_mask = np.isnan(x) | np.isnan(y)
            x = x[~na_mask]
            y = y[~na_mask]

            r, p = pearsonr(x, y)
            output_df.loc[col, "r"] = r
            output_df.loc[col, "p"] = p

        # convert values to float
        output_df = output_df.astype(float)

        # add benjamini-hochberg corrected p-values
        output_df["p_adj"] = Utils.nan_safe_bh_correction(output_df["p"].values)

        return output_df

    @staticmethod
    def boxplot(
        data: pd.DataFrame,
        value_col: str,
        grouping_col: str,
        metadata: pd.DataFrame = None,
        color_col: str = "base_color",
        single_color: str = None,
        grouping_levels: list = None,
        title: str = None,
        highlight_points: list = None,
        highlight_lookup_column: str = "_index",
        highlight_labels_column: str = "_index",
        highlight_color: str = "red",
        collate_label_and_lookup: bool = False,
        show_scatterplot: bool = True,
        normal_jitter: float = None,
        subplot_col: str = None,
        subplot_levels: list = None,
        show_facet_title: bool = True,
        show_facet_xlabel: bool = True,
        base_color=None,
        figsize_x: int = 2,
        figsize_y: int = 2,
        label_lookup_dict: dict = None,
        label_general_regex = None,
        scatter_kwargs: dict = None,
        xtick_kwargs: dict = None,
        # enumeration
        enumerate_plot: str = None,
        enumeration_xy: tuple = (-0.5, 1.25),
        ttest_pvalue_xy: tuple = None,
        pval_digits: int = 3,
        return_ttest: bool = False,
        ylim: tuple = None,
        ylim_padding: tuple = (0, 0),
        linewidth_medium = 0.5,
        fontsize_medium: int = 8,
    ):
        """Boxplot with optional highlighting and facetting

        Basic boxplot function with option to highlight and label points, and
        create multiple subplots based on a categorical column.

        Returns
        -------
        plt.Figure
            figure object
        plt.Axes
            axes object

        """

        plt.rcParams.update(
            {
                "svg.fonttype": "none",
                "font.family": "Arial",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

        highlight_points = [] if highlight_points is None else highlight_points
        scatter_kwargs = {"s": 15} if scatter_kwargs is None else scatter_kwargs
        xtick_kwargs = {} if xtick_kwargs is None else xtick_kwargs

        # base color
        if base_color is None:
            base_color = "lightgrey"

        # combine data and metadata
        if metadata is not None:
            if not all(data.index == metadata.index):
                raise ValueError("Data and metadata indices do not match")
            if any(data.columns.isin(metadata.columns)):
                raise ValueError("Data and metadata columns overlap")
            data = pd.concat([data, metadata], axis=1)
        else:
            data = data.copy()

        # missing values in the value column are dropped
        data = data.dropna(subset=[value_col])

        # ensure proper types
        data[value_col] = data[value_col].astype(float)
        data[grouping_col] = data[grouping_col].astype(str)
        if color_col != "base_color":
            data[color_col] = data[color_col].astype(str)
        else:
            data[color_col] = base_color

        # restrict to selected grouping levels
        if grouping_levels is not None:
            data = data[data[grouping_col].isin(grouping_levels)]

        if single_color is None:
            unique_colors = data[color_col].unique()
            preset_colors = ['#cb334d', '#54aead', '#3a7eb8']
            color_map_dict = {key : col for key, col in zip(unique_colors, cycle(preset_colors))}
            data["_color"] = data[color_col].apply(lambda x: color_map_dict.get(x, base_color))
        else:
            data["_color"] = base_color

        # add '_index' column if label and or lookup column are '_index'
        if highlight_labels_column == "_index" or highlight_lookup_column == "_index":
            data["_index"] = data.index

        # create facetting column if none is specified
        if subplot_col is None:
            data["_facet"] = "single"
            subplot_col = "_facet"
            subplot_levels = ["single"]

        # create subplot levels from data
        if subplot_col is not None and subplot_levels is None:
            subplot_levels = data[subplot_col].unique().tolist()

        # get unique facets, order alphabetically to ensure consistent ordering
        all_facets = np.array(
            [x for x in data[subplot_col].unique() if x in subplot_levels]
        )
        all_facets = np.sort(all_facets)
        rows = 1

        # create indexable subplots
        f, axs = plt.subplots(
            rows,
            len(all_facets),
            figsize=(len(all_facets) * figsize_x, figsize_y),
            sharey=True,
        )
        axs = np.atleast_1d(axs)

        # determine maximum and minimum values for the boxplot
        max_value = data[value_col].max()
        min_value = data[value_col].min()

        # iterate over facets and add plots
        facet_ttests = {}
        for i, facet in enumerate(all_facets):
            # prepare isolated facet data
            facet_data = data[data[subplot_col] == facet].copy()
            facet_groups = facet_data[grouping_col].unique()
            facet_positions = {group: i for i, group in enumerate(facet_groups)}

            # make scatterplot dataframe
            facet_data["_position"] = facet_data[grouping_col].apply(
                lambda x, fp=facet_positions: fp[x]
            )

            # add scatterplot and jitter
            if show_scatterplot:
                if normal_jitter is not None:
                    facet_data["_position"] = facet_data[
                        "_position"
                    ] + np.random.normal(0, normal_jitter, len(facet_data))
                axs[i].scatter(
                    facet_data["_position"],
                    facet_data[value_col],
                    c=facet_data["_color"],
                    **scatter_kwargs,
                )

            # add highlight points
            if collate_label_and_lookup:
                if highlight_labels_column == highlight_lookup_column:
                    facet_data["_label"] = facet_data[highlight_lookup_column]
                else:
                    facet_data["_label"] = (
                        facet_data[highlight_labels_column]
                        + " ("
                        + facet_data[highlight_lookup_column]
                        + ")"
                    )
            else:
                facet_data["_label"] = facet_data[highlight_labels_column]

            # extract boxplot arrays to conform with matplotlib boxplot function
            values = []
            positions = []
            position_labels = []

            for group in facet_groups:
                group_data = facet_data[facet_data[grouping_col] == group]
                positions.append(facet_positions[group])
                values.append(group_data[value_col].values)
                position_labels.append(group)

            # set linewidths
            for spine in axs[i].spines.values():
                spine.set_linewidth(linewidth_medium)

            # ticks
            axs[i].tick_params(width=linewidth_medium)

            # set boxplot linewidths
            plt.rcParams.update(
                {
                    "font.size": 5.102,
                    "axes.linewidth": 0.25,
                    "boxplot.medianprops.linewidth": 0.25,
                    "boxplot.boxprops.linewidth": 0.25,
                    "boxplot.whiskerprops.linewidth": 0.25,
                    "boxplot.capprops.linewidth": 0.25,
                    "boxplot.flierprops.linewidth": 0.25,
                }
            )

            # create boxplot
            boxprops = dict(fill=None)
            box_dict = axs[i].boxplot(
                values,
                positions=positions,
                showfliers=False,
                widths=0.5,
                patch_artist=True,
                boxprops=boxprops,
            )

            # set facecolor
            for _, (b, m) in enumerate(
                zip(
                    box_dict["boxes"],
                    box_dict["medians"],
                )
            ):
                b.set_linewidth(linewidth_medium)
                b.set_color("black")
                m.set_linewidth(linewidth_medium)
                m.set_color("black")

            # there are twice as many whiskers and caps as boxes...
            for _, (w, c) in enumerate(zip(box_dict["whiskers"], box_dict["caps"])):
                w.set_linewidth(linewidth_medium)
                w.set_color("black")
                c.set_linewidth(linewidth_medium)
                c.set_color("black")

            # add ttest annotations if necessary
            if ttest_pvalue_xy is not None:
                if len(values) != 2:
                    print(
                        f"Cannot perform ttest for {facet}, since there are not exactly two groups."
                    )
                    ttest = None
                else:
                    ttest = Utils.nan_safe_ttest_ind(
                        pd.Series(values[0]),
                        pd.Series(values[1]),
                    )

                    # add pvalue annotation
                    pval = round(ttest[1], pval_digits)
                    axs[i].text(
                        ttest_pvalue_xy[0],
                        ttest_pvalue_xy[1],
                        f"p = {pval}",
                        transform=axs[i].transAxes,
                        fontsize=fontsize_medium,
                        ha="center",
                    )

                    facet_ttests[facet] = ttest

            # after scatter and box plots are added, label selected points
            highlight_data = facet_data[
                facet_data[highlight_lookup_column].isin(highlight_points)
            ]

            # highlight points
            axs[i].scatter(
                highlight_data["_position"],
                highlight_data[value_col],
                c=highlight_color,
                s=15,
                alpha=0.5,
            )
            for j, txt in enumerate(highlight_data["_label"]):
                txt_parsed = Utils.parse_label(
                    txt, label_lookup_dict, label_general_regex
                )
                axs[i].annotate(
                    txt_parsed,
                    (
                        highlight_data["_position"].values[j],
                        highlight_data[value_col].values[j],
                    ),
                    fontsize=fontsize_medium,
                )

            # set facet xticks and xticklabels
            axs[i].tick_params(axis="both", labelsize=fontsize_medium)
            axs[i].set_xticks(positions)

            _xticklabels = Utils.parse_labels(
                position_labels, label_lookup_dict, label_general_regex
            )
            xtick_kwargs = dict(fontsize=fontsize_medium) | xtick_kwargs
            axs[i].set_xticklabels(_xticklabels, **xtick_kwargs)

            # set facet title
            if show_facet_title:
                _title = Utils.parse_label(
                    f"{subplot_col}: {facet}", label_lookup_dict, label_general_regex
                )
                axs[i].set_title(_title, fontsize=fontsize_medium)

            # set facet x axis label
            if show_facet_xlabel:
                _xlabel = Utils.parse_label(
                    grouping_col, label_lookup_dict, label_general_regex
                )
                axs[i].set_xlabel(_xlabel, fontsize=fontsize_medium)

            # set facet y axis label
            if i == 0:
                _ylabel = Utils.parse_label(
                    value_col, label_lookup_dict, label_general_regex
                )
                axs[i].set_ylabel(_ylabel, fontsize=fontsize_medium)

            # set plot title
            if title is not None:
                _suptitle = Utils.parse_label(
                    title, label_lookup_dict, label_general_regex
                )
                plt.suptitle(_suptitle, fontsize=fontsize_medium)

            # set limits
            if ylim is None:
                axs[i].set_ylim(
                    min_value - ylim_padding[0], max_value + ylim_padding[1]
                )
            else:
                axs[i].set_ylim(ylim)

            # enumerate plot
            if enumerate_plot is not None and i == 0:
                _parsed_enumeration = Utils.parse_label(
                    enumerate_plot, label_lookup_dict, label_general_regex
                )
                axs[i].text(
                    enumeration_xy[0],
                    enumeration_xy[1],
                    _parsed_enumeration,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=axs[i].transAxes,
                    fontsize=fontsize_medium,
                    color="black",
                    fontweight="bold",
                )

        plt.show()

        if return_ttest:
            return f, axs, facet_ttests
        else:
            return f, axs

    @staticmethod 
    def scatter(
            data: pd.DataFrame,
            x_col: str,
            y_col: str,
            metadata: pd.DataFrame = None,
            color_column: str = None,
            base_color: str = None,
            single_color: bool = False,
            outline_color: str = None,
            title: str = "",
            xlabel: str = "",
            ylabel: str = "",
            figsize_x: int = 2,
            figsize_y: int = 2,
            xlim: tuple = None,
            ylim: tuple = None,
            linear_regression: bool = False,
            regression_linecolor: str = "red",
            show_equation: bool = False,
            linreg_range: tuple = None,
            vline: float = None,
            hline: float = None,
            dline: float = None,
            add_lineplot: bool = False,
            lineplot_kwargs: dict = None,
            highlight_points: list = None,
            highlight_lookup_column: str = "index",
            highlight_labels_column: str = "index",
            highlight_color: str = "red",
            collate_label_and_lookup: bool = False,
            label_va: str = "random",
            label_ha: str = "random",
            hide_labels: bool = False,
            density: bool = False,
            segments: str = "color",  # 'facets' or 'color'
            facet_titles: bool = False,
            x_tick_multiple: int = None,
            y_tick_multiple: int = None,
            aspect_equal: bool = False,
            legend: bool = False,
            scatter_kwargs=None,
            legend_kwargs=None,
            label_general_regex=None,
            label_lookup_dict=None,
            enumerate_plot: str = None,
            enumeration_xy: tuple = (-0.5, 1.25),
            return_linear_regression: bool = False,
            show: bool = True,
            linewidth_large = 1,
            linewidth_medium = 0.5,
            linewidth_small = 0.5,
            fontsize_medium: int = 8,
        ):
            """Scatterplot functionality based on dataframe input"""

            plt.rcParams.update(
                {
                    "svg.fonttype": "none",
                    "font.family": "Arial",
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                }
            )

            highlight_points = [] if highlight_points is None else highlight_points
            scatter_kwargs = {"s": 5} if scatter_kwargs is None else scatter_kwargs
            legend_kwargs = {} if legend_kwargs is None else legend_kwargs

            # combine data and metadata for easier plotting
            if metadata is not None:
                if not all(data.index == metadata.index):
                    raise ValueError("Data and metadata indices do not match")
                data = pd.concat([data, metadata], axis=1)
            else:
                data = data.copy()

            # prepare basic data arrays: no missing values allowed
            data = data.dropna(subset=[x_col, y_col])

            x = data[x_col].values.astype(float)
            y = data[y_col].values.astype(float)
            c = (
                data[color_column].values
                if color_column is not None
                else np.array(["datapoint"] * len(x))
            )

            # set base color
            base_color = base_color if base_color is not None else "lightgrey"

            # create colorscale
            if single_color:
                colorscale_names = ["datapoint"]
                color_map_dict = {"datapoint": base_color}
            else:
                colorscale_names = pd.Series(c).unique()
                color_map_dict = {cn : base_color for cn in colorscale_names}

            # determine number of windows
            if segments == "facets":
                windows = len(colorscale_names)
            elif segments == "color":
                windows = 1

            # scatterplot: subplot for each color
            f, axs = plt.subplots(
                1, windows, figsize=(figsize_x, figsize_y), sharey=True, sharex=True
            )

            # lineplot if specified: all points regardless of color have to be connected
            if add_lineplot:
                if lineplot_kwargs is None:
                    lineplot_kwargs = {}
                if "lw" not in lineplot_kwargs:
                    lineplot_kwargs["lw"] = linewidth_medium
                if "color" not in lineplot_kwargs:
                    lineplot_kwargs["color"] = "black"
                axs.plot(
                    x,
                    y,
                    **lineplot_kwargs,
                )

            # iterate over colorscale names
            legend_patches = []
            r2_scores = {}
            coefficients = {}
            for i, n in enumerate(color_map_dict.keys()):
                ax = axs if windows == 1 else axs[i]
                idx = c == n

                # set linewidths
                for spine in ax.spines.values():
                    spine.set_linewidth(linewidth_medium)


                # add gaussian density
                if density:
                    point_color = Utils.gaussian_density(x=x[idx], y=y[idx])
                else:
                    point_color = np.array(
                        [color_map_dict.get(n, base_color)] * len(x[idx])
                    )

                # scatterplot
                ax.scatter(
                    x[idx], y[idx], c=point_color, **scatter_kwargs, marker=".", lw=0
                )

                # set tick multiples, e.g. to obtain labels 5, 10, 15, 20, 25, 30 instead of 10, 20, 30
                if x_tick_multiple is not None:
                    ax.set_xticks(
                        np.arange(
                            np.floor(x.min() / x_tick_multiple) * x_tick_multiple,
                            np.ceil(x.max() / x_tick_multiple) * x_tick_multiple,
                            x_tick_multiple,
                        )
                    )
                if y_tick_multiple is not None:
                    ax.set_yticks(
                        np.arange(
                            np.floor(y.min() / y_tick_multiple) * y_tick_multiple,
                            np.ceil(y.max() / y_tick_multiple) * y_tick_multiple,
                            y_tick_multiple,
                        )
                    )

                # highlight points
                if len(highlight_points) > 0:
                    lookup_col = (
                        data.index
                        if highlight_lookup_column == "index"
                        else data[highlight_lookup_column]
                    )
                    labels_col = (
                        data.index
                        if highlight_labels_column == "index"
                        else data[highlight_labels_column]
                    )

                    if any(lookup_col.duplicated()):
                        logging.warning(
                            f"Point labelling lookup column {highlight_lookup_column} contains duplicates. Taking first occurence."
                        )

                    for point in highlight_points:
                        if point not in lookup_col.values:
                            logging.warning(
                                f"Point {point} not found in lookup column {highlight_lookup_column}. Skipping."
                            )
                            continue

                        if collate_label_and_lookup:
                            collated_label = (
                                f"{labels_col.values[lookup_col == point][0]} : {point}"
                            )
                        else:
                            collated_label = labels_col.values[lookup_col == point][0]

                        point_dict = {
                            "x": x[lookup_col == point][0],
                            "y": y[lookup_col == point][0],
                            "label": collated_label,
                        }
                        ax.scatter(
                            point_dict["x"],
                            point_dict["y"],
                            color=highlight_color,
                            s=15,
                            edgecolor="black",
                            lw=1,
                        )

                        # randomly draw from left, right, top, bottom
                        if label_va == "random":
                            va = np.random.choice(["top", "bottom"])
                        else:
                            va = label_va

                        if label_ha == "random":
                            ha = np.random.choice(["left", "right"])
                        else:
                            ha = label_ha

                        if not hide_labels:
                            _parsed_labels = Utils.parse_label(
                                point_dict["label"], label_lookup_dict, label_general_regex
                            )
                            ax.text(
                                point_dict["x"],
                                point_dict["y"],
                                _parsed_labels,
                                color="white",
                                fontsize=fontsize_medium,
                                ha=ha,
                                va=va,
                                path_effects=[
                                    pe.withStroke(linewidth=3, foreground="black")
                                ],
                            )

                # regression line
                if segments == "facets":
                    if linear_regression:
                        lr_x, lr_y, r2, a, b = Utils.regression(
                            x=x[idx], y=y[idx], custom_range=linreg_range
                        )
                        ax.plot(
                            lr_x,
                            lr_y,
                            color=regression_linecolor,
                            lw=linewidth_large,
                            ls="--",
                        )
                        if show_equation:
                            ax.text(
                                0.05,
                                0.85,
                                f"y = {a:.2f}x + {b:.2f}\nR2 = {r2:.2f}",
                                transform=ax.transAxes,
                                fontsize=fontsize_medium,
                                color="grey",
                            )

                elif segments == "color" and linear_regression:
                    lr_x, lr_y, r2, a, b = Utils.regression(
                        x=x, y=y, custom_range=linreg_range
                    )
                    ax.plot(
                        lr_x,
                        lr_y,
                        color=regression_linecolor,
                        lw=linewidth_large,
                        ls="--",
                    )
                    if show_equation:
                        ax.text(
                            0.05,
                            0.85,
                            f"y = {a:.2f}x + {b:.2f}\nR2 = {r2:.2f}",
                            transform=ax.transAxes,
                            fontsize=fontsize_medium,
                        )

                # add r2 scores and coefficient to dictionary
                r2_scores[n] = r2 if linear_regression else None
                coefficients[n] = a if linear_regression else None

                # other lines
                if vline is not None:
                    ax.axvline(vline, color="black", lw=linewidth_large, ls="--")

                if hline is not None:
                    ax.axhline(hline, color="black", lw=linewidth_large, ls="--")

                # add to legend patches
                _legend_patchname_parsed = Utils.parse_label(
                    n, label_lookup_dict, label_general_regex
                )
                legend_patches.append(
                    Patch(
                        facecolor=color_map_dict.get(n, base_color),
                        label=_legend_patchname_parsed,
                        edgecolor=outline_color,
                        linewidth=linewidth_medium,
                    )
                )

                # decoration
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # set aspect ratio
                if aspect_equal:
                    ax.set_aspect("equal")

                # set axis labels for every subplot
                if xlabel is None:
                    _xcol_parsed = Utils.parse_label(
                        x_col, label_lookup_dict, label_general_regex
                    )
                    ax.set_xlabel(_xcol_parsed, fontsize=fontsize_medium)
                else:
                    _xlabel_parsed = Utils.parse_label(
                        xlabel, label_lookup_dict, label_general_regex
                    )
                    ax.set_xlabel(_xlabel_parsed, fontsize=fontsize_medium)

                if i == 0:
                    if ylabel is None:
                        _ycol_parsed = Utils.parse_label(
                            y_col, label_lookup_dict, label_general_regex
                        )
                        ax.set_ylabel(_ycol_parsed, fontsize=fontsize_medium)
                    else:
                        _ylabel_parsed = Utils.parse_label(
                            ylabel, label_lookup_dict, label_general_regex
                        )
                        ax.set_ylabel(_ylabel_parsed, fontsize=fontsize_medium)

                # for facets, set labels if needed
                if segments == "facets" and facet_titles:
                    _facet_title_parsed = Utils.parse_label(
                        n, label_lookup_dict, label_general_regex
                    )
                    ax.set_title(_facet_title_parsed, fontsize=fontsize_medium)

                # set x tick size
                ax.tick_params(axis="x", labelsize=fontsize_medium)
                ax.tick_params(axis="y", labelsize=fontsize_medium)

                # set figure enumeration if needed
                if enumerate_plot is not None and i == 0:
                    _parsed_enumeration = Utils.parse_label(
                        enumerate_plot, label_lookup_dict, label_general_regex
                    )
                    ax.text(
                        enumeration_xy[0],
                        enumeration_xy[1],
                        _parsed_enumeration,
                        horizontalalignment="left",
                        verticalalignment="center",
                        transform=ax.transAxes,
                        fontsize=fontsize_medium,
                        color="black",
                        fontweight="bold",
                    )

                # set suptitle
                if segments == "facets":
                    if title is not None:
                        _suptitle_parsed = Utils.parse_label(
                            title, label_lookup_dict, label_general_regex
                        )
                        plt.suptitle(_suptitle_parsed, fontsize=fontsize_medium)
                elif segments == "color":
                    if title is None:
                        _plot_title = f'{y_col} : {"|".join(colorscale_names)}'
                        _plot_title_parsed = Utils.parse_label(
                            _plot_title, label_lookup_dict, label_general_regex
                        )
                        ax.set_title(_plot_title_parsed, fontsize=fontsize_medium)
                    else:
                        _plot_title_parsed = Utils.parse_label(
                            title, label_lookup_dict, label_general_regex
                        )
                        ax.set_title(_plot_title_parsed, fontsize=fontsize_medium)

            if legend:
                _legend_title_parsed = Utils.parse_label(
                    color_column, label_lookup_dict, label_general_regex
                )
                ax.legend(
                    handles=legend_patches,
                    frameon=False,
                    title=_legend_title_parsed,
                    prop={"size": fontsize_medium},
                    title_fontsize=fontsize_medium,
                    **legend_kwargs,
                )

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            # add diagonal line
            if dline is not None:
                min_left = min(ax.get_xlim()[0], ax.get_ylim()[0])
                max_right = max(ax.get_xlim()[1], ax.get_ylim()[1])
                ax.plot(
                    [min_left, max_right],
                    [min_left, max_right],
                    color="black",
                    lw=linewidth_small,
                    ls="--",
                )

            if show:
                plt.show()

            if return_linear_regression:
                return f, axs, (r2_scores, coefficients)
            else:
                return f, axs

    @staticmethod    
    def annotated_heatmap(
        data: pd.DataFrame,
        sample_metadata: pd.DataFrame = None,
        feature_metadata: pd.DataFrame = None,
        colorbars_x: list = None,
        colorbars_y: list = None,
        cluster_across_y: bool = False,
        show_dendrogram_y: bool = False,
        cluster_across_x: bool = False,
        show_dendrogram_x: bool = False,
        cmap: str = "diverging",
        annotated_rows: bool = True,
        annotated_cols: bool = True,
        show_xlabels: bool = True,
        show_ylabels: bool = True,
        title: str = None,
        label_parse_dict: dict = None,
        label_parse_regex: str = None,
        legends_position: str = "right",
        legend_right_offset: float = 0.25,
        legend_right_lineheight: float = 0.02,
        legend_right_headspace: float = 0,
        enumerate_plot: str = None,
        enumerate_xy: tuple = (-0.35, 1.15),
        vmin: float = -1,
        vmax: float = 1,
        fontsize_medium: int = 8,
        linewidth_medium: float = 0.5,
    ):
        """Wrapper for scanpy's heatmap function

        Simplified input syntax to sns heatmap functionality

        Parameters:

        Returns:

        Example:

        """

        # make fonts editable in illustrator
        plt.rcParams.update(
            {
                "svg.fonttype": "none",
                "font.family": "Arial",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

        # create AnnData object
        adata = ad.AnnData(X=data)

        if sample_metadata is not None:
            adata.obs = sample_metadata

        if feature_metadata is not None:
            adata.var = feature_metadata

        # compute linkage for horizontal dendrogram
        x_linkage = None
        if cluster_across_x:
            x_linkage = linkage(adata.X.T, method="ward")
            # scale for equal sized steps
            n_samples = adata.X.shape[1]
            x_linkage[:, 2] = np.arange(1, n_samples) / (n_samples - 1)

        # compute linkage for vertical dendrogram
        y_linkage = None
        if cluster_across_y:
            y_linkage = linkage(adata.X, method="ward")
            # scale for equal sized steps
            n_samples = adata.X.shape[0]
            # y_linkage[:, 2] = np.linspace(0, 1, len(y_linkage))

        # specify palette based on shuffled spectral palette
        _pal = sns.color_palette("Spectral", 11).as_hex()
        color_palette = [_pal[i] for i in [0, 9, 10, 3, 5, 1, 8, 2, 7]]
        
        # map colors to colorbars
        color_x_frame = None
        if colorbars_x is not None:
            unique_values = adata.var[colorbars_x].apply(pd.unique)
            sorted_unique_values = sorted(pd.Series(np.concatenate(unique_values.values)).fillna('NA').unique().tolist())
            color_dict = {key: col for key, col in zip(sorted_unique_values, cycle(color_palette))}
            colorbars_x_reference_dict = {
                x: {y: color_dict[y] for y in unique_values[x]} for x in colorbars_x
            }
            color_x_frame = pd.DataFrame(
                {
                    x: adata.var[x].map(colorbars_x_reference_dict[x])
                    for x in colorbars_x
                }
            )
        else:
            colorbars_x_reference_dict = dict()

        color_y_frame = None
        if colorbars_y is not None:
            unique_values = adata.obs[colorbars_y].apply(pd.unique)
            sorted_unique_values = sorted(pd.Series(np.concatenate(unique_values.values)).fillna('NA').unique().tolist())
            color_dict = {key: col for key, col in zip(sorted_unique_values, cycle(color_palette))}
            colorbars_y_reference_dict = {
                x: {y: color_dict[y] for y in unique_values[x]} for x in colorbars_y
            }
            color_y_frame = pd.DataFrame(
                {
                    x: adata.obs[x].map(colorbars_y_reference_dict[x])
                    for x in colorbars_y
                }
            )
        else:
            colorbars_y_reference_dict = dict()

        # combine colorbar reference dictionaries
        colorbars_reference_dict = (
            colorbars_x_reference_dict | colorbars_y_reference_dict
        )

        # set colorbar scale
        cmap = "vlag"

        # if each row/column of the heatmap should have readable labels,
        # we must set the size such that the fontsize is accomodated.

        # set sensible default heatmap dimensions
        heatmap_height = 5
        heatmap_width = 5

        # convert fontsize to inches for matplotlib
        fontsize_inches = fontsize_medium / 72
        if annotated_rows:
            row_height = fontsize_inches * 2
            heatmap_height = adata.X.shape[0] * row_height
        else:
            row_height = heatmap_height / adata.X.shape[0]

        if annotated_cols:
            col_width = fontsize_inches * 2
            heatmap_width = adata.X.shape[1] * col_width
        else:
            col_width = heatmap_width / adata.X.shape[1]

        # ensure that the heatmap is not too large
        if heatmap_height > 20:
            print(
                "Heatmap height exceeds 20 inches. This is likely to cause issues with rendering."
            )
        if heatmap_width > 20:
            print(
                "Heatmap width exceeds 20 inches. This is likely to cause issues with rendering."
            )

        # set sensible margins for annotations (dendrogram, legend, names)
        plot_height = heatmap_height + 1.5  # legend
        plot_width = heatmap_width

        if cluster_across_y:
            plot_height += 1
            if show_xlabels:
                plot_height += 1
        if cluster_across_x:
            plot_width += 1
            if show_ylabels:
                plot_width += 1

        # get dataframe from anndata object
        adata_data = pd.DataFrame(
            data=adata.X, index=adata.obs_names, columns=adata.var_names
        )

        # plot sns clustermap
        hm = sns.clustermap(
            data=adata_data,
            cmap=cmap,
            row_colors=color_y_frame if colorbars_y is not None else None,
            col_colors=color_x_frame if colorbars_x is not None else None,
            row_linkage=y_linkage if cluster_across_y else None,
            col_linkage=x_linkage if cluster_across_x else None,
            figsize=(plot_width, plot_height),
            vmin=vmin,
            vmax=vmax,
            dendrogram_ratio=0.1,
            # by default, if we want to see labels we want all of them
            yticklabels=1,
            xticklabels=1,
        )

        # parse heatmap tick labels and set tick label size and tick line width
        yaxis_hm_labels_parsed = [
            Utils.parse_label(L.get_text(), label_parse_dict, label_parse_regex)
            for L in hm.ax_heatmap.get_yticklabels()
        ]
        hm.ax_heatmap.set_yticklabels(yaxis_hm_labels_parsed)

        xaxis_hm_labels_parsed = [
            Utils.parse_label(L.get_text(), label_parse_dict, label_parse_regex)
            for L in hm.ax_heatmap.get_xticklabels()
        ]
        hm.ax_heatmap.set_xticklabels(xaxis_hm_labels_parsed)
        hm.ax_heatmap.tick_params(
            labelsize=fontsize_medium, width=linewidth_medium
        )

        # set tick label size and tick line width for heatmap
        hm.ax_heatmap.tick_params(axis="x", labelsize=fontsize_medium)
        hm.ax_heatmap.tick_params(axis="y", labelsize=fontsize_medium)

        # hide labels if not needed
        if not show_xlabels:
            hm.ax_heatmap.set_xticks([])
            hm.ax_heatmap.set_xlabel("")
        if not show_ylabels:
            hm.ax_heatmap.set_yticks([])
            hm.ax_heatmap.set_ylabel("")

        # set heatmap position fixed, all other plot elements are positioned relative to this!
        heatmap_position = [0.1, 0.1, 0.8, 0.8]
        hm.ax_heatmap.set_position(heatmap_position)

        # get the width of one column of the heatmap in figure units
        heatmap_bottom = hm.ax_heatmap.get_position().y0
        heatmap_left = hm.ax_heatmap.get_position().x0
        heatmap_width = hm.ax_heatmap.get_position().width
        heatmap_height = hm.ax_heatmap.get_position().height

        # set colorbar columns, position off to the left
        if color_x_frame is not None:
            hm.ax_col_colors.tick_params(
                axis="y", labelsize=fontsize_medium, width=linewidth_medium
            )
            yaxis_bar_labels_parsed = [
                Utils.parse_label(L.get_text(), label_parse_dict, label_parse_regex)
                for L in hm.ax_col_colors.get_yticklabels()
            ]
            hm.ax_col_colors.set_yticklabels(yaxis_bar_labels_parsed)
            hm.ax_col_colors.spines[["right", "top", "bottom", "left"]].set_visible(
                True
            )
            hm.ax_col_colors.set_position(
                [
                    heatmap_left,
                    (heatmap_bottom + heatmap_height + (row_height / 10)),
                    heatmap_width,
                    hm.ax_col_colors.get_position().height,
                ]
            )
            color_x_top = hm.ax_col_colors.get_position().y1
        else:
            color_x_top = heatmap_bottom + heatmap_height

        # set colorbar rows, position off to the top
        if color_y_frame is not None:
            hm.ax_row_colors.xaxis.set_ticks_position("top")
            xaxis_bar_labels_parsed = [
                Utils.parse_label(L.get_text(), label_parse_dict, label_parse_regex)
                for L in hm.ax_row_colors.get_xticklabels()
            ]
            hm.ax_row_colors.set_xticklabels(xaxis_bar_labels_parsed)
            hm.ax_row_colors.tick_params(
                axis="x",
                rotation=90,
                labelsize=fontsize_medium,
                width=linewidth_medium,
            )
            hm.ax_row_colors.spines[["right", "top", "bottom", "left"]].set_visible(
                True
            )
            hm.ax_row_colors.set_position(
                [
                    (heatmap_left - hm.ax_row_colors.get_position().width)
                    - (col_width / 10),
                    heatmap_bottom,
                    hm.ax_row_colors.get_position().width,
                    heatmap_height,
                ]
            )
            color_y_left = hm.ax_row_colors.get_position().x0
        else:
            color_y_left = heatmap_left

        # position dendrograms
        if show_dendrogram_y:
            hm.ax_row_dendrogram.set_position(
                [
                    color_y_left - 0.1,
                    heatmap_bottom,
                    0.1,
                    heatmap_height,
                ]
            )
            dendrogram_y_left = hm.ax_row_dendrogram.get_position().x0
        else:
            dendrogram_y_left = heatmap_left

        if show_dendrogram_x:
            hm.ax_col_dendrogram.set_position(
                [
                    heatmap_left,
                    color_x_top,
                    heatmap_width,
                    0.1,
                ]
            )

        # hide dendrograms if not needed
        hm.ax_row_dendrogram.set_visible(show_dendrogram_y)
        hm.ax_col_dendrogram.set_visible(show_dendrogram_x)

        # set colorbar position on the left side of the heatmap
        hm.cax.set_position(
            [(dendrogram_y_left - 0.075), heatmap_bottom, (0.025), heatmap_height]
        )
        hm.cax.yaxis.set_ticks_position("left")

        # set colorbar tick label size and tick line width
        hm.ax_cbar.tick_params(
            labelsize=fontsize_medium, width=linewidth_medium
        )

        # add Patch legend for colorbars: requires dictionary of values
        if len(colorbars_reference_dict) > 0:
            # iterate over colorbars and make all legends, then place them with appropriate spacing
            legend_widths = []
            legend_heights = []
            for k, v in colorbars_reference_dict.items():
                # extract patches and labels from color dictionary
                _hs, _ls, _lt = Utils.multilegend_patch_generator(
                    {k: v}, single_column=False
                )
                ls_p = [
                    Utils.parse_label(L, label_parse_dict, label_parse_regex)
                    for L in _ls
                ]
                lt_p = Utils.parse_label(_lt, label_parse_dict, label_parse_regex)
                hml = hm.figure.legend(
                    _hs,
                    ls_p,
                    frameon=False,
                    bbox_to_anchor=(0, 0),
                    loc="upper left",
                    bbox_transform=hm.figure.transFigure,
                    prop={"size": fontsize_medium},
                    title=lt_p,
                    title_fontsize=fontsize_medium,
                )

                # get width and height of legend
                legend_widths.append(hml.get_window_extent().width)
                legend_heights.append(hml.get_window_extent().height)

                # remove legend again
                hml.remove()

            # get maximum height of all legends
            max_legend_height_relative = (max(legend_heights) / 350) / heatmap_height

            # iterate over colorbars and plot and individual legend for each one
            for i, ((k, v), lh, lw) in enumerate(
                zip(colorbars_reference_dict.items(), legend_heights, legend_widths)
            ):
                # extract patches and labels from color dictionary
                hs, ls, legend_title = Utils.multilegend_patch_generator(
                    {k: v}, pad_max_length=False, single_column=False
                )

                # parse all strings
                ls_parsed = [
                    Utils.parse_label(L, label_parse_dict, label_parse_regex)
                    for L in ls
                ]
                legend_title_parsed = Utils.parse_label(
                    legend_title, label_parse_dict, label_parse_regex
                )

                # position the first legend and subsequently add the others below
                lh_relative = (lh / 300) / heatmap_height
                lw_relative = (lw / 250) / heatmap_width

                # decide legend positions
                if legends_position == "right":
                    legend_alignment = "lower left"
                    legend_x = (heatmap_left + heatmap_width) + legend_right_offset
                    if i == 0:
                        legend_y = (
                            heatmap_bottom + heatmap_height
                        ) - legend_right_headspace
                    else:
                        spacing_y = lh_relative + legend_right_lineheight
                        legend_y = legend_y - spacing_y

                elif legends_position == "top":
                    legend_alignment = "upper left"
                    legend_x = heatmap_left if i == 0 else legend_x + lw_relative
                    legend_y = (
                        (heatmap_bottom + heatmap_height)
                        + max_legend_height_relative
                        + 0.1
                    )

                # plot legend
                hml = hm.figure.legend(
                    hs,
                    ls_parsed,
                    frameon=False,
                    bbox_to_anchor=(legend_x, legend_y),
                    loc=legend_alignment,
                    bbox_transform=hm.figure.transFigure,
                    prop={"size": fontsize_medium},
                    title=legend_title_parsed,
                    title_fontsize=fontsize_medium,
                )

        # enumerate plot
        if enumerate_plot is not None:
            _parsed_enumeration = Utils.parse_label(
                enumerate_plot, label_parse_dict, label_parse_regex
            )
            hm.ax_heatmap.text(
                enumerate_xy[0],
                enumerate_xy[1],
                _parsed_enumeration,
                horizontalalignment="left",
                verticalalignment="center",
                transform=hm.ax_heatmap.transAxes,
                fontsize=fontsize_medium,
                color="black",
                fontweight="bold",
            )

        # add title if necesseary
        if title is not None:
            title_parsed = Utils.parse_label(title, label_parse_dict, label_parse_regex)
            plt.suptitle(title_parsed, fontsize=fontsize_medium)

        # return heatmap object and dendrogram linkages
        if not cluster_across_y:
            x_linkage = None
        if not cluster_across_x:
            y_linkage = None

        return hm, x_linkage, y_linkage
        
    @staticmethod
    def save_figure(
        fig,
        filename="figure.png",
        output_dir: str = "../assets/figures",
        dpi: int = 300,
        width_mm: int = None,
        height_mm: int = None,
        paper_width: str = "single",  
        paper_height: str = "single",  
        transparency: bool = True,
    ):
        """

        Save figure to output_dir/filename. Set specific width in order to make the figure fit in a column of a scientific paper

        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # preset sizes for columns on pages
        fig_size_dict = {
            "single": 89,
            "double": 183,
            "triple": 277,
            "1.5": 135,
            "0.5": 45,
            "0.25": 22.5,
            "0.75": 67.5,
        }

        set_width = width_mm
        set_height = height_mm

        if width_mm is None:
            set_width = fig_size_dict[paper_width]
        if height_mm is None:
            set_height = fig_size_dict[paper_height]

        effective_width = set_width / 25.4
        effective_height = set_height / 25.4

        fig.set_size_inches(effective_width, effective_height)

        fig.savefig(
            os.path.join(output_dir, filename),
            bbox_inches="tight",
            dpi=dpi,
            transparent=transparency,
        )