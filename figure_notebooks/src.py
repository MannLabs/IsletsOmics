# Helper functions for IsletsOmics analysis

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from biomart import BiomartServer
from scipy.stats import false_discovery_control, pearsonr, ttest_ind, gaussian_kde
import warnings
import logging
import matplotlib.patheffects as pe
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.patches import Patch
from itertools import cycle

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
                    mappings[ensembl_id] = uid
                    break

        # coalesce the mappings: if no uniprot id is found, use the ensemble id
        if coalesce:
            for e in list(mappings.keys()):
                if not mappings[e]:
                    mappings[e] = e

        return mappings
    
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