# Helper functions for IsletsOmics analysis

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from biomart import BiomartServer

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