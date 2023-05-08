# Author: Hartaig Singh

import pandas as pd
import numpy as np


class MMMPreprocessing:

    def balance(self, df: pd.DataFrame, group_by: str, date: str, segments: [] = None) -> \
            pd.DataFrame:
        """
        Balances data frame so each segment has the same number of months. Missing months will get
        filled with zeros
        """
        # get range of dates
        dates = pd.period_range(start=min(df[date]), end=max(df[date]), freq='M').tolist()
        # get list of unique npis
        npi = np.unique(df[group_by])
        # create group_by, date level dataframe where each npi has the same months
        df_out = pd.DataFrame({group_by: np.repeat(npi, len(dates)), date: dates * len(npi)})
        df_out[date] = df_out[date].astype(str)
        # merge in original data
        df_out = df_out.merge(df, on=[group_by, date], how='left')
        # if static hcp level columns exist, fill the column with original values
        if segments is not None:
            for i in segments:
                df_out[i] = df_out.groupby([group_by])[i].bfill()
                df_out[i] = df_out.groupby([group_by])[i].ffill()
        # replace nan with 0 since nan is the lack of occurrence of media for that time point
        df_out = df_out.fillna(0)
        return df_out

    def one_hot(self, df: pd.DataFrame, segment: str) -> pd.DataFrame:
        """
        Performance one-hot encoding on specified feature column. Necessary for some tree based
        models
        """
        df_out = pd.concat([df, pd.get_dummies(df[[segment]])], axis=1)
        return df_out

    def top_segment(self, df: pd.DataFrame, segment: str, group_by: str, top_n: int) -> pd.DataFrame:
        """
        """
        df_out = df.copy()
        aggr = df_out.groupby(segment)[group_by].nunique().reset_index()
        other_spec = aggr[aggr[group_by] <= top_n][segment]
        df_out[segment] = np.where(df_out[segment].isin(other_spec), 'Other', df_out[segment])
        return df_out

