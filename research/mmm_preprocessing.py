# Author: Hartaig Singh

import pandas as pd
import numpy as np


class MMMPreprocessing:
    def __init__(self, group_by: str, segment: str, date: str, media_cols: [], dv: str):
        self.group_by = group_by
        self.segment = segment
        self.date = date
        self.media_cols = media_cols
        self.dv = dv

    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balances data frame so each segment as the same number of months. Missing months will get filled with zeros
        """
        # fill missing timestamps at day level with zeros for each segment
        df_out = df.set_index([self.date, self.group_by]).unstack(fill_value=0).asfreq(
            'D', fill_value=0).stack().sort_index(level=1).reset_index()
        # convert timestamp column to datetime
        df_out[self.date] = df_out[self.date].dt.to_period("M")
        # roll up to month level
        df_out = df_out.groupby([self.group_by, self.date]).sum().reset_index()
        return df_out

    def top_segment(self, df: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """
        """
        df_out = df.copy()
        aggr = df_out.groupby(self.segment)[self.date].count().reset_index()
        other_spec = aggr[aggr[self.date] <= top_n][self.segment]
        df_out[self.segment] = np.where(df_out[self.segment].isin(other_spec), 'Other', df_out[self.segment])
        return df_out

    def one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performance one-hot encoding on specified feature column. Necessary for some tree based models
        """
        df_out = pd.concat([df, pd.get_dummies(df[[self.segment]])], axis=1)
        return df_out

    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs proper groupby and sorts by date within segments
        """
        df_out = df.copy()
        df_out[self.date] = pd.to_datetime(df_out[self.date])
        # encode only top segments
        df_out = self.top_segment(df_out, 15)
        # ensure data is at segment, date level
        df_out = df_out.groupby([self.segment, self.date]).sum().reset_index()
        # balance
        df_out = self.balance(df_out)
        # one hot encode segments
        df_out = self.one_hot(df_out)
        return df_out
