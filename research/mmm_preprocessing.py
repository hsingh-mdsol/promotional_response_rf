# Author: Hartaig Singh

import pandas as pd
import numpy as np


class MMMPreprocessing:

    def remove(self, df: pd.DataFrame, group_by: str, dv: str, channels: []) -> pd.DataFrame:
        """
        """
        df_g = df.groupby(group_by).max().reset_index()
        df_g['drop'] = np.where(df_g[[dv] + channels].sum(axis=1) == 0, 1, 0)
        drop_hcps = df_g[df_g['drop'] == 1][group_by]
        return df[~df[group_by].isin(drop_hcps)]

    def balance(self, df: pd.DataFrame, group_by: str, date: str, seg_cols=None) -> pd.DataFrame:
        """
        Balances data frame so each segment has the same number of months. Missing months will get
        filled with zeros
        """
        if seg_cols is None:
            seg_cols = []
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
        if seg_cols is not None:
            for i in seg_cols:
                df_out[i] = df_out.groupby([group_by])[i].bfill()
                df_out[i] = df_out.groupby([group_by])[i].ffill()
        # replace nan with 0 since nan is the lack of occurrence of media for that time point
        return df_out.fillna(0)

    def one_hot(self, df: pd.DataFrame, seg_cols: []) -> pd.DataFrame:
        """
        Performance one-hot encoding on specified feature column. Necessary for some tree based
        models
        """
        df_out = df.copy()
        for i in seg_cols:
            df_out = pd.concat([df_out, pd.get_dummies(df_out[[i]])], axis=1)
        return df_out

    def top_segment(self, df: pd.DataFrame, seg_cols: [], metric: str, top_n: int) -> pd.DataFrame:
        """
        """
        df_out = df.copy()
        for i in seg_cols:
            aggr = df_out.groupby(i)[metric].nunique().reset_index()
            other_spec = aggr[aggr[metric] <= top_n][i]
            df_out[i] = np.where(df_out[i].isin(other_spec), 'Other', df_out[i])
        return df_out

    def hcp_time_agg(self, df: pd.DataFrame, group_by: str, date: str, dv: str, channels: [],
                     seg_cols=None) -> pd.DataFrame:
        """
        """
        if seg_cols is None:
            seg_cols = []
        df_out = df[[group_by, date, dv] + channels + seg_cols]
        df_out = self.balance(df_out, group_by, date, seg_cols)
        if seg_cols is not None:
            df_out = self.top_segment(df_out, seg_cols, group_by, 15)
            df_out = self.one_hot(df_out, seg_cols)
        return df_out

    def segment_time_agg(self, df: pd.DataFrame, segment: str, date: str, dv: str, channels: []) \
            -> pd.DataFrame:
        """
        """
        df_out = df[[segment, date, dv] + channels].groupby([segment, date]).sum().reset_index(). \
            sort_values([segment, date])
        df_out = self.top_segment(df_out, [segment], date, 15)
        # ensure data is at segment, date level
        df_out = df_out.groupby([segment, date]).sum().reset_index()
        # balance
        df_out = self.balance(df_out, segment, date)
        # one hot encode segments
        df_out = self.one_hot(df_out, [segment])
        return df_out

    def aggregation(self, df: pd.DataFrame, group_by: str, date: str, dv: str, channels: [],
                    seg_cols=None) -> dict:
        """
        """
        final_dct = {}
        if seg_cols is None:
            seg_cols = []
        df_out = df.copy()
        # hcp, time level aggregation
        df_hcp_time = self.hcp_time_agg(df_out, group_by, date, dv, channels, seg_cols)
        final_dct.update({'hcp_time': df_hcp_time})
        # segment time
        if seg_cols:
            for i in seg_cols:
                df_seg_time = self.segment_time_agg(df_out, i, date, dv, channels)
                final_dct.update({f"segment_time_{i}": df_seg_time})
                # time filtered by segment
                seg_filt = {}
                drop_cols = [x for x in df_seg_time.columns if x.startswith(f"{i}_")]
                for s in np.unique(df_seg_time[i]):
                    df_filt = df_seg_time[df_seg_time[i] == s].drop(drop_cols, axis=1)
                    seg_filt.update({f"time_{s}": df_filt})
                final_dct.update({'time_filtered_on_segment': seg_filt})
        return final_dct
