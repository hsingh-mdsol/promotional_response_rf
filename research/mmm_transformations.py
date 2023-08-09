# Author: Hartaig Singh

import pandas as pd
import numpy as np
from scipy.stats import halfnorm


class MMMTransformations(object):
    def add_noise(self, df: pd.DataFrame, columns: [], perc: float = 0.05):
        df_out = df.copy()
        for i in columns:
            df_samp = df_out.sample(frac=perc)
            indx = df_samp.reset_index()['index']
            repl_with = halfnorm.rvs(loc=0, scale=np.std(df_out[i]), size=len(df_samp))
            df_out.loc[indx, i] = repl_with
        return df_out

    def geometric_adstock(self, data: [], decay_rate: float, max_duration: int = 3) -> []:
        """
        Performs geometric adstock on a list of values given a specified decay rate and max time steps for the lookback
        """
        adstock_values = []
        for i in range(len(data)):
            adstock_value = data[i]
            for j in range(1, min(i + 1, max_duration + 1)):
                adstock_value += data[i - j] * decay_rate ** j
            adstock_values.append(adstock_value)
        return adstock_values

    def segment_adstock(self, df: pd.DataFrame, cols_to_adstock: str, segment: str,
                        decay_rates: float):
        segments = np.unique(df[segment].tolist())
        segment_adstock = []
        for i in segments:
            df_filt = df[df[segment] == i]
            adstock = self.geometric_adstock(df_filt[cols_to_adstock].tolist(), decay_rates, 3)
            segment_adstock = segment_adstock + adstock
        return segment_adstock

    def segment_adstock_df(self, df: pd.DataFrame, cols_to_adstock: [], segment: str,
                           decay_rates: []) -> pd.DataFrame:
        """
        Performs segment level geometric adstock.
        """
        df_ads = df.copy()
        for i in cols_to_adstock:
            for j in decay_rates:
                segment_adstock = self.segment_adstock(df_ads, i, segment, j)
                df_ads[f"{i}_adstock_{j}"] = segment_adstock
        return df_ads

    def lag_dv(self, df: pd.DataFrame, dv: str, periods: int = 3,
               group: str = None) -> pd.DataFrame:
        """
        """
        df_lag = df.copy()
        for i in range(1, periods + 1):
            if group is None:
                df_lag[f"{dv}_lag{i}"] = df_lag[dv].shift(i, fill_value=0)
            else:
                df_lag[f"{dv}_lag{i}"] = df_lag.groupby(group)[dv].shift(i, fill_value=0)
        return df_lag

    def winsorize(self, df: pd.DataFrame, cols: [], percentile: int = 95) -> pd.DataFrame:
        """
        """
        df_out = df.copy()
        for i in cols:
            threshold = np.percentile(df_out[i], percentile)
            df_out[i] = np.where(df_out[i] > threshold, threshold, df_out[i])
        return df_out
