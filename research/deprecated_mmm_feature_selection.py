# Author: Hartaig Singh

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from tqdm import tqdm


class MMMFeatureSelection(object):
    def _results(self, df: pd.DataFrame):
        """
        Private function to aggregate and return optimal transformed features after model fitting
        """
        results = df.copy()
        results.reset_index(drop=True, inplace=True)
        return results.loc[results.groupby('type')['avg_rmse'].idxmin()]

    def rf(self, df: pd.DataFrame, dv_col: str, transformed_cols: [], cv: int = 10) -> dict:
        """
        Fits a random forest model using 5-fold cv to obtain an average performance score for
        each transformed feature. For each media channel, the feature with the best average
        performance is used for the final model fit.
        """
        df_fs = df.copy()
        results = pd.DataFrame()
        for i in transformed_cols:
            model = RandomForestRegressor()
            X = df_fs[[i]]
            y = df_fs[[dv_col]]
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
            avg_score = pd.DataFrame(
                {'feature': [i], 'type': [i.split('adstock')[0].replace('_', '')],
                 'avg_rmse': [np.mean(scores * -1)]})
            results = pd.concat([results, avg_score])
        results_final = self._results(results)
        return {'best_feats': results_final, 'avg_rmse': results}

    def best_feat_distribution(self, df: pd.DataFrame, transformed_cols: [], n: int = 100, cv: int = 10):
        """
        """
        df_out = pd.DataFrame()
        for i in tqdm(range(0, n)):
            results = self.rf(df, transformed_cols, cv)
            df_out = pd.concat([df_out, results['best_feats']])
        df_out.reset_index(inplace=True)
        return df_out.groupby(['type', 'feature']).size().reset_index(name='counts')



