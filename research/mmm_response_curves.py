# Author: Hartaig Singh

import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


class MMMResponseCurves(object):
    def _scaling(self, x: [], lb: int = 0, ub: int = 1):
        """
        """
        scaler = MinMaxScaler(feature_range=(lb, ub))
        x_scaled = scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten()
        return {'scaler': scaler, 'scaled_vals': x_scaled}

    def _response_curve_hill(self, x: [], beta: float, k: float, s: float) -> []:
        """
        k (or gamma) shifts inflection point, s (or alpha) controls c or s shape
        """
        return beta * (1 / (1 + (x / k) ** (-s)))

    def prediction_interval(self, x: pd.DataFrame, y: np.array, n_samples: int = 100,
                            lwr_perc: float = 2.5, upr_perc: float = 97.5) -> np.array:
        """
        """
        # Generate bootstrapped predictions for each test observation
        bootstrapped_preds = np.zeros((x.shape[0], n_samples))
        # this will create a 120 x 100 array where each row is the 100 predictions of y at that row
        for i in tqdm(range(n_samples)):
            bootstrapped_samples = np.random.choice(x.shape[0], size=x.shape[0], replace=True)
            X_bootstrapped = x.iloc[bootstrapped_samples]
            y_bootstrapped = y[bootstrapped_samples]
            rf_bootstrapped = RandomForestRegressor()
            rf_bootstrapped.fit(X_bootstrapped, y_bootstrapped)
            bootstrapped_preds[:, i] = rf_bootstrapped.predict(x)
        # Calculate the prediction interval for each test observation
        lower_bounds = np.percentile(bootstrapped_preds, lwr_perc, axis=1)
        upper_bounds = np.percentile(bootstrapped_preds, upr_perc, axis=1)
        return np.column_stack((lower_bounds, upper_bounds))

    def error_propagation(self, pred_intervals: np.array, prop_type: str = 'mean'):
        """
        """
        if prop_type == 'mean':
            errs = pred_intervals[:, 1] - pred_intervals[:, 0]
            agg_error = np.sqrt(sum(errs ** 2) / len(errs))
        return agg_error

    def plot(self, df: pd.DataFrame, x: str, y: []):
        """
        """
        fig = px.line(df, x=x, y=df[[x] + y].columns)
        fig.update_layout(yaxis_title="Rx Impact")
        return fig

    def _response_aggr_errors(self, df: pd.DataFrame, feature: str, max_freq: int,
                              increment: int) -> pd.DataFrame:
        """
        """
        # create final response data frame
        resp_final = pd.DataFrame({'touches': np.arange(0, max_freq + 1, increment)})
        # average predictions at each frequency
        pred_cols = [x for x in df.columns if ('preds' in x) & ('lb' not in x) & ('ub' not in x)]
        mean_pred_err = []
        for i in pred_cols:
            mean_pred_err.append(self.error_propagation(np.column_stack((df[f"{i}_lb"],
                                                                         df[f"{i}_ub"]))))
        # add to final output data frame
        responses = pd.DataFrame({f"{feature}_errors": mean_pred_err})
        return pd.concat([resp_final, responses], axis=1)

    def _response_aggr(self, df: pd.DataFrame, feature: str, max_freq: int, increment: int) -> []:
        """
        """
        # create final response data frame
        resp_final = pd.DataFrame({'touches': np.arange(0, max_freq + 1, increment)})
        # average predictions at each frequency
        pred_cols = [x for x in df.columns if ('preds' in x) & ('lb' not in x) & ('ub' not in x)]
        mean_pred = []
        for i in pred_cols:
            mean_pred.append(np.mean(df[i]))
        # add to final output data frame
        responses = pd.DataFrame({feature: mean_pred})
        resp_final = pd.concat([resp_final, responses], axis=1)
        resp_og = resp_final.copy()
        # to get attributions remove response at 0 from all other responses
        resp_final = resp_final - resp_final.iloc[0]
        # sometimes response at 0 could be greater than first few frequency responses - replace w/ 0
        resp_final[resp_final < 0] = 0
        # fit hill estimates and output final response data frame
        touches = self._scaling(resp_final['touches'])
        y_axis = self._scaling(resp_final[feature])
        popt = self._hill_fitting(touches['scaled_vals'], y_axis['scaled_vals'])
        # create hill curve
        hill_est = np.array(
            self._response_curve_hill(touches['scaled_vals'], popt[0], popt[1], popt[2])).reshape(
            -1, 1)
        # convert hill curve to original scale
        resp_final[f"{feature}_hill_estimate"] = y_axis['scaler'].inverse_transform(
            hill_est).flatten()
        # refit hill on final hill curve to obtain parameter estimates on the correct scale
        popt = self._hill_fitting(resp_final['touches'], resp_final[f"{feature}_hill_estimate"])
        resp_final["touches_scaled"] = touches['scaled_vals']
        resp_final[f"{feature}_hill_estimate_minmax"] = resp_final[f"{feature}_hill_estimate"] / \
                                                        (max(resp_final[f"{feature}_hill_estimate"])
                                                         - min(resp_final[
                                                                   f"{feature}_hill_estimate"]))
        # if errors specified
        if len([x for x in df.columns if ('lb' in x) | ('ub' in x)]) > 0:
            err_df = self._response_aggr_errors(df, feature, max_freq, increment)
            resp_og = resp_og.merge(err_df, on='touches', how='left')
        return [resp_final, popt, resp_og]

    def _hill_fitting(self, x: [], y: []):
        """
        """
        try:
            popt, pcov = curve_fit(self._response_curve_hill, x, y, bounds=(0, [np.inf, np.inf,
                                                                                np.inf]))
        except RuntimeError:
            print("Error - curve_fit failed")
            popt = [0, 0, 0]
        return popt

    def responses(self, model, x: pd.DataFrame, feature: str, max_freq: int,
                  increment: int, **errors) -> dict:
        """
        """
        df_preds = x.copy()
        # get predictions
        for i in tqdm(np.arange(0, max_freq + 1, increment)):
            df_sim = x.copy()
            # replace channel and its lags with desired frequency
            df_sim[[feature] + [c for c in x.columns if f"{feature}_lag" in c]] = i
            df_preds[f"{feature}_preds_{i}"] = model.predict(df_sim)
            if errors:
                errs = self.prediction_interval(df_sim, errors["y"], errors["samples"])
                df_preds[f"{feature}_preds_{i}_lb"] = errs[:, 0]
                df_preds[f"{feature}_preds_{i}_ub"] = errs[:, 1]
        # aggregate predictions
        resp_final, popt, resp_og = self._response_aggr(df_preds, feature, max_freq, increment)
        return {'resp_df': resp_final, 'resp_og': resp_og, 'optimal_hill': popt}

    def responses_segment(self, model, x: pd.DataFrame, feature: str, max_freq: int, increment: int,
                          segment: str) -> dict:
        """
        """
        segments = [s for s in x.columns if s.startswith(f'{segment}_')]
        resp_final = pd.DataFrame({'touches': range(0, max_freq + 1, increment)})
        # get predictions
        for s in segments:
            pred = []
            for i in range(0, max_freq + 1, increment):
                df_sim = x.copy()
                df_sim[s] = 1
                df_sim[[feature] + [c for c in x.columns if f"{feature}_lag" in c]] = i
                pred.append(np.mean(model.predict(df_sim.drop([segment], errors='ignore', axis=1))))
            resp_final = pd.concat([resp_final, pd.DataFrame({f"{s}_{feature}": pred})], axis=1)
        resp_og = resp_final.copy()
        # aggregate predictions
        resp_final = resp_final - resp_final.iloc[0]
        resp_final[resp_final < 0] = 0
        # hill response curve
        opt_hill = {}
        raw_pred_cols = resp_final.drop(['touches'], axis=1).columns.tolist()
        for i in raw_pred_cols:
            touches = self._scaling(resp_final['touches'])
            y_axis = self._scaling(resp_final[i])
            popt = self._hill_fitting(touches['scaled_vals'], y_axis['scaled_vals'])
            hill_est = np.array(self._response_curve_hill(touches['scaled_vals'],
                                                          popt[0], popt[1], popt[2])).reshape(-1, 1)
            resp_final[f"{i}_hill_estimate"] = y_axis['scaler'].inverse_transform(
                hill_est).flatten()
            # refit hill on final hill curve to obtain parameter estimates on the correct scale
            popt = self._hill_fitting(resp_final['touches'], resp_final[f"{i}_hill_estimate"])
            resp_final[f"{i}_hill_estimate_minmax"] = resp_final[f"{i}_hill_estimate"] / \
                                                      (max(resp_final[f"{i}_hill_estimate"])
                                                       - min(resp_final[f"{i}_hill_estimate"]))
            opt_hill.update({i: popt})
        # plot final curve
        fig_hill = self.plot(resp_final, x='touches',
                             y=[x for x in resp_final.columns if ("hill" in x) & ('min' not in x)])
        fig_raw = self.plot(resp_final, x='touches', y=raw_pred_cols)
        # final out
        return {'resp_df': resp_final, 'optimal_hill': opt_hill, 'fig_hill': fig_hill,
                'fig_raw': fig_raw, 'resp_og': resp_og}
