# Author: Hartaig Singh

import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler


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
        imp = beta * (1 / (1 + (x / k) ** (-s)))
        return imp

    def plot(self, df: pd.DataFrame, x: str, y: []):
        """
        """
        fig = px.line(df, x=x, y=df[[x] + y].columns)
        fig.update_layout(yaxis_title="Rx Impact")
        return fig

    def _response_aggr(self, df: pd.DataFrame, feature: str, max_freq: int, increment: int) -> []:
        """
        """
        # create final response data frame
        resp_final = pd.DataFrame({'touches': range(0, max_freq + 1, increment)})
        # average predictions at each frequency
        pred_cols = [x for x in df.columns if 'preds' in x]
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
        resp_final, popt, touches, y_axis = self._hill_fitting(feature, resp_final)
        return [resp_final, popt, touches['scaler'], y_axis['scaler'], resp_og]

    def _hill_fitting(self, feature: str, resp_final: pd.DataFrame):
        """
        """
        # hill response curve
        touches = self._scaling(resp_final['touches'])
        y_axis = self._scaling(resp_final[feature])
        # fill hill on scaled x and y
        try:
            popt, pcov = curve_fit(self._response_curve_hill, touches['scaled_vals'],
                                   y_axis['scaled_vals'],
                                   bounds=(0, [np.inf, np.inf, np.inf]))
        except RuntimeError:
            print("Error - curve_fit failed")
            popt = [0, 0, 0]
        # create hill curve
        hill_est = np.array(
            self._response_curve_hill(touches['scaled_vals'], popt[0], popt[1], popt[2])).reshape(
            -1, 1)
        # convert hill curve to original scale
        resp_final[f"{feature}_hill_estimate"] = y_axis['scaler'].inverse_transform(
            hill_est).flatten()
        # refit hill on final hill curve to obtain parameter estimates on the correct scale
        popt, pcov = curve_fit(self._response_curve_hill, resp_final['touches'],
                               resp_final[f"{feature}_hill_estimate"],
                               bounds=(0, [np.inf, np.inf, np.inf]))
        resp_final["touches_scaled"] = touches['scaled_vals']
        resp_final[f"{feature}_hill_estimate_minmax"] = resp_final[f"{feature}_hill_estimate"] / \
                                                        (max(resp_final[f"{feature}_hill_estimate"])
                                                         - min(resp_final[
                                                                   f"{feature}_hill_estimate"]))
        return resp_final, popt, touches, y_axis

    def responses(self, model, x: pd.DataFrame, feature: str, max_freq: int, increment: int) -> dict:
        """
        """
        df_preds = x.copy()
        # get predictions
        for i in range(0, max_freq + 1, increment):
            df_sim = x.copy()
            # replace channel and its lags with desired frequency
            df_sim[[feature] + [c for c in x.columns if f"{feature}_lag" in c]] = i
            df_preds[f"{feature}_preds_{i}"] = model.predict(df_sim)
        # aggregate predictions
        resp_final, popt, scaler_x, scaler_y, resp_og = self._response_aggr(df_preds, feature,
                                                                            max_freq, increment)
        return {'resp_df': resp_final, 'resp_og': resp_og, 'optimal_hill': popt,
                'scaler_x': scaler_x, 'scaler_y': scaler_y}

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
        # aggregate predictions
        resp_final = resp_final - resp_final.iloc[0]
        resp_final[resp_final < 0] = 0
        # hill response curve
        opt_hill = {}
        raw_pred_cols = resp_final.drop(['touches'], axis=1).columns.tolist()
        for i in raw_pred_cols:
            touches = self._scaling(resp_final['touches'])
            y_axis = self._scaling(resp_final[i])
            try:
                popt, pcov = curve_fit(self._response_curve_hill, touches['scaled_vals'],
                                       y_axis['scaled_vals'],
                                       bounds=(0, [np.inf, np.inf, np.inf]))
            except RuntimeError:
                print("Error - curve_fit failed")
                popt = [0, 0, 0]
            hill_est = np.array(self._response_curve_hill(touches['scaled_vals'],
                                                          popt[0], popt[1], popt[2])).reshape(-1, 1)
            resp_final[f"{i}_hill_estimate"] = y_axis['scaler'].inverse_transform(
                hill_est).flatten()
            # refit hill on final hill curve to obtain parameter estimates on the correct scale
            try:
                popt, pcov = curve_fit(self._response_curve_hill, resp_final['touches'],
                                       resp_final[f"{i}_hill_estimate"],
                                       bounds=(0, [np.inf, np.inf, np.inf]))
            except RuntimeError:
                print("Error - curve_fit failed")
                popt = [0, 0, 0]
            resp_final[f"{i}_hill_estimate_minmax"] = resp_final[f"{i}_hill_estimate"] / \
                                                      (max(resp_final[f"{i}_hill_estimate"])
                                                       - min(resp_final[f"{i}_hill_estimate"]))
            opt_hill.update({i: popt})
        # plot final curve
        fig_hill = self.plot(resp_final, x='touches',
                             y=[x for x in resp_final.columns if "hill" in x])
        fig_raw = self.plot(resp_final, x='touches', y=raw_pred_cols)
        # final out
        return {'resp_df': resp_final, 'optimal_hill': opt_hill, 'fig_hill': fig_hill,
                'fig_raw': fig_raw}
