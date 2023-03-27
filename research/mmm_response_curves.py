# Author: Hartaig Singh

import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
import mmm_transformations


class MMMResponseCurves(object):
    def _scaling(self, x: [], lb: int = 0, ub: int = 1):
        scaler = MinMaxScaler(feature_range=(lb, ub))
        x_scaled = scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten()
        return {'scaler': scaler, 'scaled_vals': x_scaled}

    def _response_curve_hill(self, x: [], beta: float, k: float, s: float) -> []:
        """
        k shifts inflection point, s controls c or s shape
        """
        imp = beta * (1 / (1 + (x / k) ** (-s)))
        return imp

    def plot(self, df: pd.DataFrame, x: str, y: []):
        """
        """
        fig = px.line(df, x=x, y=df[[x] + y].columns)
        fig.update_layout(yaxis_title="Rx Impact")
        return fig

    def _segment_response_aggr(self, df: pd.DataFrame, feature: str, max_freq: int, increment: int,
                               segment: str) -> pd.DataFrame:
        """
        """
        # rf response curve
        seg_resp = pd.DataFrame({'touches': range(0, max_freq + 1, increment)})
        for i in range(0, len(df), 1):
            preds = [x for x in df.columns if 'preds' in x]
            row = list(dict(df[[segment] + preds].iloc[i]).values())
            spec = row[0]
            df_out = pd.DataFrame({f"{spec}_{feature}": row[1:len(row)]})
            seg_resp = pd.concat([seg_resp, df_out], axis=1)
        seg_resp = seg_resp - seg_resp.iloc[0]
        seg_resp[seg_resp < 0] = 0
        # hill response curve
        seg_cols = seg_resp.drop(['touches'], axis=1).columns.tolist()
        opt_hill, scaler_x, scaler_y = {}, {}, {}
        for i in seg_cols:
            touches = self._scaling(seg_resp['touches'])
            y_axis = self._scaling(seg_resp[i])
            try:
                #popt, pcov = curve_fit(self._response_curve_hill, seg_resp['touches'], seg_resp[i],
                #                       bounds=(0, [np.inf, np.inf, np.inf]))
                popt, pcov = curve_fit(self._response_curve_hill, touches['scaled_vals'], y_axis['scaled_vals'],
                                       bounds=(0, [np.inf, np.inf, np.inf]))
            except RuntimeError:
                print("Error - curve_fit failed")
                popt = [0, 0, 0]
            # seg_resp[f"{i}_hill_estimate"] = self._response_curve_hill(seg_resp['touches'], popt[0], popt[1], popt[2])
            hill_est = np.array(self._response_curve_hill(touches['scaled_vals'],
                                                          popt[0], popt[1], popt[2])).reshape(-1, 1)
            seg_resp[f"{i}_hill_estimate"] = y_axis['scaler'].inverse_transform(hill_est).flatten()
            scaler_x.update({i: touches['scaler']})
            scaler_y.update({i: y_axis['scaler']})
            opt_hill.update({i: popt})
        # plot final curve
        hill_cols = [x for x in seg_resp if x.endswith('_hill_estimate')]
        fig_hill = self.plot(seg_resp, x='touches', y=hill_cols)
        fig_raw = self.plot(seg_resp, x='touches', y=seg_cols)
        return [seg_resp, opt_hill, fig_hill, fig_raw, scaler_x, scaler_y]

    def _overall_response_aggr(self, df: pd.DataFrame, feature: str, max_freq: int, increment: int) -> []:
        """
        """
        # rf response curve
        resp_final = pd.DataFrame({'touches': range(0, max_freq + 1, increment)})
        pred_cols = [x for x in df.columns if 'preds' in x]
        mean_pred = []
        for i in pred_cols:
            mean_pred.append(np.mean(df[i]))
        responses = pd.DataFrame({feature: mean_pred})
        resp_final = pd.concat([resp_final, responses], axis=1)
        resp_final = resp_final - resp_final.iloc[0]
        resp_final[resp_final < 0] = 0
        # hill response curve
        touches = self._scaling(resp_final['touches'])
        y_axis = self._scaling(resp_final[feature])
        try:
            #popt, pcov = curve_fit(self._response_curve_hill, resp_final['touches'], resp_final[feature],
            #                       bounds=(0, [np.inf, np.inf, np.inf]))
            popt, pcov = curve_fit(self._response_curve_hill, touches['scaled_vals'], y_axis['scaled_vals'],
                                   bounds=(0, [np.inf, np.inf, np.inf]))
        except RuntimeError:
            print("Error - curve_fit failed")
            popt = [0, 0, 0]
        #resp_final[f"{feature}_hill_estimate"] = self._response_curve_hill(resp_final['touches'],
        #                                                                   popt[0], popt[1], popt[2])
        hill_est = np.array(self._response_curve_hill(touches['scaled_vals'], popt[0], popt[1], popt[2])).reshape(-1, 1)
        resp_final[f"{feature}_hill_estimate"] = y_axis['scaler'].inverse_transform(hill_est).flatten()
        resp_final["touches_scaled"] = touches['scaled_vals']
        # plot final curve
        fig_raw = self.plot(resp_final, x='touches', y=[f"{feature}_hill_estimate"])
        fig_hill = self.plot(resp_final, x='touches', y=[feature])
        return [resp_final, popt, fig_hill, fig_raw, touches['scaler'], y_axis['scaler']]

    def responses(self, curve_type: str, model, x: pd.DataFrame, feature: str, max_freq: int, increment: int,
                  segment: str, date: str, lag_channels=[]) -> dict:
        """

        """
        if lag_channels is None:
            lag_channels = []
        transform = mmm_transformations.MMMTransformations()
        df_preds = x.copy()
        # get predictions
        for i in range(0, max_freq + 1, increment):
            df_sim = x.copy()
            df_sim[[feature] + lag_channels] = i
            # re adstock if applied
            if 'adstock' in feature:
                df_sim[feature] = transform.segment_adstock(df_sim, feature, segment, float(feature.split('_')[-1]))
            df_preds[f"{feature}_preds_{i}"] = model.predict(df_sim.drop([segment, date], errors='ignore', axis=1))
        # aggregate predictions
        if curve_type == 'segment':
            df_aggr = df_preds.drop([date], axis=1).groupby(segment).mean().reset_index()
            resp_final, popt, fig_hill, fig_raw, scaler_x, scaler_y = self._segment_response_aggr(df_aggr, feature,
                                                                                                  max_freq, increment,
                                                                                                  segment)
        elif curve_type == 'overall':
            # df_aggr = df_preds.drop([segment], axis=1).groupby(date).sum().reset_index()
            resp_final, popt, fig_hill, fig_raw, scaler_x, scaler_y = self._overall_response_aggr(df_preds, feature,
                                                                                                  max_freq, increment)
        resp_dict = {'resp_df': resp_final, 'optimal_hill': popt, 'fig_hill': fig_hill, 'fig_raw': fig_raw,
                     'scaler_x': scaler_x, 'scaler_y': scaler_y}
        return resp_dict

    def responses_segment(self, model, x: pd.DataFrame, feature: str, max_freq: int, increment: int, segments: [],
                          segment: str, lag_channels=[]) -> dict:
        """
        """
        transform = mmm_transformations.MMMTransformations()
        resp_final = pd.DataFrame({'touches': range(0, max_freq + 1, increment)})
        # get predictions
        for s in segments:
            pred = []
            for i in range(0, max_freq + 1, increment):
                df_sim = x.copy()
                df_sim[s] = 1
                df_sim[[feature] + lag_channels] = i
                # re adstock if applied
                if 'adstock' in feature:
                    df_sim[feature] = transform.segment_adstock(df_sim, feature, segment, float(feature.split('_')[-1]))
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
                #popt, pcov = curve_fit(self._response_curve_hill, resp_final['touches'], resp_final[i],
                #                       bounds=(0, [np.inf, np.inf, np.inf]))
                popt, pcov = curve_fit(self._response_curve_hill, touches['scaled_vals'], y_axis['scaled_vals'],
                                       bounds=(0, [np.inf, np.inf, np.inf]))
            except RuntimeError:
                print("Error - curve_fit failed")
                popt = [0, 0, 0]
            hill_est = np.array(self._response_curve_hill(touches['scaled_vals'],
                                                          popt[0], popt[1], popt[2])).reshape(-1, 1)
            resp_final[f"{i}_hill_estimate"] = y_axis['scaler'].inverse_transform(hill_est).flatten()
            opt_hill.update({i: popt})
        # plot final curve
        fig_hill = self.plot(resp_final, x='touches', y=[x for x in resp_final.columns if "hill" in x])
        fig_raw = self.plot(resp_final, x='touches', y=raw_pred_cols)
        # final out
        resp_dict = {'resp_df': resp_final, 'optimal_hill': opt_hill, 'fig_hill': fig_hill, 'fig_raw': fig_raw}
        return resp_dict
