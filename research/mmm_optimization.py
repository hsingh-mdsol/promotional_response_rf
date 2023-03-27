# Author: Hartaig Singh

import pandas as pd
import numpy as np
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from scipy.optimize import minimize


class MMMOptimization:
    def __init__(self, budget: int, params: dict = None, scalers: dict = None, x_scaler=None):
        self.budget = budget
        self.params = params
        self.scalers = scalers
        self.x_scaler = x_scaler

    def _objective_string(self, imp_calc: [] = None) -> str:
        beta = self.params['beta'].astype(str).tolist()
        a = self.params['alpha'].astype(str).tolist()
        g = self.params['gamma'].astype(str).tolist()
        if imp_calc is None:
            n = [f"n[{i}]" for i in range(len(self.params))]
            res = ['(' + i + '*(1/(1+(' + j + '/' + k + ')**' + '(-' + l + '))))' for i, j, k, l in zip(beta, n, a, g)]
            res = '-1*(' + '+'.join(res) + ')'
        else:
            res = ['(' + i + '*(1/(1+(' + j + '/' + k + ')**' + '(-' + l + '))))' for i, j, k, l in zip(beta, imp_calc,
                                                                                                        a, g)]
        return res

    def _constraint_string(self) -> str:
        n = [f"n[{i}]" for i in range(len(self.params))]
        n = '-'.join(n)
        # WARNING: Need to address which min max scaler to use for scaling budget
        budget_scaled = float(self.x_scaler.transform([[self.budget]]))
        return str(budget_scaled) + '-' + n

    def _bounds_string(self) -> str:
        # WARNING: Need to address which min max scaler to use for scaling budget
        budget_scaled = float(self.x_scaler.transform([[self.budget]]))
        return '(' + ','.join([f"(0, {budget_scaled})"] * len(self.params)) + ')'

    def _scale_output_x(self, x: []):
        j = 0
        scaled_x = []
        for i in self.scalers:
            scaled_x.append(float(self.scalers[i]['x'].inverse_transform([[x[j]]])))
            j += 1
        return scaled_x

    def _scale_output_y(self, x: []):
        x_str = np.array(x).astype(str).tolist()
        imp_funcs = self._objective_string(x_str)
        j, scaled_y = 0, 0
        for i in self.scalers:
            scaled_y += float(self.scalers[i]['y'].inverse_transform([[eval(imp_funcs[j])]]))
            j += 1
        return scaled_y

    def optimize_hill(self, start_vals: []):
        # WARNING: Need to address which min max scaler to use for scaling budget
        start_vals_scaled = self.x_scaler.transform(np.array(start_vals).reshape(-1, 1)).flatten().tolist()
        opt_code = \
            f"def objective(n):return {self._objective_string()}\n" + \
            f"def constraint(n):return {self._constraint_string()}\n" + \
            f"bnds = {self._bounds_string()}\n" + \
            f"cons = [{{'type': 'ineq', 'fun': constraint}}]\n" + \
            f"sol = minimize(objective, {start_vals_scaled}, bounds=bnds, constraints=cons)\n" + \
            f"print(self._objective_string())\n" + \
            f"print(self._scale_output_x(sol.x))\n" + \
            f"#print(self._scale_output_y(sol.x))\n" + \
            f"print(sol)"
        return exec(opt_code)

    def optimize_hyperopt_hill(self, channels: dict, max_evals: int):
        """
        channels.columns = channel, max tp, incr
        """

        space = {}
        for i in channels:
            space.update({i: hp.choice(i, [x for x in range(0, channels[i]['max_tp'], channels[i]['incr']) if x > 0])})

        # define function to be minimized
        def objective(n):
            imp = float(self.scalers['P1_Arikayce']['y'].inverse_transform(
                [[(0.962494 * (1 / (1 + (float(self.scalers['P1_Arikayce']['x'].transform(
                    [[n['P1_Arikayce']]])) / 0.275455) ** (-3.189146e+01))))]])) + \
                  float(self.scalers['P2_Arikayce']['y'].inverse_transform(
                      [[(0.991836 * (1 / (1 + (float(self.scalers['P2_Arikayce']['x'].transform(
                          [[n['P2_Arikayce']]])) / 0.140809) ** (-9.817393))))]])) + \
                  float(self.scalers['sfmc_opened_email']['y'].inverse_transform(
                      [[(10.120613 * (1 / (1 + (float(self.scalers['sfmc_opened_email']['x'].transform(
                          [[n['sfmc_opened_email']]])) / 19.025624) ** (-7.315834e-01))))]])) + \
                  float(self.scalers['publ_pulsepoint_count']['y'].inverse_transform(
                      [[(0.837760 * (1 / (1 + (float(self.scalers['publ_pulsepoint_count']['x'].transform(
                          [[n['publ_pulsepoint_count']]])) / 2.0552716) ** (-9.123839e-13))))]])) + \
                  float(self.scalers['deep_intent_count']['y'].inverse_transform(
                      [[(0.852862 * (1 / (1 + (float(self.scalers['deep_intent_count']['x'].transform(
                          [[n['deep_intent_count']]])) / 0.368371) ** (-9.599313e+01))))]]))
            if sum([n[z] for z in channels]) > self.budget:
                imp = imp - 99999999
            return -1 * imp

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        output = {'mix': best, 'trials': trials.results, 'space': space}
        return output

    def optimize_predict(self, x: pd.DataFrame, channels: [], max_evals: int, model, incr: int = 1, lag_channels=[]):
        """
        """
        df_sim = x.copy()

        space = {}
        for i in channels:
            space.update({i: hp.choice(i, range(0, self.budget, incr))})

        # define function to be minimized
        def objective(n):
            for c in channels + lag_channels:
                df_sim[c] = n[c]
            pred = model.predict(df_sim)
            imp = np.mean(pred)
            if sum([n[z] for z in channels]) > self.budget:
                imp = imp - 99999999
            return -1 * imp

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        output = {'mix': best, 'trials': trials.results, 'space': space}
        return output

    def optimize_predict_v2(self, x: pd.DataFrame, channels: [], max_evals: int, model, incr: int = 1, lag_channels=[]):
        """
        """
        df_sim = x.copy()

        space = {}
        for i in channels:
            space.update({i: hp.choice(i, range(0, self.budget, incr))})

        # define function to be minimized
        def objective(n):
            for c in channels + lag_channels:
                df_sim[c] = n[c]
            pred = model.predict(df_sim)
            imp = np.mean(pred)
            #channels = ['P1_Arikayce', 'P2_Arikayce', 'sfmc_opened_email', 'publ_pulsepoint_count', 'deep_intent_count']
            cost = n['P1_Arikayce']
            if sum([n[z] for z in channels]) > self.budget:
                imp = imp - 99999999
            return -1 * imp

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        output = {'mix': best, 'trials': trials.results, 'space': space}
        return output

    def optimize_predict_segments(self, x: pd.DataFrame, channels: [], segments: [], max_evals: int, model,
                                  incr: int = 1):
        """
        """
        df_sim = x.copy()

        space = {}
        # for i in channels:
        #    for j in segments:
        #        space.update({j: hp.choice(j, {j: [0, 1], i: hp.choice(i, range(0, self.budget, incr))})})

        for i in channels:
            space.update({i: hp.choice(i, range(0, self.budget, incr))})
        for j in segments:
            space.update({j: hp.choice(j, [0, 1])})

        # define function to be minimized
        def objective(n):
            for c in channels + segments:
                df_sim[c] = n[c]
            pred = model.predict(df_sim)
            imp = np.mean(pred)
            if sum([n[z] for z in channels]) > self.budget:
                imp = imp - 99999999
            return -1 * imp

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        output = {'mix': best, 'trials': trials.results, 'space': space}

        return output
