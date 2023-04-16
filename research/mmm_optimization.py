# Author: Hartaig Singh

import pandas as pd
import numpy as np
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from scipy.optimize import minimize


class MMMOptimization:
    def __init__(self, budget: int, params: dict = None):
        self.budget = budget
        self.params = params

    def _objective_string(self) -> str:
        beta = self.params['beta'].astype(str).tolist()
        a = self.params['alpha'].astype(str).tolist()
        g = self.params['gamma'].astype(str).tolist()
        n = [f"n[{i}]" for i in range(len(self.params))]
        res = ['(' + i + '*(1/(1+(' + j + '/' + k + ')**' + '(-' + l + '))))' for i, j, k, l in zip(beta, n, g, a)]
        return '-1*(' + '+'.join(res) + ')'

    def _constraint_string(self) -> str:
        n = [f"n[{i}]" for i in range(len(self.params))]
        n = '-'.join(n)
        return str(self.budget) + '-' + n

    def _bounds_string(self) -> str:
        return '(' + ','.join([f"(0, {self.budget})"] * len(self.params)) + ')'

    def calc_impact(self, freq: []):
        beta = self.params['beta'].astype(str).tolist()
        a = self.params['alpha'].astype(str).tolist()
        g = self.params['gamma'].astype(str).tolist()
        res = ['(' + i + '*(1/(1+(' + str(j) + '/' + k + ')**' + '(-' + l + '))))' for i, j, k, l in zip(beta, freq, g, a)]
        return {'impact': eval('+'.join(res)), 'total spend': sum(freq)}

    def optimize_hill(self, start_vals: []):
        opt_code = \
            f"def objective(n):return {self._objective_string()}\n" + \
            f"def constraint(n):return {self._constraint_string()}\n" + \
            f"bnds = {self._bounds_string()}\n" + \
            f"cons = [{{'type': 'ineq', 'fun': constraint}}]\n" + \
            f"sol = minimize(objective, {start_vals}, bounds=bnds, constraints=cons)\n" + \
            f"print(self._objective_string())\n" + \
            f"print(sol.x)\n" + \
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
