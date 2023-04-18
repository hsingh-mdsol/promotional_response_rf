# Author: Hartaig Singh

import pandas as pd
import numpy as np
from hyperopt import tpe, rand, hp, fmin, STATUS_OK, Trials
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

    def optimize_hyperopt_hill(self, channels: [], max_evals: int, incr: int = 1):
        """
        Note! Optimization function cannot have 0 raised to negative power and no division by 0
        """

        # create optimization space
        space = {}
        for i in channels:
            space.update({i: hp.choice(i, [x for x in range(0, self.budget, incr) if x > 0])})

        # create equation to optimize
        c_list = [f"n['{c}']" for c in channels]
        beta = self.params['beta'].astype(str).tolist()
        a = self.params['alpha'].astype(str).tolist()
        g = self.params['gamma'].astype(str).tolist()
        eq = ['(' + i + '*(1/(1+(' + j + '/' + k + ')**' + '(-' + l + '))))' for i, j, k, l in zip(beta, c_list, g, a)]

        # define function to be minimized
        def objective(n):
            imp = eval('+'.join(eq))
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

        # create optimization space
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
        best = fmin(fn=objective, space=space, algo=rand.suggest, max_evals=max_evals,
                    trials=trials)
        output = {'mix': best, 'trials': trials.results, 'space': space}
        return output