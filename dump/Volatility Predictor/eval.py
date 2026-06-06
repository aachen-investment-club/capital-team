'''This is the evaluation file for comparing all models together

I will be evaluating the models based off of:
QLIKE:
This is the quasi-likelihood loss function
QLIKE = log(forecasted variance) + (realised variance / forecasted variance)


MSE:
This is the mean squared error from the actual realised variance (intraday variance)
(1/n) * sum((forecasted variance - realised variance)^2)


Diebold-Mariano test:
Compares two forecasting models to see if there is a significant difference in their forecasting performance
The difference, d_t = L(e1_t) - L(e2_t), where L is the loss function (e.g. QLIKE or MSE) and e1_t and e2_t are the forecast errors from the two models at time t
H0: the two models have the same forecasting performance
H1: the two models have different forecasting performance

The benchmark will be HAR-RV

'''
import numpy as np
from scipy.stats import norm


class Evaluation:
    def QLIKE(self, forecast, realised):
        return np.log(forecast) + (realised / forecast)

    def MSE(self, forecast, realised):
        return np.mean((forecast - realised) ** 2)

    def QLIKE_MSE(self, forecast, realised):
        return self.QLIKE(forecast, realised), self.MSE(forecast, realised)

    def Diebold_Mariano(self, forecast1, forecast2, realised):
        d = self.QLIKE(forecast1, realised) - self.QLIKE(forecast2, realised)
        d = np.asarray(d)
        d = d[np.isfinite(d)]
        n = len(d)
        if n < 2:
            return np.nan, np.nan
        stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n))
        p = 2 * (1 - norm.cdf(abs(stat)))
        return stat, p
    
'''forecast is the predicted variance from the model (output)
realised is the actual variance calculated from intraday data (realised variance)
I will fit all models on data, then test it on out of sample data and compare. I will plot the evaluation outputs (QLIKE and MSE) each day over time to see who won each day. '''