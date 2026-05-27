I have built and compared several volatility prediction models. All models are benchmarked to the HAR_RV and evaluated off of QLIKE and MSE. Realised variance is calculated by intraday pricing information from SPY.

Overall structure: 

1. GARCH model family including garch, rgarch, gjr garch
2. HAR model family including HAR_RV, HAR_CJ
3. combine.py is a combination of HAR_CJ and rgarch. These models are dynamically weighted (DMA) with Bayesian logscoring
4. eval has methods of evaluation for each model, including Diebold Marino Tests
5. loader takes data from the lseg api, including data cleaning
6. main is where I run all files, and plot and compare results
7. testing includes tests for stationarity in the price data, as well as arch effects. Tests used: ADF, KPSS, PP, Ljung-Box, Engle Arch
