from engine import PortfolioManager, Basket
from analysis import Analysis
import IPython.display as display
from visuals import Visuals
import pandas as pd
import numpy as np
from barra_factor import BarraFactor
from data import DataLoaderLSEG, DataProcessor

baskets = [
    Basket("Consumer_Staples", [0.2, 0.2, 0.15, 0.15, 0.15, 0.15],
           ['PG', 'KO', 'PEP', 'COST', 'WMT', 'CL'], 'SPY'),
    Basket("Healthcare_Giants", [0.2, 0.2, 0.15, 0.15, 0.15, 0.15],
           ['JNJ', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO'], 'SPY'),
    Basket("Utilities", [0.2, 0.2, 0.15, 0.15, 0.15, 0.15],
           ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE'], 'SPY'),
    Basket("Mega_Cap_Tech", [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
           ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META'], 'SPY'),
    Basket("Semiconductors", [0.2, 0.2, 0.15, 0.15, 0.3],
           ['NVDA', 'TXN', 'QCOM', 'ADI', 'AMAT'], 'SPY'),
    Basket("Financial_Majors", [0.2, 0.2, 0.15, 0.15, 0.15, 0.15],
           ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C'], 'SPY'),
    Basket("Industrials", [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
           ['GE', 'CAT', 'HON', 'MMM', 'UPS', 'LMT'], 'SPY'),
    Basket("Energy_Majors", [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
           ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC'], 'SPY'),
]

RIC_MAP = {
    'AAPL': 'AAPL.O', 'MSFT': 'MSFT.O', 'NVDA': 'NVDA.O', 'GOOGL': 'GOOGL.O',
    'AMZN': 'AMZN.O', 'META': 'META.O', 'COST': 'COST.O', 'QCOM': 'QCOM.O',
    'TXN':  'TXN.O',  'ADI':  'ADI.O',  'AMAT': 'AMAT.O',
    'JPM':  'JPM.N',  'BAC':  'BAC.N',  'GS':   'GS.N',   'MS':   'MS.N',
    'WFC':  'WFC.N',  'C':    'C.N',    'JNJ':  'JNJ.N',  'UNH':  'UNH.N',
    'ABBV': 'ABBV.N', 'MRK':  'MRK.N',  'LLY':  'LLY.N',  'TMO':  'TMO.N',
    'PG':   'PG.N',   'KO':   'KO.N',   'PEP':  'PEP.N',  'WMT':  'WMT.N',
    'CL':   'CL.N',   'NEE':  'NEE.N',  'DUK':  'DUK.N',  'SO':   'SO.N',
    'D':    'D.N',    'AEP':  'AEP.N',  'SRE':  'SRE.N',  'XOM':  'XOM.N',
    'CVX':  'CVX.N',  'COP':  'COP.N',  'SLB':  'SLB.N',  'EOG':  'EOG.N',
    'MPC':  'MPC.N',  'GE':   'GE.N',   'CAT':  'CAT.N',  'HON':  'HON.N',
    'MMM':  'MMM.N',  'UPS':  'UPS.N',  'LMT':  'LMT.N',
}

NASDAQ = ['AAPL.O', 'MSFT.O', 'NVDA.O', 'GOOGL.O', 'AMZN.O', 'META.O',
          'COST.O', 'QCOM.O', 'TXN.O', 'ADI.O', 'AMAT.O']
NYSE   = ['JPM.N', 'BAC.N', 'GS.N', 'MS.N', 'WFC.N', 'C.N', 'JNJ.N',
          'UNH.N', 'ABBV.N', 'MRK.N', 'LLY.N', 'TMO.N', 'PG.N', 'KO.N',
          'PEP.N', 'WMT.N', 'CL.N', 'NEE.N', 'DUK.N', 'SO.N', 'D.N',
          'AEP.N', 'SRE.N', 'XOM.N', 'CVX.N', 'COP.N', 'SLB.N', 'EOG.N',
          'MPC.N', 'GE.N', 'CAT.N', 'HON.N', 'MMM.N', 'UPS.N', 'LMT.N']

data = DataLoaderLSEG()
data.open_session_lseg()

universe     = list(set(ticker for basket in baskets for ticker in basket.tickers))
ric_universe = [RIC_MAP.get(t, t) for t in universe]

df = data.get_barra_fundamentals_lseg(universe=ric_universe, nasdaq=NASDAQ, nyse=NYSE)

processor = DataProcessor()
df = processor.process_lseg(df)

print("\n--- DATA CHECK ---")
print(f"DataFrame Shape: {df.shape}")
print("Columns:", df.columns.tolist())
print("------------------\n")

barra = BarraFactor(df)

print("Calculating factor returns...")
factor_returns = barra.calculate_factor_returns()

print("Calculating risk matrix...")
history = barra.factor_return_history()
barra.calculate_ewma_risk_matrix(history, halflife=60)

attribution = barra.match_to_baskets(baskets)
risk_metrics = barra.risk_results(baskets)

for basket in baskets:
    if basket.name in attribution:
        print(f"\n--- {basket.name.upper()} ---")
        pred_ret = attribution[basket.name]['Total_Return']
        print(f"Factor-Predicted Return: {pred_ret:.6f}")
        sys_vol = risk_metrics[basket.name]['Systematic_Vol']
        print(f"Annualized Systematic Vol: {sys_vol:.2%}")
        print("Top 3 Risk Drivers:")
        risk_contrib = risk_metrics[basket.name]['Risk_Contribution']
        top_risks = risk_contrib.abs().sort_values(ascending=False).head(3)
        for factor in top_risks.index:
            print(f"  > {factor}: {risk_contrib[factor]:.6f}")
    else:
        print(f"\n--- {basket.name} skipped (No matching data found) ---")