import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cvx
import analysis 
import data
import seaborn as sns


class Visuals:
    def __init__(self, data_loader: data.DataLoaderYfinance, portfolio_analysis: analysis.Analysis):
        self.data = data_loader
        self.analysis = portfolio_analysis
        
    @staticmethod
    def plot_efficient_frontier(basket_stats, current_portfolio_metrics):
        plt.figure(figsize=(10, 7))
        sns.set_style("whitegrid")
        
        # 1. Plot Individual Baskets
        vols = [b['Ann Vol'] for b in basket_stats]
        rets = [b['Ann Return'] for b in basket_stats]
        plt.scatter(vols, rets, marker='o', color='gray', alpha=0.5, label='Individual Baskets')

        # 2. Extract Data once, carefully
        try:
            # We must ensure we only use columns that actually exist in the data
            valid_rets_list = [b['Daily Rets'] for b in basket_stats if b['Daily Rets'] is not None]
            daily_rets_df = pd.concat(valid_rets_list, axis=1).dropna()
            
            # Match the return/cov data to the available columns
            mu = daily_rets_df.mean().values * 252
            S = daily_rets_df.cov().values * 252
            n = len(mu)
        except Exception as e:
            print(f"❌ Data Error: {e}")
            return

        frontier_vols = []
        frontier_rets = []
        
        # Wider range of Risk Aversion to force the curve to stretch
        ra_levels = np.logspace(-3, 2, 20) 

        for ra in ra_levels:
            try:
                w = cvx.Variable(n)
                prob = cvx.Problem(cvx.Maximize(mu @ w - ra * cvx.quad_form(w, S)),
                                 [cvx.sum(w) == 1, w >= 0])
                prob.solve(solver=cvx.ECOS)
                
                if w.value is not None:
                    # Annualized Stats for the point
                    r = mu @ w.value
                    v = np.sqrt(w.value.T @ S @ w.value)
                    frontier_rets.append(r)
                    frontier_vols.append(v)
            except:
                continue

        # 3. Sort and Plot
        if len(frontier_vols) > 1:
            data_points = sorted(zip(frontier_vols, frontier_rets))
            sorted_vols, sorted_rets = zip(*data_points)
            plt.plot(sorted_vols, sorted_rets, color='green', linestyle='--', linewidth=2, label='Efficient Frontier')
            print(f"✅ SUCCESS: Plotted line with {len(sorted_vols)} points.")
        else:
            print("❌ FAIL: Optimizer couldn't find points. Check if your baskets have overlapping data.")

        # 4. Plot Your Current Portfolio
        plt.scatter(current_portfolio_metrics['Annualized Volatility'], 
                    current_portfolio_metrics['Annualized Return'], 
                    color='red', marker='*', s=300, label='Your Portfolio', zorder=10)

        plt.title("Efficient Frontier Curve")
        plt.xlabel("Annualized Volatility")
        plt.ylabel("Annualized Return")
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_cum_ret_historical(basket_stats,weights):
        plt.figure(figsize=(10, 7))
        sns.set_style("whitegrid")
        
        # 1. Plot Cumulative Returns for Each Basket
        for b in basket_stats:
            if b['Daily Rets'] is not None:
                cum_rets = (1 + b['Daily Rets']).cumprod() - 1
                plt.plot(cum_rets.index, cum_rets.values, label=b['Name'], alpha=0.5)

        # 2. Plot Cumulative Return for Your Portfolio
        rets_dict = {b['Name']: b['Daily Rets'] for b in basket_stats if b['Daily Rets'] is not None}

        # 2. Create the DataFrame
        # Each column will now be named after the basket (e.g., 'Basket A', 'Basket B')
        port_rets_df = pd.concat(rets_dict, axis=1).dropna()

        weight_series = pd.Series(weights, index=[b['Name'] for b in basket_stats])

        aligned_weights = weight_series.reindex(port_rets_df.columns)
        port_daily_rets = port_rets_df @ aligned_weights
        port_cum_rets = (1 + port_daily_rets).cumprod() - 1
        plt.plot(port_cum_rets.index, port_cum_rets.values, label='Your Portfolio', color='red', linewidth=2)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.show()