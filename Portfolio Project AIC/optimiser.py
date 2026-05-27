import pandas as pd
import numpy as np
import cvxpy as cvx

class Optimiser:
    @staticmethod
    def get_weights(basket_stats, max_beta=1.0, risk_aversion=0.5):
        """
        Optimises portfolio weights with a relaxation loop to handle 
        infeasibility by gradually loosening constraints.
        """
        n = len(basket_stats)
        returns = np.array([b['Ann Return'] for b in basket_stats])
        betas = np.array([b['Beta'] for b in basket_stats])
        
        # Combine daily returns into a matrix to calculate covariance
        daily_rets_df = pd.concat([b['Daily Rets'] for b in basket_stats], axis=1).dropna()
        cov_matrix = daily_rets_df.cov().values
        
        # Setup CVXPY variables
        w = cvx.Variable(n)
        risk_penalty = cvx.quad_form(w, cov_matrix)
        
        # Initial parameters for the relaxation loop
        current_max_beta = max_beta
        current_ra = risk_aversion
        
        current_max_w = 0.25  # Initial concentration cap,but we are always aiming for more than 4 baskets

        
        for attempt in range(25):
            # Define Constraints
            constraints = [
                cvx.sum(w) == 1,           # Must be fully invested
                w <= current_max_w,        # Concentration cap
                w >= 0.05,                 # Long-only (dropped min 0.05 for feasibility)
                betas @ w <= current_max_beta  # Beta constraint
            ]
            
            # Define Objective
            obj = cvx.Maximize(returns @ w - current_ra * risk_penalty)
            
            # Solve
            prob = cvx.Problem(obj, constraints)
            # Using ECOS solver as it handles small precision issues better
            prob.solve(solver=cvx.ECOS)

            # Check if a solution was found
            if w.value is not None:
                if attempt > 0:
                    print(f"✅ Solution found on attempt {attempt+1}")
                    print(f"Final Beta Cap: {current_max_beta:.2f}, Risk Aversion: {current_ra:.2f}")
                return w.value

            # loosening parameters for further loop
            current_max_beta += 0.1        
            current_max_w += 0.05          
            current_ra = max(0.01, current_ra - 0.01) # Step down risk aversion (floor at 0.01)

        # If the loop finishes without returning w.value
        raise ValueError("Optimisation Infeasible even after 25 relaxation attempts. Check your data for NaNs or errors.")