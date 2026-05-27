This is the folder for the AIC capital management team. We are trying to build systems that can help to manage risk, and increase our returns. 

Brief overview:

1. Barra factor model: A  combination of industry factors and style factors fitted with a weighted linear regression (weight being market cap). Outputs a percentage contribution of daily returns according to factors.

2. Optimiser: Based off of Classical Markowitz Optimisation, with a few added constraints. Uses convex optimisation to calculate the portfolio's Efficient Frontier. In-built loosening of constraints to be feasible with low number of baskets.

3. Visuals: plots efficient frontier (Risk vs Return) and historical returns of weighted portfolio. 

4. Analysis: calculates and colour-codes historical metrics including value at risk, max drawdown, max drawdown duration, ...

5. Data: loads data and fundamentals for models from yahoo finance and LSEG refinitive workspace, pulls login data from external .json file

6. Engine: manages baskets and the combined portfolio

7. Main: here I access and run my files
