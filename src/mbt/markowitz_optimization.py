import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import os

# Create target directory if it doesn't exist
os.makedirs(r"c:\Users\ADMIN\Desktop\Kaggle\src\mbt", exist_ok=True)

# Load the VN46 data
data_path = r"c:\Users\ADMIN\Desktop\Kaggle\data\processed\m1_vn46.csv"
df = pd.read_csv(data_path)

# Pivot data to get closing prices for each ticker per day
df['time'] = pd.to_datetime(df['time']).dt.normalize()
df_close = df.pivot(index='time', columns='ticker', values='close')
df_close = df_close.dropna(axis=1) # Drop any ticker that has missing data across all rows

# Calculate daily and annual returns
returns = df_close.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.03 # Assuming a risk free rate of 3%

# Functions for portfolio evaluation
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns_ann = np.sum(mean_returns*weights ) * 252
    std_ann = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std_ann, returns_ann

# Simulate random portfolios
p_ret, p_std, p_weights = [], [], []
num_assets = len(mean_returns)

for _ in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    std, ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    p_ret.append(ret)
    p_std.append(std)
    p_weights.append(weights)

p_ret = np.array(p_ret)
p_std = np.array(p_std)
p_weights = np.array(p_weights)
sharpe_ratio = (p_ret - risk_free_rate) / p_std

# Find the maximum sharpe ratio portfolio and minimum volatility portfolio
max_sharpe_idx = np.argmax(sharpe_ratio)
max_sharpe_ret = p_ret[max_sharpe_idx]
max_sharpe_std = p_std[max_sharpe_idx]

min_vol_idx = np.argmin(p_std)
min_vol_ret = p_ret[min_vol_idx]
min_vol_std = p_std[min_vol_idx]

# --- SciPy Optimization for the Efficient Frontier Curve ---
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def portfolio_return(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
init_guess = num_assets * [1./num_assets,]

target_returns = np.linspace(min_vol_ret, p_ret.max(), 50)
efficient_portfolios = []

for target in target_returns:
    cons = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns, cov_matrix) - target},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(portfolio_volatility, init_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=cons)
    efficient_portfolios.append(res.fun)

# Plotting
plt.figure(figsize=(10, 7))
plt.scatter(p_std, p_ret, c=sharpe_ratio, cmap='YlGnBu', marker='o', s=10, alpha=0.3, label='Random Portfolios')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_std, max_sharpe_ret, marker='*', color='r', s=500, label='Maximum Sharpe Ratio')
plt.scatter(min_vol_std, min_vol_ret, marker='*', color='g', s=500, label='Minimum Volatility')
plt.plot(efficient_portfolios, target_returns, 'k--', linewidth=2, label='Efficient Frontier')
plt.title('Markowitz Portfolio Optimization - Efficient Frontier (VN46)')
plt.xlabel('Annualised Volatility')
plt.ylabel('Annualised Returns')
plt.legend(labelspacing=0.8)

# Save the plot
output_img = r"c:\Users\ADMIN\Desktop\Kaggle\src\mbt\markowitz_efficient_frontier.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close()

# Save maximum sharpe ratio weights
max_sharpe_weights = pd.Series(p_weights[max_sharpe_idx], index=df_close.columns)
max_sharpe_weights = max_sharpe_weights[max_sharpe_weights > 0.01].sort_values(ascending=False)
output_csv = r"c:\Users\ADMIN\Desktop\Kaggle\src\mbt\markowitz_optimal_weights.csv"
max_sharpe_weights.to_csv(output_csv, header=['Weight'])

print(f"Chart saved to {output_img}")
print(f"Optimal weights saved to {output_csv}")
