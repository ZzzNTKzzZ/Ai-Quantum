import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

# Turn off progress printing 
solvers.options['show_progress'] = False
np.random.seed(123)

# ---------------------------------------------------------
# 1. LOAD VN46 DATA
# ---------------------------------------------------------
df = pd.read_csv(r'c:\Users\ADMIN\Desktop\Kaggle\data\processed\m1_vn46.csv')
df['time'] = pd.to_datetime(df['time']).dt.normalize()

# Lọc dữ liệu 4 năm gần nhất
max_date = df['time'].max()
cutoff_date = max_date - pd.DateOffset(years=4)
df = df[df['time'] >= cutoff_date]

df_close = df.pivot(index='time', columns='ticker', values='close').dropna(axis=1)
returns_df = df_close.pct_change().dropna()
return_vec = returns_df.T.values
tickers = df_close.columns.tolist()

print(f'Loaded {return_vec.shape[0]} assets, {return_vec.shape[1]} observations from {cutoff_date.date()} to {max_date.date()}.')

# ---------------------------------------------------------
# OPTIONAL: GENERATE RANDOM DATA (Bỏ comment để dùng)
# ---------------------------------------------------------
# n_assets = 4
# n_obs = 1000
# return_vec = np.random.randn(n_assets, n_obs)
# tickers = [f'Asset_{i}' for i in range(n_assets)]
# print(f'Using {n_assets} random assets with {n_obs} observations.')

# ---------------------------------------------------------
# 2. RANDOM PORTFOLIOS SIMULATION
# ---------------------------------------------------------
def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def random_portfolio(returns):
    ''' Returns the mean and standard deviation of returns for a random portfolio '''
    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    return mu[0,0], sigma[0,0]

n_portfolios = 5000
print(f"Generating {n_portfolios} random portfolios...")
means, stds = np.column_stack([random_portfolio(return_vec) for _ in range(n_portfolios)])

# ---------------------------------------------------------
# 3. OPTIMAL PORTFOLIO ALGORITHM (EFFICIENT FRONTIER)
# ---------------------------------------------------------
def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
    
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    frontier_returns = [blas.dot(pbar, x) for x in portfolios]
    frontier_risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(frontier_returns, frontier_risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), frontier_returns, frontier_risks

print("Optimizing portfolio to find the Efficient Frontier...")
weights, opt_returns, opt_risks = optimal_portfolio(return_vec)

# ---------------------------------------------------------
# 4. PLOTTING
# ---------------------------------------------------------
plt.figure(figsize=(10, 7))
plt.plot(stds, means, 'o', markersize=3, alpha=0.5, label='Random Portfolios')
plt.plot(opt_risks, opt_returns, 'y-o', markersize=5, label='Efficient Frontier')
plt.ylabel('Mean Return')
plt.xlabel('Standard Deviation')
plt.title('Markowitz Portfolio Optimization - Efficient Frontier')
plt.legend()
plt.savefig('markowitz_frontier_vn46.png', dpi=300, bbox_inches='tight')
print("Saved plot to 'markowitz_frontier_vn46.png'")

# ---------------------------------------------------------
# 5. OUTPUT WEIGHTS
# ---------------------------------------------------------
optimal_w = pd.Series(weights.flatten(), index=tickers)
optimal_w = optimal_w[optimal_w > 0.001].sort_values(ascending=False)
print('\nOptimal Weights for Max Sharpe / Frontier Point:')
print(optimal_w)
optimal_w.to_csv('optimal_weights.csv')
print("Saved weights to 'optimal_weights.csv'")
