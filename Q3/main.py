import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


Rf=0.07   # 7% approx yield by a gov bond 
start='2023-01-01' # start date
end='2024-01-01'  # end date
stonks=['RELIANCE.NS','^NSEI'] # Stock vs Market Index (Nifty 50)


data = yf.download(stonks,start=start,end=end,auto_adjust=False)['Adj Close']

returns = data.pct_change().dropna()  # Calculate Daily Returns:(Price_Today - Price_Yesterday)/Price_Yesterday

# Separate the data into X (Market) and Y (Stock)
stock_ret = returns['RELIANCE.NS'].values # i used .values to convert pd series to np array
market_ret = returns['^NSEI'].values

X = market_ret.reshape(-1, 1) # just some formatting for the input into the linera regrssion 
y = stock_ret

model = LinearRegression()  # inintailize the model
model.fit(X, y)

beta=model.coef_[0] # calculating the beta by slope 
alpha=model.intercept_ # calculating the alpha by the intercept 

#CAPM thingy 
Rm=market_ret.mean()*252 # calulation of the market return to use in formula 

# The CAPM Formula: E(R)=Rf+Beta*(Rm-Rf)
Er=Rf+beta*(Rm-Rf)

# this is just a way to prit cool result on terminal 
print("\n" + "="*40)
print("CAPM ANALYSIS REPORT: Reliance vs Nifty 50")
print("="*40)
print(f"Beta (Systematic Risk):      {beta:.4f}")
print(f"Alpha (Abnormal Return):     {alpha:.4f}")
print("-" * 40)
print(f"Market Return (Annualized):  {Rm:.2%}")
print(f"Risk-Free Rate Used:         {Rf:.2%}")
print("-" * 40)
print(f"EXPECTED RETURN (CAPM):      {Er:.2%}")
print("="*40)

# interpretation of the beta we got 
if beta>1:
    print("CONCLUSION: Reliance is AGGRESSIVE (Riskier than Market).")
elif beta<1:
    print("CONCLUSION: Reliance is DEFENSIVE (Safer than Market).")
else:
    print("CONCLUSION: Reliance moves EXACTLY with the Market.")

# grapgh plotting stuff
plt.figure(figsize=(10, 6))

plt.scatter(market_ret, stock_ret, alpha=0.5, color='#1f77b4', label='Daily Returns') # Scatter plot

predictions = model.predict(X) # regression line 
plt.plot(market_ret, predictions, color='red', linewidth=2, label=f'Regression Line (Beta={beta:.2f})')

plt.title(f'CAPM Analysis: Reliance Industries (Beta: {beta:.2f})') # formatting just so the graph doesnt look fucked
plt.xlabel('Market Returns (Nifty 50) - "The Bus"')
plt.ylabel('Stock Returns (Reliance) - "The Passenger"')
plt.axhline(0,color='black',linewidth=0.5)
plt.axvline(0,color='black',linewidth=0.5)
plt.legend()
plt.grid(True,linestyle='--',alpha=0.3)

plt.show()
