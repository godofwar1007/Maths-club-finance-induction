import numpy as np
import matplotlib.pyplot as plt 

# setting ut the parameters 
start=100  # starting price for the stocks
mu=0.08    
sigma = 0.20   # volatatity 
T = 1.0        # Time horizon (no of yrs u want to simulate the code for )
N = 1000       # no of simulations u want to perform 
dt=(1/252)     # time step 
days = int(T*252) # just doing this so we dont have to adjust the days manually it will get changed as we change the T

np.random.seed(42) # added this so that i can produce some the same results again and again everytime

sak=np.random.normal(0, 1, (days, N)) # this is just a matrix of random numbers 
                                      # added the .normal here so that the results of the random thingy arent too random (abnormal)

prices=np.zeros((days+1,N))  # this is an array to hold all the prices 
prices[0]=start      # the first day will start with 'start' obviously :p

drift=(mu-0.5*sigma**2)*dt  # the drift term 

for i in range(days):

    shock=sigma*np.sqrt(dt)*sak[i]  # the shock term 

    prices[i+1]=prices[i]*np.exp(drift+shock) # the caluclation for the next day price


# plottin the line graph 
time_axis = np.linspace(0, T, days + 1)

plt.figure(figsize=(10, 6))
plt.plot(time_axis, prices[:, :20], lw=1.5, alpha=0.6)   # just plotting the first 20 paths so that the graph doesnt get messy 
plt.plot(time_axis, prices.mean(axis=1), 'k--', linewidth=3, label='Average Path')  # this will create the avg path of all the paths 
plt.title(f'Monte Carlo Simulation: {N} Price Paths (GBM)')
plt.xlabel('Time (Years)')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# some calculations and neat output at the terminal 

fin=prices[-1]  # the final prices after 1 yr 

K = 105 # strike price for an option 

payoffs = np.maximum(fin-K,0) # calculation of payoff for call 

option_price = np.mean(payoffs)  # The Fair Price is the average of all these potential payoffs


print("-" * 30)
print("SIMULATION RESULTS")
print("-" * 30)
print(f"Start Price:       ${start}")
print(f"Mean Final Price:  ${fin.mean():.2f}")
print(f"Simulated Call Price: ${option_price:.2f}")
print("-" * 30)



# plotting the histogram for the overall ananlysis of the final prices 
plt.figure(figsize=(10, 6))
plt.hist(fin, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike Price (${K})')
plt.title('Distribution of Final Stock Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
