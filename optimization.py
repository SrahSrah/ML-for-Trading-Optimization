"""MC1-P2: Optimize a portfolio. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Copyright 2018, Georgia Institute of Technology (Georgia Tech) 			  		 			     			  	   		   	  			  	
Atlanta, Georgia 30332 			  		 			     			  	   		   	  			  	
All Rights Reserved 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Template code for CS 4646/7646 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Georgia Tech asserts copyright ownership of this template and all derivative 			  		 			     			  	   		   	  			  	
works, including solutions to the projects assigned in this course. Students 			  		 			     			  	   		   	  			  	
and other users of this template code are advised not to share it with others 			  		 			     			  	   		   	  			  	
or to make it available on publicly viewable websites including repositories 			  		 			     			  	   		   	  			  	
such as github and gitlab.  This copyright statement should not be removed 			  		 			     			  	   		   	  			  	
or edited. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
We do grant permission to share solutions privately with non-students such 			  		 			     			  	   		   	  			  	
as potential employers. However, sharing with other current or future 			  		 			     			  	   		   	  			  	
students of CS 7646 is prohibited and subject to being investigated as a 			  		 			     			  	   		   	  			  	
GT honor code violation. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
-----do not edit anything above this line--- 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Student Name: Sarah Hernandez                                                         
GT User ID: shernandez43                                                          
GT ID: 903458532         			     			  	   		   	  			  	
""" 			  		 			     			  	   		   	  			  	
 			  			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
import pandas as pd 			  		 			     			  	   		   	  			  	
import matplotlib.pyplot as plt 			  		 			     			  	   		   	  			  	
import numpy as np 			  		 			     			  	   		   	  			  	
import datetime as dt 			  		 			     			  	   		   	  			  	
from util import get_data, plot_data 			  		 			     			  	   		   	  			  	
import scipy.optimize as spo



def optimize_allocs(prices):
    """
    Input: prices: df with portfolio values

    Output: allocs: a 1D np array of optimized allocations for portfolio value
    """


    def get_drs(allocs):

        start_val = 1000000.0

        normalized = prices/prices.iloc[0]

        port_vals = (normalized*allocs*start_val).sum(axis =1)

        drs = (port_vals/port_vals.shift(1)) - 1
        drs = drs[1:]

        return drs

    def get_sr(allocs):
   
        drr = 0.0
        drs = get_drs(allocs)
        drs = drs - drr
        k = np.sqrt(252)
        sr = drs.mean()/drs.std() * -1 * k

        return sr


    cols = prices.shape[1]
    alloc_guess = np.ones(cols)/cols
    bounds = tuple([(0.0,1.0) for i in range(cols)])

    def sum_constraint(inputs):
        return 1.0-np.sum(np.abs(inputs))

    my_constraint = ({'type': 'eq', "fun": sum_constraint})

    min_err = spo.minimize(get_sr, alloc_guess, method = "SLSQP", options = {"disp" : True}, 
                            bounds = bounds, constraints = my_constraint)

    return min_err.x


def get_stats(prices, allocs):

    port_vals = get_daily_port_value(prices, allocs)

    drs = (port_vals/port_vals.shift(1)) - 1
    drs = drs[1:]

    cr = (port_vals.iloc[-1]/port_vals.iloc[0]) - 1
    adr = drs.mean()
    sddr = drs.std()
    k = np.sqrt(252)
    sr = drs.mean()/drs.std() * k # drr is 0.0

    return [cr, adr, sddr, sr]


def get_daily_port_value(prices, allocs):
    start_val = 1000000.0

    normalized = prices/prices.iloc[0]

    port_vals = (normalized*allocs*start_val).sum(axis =1)

    return port_vals

def fillNA(prices):
    prices = prices.fillna(method = "ffill")
    prices = prices.fillna(method = "bfill")
    return prices


# This is the function that will be tested by the autograder 			  		 			     			  	   		   	  			  	
# The student must update this code to properly implement the functionality 			  		 			     			  	   		   	  			  	
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False): 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    # Read in adjusted closing prices for given symbols, date range 			  		 			     			  	   		   	  			  	
    dates = pd.date_range(sd, ed) 			  		 			     			  	   		   	  			  	
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all = fillNA(prices_all) # forward and backfills nan values as neccessary

    prices = prices_all[syms]  # only portfolio symbols 			  		 			     			  	   		   	  			  	
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    # find the allocations for the optimal portfolio 			  		 			     			  	   		   	  			  	
    allocs = optimize_allocs(prices) 

    # computes stats based on optimal allocation
    cr, adr, sddr, sr = get_stats(prices, allocs) 	  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    # Get daily portfolio value 			  		 			     			  	   		   	  			  	
    port_val = get_daily_port_value(prices, allocs)  			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    # Compare daily portfolio value with SPY using a normalized plot 			  		 			     			  	   		   	  			  	
    if gen_plot: 			  		 			     			  	   		   	  			  	
        # add code to plot here
        port_val = port_val/port_val.iloc[0] 
        prices_SPY = prices_SPY/prices_SPY.iloc[0]			  		 			     			  	   		   	  			  	
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp.plot()

        plt.xlim(sd, ed)
        plt.xlabel("Date")
        plt.ylabel("Normalized Return")
        plt.legend()
        plt.title("Portfolio Returns vs. SPY Returns")

        plt.savefig("plot.png")
         			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    return allocs, cr, adr, sddr, sr 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
def test_code(): 			  		 			     			  	   		   	  			  	
    # This function WILL NOT be called by the auto grader 			  		 			     			  	   		   	  			  	
    # Do not assume that any variables defined here are available to your function/code 			  		 			     			  	   		   	  			  	
    # It is only here to help you set up and test your code 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    # Define input parameters 			  		 			     			  	   		   	  			  	
    # Note that ALL of these values will be set to different values by 			  		 			     			  	   		   	  			  	
    # the autograder! 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    start_date = dt.datetime(2008,6,1)                                                                              
    end_date = dt.datetime(2009,6,1)                                                                                
    symbols =  ['IBM', 'X', 'GLD', 'JPM']
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

 			  		 			     			  	   		   	  			  	
    # Print statistics 			  		 			     			  	   		   	  			  	
    print "Start Date:", start_date 			  		 			     			  	   		   	  			  	
    print "End Date:", end_date 			  		 			     			  	   		   	  			  	
    print "Symbols:", symbols 			  		 			     			  	   		   	  			  	
    print "Allocations:", allocations 			  		 			     			  	   		   	  			  	
    print "Sharpe Ratio:", sr 			  		 			     			  	   		   	  			  	
    print "Volatility (stdev of daily returns):", sddr 			  		 			     			  	   		   	  			  	
    print "Average Daily Return:", adr 			  		 			     			  	   		   	  			  	
    print "Cumulative Return:", cr 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
if __name__ == "__main__": 			  		 			     			  	   		   	  			  	
    # This code WILL NOT be called by the auto grader 			  		 			     			  	   		   	  			  	
    # Do not assume that it will be called 		
    print
    print "START OF RUN:  "	  		
    print

    test_code() 

    print
    print "END OF RUN"
    print
    			  		 			     			  	   		   	  			  	
