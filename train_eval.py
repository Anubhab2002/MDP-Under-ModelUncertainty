import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from tensorflow_probability import distributions as tfd
import numpy as np
import random
from tqdm import tqdm
from constants import ticker_list, training_end, test_periods, epsilons, scale_parametric
from plot_data import plot_eval
from mdp import Robust_Portfolio_Optimization

import os
import warnings

# Suppress TensorFlow and Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
warnings.filterwarnings("ignore")  # Suppress Python warnings

# Optionally, disable Keras's own logging
tf.get_logger().setLevel('ERROR')


test_period_1_start, test_period_1_end = test_periods[0]
test_period_2_start, test_period_2_end = test_periods[1]
test_period_3_start, test_period_3_end = test_periods[2]

# Load data from CSVs into a list of DataFrames
stocks = [pd.read_csv(f"{ticker}.csv") for ticker in ticker_list]

periods = 10 

Returns_10periods = []

for stock in stocks:
    # Ensure the 'Close' column is numeric
    stock["Close"] = pd.to_numeric(stock["Close"], errors="coerce")
    stock.dropna(subset=["Close"], inplace=True)  # Remove rows with NaN in 'Close'

    # Calculate 10-period returns
    stock_returns = [
        [
            (stock["Close"].iloc[j + k] - stock["Close"].iloc[j - 1 + k]) / stock["Close"].iloc[j - 1 + k]
            for k in range(periods)
            if j + k < len(stock)  # Ensure j + k does not exceed bounds
        ]
        for j in range(1, len(stock) - periods + 1)
    ]
    
    Returns_10periods.append(stock_returns)


# Convert Returns_10periods to a TensorFlow constant for efficient handling in models
Returns_10periods = tf.constant(Returns_10periods, dtype=tf.float32)

# Split into training and testing sets using tf.gather
Returns_10periods_train = tf.gather(Returns_10periods, range(training_end), axis=1)
Returns_10periods_test_1 = tf.gather(Returns_10periods, range(test_period_1_start, test_period_1_end), axis=1)
Returns_10periods_test_2 = tf.gather(Returns_10periods, range(test_period_2_start, test_period_2_end), axis=1)
Returns_10periods_test_3 = tf.gather(Returns_10periods, range(test_period_3_start, test_period_3_end), axis=1)

################################################################
# TRAIN

a_W, a_P = [], []
for epsilon in epsilons:
    print("******************Running for Error Margin*****************", epsilon)
    RL_W = Robust_Portfolio_Optimization(Returns_10periods_train,uncertainty = "Wasserstein",
                                        Nr_measures= 10,
                                        epsilon =epsilon,
                                        Nr_Batch= 2**8,
                                        Nr_MC  = 2**3,
                                        alpha = 0.45)
    RL_P = Robust_Portfolio_Optimization(Returns_10periods_train,uncertainty = "Parametric",
                                        Nr_measures= 10,
                                        epsilon =epsilon*scale_parametric,
                                        Nr_Batch= 2**8,
                                        Nr_MC  = 2**3,
                                        alpha = 0.45)

    print("Training with Wasserstein mode")
    RL_W.train(Epochs = 30,iterations_a=5,iterations_v=5)
    a_W.append(RL_W.a)
    print("Training with Parametric mode")
    RL_P.train(Epochs = 30,iterations_a=5,iterations_v=5)
    a_P.append(RL_P.a)

################################################################


# Define the test periods you want to evaluate
test_periods = [1, 2, 3]

# Initialize a dictionary to store results for each test period
results = {}

for period in test_periods:
    print(f"Evaluating for test period: {period}")
    
    # Create Returns_test for the current test period
    if period==1:
        Returns_test = Returns_10periods_test_1
    if period==2:
        Returns_test = Returns_10periods_test_2
    if period==3:
        Returns_test = Returns_10periods_test_3

    N_test = Returns_test.shape[1]
    
    profits_W = []
    profits_P = []

    for j in range(len(epsilons)):
        trades_W_eps = []
        profits_W_eps = []
        trades_P_eps = []
        profits_P_eps = []

        for i in range(N_test - period):  # Adjust the range to prevent out-of-bounds errors
            trades_W_eps.append(a_W[j](tf.expand_dims(Returns_test[:, i, :], 0)))
            profits_W_eps.append(tf.tensordot(trades_W_eps[-1], Returns_test[:, i + period, -1], 1)[0].numpy())

            trades_P_eps.append(a_P[j](tf.expand_dims(Returns_test[:, i, :], 0)))
            profits_P_eps.append(tf.tensordot(trades_P_eps[-1], Returns_test[:, i + period, -1], 1)[0].numpy())


        profits_W.append(profits_W_eps)
        profits_P.append(profits_P_eps)
    
    plot_eval(profits_W, epsilons, scale_parametric, mode2="Wasserstein", period=period)
    plot_eval(profits_P, epsilons, scale_parametric, mode2="Parametric", period=period)

    print("LEN: ", profits_P)
    # Statistics for Wasserstein 
    profits_W = np.array(profits_W)
    overall_W = np.cumsum(profits_W, 1)[:, -1]
    average_W = np.mean(profits_W, 1)
    percentage_W = np.sum((profits_W > 0) / profits_W.shape[1], 1)
    Sharpe_W = np.mean(profits_W, 1) / np.sqrt(np.mean(profits_W**2, 1))
    Sortino_W = np.mean(profits_W, 1) / np.sqrt(np.mean((profits_W * (profits_W < 0))**2, 1))

    data_W = {
        "Overall Profit": overall_W,
        "Average Profit": average_W,
        "% of Profitable Trades": np.round(percentage_W * 100, 2),
        "Sharpe Ratio": Sharpe_W,
        "Sortino Ratio": Sortino_W
    }

    df_W = pd.DataFrame(data=data_W, index=["Epsilon = {}".format(str(eps)) for eps in epsilons])
    results[f'Wasserstein - Period {period}'] = df_W


    # Statistics for Parametric
    profits_P = np.array(profits_P)
    overall_P = np.cumsum(profits_P, 1)[:, -1]
    average_P = np.mean(profits_P, 1)
    percentage_P = np.sum((profits_P > 0) / profits_P.shape[1], 1)
    Sharpe_P = np.mean(profits_P, 1) / np.sqrt(np.mean(profits_P**2, 1))
    Sortino_P = np.mean(profits_P, 1) / np.sqrt(np.mean((profits_P * (profits_P < 0))**2, 1))

    data_P = {
        "Overall Profit": overall_P,
        "Average Profit": average_P,
        "% of Profitable Trades": np.round(percentage_P * 100, 2),
        "Sharpe Ratio": Sharpe_P,
        "Sortino Ratio": Sortino_P
    }

    df_P = pd.DataFrame(data=data_P, index=["Epsilon = {}".format(np.round(eps * scale_parametric, 3)) for eps in epsilons])
    results[f'Parametric - Period {period}'] = df_P


# Output all results
for key, df in results.items():
    print(f"\n{key} Results:")
    print(df)