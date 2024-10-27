import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from constants import *
import sys

mode = sys.argv[1]

if mode=='in':
    ticker_list = ticker_list_india
    training_end = training_end_india
    test_periods = test_periods_india
else:
    ticker_list = ticker_list
    training_end = training_end
    test_periods = test_periods


def load_and_normalize_stocks(ticker_list):
    # Normalize the stock prices to start from 1.0
    normalized_stocks = []
    for ticker in ticker_list:
        print(ticker)
        df = pd.read_csv(f'{ticker}.csv')

        print(df.head())
        if 'TCS' in ticker:
            df['Close'] = pd.to_numeric(df['Close'].str.replace('"', ''), errors='coerce')
        else:
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        
        # Drop rows with NaN values in 'Close' (optional)
        df.dropna(subset=['Close'], inplace=True)

        # Normalize closing prices
        df['normalized_close'] = df['Close'] / df['Close'].iloc[0]
        normalized_stocks.append(df)
    return normalized_stocks

def get_price_plot(normalized_stocks):
    plt.figure(figsize=(12, 8))  

    # Define colors for each period
    colors = ['lightcoral', 'darkkhaki', 'mediumaquamarine']
    labels_added = set()

    for i, stock_data in enumerate(normalized_stocks):
        close_prices = stock_data['normalized_close'].tolist()
        
        # Plot the training period
        plt.plot(range(training_end), close_prices[:training_end],
                 color="steelblue", linewidth=1, label="Training Period" if "Training Period" not in labels_added else None)
        
        # Plot each test period
        for j, (start, end) in enumerate(test_periods):
            plt.plot(range(start, end), close_prices[start:end],
                     color=colors[j], linewidth=1, label=f"Testing Period {j + 1}" if f"Testing Period {j + 1}" not in labels_added else None)
            labels_added.add(f"Testing Period {j + 1}")
        
        # Plot the regions not in training or testing periods
        not_test_not_train = [
            [idx for idx in range(len(close_prices)) if (idx > training_end and idx < test_periods[0][0])],
            [idx for idx in range(len(close_prices)) if (idx > test_periods[0][1] and idx < test_periods[1][0])],
            [idx for idx in range(len(close_prices)) if (idx > test_periods[1][1] and idx < test_periods[2][0])],
            [idx for idx in range(len(close_prices)) if idx > test_periods[2][1]]
        ]
        
        for ntnt in not_test_not_train:
            plt.plot(ntnt, [close_prices[idx] for idx in ntnt], color="black", linewidth=1)

    # Set title, labels, and legend
    plt.title("Normalized Evolution of the Considered Stocks")
    plt.xlabel("Date")
    plt.ylabel("Normalized Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Set x-ticks for the start and end of the training period
    xtick_numbers = [0, training_end]
    plt.xticks(xtick_numbers, [normalized_stocks[0]["Date"].iloc[idx] for idx in xtick_numbers])
    
    # Save and display the plot
    plt.savefig(f'portfolio_train_test_split_{mode}.png', format='png')
    plt.show()

def plot_test_periods(stocks):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

    # Lists to store normalized data for each test period
    normalized_period_1, normalized_period_2, normalized_period_3 = [], [], []

    # Normalize each test period for each stock
    for stock_data in stocks:
        normalized_period_1.append((stock_data["Close"] / stock_data["Close"].iloc[test_periods[0][0]]).tolist())
        normalized_period_2.append((stock_data["Close"] / stock_data["Close"].iloc[test_periods[1][0]]).tolist())
        normalized_period_3.append((stock_data["Close"] / stock_data["Close"].iloc[test_periods[2][0]]).tolist())

    # Plot each test period in separate subplots
    for j, normalized_period in enumerate([normalized_period_1, normalized_period_2, normalized_period_3]):
        start, end = test_periods[j]
        for data in normalized_period:
            axs[j].plot(range(start, end), data[start:end], linewidth=2,
                        color=["lightcoral", "darkkhaki", "mediumaquamarine"][j])
        
        # Set title and xticks for each subplot
        axs[j].set_title(f"Testing Period {j + 1}")
        axs[j].set_xticks([start, end])
        axs[j].set_xticklabels(np.array(stocks[0]["Date"].iloc[[start, end]], dtype='datetime64[D]'))
        
        # Format dates for readability
        axs[j].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
        for label in axs[j].get_xticklabels(which='major'):
            label.set(rotation=20, horizontalalignment='right')

    plt.tight_layout()
    plt.savefig(f'portfolio_test_periods_{mode}.png', format='png')
    plt.show()

def plot_eval(profits, epsilons, scale_parametric, mode2, period):
    print(profits)
    linestyle_list=["solid","dashed","dotted","dashdot"]*10
    for i in range(len(epsilons)):
        if mode2=="Wasserstein":
            plt.plot(np.cumsum(profits[i]),
                    label = r'$\varepsilon =$ ' + str(epsilons[i]),
                    linestyle = linestyle_list[i]
                    )
        if mode2=="Parametric":
            plt.plot(np.cumsum(profits[i]),
                label = r'$\varepsilon =$ ' + str(np.round(epsilons[i]*scale_parametric,3)),
                linestyle = linestyle_list[i]
                )
    plt.grid(True)
    plt.title(f"Cumulated Profit of Trained Strategy in Testing Period {period},\n {mode2} Approach")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'trades_period_{period}_{mode2}.png')
    plt.show()
    plt.clf()


# Main function to load data and create both plots
if __name__ == "__main__":
    normalized_stocks = load_and_normalize_stocks(ticker_list)
    get_price_plot(normalized_stocks)
    plot_test_periods(normalized_stocks)
