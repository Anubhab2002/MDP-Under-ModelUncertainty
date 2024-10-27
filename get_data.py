import yfinance
import pandas as pd
from constants import *
import sys

mode = sys.argv[1] # us or india

start_date = '2010-01-01'
end_date = '2024-09-30'

def download_data(ticker, start_date, end_date):
    dataset = yfinance.download(ticker, start=start_date, end=end_date)
    return dataset

def collate_data():
    stocks = []
    for ticker in ticker_list:
        stocks += [download_data(ticker, start_date, end_date)]
    
    for i in range(len(stocks)):
        stocks[i] = stocks[i].dropna()
        stocks[i] = stocks[i].reset_index()
        stocks[i]  = stocks[i] [["Date","Close"]]
        stocks[i].to_csv(f'{ticker_list[i]}.csv')

    return stocks

import pandas as pd

def collate_data_india():
    stocks = []
    for ticker in ticker_list_india:
        stocks.append(pd.read_csv(f'{ticker}.csv'))
    
    for i in range(len(stocks)):
        # Drop NaN values and reset index
        stocks[i] = stocks[i].dropna().reset_index(drop=True)

        # Strip whitespace from column names
        stocks[i].columns = stocks[i].columns.str.strip()

        # Select relevant columns
        stocks[i] = stocks[i][["Date", "close"]]
        
        # Rename columns
        stocks[i].rename(columns={'close': 'Close', 'Date': 'Date'}, inplace=True)

        # Convert 'Date' column to datetime and set timezone to UTC
        stocks[i]['Date'] = pd.to_datetime(stocks[i]['Date'], format='%d-%b-%Y').dt.tz_localize('UTC')

        # Save the DataFrame to a CSV file, including the index
        stocks[i].to_csv(f'{ticker_list_india[i]}.csv', index=True)

    return stocks


if __name__ == '__main__':
    if(mode=='us'):
        dataset = collate_data()
    else:
        dataset = collate_data_india()
    print(len(dataset), dataset[0].shape)


