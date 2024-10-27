import yfinance

# ticker list for data
ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'EBAY', 'AMZN']
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

if __name__ == '__main__':
    dataset = collate_data()
    print(len(dataset), dataset[0].shape)


