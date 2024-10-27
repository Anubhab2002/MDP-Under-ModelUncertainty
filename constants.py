ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'EBAY', 'AMZN']

epsilons = [0,0.01,0.05,0.1,0.3]

scale_parametric = 0.5

# Define training and testing periods
training_end = 2200

test_periods = [
    (2201, 2270),  # Testing Period 1
    (2350, 2450),  # Testing Period 2
    (2601, 2680),  # Testing Period 3
]