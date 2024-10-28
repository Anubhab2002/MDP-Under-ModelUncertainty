ticker_list = ['AAPL', 'GOOGL', 'EBAY', 'AMZN' , 'AMAT']
ticker_list_india = ['Quote-Equity-BPCL-EQ-EQ-20-05-2023-to-20-08-2023', 'Quote-Equity-ICICIBANK-EQ-20-05-2023-to-20-08-2023', 'Quote-Equity-TATASTEEL-EQ-20-05-2023-to-20-08-2023']

epsilons = [0,0.05,0.3]
# epsilons = [0]

scale_parametric = 0.5

# Define training and testing periods
training_end = 2200

test_periods = [
    (2201, 2270),  # Testing Period 1
    (2350, 2450),  # Testing Period 2
    (2601, 2680),  # Testing Period 3
]

training_end_india = 40

test_periods_india = [
    (41, 51),  # Testing Period 1
    (51, 55),  # Testing Period 2
    (55, 60),  # Testing Period 3
]