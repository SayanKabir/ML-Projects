from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather Data
boston_dataset = load_boston()
data_frame = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)

features = data_frame.drop(["INDUS", "AGE"], axis=1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=["PRICE"])

CRIME_IDX=0
CHAS_IDX=2
RM_IDX=4
PTRATIO_IDX=8

property_stats = features.mean().values.reshape(1, 11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)
MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river=False,
                    high_confidence=True):
    
    # Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    property_stats[0][CHAS_IDX] = 1 if (next_to_river) else 0
    # For all other values, we're using average values
    
    # Log price prediction 
    log_estimate = regr.predict(property_stats)[0][0]
    
    # Calculate range
    if high_confidence:
        # 2 SD calculation
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        # 1 SD calculation
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    return log_estimate, lower_bound, upper_bound, interval

# Convert log price estimates to today's prices
ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE/np.median(boston_dataset.target)
def convert_to_todays_dollars(old_log_price):
    dollar_est = np.e**old_log_price *1000* SCALE_FACTOR
    rounded_est = np.around(dollar_est, -3)
    return rounded_est

def getEstimate(rm, ptratio, chas=False, large_range=True):
    
    """
    Estimate the price of a property in Boston.
    
    Keyword arguments:
    rm -- Number of rooms in the property
    ptratio -- Pupil to Teacher ratio of school in the area
    chas -- True if the property is next to the river. False otherwise
    large_range -- true for a 95% prediction interval. False for a 68% prediction interval
    """
    if rm<1 or ptratio<1:
        print("Unrealistic arguments, Try Again")
        return
    log_est, lower, upper, conf = get_log_estimate(nr_rooms=rm, students_per_classroom=ptratio, next_to_river=chas, high_confidence=large_range)
    dollar_est = convert_to_todays_dollars(log_est)
    dollar_low = convert_to_todays_dollars(lower)
    dollar_upp = convert_to_todays_dollars(upper)
    print(f"The estimated property value is {dollar_est}.")
    print(f"At {conf}% confidence the valuation range is: ")
    print(f"USD {dollar_low} to USD {dollar_upp}")
    
    
