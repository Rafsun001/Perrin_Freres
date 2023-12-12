# Perrin_Freres

# What is time series?
Time series refers to a set of data points recorded in a sequence over regular intervals of time. It's like taking measurements or observations of something at specific moments, like every hour, day, month, or year. This data is arranged in chronological order, showing how things change or behave over time.

**Example:**
1. Imagine you are tracking your weight every week. This means you have a series of data points, each representing your weight at a specific point in time. This is an example of a time series.
2. If you measure the temperature every hour in your city for a week, you'd have a time series dataset showing how the temperature changes throughout the day and from day to day. 
3. The daily closing price of the Dow Jones Industrial Average
4. The monthly sales figures for a retail store
5. The hourly temperature readings at a weather station
6. The number of website visitors per day

# Components of Time Series Analysis
**1.Trend:** The long-term movement or direction in data. It shows whether the  data is increasing, decreasing, or staying relatively constant over time.

> Ex 1:Population Growth: Over several decades, the global population has been consistently increasing, showing a clear upward trend.

> Ex 2: Technology Adoption: The increasing number of smartphone users over the years showcases a rising trend in technology adoption.

**Seasonality:** Patterns that repeat at regular intervals. For instance, sales might increase during holiday seasons every year.

> Ex 1: Ice Cream Sales: Sales might spike during summer months and drop during colder seasons, demonstrating a seasonal pattern influenced by weather.

> Ex 2: Retail Sales during Holidays: Retail sales often surge during holiday seasons like Christmas or Thanksgiving due to increased shopping.


**Cyclical Patterns:** Repeating but not fixed patterns over time, like economic cycles that might occur irregularly.

>Ex 1: Real Estate Market: The real estate market experiences cycles of booms and busts, where property prices rise, peak, and then might fall before rising again.

>Ex 2: Business Cycles: Economic expansions and contractions that occur irregularly but show a cyclical pattern of growth and recession.


**Irregular or Random Movements:** Unexpected fluctuations or variations that don't follow a specific pattern. These could be due to random events or unforeseen factors affecting the data.

> Ex 1: Stock Market Fluctuations: Sudden drops or increases in stock prices due to unexpected news or events, like company announcements or geopolitical changes.

> Ex 2: Weather-Related Events: An unanticipated storm impacting agricultural production leading to irregular variations in crop yields.

# When not to use time series analysis?
1. Can't apply time series analysis when values are constant
Example: Suppose sells of a coffee shop each month is 500. So now if you want to predict the sells of the next month then it can be use again 500. So here to predict, the data we used all are 500. So the next value will automatically becomes 500. So in this type of case we should not use time series analysis.
2. Don't use time series when you just get the predict value by using a function like.

# ARIMA Model
The ARIMA (Autoregressive Integrated Moving Average) model is a widely used statistical method for analyzing and forecasting time series data. It combines the concepts of autoregression (AR), differencing (I for Integrated), and moving averages (MA) to model and predict future values based on past observations.

#### Components of ARIMA:
**1. AutoRegression (AR):**
The AR component models the relationship between the current observation and a certain number of lagged observations (autoregressive terms). It assumes that the current value depends linearly on its own previous values. This is the number  of past values of the time series that are used to predict the future value.

**2. Integrated (I):**
The differencing component represents the number of differences needed to make the time series data stationary. Stationarity implies that the statistical properties of the data (like mean, variance) remain constant over time. The 'I' term is the number of differences required to achieve stationarity. The degree of differencing. Differencing is a method of removing trend or seasonality from a time series.

**3. Moving Average (MA):**
The MA component captures the relationship between the current value and the residual errors obtained by using past forecast errors in a regression-like model. In short this is the number of past forecast errors that are used to predict the future value.

*So an ARIMA(1, 1, 1) model would have one autoregressive term, one differencing term, and one moving average term.*

**Select ARIMA model only when your data is not seasonal.**

# How ARIMA Works?
**1. Identifying Stationarity:**
The first step is to check and achieve stationarity in the time series data using differencing. If the series is not stationary, differencing is applied until stationarity is achieved.

**2. Determining Parameters:**
Determining the parameters (p, d, q) of the ARIMA model:
>p (AR): The number of autoregressive terms (lags) in the model, indicating how many past observations to consider.

>d (I): The number of differences required to achieve stationarity.

>q (MA): The number of moving average terms, representing the number of lagged forecast errors to include in the model.

**3. Model Fitting:**
Once the parameters are determined, the ARIMA model is fitted to the stationary time series data, typically using methods like least squares estimation or maximum likelihood estimation.

**4. Forecasting:**
After fitting the model, it can be used to forecast future values based on the identified patterns in the historical data.

# What is stationarity?
Stationarity is a property of a time series that means that its statistical properties do not change over time means constant. This means that the mean, variance, and autocorrelation of the time series are constant. If a time series is not stationary, then some  models may not be able to accurately forecast the future values of the time series.

**Here are some examples of stationary time series:**
1. The daily closing price of a stock that does not have a trend or seasonality
2. The monthly sales figures for a retail store that do not have a trend or seasonality
3. The hourly temperature readings at a weather station that do not have a trend or seasonality

**Here are some examples of non-stationary time series:**
1. The daily closing price of a stock that has an upward trend
2. The monthly sales figures for a retail store that has a seasonal pattern
3. The hourly temperature readings at a weather station that have a seasonal pattern

### Test to check stationarity:

**1. Rolling Statistics:** Rolling statistics provide a visual representation of the time-varying mean and standard deviation of a time series. This can be helpful for identifying trends, seasonality, and other patterns in the data.

*Purpose:* Visualize time-varying properties	

*Output:* Plot of mean and standard deviation	

*Interpretation:* Identify trends, seasonality, and patterns

*Limitations:* Not a formal statistical test

**2.ADF test:** The ADF test is a statistical test that determines whether a time series has a unit root. A unit root is a type of non-stationarity that occurs when the mean of the time series increases or decreases without bound. The ADF test rejects the null hypothesis of a unit root if the test statistic is less than a critical value.

*Purpose:* Determine stationarity

*Output:* Test statistic and p-value

*Interpretation:* Reject null hypothesis for stationarity

*Limitations:* May not detect all types of non-stationarity

### Methods to make data stationary
**1. Differencing:**

*First Difference:* Subtracting each value from its previous value.
```
df['First Difference'] = df['Column'].diff()
```

*Seasonal Difference:* Subtracting each value from the value at the same season in the previous cycle (for seasonal data).
```
df['Seasonal Difference'] = df['Column'] - df['Column'].shift(seasonal_period)
```
**2. Log Transformation:**
Applying a logarithmic function to the data to stabilize the variance.
```
import numpy as np
df['Log_Column'] = np.log(df['Column'])
```
**3. Moving Average Smoothing:**
Calculating the rolling mean or moving average to reduce noise.
```
window_size = 3
df['Rolling_Mean'] = df['Column'].rolling(window=window_size).mean()
```
**4. Seasonal Decomposition:**
Decompose the time series into trend, seasonal, and residual components.
```
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['Column'], model='additive')
seasonal_adjusted = df['Column'] - result.seasonal
```
**5. Box-Cox Transformation:**
A power transformation to stabilize variance.
```
from scipy.stats import boxcox
transformed_data, lambda_value = boxcox(df['Column'])
```
**6. Detrending:**
Removing the trend component from the data.
```
from scipy import signal
detrended_data = signal.detrend(df['Column'])
```

**7. Deseasonalization:**
Removing seasonal effects using averaging or differencing techniques.
```
# Example of deseasonalization by averaging
seasonal_index = df['Column'] / df['Seasonal Component']
deseasonalized_data = df['Column'] / seasonal_index.mean()
```

**8. Taking Second or Higher Order Differences:**
Useful for removing quadratic or higher trends.
```
df['Second Difference'] = df['Column'].diff().diff()
```

# Steps in Building an ARIMA Model:

**1. Visualize the Time Series:** Understand trends, seasonality, and irregularities in the data.

**2. Check Stationarity:** Use tests like the Augmented Dickey-Fuller (ADF) test or visual inspection to determine stationarity.

**3. Make Data Stationary:** Apply differencing to achieve stationarity if necessary.

**4. Identify Parameters:** Use ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots to estimate p, d, q values.

**5. Fit the ARIMA Model:** Apply the determined parameters and fit the model.

**6. Validate the Model:** Evaluate the model's performance using metrics like RMSE, MAE, etc., and validate against a test dataset.

**7. Forecasting:** Use the model to forecast future values.
