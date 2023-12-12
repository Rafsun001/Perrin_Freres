import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset



def load_data(file_path):
    return pd.read_csv(file_path)
df=load_data('data.csv')

st.title('Restaurant data analysis report')
st.write(df.head())

df.columns=['Month','Sales']
df.drop([105,106],axis=0,inplace=True)
df['Month']=pd.to_datetime(df['Month'])
df.set_index('Month',inplace=True)


st.write(round(df.describe(),3))

st.header("Perrin freres Monthly Sales")
fig1, ax=plt.subplots(figsize=(12,8))
sns.lineplot(x=df.index,y="Sales",data=df)
plt.title(f'Perrin freres Monthly Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid()
st.pyplot(fig1)

st.header("Perrin freres Monthly Sales Using Box-Plot")
fig2, ax=plt.subplots(figsize=(12,8))
sns.boxplot(x=df.index.year, y=df.values[:,0],ax=ax)
plt.title(f'Perrin freres Monthly Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=90)
plt.grid()
st.pyplot(fig2)

st.header("Perrin freres Monthly Average Sales")
fig3, ax=plt.subplots(figsize=(12,8))
df_yearly_sum = df.resample('A').mean()
sns.lineplot(x=df_yearly_sum.index,y="Sales",data=df_yearly_sum)
plt.title(f"Perrin freres Monthly Average Sales  ")
plt.xlabel("Date")
plt.ylabel('Sales')
sns.set(style="darkgrid")
plt.show()
st.pyplot(fig3)

st.header("Perrin freres Monthly Quaterly Sales")
fig4, ax=plt.subplots(figsize=(12,8))
df_quaterly_sum2=df.resample('Q').mean()
sns.lineplot(x=df_quaterly_sum2.index,y="Sales",data=df_quaterly_sum2)
plt.title(f'Perrin freres Monthly Average Quaterly Sales')
plt.xlabel('Quarter')
plt.ylabel('Sales')
plt.grid()
st.pyplot(fig4)

st.header("Perrin freres Monthly Average Sales Per 2 Years")
fig5, ax=plt.subplots(figsize=(12,8))
df_decade_sum=df.resample('2Y').mean()
sns.lineplot(x=df_decade_sum.index,y="Sales",data=df_decade_sum)
plt.title(f'Perrin freres Monthly Average Sales Per 2 Years')
plt.xlabel('Decade')
plt.ylabel('Sales')
plt.grid()
st.pyplot(fig5)

df_1=df.groupby(df.index.year).mean().rename(columns={'Sales':'Mean'})
df_1= df_1.merge(df.groupby(df.index.year).std().rename(columns={'Sales':'Std'}), left_index=True, right_index=True)
df_1['Cov_pct']=((df_1["Std"]/df_1["Mean"])*100).round(2)
st.write(df_1.head())

st.header("Perrin freres Monthly Average Sales")
fig6, ax = plt.subplots(figsize=(12,8))
df_1['Cov_pct'].plot()
plt.title(f'Perrin freres Monthly Average Sales')
plt.xlabel('Year')
plt.ylabel('cv in %')
plt.grid()
st.pyplot(fig6)

st.header("Rolling Mean & Standard Deviation")
rolmean= df.rolling(window=12).mean()
rolstd= df.rolling(window=12).std()
fig7, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=df.index,y="Sales",data=df,label='Original')
sns.lineplot(x=rolmean.index,y="Sales",data=rolmean,label='Rolling Mean')
sns.lineplot(x=rolstd.index,y="Sales",data=rolstd,label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
st.pyplot(fig7)





test_result = adfuller(df['Sales'])

def adfuller_test(sales):
    result= adfuller(sales)
    labels=['ADF Test Statistic','p-value', '#Lags Used', 'Number of Observaations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value))
        if result[1]<=0.05:
            print('Data is Stationary')
        else:
            print('Data is not stationary')
        
adfuller_test(df['Sales'])

df['Seasonal First Difference'] = df['Sales'] - df['Sales'].shift(12)
adfuller_test(df['Seasonal First Difference'].dropna())


st.header("Line Plot After Making the Data Stationary")
fig8, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=df.index,y="Seasonal First Difference",data=df)
plt.xlabel('Year')
plt.ylabel('Seasonal First Difference')
st.pyplot(fig8)

st.header("Rolling Mean & Standard Deviation After Making the Data Stationary")
rolmean= df.rolling(window=12).mean()
rolstd= df.rolling(window=12).std()
fig7, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=df.index,y="Seasonal First Difference",data=df,label='Original')
sns.lineplot(x=rolmean.index,y="Seasonal First Difference",data=rolmean,label='Rolling Mean')
sns.lineplot(x=rolstd.index,y="Seasonal First Difference",data=rolstd,label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation After Making Data Stationary')
st.pyplot(fig7)

st.header("Creating Autocorrelation and Partial Autocorrelation Graph")
fig9 = plt.figure(figsize=(12,8))
ax2 = fig9.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)
#Here iloc[13:] used because first 12 values are nan.
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
st.pyplot(fig9)


model = ARIMA(df['Sales'], order=(1, 1, 1))
model_fit = model.fit()


model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=90,end=103,dynamic=True)


st.header("Plotting Forecast and Actual Value")
fig10, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=df.index,y="Sales",data=df,label='Original')
sns.lineplot(x=df.index,y="forecast",data=df,label='forecast')
st.pyplot(fig10)



future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
future_df=pd.concat([df,future_datest_df])
future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)
st.header("Plotting New Forecasted Data Given By the Model")
fig11, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=future_df.index,y="Sales",data=future_df,label='Original')
sns.lineplot(x=future_df.index,y="forecast",data=future_df,label='forecast')
st.pyplot(fig11)
