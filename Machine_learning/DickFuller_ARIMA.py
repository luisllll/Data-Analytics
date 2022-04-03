# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 12:00:49 2022

@author: luisl
"""


import os
os.chdir('C:/Users/luisl/OneDrive/Escritorio/ml-scripts')


# importing required libraries
import pandas as pd
import numpy as np

data = pd.read_csv('AirPassengers.csv')
print (data.head())
print ('\n Data Types:')
print (data.dtypes)


data.tail()

#dar formato as la frcha del dt
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
print ('\n Parsed Data:')
print (data.head())


data.index


ts=data['#Passengers']

ts['1949-01-01']



#libreria datetime para manejar mejor las frchas
!pip install datetime
from datetime import datetime

ts[datetime(1949,1,1)]


#desde incio hasta 1949-05-01'
ts[:'1949-05-01']



import matplotlib.pyplot as plt

#plot serie tmeporal
plt.plot(ts)



#Dick fuller test para comprobar la estacionalidad

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(ts).rolling(window=12).mean()
    rolstd = pd.Series(ts).rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)



test_stationarity(ts)




#pasar a logaritmico apra comparar crecimeintos
ts_log = np.log(ts)
plt.plot(ts_log)



##DESCOMPONER LA SERIE TMEPORAL)


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()



#hacemos dickfuller sobre el residual
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)
#este si que es muy stationary( prox to 0)





##MODELO DE PREDICCIÓN ARIMA##
from statsmodels.tsa.arima_model import ARIMA



model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(results_ARIMA.fittedvalues, color='red')




##devolverlos a la esclaa original para comparar predicciones y original

#guarrdar predicciones
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff)


#pasar de dif a escala logaritmica pimero determinamos la suma acumulativa
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head)

#después la añadimos al númeor base las diferncias
predictions_ARIMA_log = pd.Series(ts_log[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


#finalmente aplicamos exponencial para sacar el valor a escala inicial
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))






































