import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from numpy import log
import seaborn as sns


#Integrated (d)
from statsmodels.tsa.stattools import adfuller

#Auto Regressive (p)
from statsmodels.graphics.tsaplots import  plot_pacf
from statsmodels.tsa.stattools import pacf

#Moving Average (q)
from statsmodels.graphics.tsaplots import  plot_acf
from statsmodels.tsa.stattools import acf

#grid search
import itertools

#Performance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class ARIMA_MODEL():

    def __init__(self,df):
        self.df = df
        self.p = 0
        self.d = 0
        self.q = 0

        self.train = df[df["datetime"] < "2019-07-29"].reset_index(drop=True)
        self.test = df[df["datetime"] >= "2019-08-10"].reset_index(drop=True)

    
    def find_d(self):
        result = adfuller(self.df["delta_t"])   #check data is stationary or not 
        print('p-value: {}'.format(result[1]))
        if result[1] < 0.05:
            self.d = 0
            print("d = 0")
        else : 
            for i in range(1,3):
                tmp = self.df['delta_t'].diff(i)
                result = adfuller(tmp.dropna(), autolag = 'AIC')
                if result[1] < 0.05:
                    print("d = " + str(i) )
                    print('p-Values:' + str(result[1]))
                    self.d = i
                    break
                

    def find_p(self):
        plt.rcParams.update({'figure.figsize':(10,4)})
        plot_pacf(self.df['delta_t'].dropna(), method = 'ols')
        df_pacf = pacf(self.df['delta_t'].dropna(), method = 'ols')

        for i in range(0, len(df_pacf)):
            if df_pacf[i] < 1.96 / np.sqrt( len(self.df['delta_t']) ):
                print('p = ', i - 1)
                self.p = i-1
                break
    

    def find_q(self):
        plt.rcParams.update({'figure.figsize':(10,4)})
        plot_acf(self.df['delta_t'], fft = True)
        df_acf = acf(self.df['delta_t'], fft = True)

        for i in range(0, len(df_acf)):
            if df_acf[i] < 1.96 / np.sqrt(len(self.df['delta_t'])):
                print('q=', i - 1)
                self.q = i-1
                break

    def grid_search(self):
        p = range(3, 10)
        d = range(0,1)
        q = range(0, 12)
        pdq = list(itertools.product(p, d, q))
        lowest = 100000000.0
        aic = []

        for param in pdq:
            try:
                model = ARIMA(self.train['delta_t'].dropna(), order = param)
                results = model.fit()
                #print('Order = {}'.format(param))
                #print('AIC = {}'.format(results.aic))
                if results.aic < lowest :
                    lowest = results.aic
                    print('Order = {}'.format(param))
                    print('AIC = {}'.format(results.aic))
                    self.results = results
                    self.p = param[0]
                    self.d = param[1]
                    self.q = param[2]
                a = 'Order: '+str(param) +' AIC: ' + str(results.aic)
                aic.append(a)
            except:
                continue


    def train_model(self):
        model = ARIMA(self.train["delta_t"].dropna(), order=(self.p,self.d,self.q))
        self.results = model.fit()

        #Prediction
        plt.figsize = (25,5)
        plt.plot(self.train['delta_t'], color = 'green', label = 'Actual diff')
        plt.plot(self.results.predict(), color= 'orange', label = 'Predicted diff')
        plt.legend()
        plt.title("ARIMA({},{},{})".format(self.p,self.d,self.q))
        

    
    def test_model(self):

        start_index = 0
        end_index = len(self.test)

        test_results = self.results.predict(start=start_index, end=end_index)
        prediction = pd.DataFrame(test_results, columns = ['Predicted'])
        df_pred = pd.merge(self.test, prediction, how = 'left', left_index = True, right_index = True)

        rmse = mean_squared_error(df_pred['delta_t'], df_pred['Predicted'],squared=False)
        mse = mean_squared_error(df_pred['delta_t'], df_pred['Predicted'])
        mae = mean_absolute_error(df_pred['delta_t'], df_pred['Predicted'])
        

        plt.figure(figsize=(25,5))
        plt.plot(df_pred['delta_t'], color = 'green', label = 'Actual delta_t')
        plt.plot(df_pred['Predicted'], color='orange', label = 'Predicted delta_t')
        plt.legend()
        plt.title("ARIMA({},{},{}) , RMSE = {} ,  MSE = {} , MAE = {}".format(self.p,self.d,self.q,rmse,mse,mae))

       

        print("MSE = " + str(mse) )
        print("MAE = " + str(mae) )
        

    def get_model(self):
        return self.results


