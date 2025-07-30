# Elastic Net - Lasso + Ridge ( Performs Feature Selection & Deals With Multicollinearity)

#Step 1: Import all required libraries
#Step 2: Define Features and Target Variables
#Step 3: Train Test Split -> Divide dataset into two parts
#Step 4: Apply Elastic Net Regression -> Or Whichever Model You Are Using
#Step 5: Get Intercept and Coeff for Elastic Net Regression
#Step 6: Predict using Elastic Net Regression
#Step 7: Create a Dataframe with Actual & Predicted Values
#Step 8: Plot Actual & Predicted Values
#Step 9: Evaluate the Model - R square, mse, rmse


#Step 1: Import The Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#Create Base Dataframe
from logic.data_import import import_fcn
df = import_fcn()
print(df.head())


#Step 2: Define Features and Target Variables
x = df[['AAPL(t-1)','AMZN(t-1)','MSFT(t-1)','QQQ(t-1)','^GSPC(t-1)','AAPL_MA_5','AMZN_MA_5','MSFT_MA_5', 'QQQ_MA_5','^GSPC_MA_5']]
print(x)

y = df["Target"]

#Train Test Split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.044, shuffle = False)#Cannot Shuffle TimeSeries Data

#Step 4: Apply Elastic Net Regression -> Or Whichever Model You Are Using

elastic_net = ElasticNet(alpha = 1, l1_ratio = 0.5)
elastic_net.fit(x_train, y_train) #training the model 

#Alpha = 1, aplha control the strength of regulariatiion (higher alpha means stronger penalty ) - Lambda Parameter
#l1_ratio = 0.5 => Applying 50% lasso and 50% as ridge regression - aplha parameter (theory)

#Step 5: Get Intercept and Coeff for Elastic Net Regression

coefficients = elastic_net.coef_
print(coefficients)

intercept = elastic_net.intercept_
print(intercept)

coeff_df = pd.DataFrame({"Feature":x.columns, "Coefficients":coefficients})
print(coeff_df)

#Step 6: Predict Using Elastic Net Regression

y_pred = elastic_net.predict(x_test)
print(y_pred)

#Step 7: Create a Dataframe with Actual & Predicted Values

df_result = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})

#Step 8: Plot Actual & Predicted Values

#Step 8: Plot Actual & Predicted Values

plt.figure(figsize = (14,6))
plt.plot(df_result.index, df_result["Actual"], label ="Actual",color = "black")
plt.plot(df_result.index, df_result['Predicted'], label = 'Predicted',color = 'red')
plt.title("Actual vs Prediction for AAPL Stock (2025) - Elastic Net Regression")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.grid(True)
plt.tight_layout()
plt.show()

#Step 9: Evaluate the Model - R square, mse, rmse

r2 = r2_score(y_test,y_pred)
print("R squared = ", r2)

mse = mean_squared_error(y_test,y_pred)
print("mse = ", mse)

rmse = np.sqrt(mse)
print("rmse = ", rmse)


#Performance For All Models

#Elastic Net Regression
#R squared =  0.41863269081458065
#mse =  21.26655944796058
#rmse =  4.61156800318076


#Ridge Regression 
#R squared =  0.4494684819690341
#mse =  20.13857860805076
#rmse =  4.487602768522495


#Lasso Regression
#R squared =  0.4230792130752915
#mse =  21.103904567818486
#rmse =  4.593898624024967


#OLS Regression
#R squared =  0.9074352189361041
#mse =  24.123529497307146
#rmse =  4.911570980583213

#Sometimes simple models may perform bette rthan complex models based on the usecase
#Hedge funds sometimes have 200 indpendent variable in their feature enginner phase
#The more time you spend finding the relevant technical indicators which apply to your use case the more accurate your preduictions and model will be

