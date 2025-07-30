#Ridge Regression in Python 

#Step 1: Import all required libraries
#Step 2: Define Features and Target Variables
#Step 3: Train Test Split -> Divide dataset into two parts
#Step 4: Apply Ridge Regression -> Or Whichever Model You Are Using
#Step 5: Get Intercept and Coeff for Ridge Regression
#Step 6: Predict using Ridge Regression
#Step 7: Create a Dataframe with Actual & Predicted Values
#Step 8: Plot Actual & Predicted Values
#Step 9: Evaluate the Model - R square, mse, rmse

#Step 1: Import The Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
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

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.044, shuffle = False)


#Step 4: Apply Ridge Regression

ridge = Ridge(alpha = 0.1)
ridge.fit(x_train, y_train) #training the model 

#Step 5: Get Intercept and Coeff for Ridge Regression

coefficients = ridge.coef_
print(coefficients)

intercept = ridge.intercept_
print(intercept)

coeff_df = pd.DataFrame({"Feature":x.columns, "Coefficients":coefficients})
print(coeff_df)



#Step 6: Predict Using Ridge Regression

y_pred = ridge.predict(x_test)
print(y_pred)


#Step 7: Create a Dataframe with Actual & Predicted Values

df_result = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})


#Step 8: Plot Actual & Predicted Values

plt.figure(figsize = (14,6))
plt.plot(df_result.index, df_result["Actual"], label ="Actual",color = "black")
plt.plot(df_result.index, df_result['Predicted'], label = 'Predicted',color = 'red')
plt.title("Actual vs Prediction for AAPL Stock (2025) - Ridge Regression")
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
