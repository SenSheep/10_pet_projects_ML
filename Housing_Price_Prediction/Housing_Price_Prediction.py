import pandas as pd
from sklearn.linear_model import LinearRegression

hpptrain = pd.read_csv('Housing_Price_Prediction/train.csv')
hpptest = pd.read_csv('Housing_Price_Prediction/test.csv')

X_train = hpptrain.drop("SalePrice", axis=1)
y_train = hpptrain['SalePrice']

X_test = hpptest

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

hpptest['SalePrice'] = y_pred
print(hpptest.head())