import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Housing_Price_Prediction/train.csv')
dftest = pd.read_csv('Housing_Price_Prediction/test.csv')

categorical_column = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                    'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
                    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 
                    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
                    'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                    'GarageQual', 'GarageCond', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'PoolQC']
numeric_column = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 
                  'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
                  'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
                  'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
                  'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

### !!!ОБУЧЕНИЕ!!!

#   Работа с NaN
# Для числовых данных (заполнение средним)
imputer_num = SimpleImputer(strategy='mean')
df[numeric_column] = imputer_num.fit_transform(df[numeric_column])
# Для категориальных данных (заполнение самой частой категорией)
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_column] = imputer_cat.fit_transform(df[categorical_column])
#   Работа с NaN

#   Нормирование
# One-Hot Encoding
df = pd.get_dummies(df, columns=categorical_column)
# Скалируем
scaler = StandardScaler()
# Масштабируем числовые данные
df[numeric_column] = scaler.fit_transform(df[numeric_column])   
#   Нормирование
one_hot_columns = df.columns

df = df.drop('Id', axis=1)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
### !!!ОБУЧЕНИЕ!!!

### !!!ПРЕДСКАЗАНИЯ!!!
# Подготовка тестового набора данных
dftest['SalePrice'] = None

# NaN
dftest[numeric_column] = imputer_num.transform(dftest[numeric_column])
dftest[categorical_column] = imputer_cat.transform(dftest[categorical_column])

# Нормализация и Скалирование
dftest = pd.get_dummies(dftest, columns=categorical_column)
dftest = dftest.reindex(columns=one_hot_columns, fill_value=0)
dftest[numeric_column] = scaler.transform(dftest[numeric_column]) 

dftesttemp = dftest.drop("Id", axis=1)
dftesttemp = dftesttemp.drop("SalePrice", axis=1)

# Предсказания и сохранение результатов
dftest['SalePrice'] = model.predict(dftesttemp)
dftest[['Id', 'SalePrice']].to_csv('submission.csv', index=False)