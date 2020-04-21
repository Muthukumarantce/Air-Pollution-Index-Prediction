#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2, RFE, RFECV
import xgboost as xgb
import statsmodels.api as sm


#Reading Input data
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
test_data = pd.read_csv("data/test.csv")
train_data = pd.read_csv("data/train.csv")
print(test_data.head(5))
print(train_data.head(5))

#Check number of Categorical and numberical variables
train_data.select_dtypes(include=['int64','float64']).columns
train_data.select_dtypes(include=['object']).columns


#Categorical Feature Transformation
df1 = pd.get_dummies(train_data['is_holiday']).rename(columns={'None': 'Not_Holiday'})
df1.head(1)
df2 = pd.get_dummies(train_data['weather_type'],prefix="Weather")
df2.head(1)
train_data=train_data.drop(["is_holiday","weather_type"],axis=1)
df1.info()
df2.info()
train_data=pd.concat([train_data,df1,df2],axis=1)
#Weather_squall is not present in test data
train_data=train_data.drop('Weather_Squall',axis=1)

train_data.info()


#Date Feature Engineering
train_data['date_time'] = pd.to_datetime(train_data['date_time'])
train_data['year'] = pd.DatetimeIndex(train_data['date_time']).year
train_data['month'] = pd.DatetimeIndex(train_data['date_time']).month
train_data['day'] = pd.DatetimeIndex(train_data['date_time']).day
train_data['dayofweek'] = pd.DatetimeIndex(train_data['date_time']).dayofweek
train_data['hour'] = pd.DatetimeIndex(train_data['date_time']).hour

#Univariate Analysis
train_data['air_pollution_index'].describe()
train_data['humidity'].describe()
plt.hist(train_data['air_pollution_index'])
plt.hist(train_data['humidity'])
plt.hist(train_data['wind_speed'])
plt.hist(train_data['wind_direction'])
plt.hist(train_data['visibility_in_miles'])
plt.hist(train_data['dew_point'])
#Both Visibility in miles and Dew points are Equal in Values. One can be ignored
train_data= train_data.drop(['dew_point'],axis=1)

plt.hist(train_data['temperature'])
plt.hist(train_data['rain_p_h'])
plt.hist(train_data['snow_p_h'])


#Missing features
missing_features = train_data.columns[train_data.isnull().any()]
missing_features



#Train Test Data Splitup
X=train_data.drop(['air_pollution_index','date_time'],axis=1)
y=train_data['air_pollution_index']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=20,test_size=0.2,stratify=y)


X.describe()
#Feature Scaling

##Apply standardization to numerical features
num_cols = ['humidity','wind_speed','wind_direction','visibility_in_miles','temperature','rain_p_h','snow_p_h','clouds_all','traffic_volume','year','month','day','dayofweek','hour']

X_train_stand = X_train.copy()
X_test_stand = X_test.copy()

for i in num_cols:
    
    # fit on training data column
    scale = StandardScaler().fit(X_train_stand[[i]])
    
    # transform the training data column
    X_train_stand[i] = scale.transform(X_train_stand[[i]])
    
    # transform the testing data column
    X_test_stand[i] = scale.transform(X_test_stand[[i]])


X_train_stand.describe()

##Apply MinMaxScaling to numerical features

X_train_norm = X_train.copy()
X_test_norm = X_test.copy()

for i in num_cols:
    
    # fit on training data column
    norm = MinMaxScaler().fit(X_train_norm[[i]])
    
    # transform the training data column
    X_train_norm[i] = norm.transform(X_train_norm[[i]])
    
    # transform the testing data column
    X_test_norm[i] = norm.transform(X_test_norm[[i]])


X_train_norm.describe()
#Correlation
import seaborn as sns
corrmat = X_train_norm.corr()
g=sns.heatmap(corrmat)

topcorr_features = corrmat.index[abs(corrmat['air_pollution_index']>0.5)]


# raw, normalized and standardized training and testing data
trainX = [X_train, X_train_norm, X_train_stand]
testX = [X_test, X_test_norm, X_test_stand]
#Building the model
#Linear Models

#Linear Regression
rmse=[]
lin = LinearRegression()
#lin.fit(X_train,y_train)
#lin_pred=lin.predict(X_test)
#lin_pred = np.round(lin_pred)
# model fitting and measuring RMAE
for i in range(len(trainX)):
    
    # fit
    lin.fit(trainX[i],y_train)
    # predict
    pred = np.round(lin.predict(testX[i]))
    # RMSE
    rmse.append(100-np.sqrt(mean_absolute_error(y_test,pred)))
    
df_lin = pd.DataFrame({'RMAE':rmse},index=['Original','Normalized','Standardized'])


#Ridge Regression
rmse=[]
ridge=Ridge(max_iter=200,alpha=5.0)
#ridge.fit(X_train,y_train)
#ridge_pred = ridge.predict(X_test)
#ridge_pred = np.round(ridge_pred)
# model fitting and measuring RMAE
for i in range(len(trainX)):
    
    # fit
    ridge.fit(trainX[i],y_train)
    # predict
    pred = np.round(ridge.predict(testX[i]))
    # RMSE
    rmse.append(100-np.sqrt(mean_absolute_error(y_test,pred)))
    
df_ridge = pd.DataFrame({'RMAE':rmse},index=['Original','Normalized','Standardized'])




#LASSO Regression
rmse=[]
#max_iter=300 91.51447
#275 91.51447
#2,200 91.51466
#1.18,200 91.5151
lasso = Lasso(alpha=1.18,max_iter=200) 
#lasso.fit(X_train,y_train)
#lasso_pred = lasso.predict(X_test)
#lasso_pred = np.round(lasso_pred)
for i in range(len(trainX)):
    
    # fit
    lasso.fit(trainX[i],y_train)
    # predict
    pred = np.round(lasso.predict(testX[i]))
    # RMSE
    rmse.append(100-np.sqrt(mean_absolute_error(y_test,pred)))
    
df_lasso = pd.DataFrame({'RMAE':rmse},index=['Original','Normalized','Standardized'])



enet = ElasticNet(alpha=0.01)
enet.fit(X_train,y_train)
enet_pred = enet.predict(X_test)
enet_pred = np.round(enet_pred)


#Regression with Tree models 
#XGBoost
#91.51399 - n_estimators = 377,max_depth=1, learning_rate=0.0367,gamma=0.22,min_child_weight=4.44,colsample_bytree=1
#91.51323 - n_estimators = 50
#91.51353 - n_estimatiors = 40
#91.51226 - n_estimators = 35
#91.51265 Average of xgb(40) and lasso()
#91.51389 0.8 and 0.2
rmse=[]
model_xgb = xgb.XGBRegressor(objective = 'reg:squarederror',n_estimators=377, max_depth=1, learning_rate=0.0367,gamma=0.22,min_child_weight=4.44,colsample_bytree=1) 
#model_xgb.fit(X_train,y_train)
#xgb_pred = model_xgb.predict(X_test)
#xgb_pred = np.round(xgb_pred)
for i in range(len(trainX)):
    
    # fit
    model_xgb.fit(trainX[i],y_train)
    # predict
    pred = np.round(model_xgb.predict(testX[i]))
    # RMSE
    rmse.append(100-np.sqrt(mean_absolute_error(y_test,pred)))
    
df_xgb = pd.DataFrame({'RMAE':rmse},index=['Original','Normalized','Standardized'])





#Evaluation of Metrics
print("linear_regression:",np.sqrt(mean_absolute_error(lin_pred,y_test)))
print("ridge_regression:", np.sqrt(mean_absolute_error(ridge_pred,y_test)))
print("lasso_regression:" ,100-np.sqrt(mean_absolute_error(lasso_pred,y_test)))
print("Elastic_regression:" ,np.sqrt(mean_absolute_error(enet_pred,y_test)))
print("XGB Regression:", 100-np.sqrt(mean_absolute_error(xgb_pred,y_test)))
print("Combined:",100-np.sqrt(mean_absolute_error(test_consol['average'],y_test)))
print("lasso_regression:" ,100-np.sqrt(mean_absolute_error(lasso_poly_pred,y_test)))


#HyperParameter Tuning for Lasso & Ridge
max_iter = [int(x) for x in np.linspace(start =200,stop=2000,num=10)]
alpha = [x for x in np.linspace(0.1,5,num=10)]
param_grid = dict(max_iter=max_iter,alpha=alpha)
grid = GridSearchCV(estimator=lasso,param_grid = param_grid,cv=4,n_jobs=-1)
grid_result = grid.fit(X_train_stand,y_train)
print(grid_result.best_params_)
print(grid_result.best_score_)


#HyperParameter Tuning for Xgboost
n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
max_depth = [int(x) for x in np.linspace(1,20, num=10)]
gamma = [x for x in np.linspace(0, 0.4, num=10)]
learning_rate = [float(x) for x in np.linspace(0.005, 0.1, num=10)]
min_child_weight = [float(x) for x in np.linspace(0, 10, num=10)]
colsample_bytree = [0.3, 0.5, 0.7, 1]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'learning_rate':learning_rate,
               'gamma':gamma,
               'min_child_weight':min_child_weight,
               'colsample_bytree':colsample_bytree}
model_xgb = xgb.XGBRegressor(random_state=1, objective='reg:squarederror', no_omp=1) #singlethread

xgb_random = RandomizedSearchCV(estimator=model_xgb,#swap in with whatever model you're using
                                param_distributions=random_grid,
                                scoring='neg_mean_absolute_error',
                                n_iter=10,
                                cv=4,
                                n_jobs=-1,
                                verbose=10
                                )

result=xgb_random.fit(X_train_stand, y_train)
params = result.best_params_
params = result.best_score_
y_pred = xgb_random.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred)}")




######TEST DATA
df3 = pd.get_dummies(test_data['is_holiday'])
df3=df3.rename(columns={'None': 'Not_Holiday'})
df3.head(1)
df4 = pd.get_dummies(test_data['weather_type'],prefix="Weather")
df4.head(2)
test_data=test_data.drop(["is_holiday","weather_type"],axis=1)
df3.info()
df4.info()
test_data=pd.concat([test_data,df3,df4],axis=1)
#Weather_squall is not present in test data
test_data=test_data.drop('Weather_Squall',axis=1)


test_data['date_time'] = pd.to_datetime(test_data['date_time'])


#Date Feature Engineering
test_data['year'] = pd.DatetimeIndex(test_data['date_time']).year
test_data['month'] = pd.DatetimeIndex(test_data['date_time']).month
test_data['day'] = pd.DatetimeIndex(test_data['date_time']).day
test_data['dayofweek'] = pd.DatetimeIndex(test_data['date_time']).dayofweek
test_data['hour'] = pd.DatetimeIndex(test_data['date_time']).hour
test_x=test_data.drop(['date_time','dew_point'],axis=1)


#Feature Scaling

##Apply standardization to numerical features
num_cols = ['humidity','wind_speed','wind_direction','visibility_in_miles','temperature','rain_p_h','snow_p_h','clouds_all','traffic_volume','year','month','day','dayofweek','hour']

test_x_stand = test_x.copy()


for i in num_cols:
    
    # fit on training data column
    scale = StandardScaler().fit(test_x_stand[[i]])
    
    # transform the training data column
    test_x_stand[i] = scale.transform(test_x_stand[[i]])
    
    # transform the testing data column
    test_x_stand[i] = scale.transform(test_x_stand[[i]])


test_x_stand.describe()

##Apply MinMaxScaling to numerical features


test_x_norm = test_x.copy()

for i in num_cols:
    
    # fit on training data column
    norm = MinMaxScaler().fit(test_x_norm[[i]])
    
    # transform the training data column
    test_x_norm[i] = norm.transform(test_x_norm[[i]])
    
    # transform the testing data column
    test_x_norm[i] = norm.transform(test_x_norm[[i]])


test_x_norm.describe()


#PREDICTION
pred_test_x_lin = lin.predict(test_x)
pred_test_x_ridge = ridge.predict(test_x)
pred_test_x_lasso = lasso.predict(test_x_stand)
pred_test_x_enet = enet.predict(test_x)
pred_test_x_xgb = model_xgb.predict(test_x_norm)

pred_test_x_lin = np.round(pred_test_x_lin)
pred_test_x_ridge = np.round(pred_test_x_ridge)
pred_test_x_lasso = np.round(pred_test_x_lasso)
pred_test_x_enet = np.round(pred_test_x_enet)
pred_test_x_xgb = np.round(pred_test_x_xgb )


#Combining XGB and lasso - 91.5151
consolidated = pd.DataFrame({'lasso':pred_test_x_lasso,'xgb':pred_test_x_xgb})

consolidated['results'] = (0.8*consolidated['lasso'] + 0.2*consolidated['xgb'])
results = np.round(consolidated['results'])
submission_file = pd.DataFrame({'date_time':test_data['date_time'],'air_pollution_index':results})

np.mean(results)

submission_file.to_csv('output/submission_file.csv')
    

#


# In[17]:





# In[23]:




# In[26]:


max(0,100-mean_absolute_error(y_pred,y_test))

