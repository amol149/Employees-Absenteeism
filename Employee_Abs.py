#################################### load Packages ##################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os
import csv

##Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

############################ change working Directory ################
os.chdir('D:\Python Basics\Pycharm\Employee_Absentisum')

################################ Load Data #################################
df = pd.read_excel('Absenteeism_at_work.xls')
df.head()

df.shape

#Rename Column
df.rename(columns=({'Work load Average/day ': 'Work load Average per day'}), inplace=True)
df.head()

############################ Missing Value DataFrame #########################
miss_value = pd.DataFrame({'Actual_value' : df.isnull().sum(),
                           'Miss_Percentage': (df.isnull().sum()/len(df)*100),
                           'Data_Types' : df.dtypes}).sort_values('Actual_value', ascending=True)
miss_value

num = ['Distance from Residence to Work', 'Service time', 'Age','Work load Average per day','Transportation expense',
       'Hit target', 'Son', 'Pet', 'Weight', 'Height', 'Body mass index']

cat = list(set(df.columns) - set(num))
cat

########################### find out no of categories in column ##########################
for i in cat:
    print(i, ':', len(df[i].unique()), 'labels')


#Imputing missing value
#for var in cat:
#     df[var].fillna(df[var].mode(), inplace=True)

#Mean = 79.05
#median = 83

#df['Weight'].loc[70] #= np.nan


###################### Imputing Missing Values ###########################
from sklearn.preprocessing import Imputer
impute_cat = Imputer(missing_values=np.nan, strategy='most_frequent')
impute_cat = impute_cat.fit(df[cat])
df[cat] = impute_cat.transform(df[cat])
impute_num = Imputer(missing_values=np.nan, strategy='median')
impute_num = impute_num.fit(df[num])
df[num] = impute_num.transform(df[num])

######################## Box Plot for Every Numerical Variable ########################
fig = plt.figure(figsize=(13,13))
for i, var_name in enumerate(num):
    ax= fig.add_subplot(5,3, i+1)
    flierprops = dict(marker='.', markersize=6, markerfacecolor='black')
    df.boxplot(column=var_name, ax=ax, flierprops=flierprops)
plt.tight_layout()
plt.show()

##################################### Corelation Plot(HeatMap) ######################
mask = np.zeros_like(df[num].corr())
traingle_indices = np.triu_indices_from(mask)
mask[traingle_indices] = True
plt.figure()
sns.heatmap(round(df[num].corr(), 2), mask = mask, annot=True, annot_kws={'size': 6}, linewidths=1.6)
plt.xticks()
plt.yticks()
plt.tight_layout()
plt.show()

################### Histogram plot for all numerical variable #############################

fig = plt.figure(figsize=(9,9))
for i, var_name in enumerate(num):
    mu, std = norm.fit(df[var_name])
    ax = fig.add_subplot(5, 3, i+1)
    df[var_name].hist(density= True, color= 'w', edgecolor= 'b', ax= ax)
    xmin, xmax = min(df[var_name]), max(df[var_name])
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p , 'k', linewidth = 2)
    ax.set_title(var_name, fontsize=8)
fig.tight_layout()
plt.show()

df.shape

#################### Outlier removal ##################################
def outlier_removal(df, cols):
    for i in cols:
        print(i)
        q75, q25 = np.percentile(df.loc[:, i], [75, 25])
        IQR = q75 - q25
        min_value = q25 - (1.5 * IQR)
        max_value = q75 + (1.5 * IQR)
        print(q75,q25, IQR, min_value, max_value)
        df = df.drop(df[df.loc[:, i] < min_value].index)
        df = df.drop(df[df.loc[:, i] > max_value].index)
        print(df.shape[0])
    return df
  
###### drop target variable from categorical variable
cat_dummies = cat
cat_dummies.pop(1)
cat_dummies

df.columns

#Dummies variable
df = pd.get_dummies(columns=cat_dummies, data=df)

#### Sliceing of data set
X = df.drop(['Absenteeism time in hours', 'ID'], axis=1).values
y = df['Absenteeism time in hours'].values.reshape(-1, 1)

######################### Normalize data ############################
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

### split dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X, y,  test_size = 0.20, random_state=2)

#cross_check splitting
print(len(X_train)/len(df)*100)
print(len(X_test)/len(df)*100)
print(len(y_train)/len(df)*100)
print(len(y_test)/len(df)*100)

##################################### Linear Regression #####################################
LR = LinearRegression()
LR.fit(X_train, y_train)
predict_LR_train = LR.predict(X_train)
predict_LR_test = LR.predict(X_test)

l_train = np.sqrt(mean_squared_error(y_train, predict_LR_train))
l_test = np.sqrt(mean_squared_error(y_test, predict_LR_test))

###################################### Support Vector Regression ##########################################
svr = SVR(kernel='linear')
svr.fit(X_train, y_train)
predict_svr_train = svr.predict(X_train)
predict_svr_test = svr.predict(X_test)

s_train = np.sqrt(mean_squared_error(y_train, predict_svr_train))
s_test = np.sqrt(mean_squared_error(y_test, predict_svr_test))

############################################# Decision Tree Regression ##################################
DTR = DecisionTreeRegressor(min_impurity_decrease=0.01, random_state=0)
DTR.fit(X_train, y_train)
predict_DTR_train = DTR.predict(X_train)
predict_DTR_test = DTR.predict(X_test)

d_train = np.sqrt(mean_squared_error(y_train, predict_DTR_train))
d_test = np.sqrt(mean_squared_error(y_test, predict_DTR_test))

############################################ Random Forest Regression ####################################
RFR = RandomForestRegressor(n_estimators=500, random_state=0)
RFR.fit(X_train, y_train)
predict_RFR_train = RFR.predict(X_train)
predict_RFR_test = RFR.predict(X_test)

r_train = np.sqrt(mean_squared_error(y_train, predict_RFR_train))
r_test = np.sqrt(mean_squared_error(y_test, predict_RFR_test))

#################################### Gradient Boosting Regression ##################################
GBR = GradientBoostingRegressor(max_depth=2, random_state=0)
GBR.fit(X_train, y_train)
predict_GBR_train = GBR.predict(X_train)
predict_GBR_test = GBR.predict(X_test)

g_train = np.sqrt(mean_squared_error(y_train, predict_GBR_train))
g_test = np.sqrt(mean_squared_error(y_test, predict_GBR_test))

################################### Logistic Regression ##########################################
lor = GradientBoostingRegressor()
lor.fit(X_train, y_train)
predict_lor_train = lor.predict(X_train)
predict_lor_test = lor.predict(X_test)

lr_test = np.sqrt(mean_squared_error(y_test, predict_lor_test))
lr_train = np.sqrt(mean_squared_error(y_train, predict_lor_train))
#RMSE_test > RMSE_train Overfitting
#RMSE_test < RMSE_train Underfitting

################### Create DataFrame of Model Result of RMSE
model = pd.DataFrame({'Train_RMSE':[l_train, s_train, d_train, r_test, g_train, lr_train],
              'Test_RMSE':[l_test, s_test, d_test, r_test, g_test, lr_test]},
             index=['Linear R', 'SVR', 'DT', 'RF', 'GBR', 'Logistic R']).reset_index()

model = model.rename(columns={'index' : 'Model Type'})

model.to_csv('Model_result.csv', header=True, sep=',')

















