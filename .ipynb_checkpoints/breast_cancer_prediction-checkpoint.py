                                        # BREAST CANCER DETECTION MACHINE LEARNING MODEL
### Import the Dependencies
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn
import warnings
import joblib
### Datacollection and Preprocessing
# loading dataset from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)
# loading the data into pandas datafram
df = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names )
df.head()
# adding the target coloum in dataframe
df['label'] = breast_cancer_dataset.target
df.tail()
# number of rows and columns in dataset
df.shape
# getting some information about the data
df.info()
# checking the missing values
df.isnull().sum()
# statistical measures of the data
df.describe()
# checking the distribution of target variable
df['label'].value_counts()
df.groupby('label').mean()
### seperating the feature and target columns
x = df.drop(columns= 'label', axis = 1)
y = df['label']
print(y)
### splitting the data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
print(x.shape, x_train.shape, x_test.shape)
# Model training ---> Logistic regression
model = LogisticRegression()
# training the logistict Regression model using training data
model.fit(x_train,y_train)

# save the trained model
joblib.dump(model, 'breast,cancer_model.pkl')
print("Model saved successfully as'breast,cancer_model.pkl'")
warnings.filterwarnings("ignore")
### model evaluation
x_train_prediction = model.predict(x_train)
accuracy = accuracy_score(y_train, x_train_prediction)
print("The accuracy score of the modle is", accuracy)
x_train_prediction_test = model.predict(x_test)
accuracy = accuracy_score(y_test, x_train_prediction_test)
print("The accuracy score of the modle is", accuracy)
### building a predictive sysytem
input_user = (17.99,	10.38,	122.8,	1001,	0.1184,	0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1.095,	0.9053,	8.589,	153.4,	0.006399,	0.04904,	0.05373,	0.01587,	0.03003,	0.006193,	25.38,	17.33,	184.6,	2019,	0.1622,	0.6656,	0.7119,	0.2654,	0.4601,	0.1189,)
#Input Columns are
print(" Columns Which are used as input:-\n radius_mean\n texture_mean\n perimeter_mean\n area_mean\n smoothness_mean\n compactness_mean\n concavity_mean\\nconcave points_mean\n symmetry_mean\n fractal_dimension_mean\n radius_se\n texture_se\n perimeter_se\n area_se\n smoothness_se\n compactness_se\n concavity_se\\nconcave points_se\n symmetry_se\n fractal_dimension_se\n radius_worst\n texture_worst\n perimeter_worst\n area_worst\n smoothness_worst\n compactness_worst\n concavity_worst\\nconcave points_worst\n symmetry_worst\n fractal_dimension_worst")
# coverting it into numpy array
input_data_numpy_array = np.asarray(input_user)
# reshape the numpy array array as we are expecting the one datapoint from the user
input_data_reshape = input_data_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshape)
if prediction[0] == 0:
    print(f"The cancer is Malignanat !!! just go for treatment" )
else:
    print("The Cancer is Benign")
### Some more sample data for testing:-
'''
You can select a row from the below and test the model:-

17.99,	10.38,	122.8,	1001,	0.1184,	0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1.095,	0.9053,	8.589,	153.4,	0.006399,	0.04904,	0.05373,	0.01587,	0.03003,	0.006193,	25.38,	17.33,	184.6,	2019,	0.1622,	0.6656,	0.7119,	0.2654,	0.4601,	0.1189,
20.57,	17.77,	132.9,	1326,	0.08474,	0.07864,	0.0869,	0.07017,	0.1812,	0.05667,	0.5435,	0.7339,	3.398,	74.08,	0.005225,	0.01308,	0.0186,	0.0134,	0.01389,	0.003532,	24.99,	23.41,	158.8,	1956,	0.1238,	0.1866,	0.2416,	0.186,	0.275,	0.08902,
19.69,	21.25,	130,	1203,	0.1096,	0.1599,	0.1974,	0.1279,	0.2069,	0.05999,	0.7456,	0.7869,	4.585,	94.03,	0.00615,	0.04006,	0.03832,	0.02058,	0.0225,	0.004571,	23.57,	25.53,	152.5,	1709,	0.1444,	0.4245,	0.4504,	0.243,	0.3613,	0.08758,
11.42,	20.38,	77.58,	386.1,	0.1425,	0.2839,	0.2414,	0.1052,	0.2597,	0.09744,	0.4956,	1.156,	3.445,	27.23,	0.00911,	0.07458,	0.05661,	0.01867,	0.05963,	0.009208,	14.91,	26.5,	98.87,	567.7,	0.2098,	0.8663,	0.6869,	0.2575,	0.6638,	0.173
20.29,	14.34,	135.1,	1297,	0.1003,	0.1328,	0.198,	0.1043,	0.1809,	0.05883,	0.7572,	0.7813,	5.438,	94.44,	0.01149,	0.02461,	0.05688,	0.01885,	0.01756,	0.005115,	22.54,	16.67,	152.2,	1575,	0.1374,	0.205,	0.4,	0.1625,	0.2364,	0.07678,

13.05,	19.31,	82.61,	527.2,	0.0806,	0.03789,	0.000692,	0.004167,	0.1819,	0.05501,	0.404,	1.214,	2.595,	32.96,	0.007491,	0.008593,	0.000692,	0.004167,	0.0219,	0.00299,	14.23,	22.25,	90.24,	624.1,	0.1021,	0.06191,	0.001845,	0.01111,	0.2439,	0.06289
8.618,	11.79,	54.34,	224.5,	0.09752,	0.05272,	0.02061,	0.007799,	0.1683,	0.07187,	0.1559,	0.5796,	1.046,	8.322,	0.01011,	0.01055,	0.01981,	0.005742,	0.0209,	0.002788,	9.507,	15.4,	59.9,	274.9,	0.1733,	0.1239,	0.1168,	0.04419,	0.322,	0.09026
10.17,	14.88,	64.55,	311.9,	0.1134,	0.08061,	0.01084,	0.0129,	0.2743,	0.0696,	0.5158,	1.441,	3.312,	34.62,	0.007514,	0.01099,	0.007665,	0.008193,	0.04183,	0.005953,	11.02,	17.45,	69.86,	368.6,	0.1275,	0.09866,	0.02168,	0.02579,	0.3557,	0.0802
8.598,	20.98,	54.66,	221.8,	0.1243,	0.08963,	0.03,	0.009259,	0.1828,	0.06757	0.3582	2.067,	2.493,	18.39,	0.01193,	0.03162,	0.03,	0.009259,	0.03357,	0.003048,	9.565,	27.04,	62.06,	273.9,	0.1639,	0.1698,	0.09001,	0.02778,	0.2972,	0.07712
11.52,	18.75,	73.34,	409,	0.09524,	0.05473,	0.03036	0.02278,	0.192,	0.05907,	0.3249,	0.9591,	2.183,	23.47,	0.008328,	0.008722,	0.01349,	0.00867,	0.03218,	0.002386,	12.84,	22.47,	81.81,	506.2,	0.1249,	0.0872,	0.09076,	0.06316,	0.3306,	0.07036'''

print(np.__version__, "numpy")
print(pd.__version__, "numpy")
print(sklearn.__version__, "numpy")