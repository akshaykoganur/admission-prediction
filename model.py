import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Admission_Predict.csv')

for col in dataset.columns:
    if ' ' in col:
        dataset = dataset.rename(columns={col:col.replace(' ','_')})
dataset = dataset.drop(['Serial_No.'], axis=1)

X = dataset.drop('Chance_of_Admit_',axis='columns')
y = dataset['Chance_of_Admit_']

regressor = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[337, 118, 4, 4.5, 4.5, 9.65, 0]]))