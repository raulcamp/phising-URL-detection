import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import re
import math
import train_model


model = pickle.load(open('xgb_model', 'rb'))

# testing
test_data = pd.read_csv('./Phishing_Test.csv')
ids = test_data['Unnamed: 0']
urls = test_data['URL']
del test_data['Unnamed: 0']
del test_data['Label']

train_model.feature_transform(test_data)
labels = model.predict(test_data)

def unfeaturize(data):
    del data['numSD']
    del data['hasPercent']
    del data['special']
    del data['numerical']
    del data['len_url']

unfeaturize(test_data)
test_data.insert(0, '',ids )
test_data.insert(4, 'URL', urls)
test_data.insert(5, 'Label', labels)

test_data.to_csv(r'./Phishing_Labels.csv', index=False)

if __name__ == '__main__':
    pass