import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import re
import math

def checkSpecial(url):
    """Returns number of special characters in string"""
    regex = re.compile('[@_!#$%^&*()<>?|}{~]')
    return len([c for c in url if regex.search(c)])

def getNums(url):
    """Returns number of digits in string"""
    return len([c for c in url if c.isdigit()])

def entropy(url):
    """Returns entropy of string"""
    s = url.strip()
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    ent = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
    return ent

def numSubDomains(url):
    """Returns number of subdomains in the given URL"""
    subdomains = url.split('http')[-1].split('//')[-1].split('/')
    return len(subdomains)-1


def feature_transform(df):
    """Featurizes the URL string into the data frame"""
    df.insert(2, 'len_url', [len(url) for url in df['URL']])
    df.insert(2, 'numerical', [getNums(url) for url in df['URL']])
    df.insert(2, 'special', [checkSpecial(url) for url in df['URL']])
    df.insert(2, 'hasPercent', [('%' in url) for url in df['URL']])
    #df.insert(2, 'entropy', [entropy(url) for url in df['URL']])
    df.insert(2, 'numSD', [numSubDomains(url) for url in df['URL']])
    del df['URL']

def train():
    df = pd.read_csv('script/data/Phishing_Dataset.csv')
    feature_transform(df)
    #split into parameters and label for supervised learning
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    #split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    #train model
    model = xgb.XGBClassifier(learning_rate=0.2, use_label_encoder=False)
    model.fit(X_train, y_train)
    #predictions /validation
    preds = model.predict(X_test)
    predictions = [round(value) for value in preds]
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    accuracy = accuracy_score(y_test,predictions)
    f1 = f1_score(y_test, predictions)
    return model, predictions, rmse, f1, accuracy

def save(model):
    pickle.dump(model, open("script/xgb_model", "wb"))

if __name__ == '__main__':
    # train
    model, _, rmse, f1, accuracy = train()
    print("RMSE: %f" % (rmse))
    print("f1: %f" % (f1))
    print("accuracy: %f" % (accuracy))
    # save model
    save(model)