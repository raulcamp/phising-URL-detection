from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from url_featurizer import URLFeaturizer
import pandas as pd
import xgboost as xgb
import pickle

def transform(df):
    rows = [URLFeaturizer(URL).extract() for URL in df['URL']]
    return pd.concat([pd.DataFrame(rows), df.drop(columns=['URL'])], axis=1, join='inner')

def train(input_path):
    df = transform(pd.read_csv(input_path))
    # split into parameters and label for supervised learning
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    # train model
    model = xgb.XGBClassifier(learning_rate=0.2, use_label_encoder=False)
    model.fit(X_train, y_train)
    # generate predicted labels
    predictions = model.predict(X_test)
    return (model,
            mean_squared_error(y_test, predictions), 
            accuracy_score(y_test, predictions, normalize=True),
            precision_score(y_test, predictions),
            recall_score(y_test, predictions),
            f1_score(y_test, predictions))

def save(model, path):
    pickle.dump(model, open(path, "wb"))

if __name__ == '__main__':
    input_data_path = 'script/data/Phishing_Dataset.csv'
    output_model_path = 'script/models/xgb_model'
    # train
    model, rmse, accuracy, precision, recall, f1 = train(input_data_path)
    print("RMSE: %f" % (rmse))
    print("f1: %f" % (f1))
    print("accuracy: %f" % (accuracy))
    print("precision: %f" % (precision))
    print("recall: %f" % (recall))
    # save model
    save(model, output_model_path)