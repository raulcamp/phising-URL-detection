import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from url_featurizer import URLFeaturizer
    
def transform(df):
    rows = [URLFeaturizer(URL).extract() for URL in df['URL']]
    return pd.concat([pd.DataFrame(rows), df.drop(columns=['URL'])], axis=1, join='inner')

def test(input_model, input_test_data_path, full_test_data_path): 
    # load model
    model = pickle.load(open(input_model, 'rb'))
    # load and transform test data
    raw_data = pd.read_csv(input_test_data_path)
    test_data = transform(raw_data.drop(columns=['index', 'Label']))
    # run model on test data
    predictions = model.predict(test_data)
    # validate predicted labels
    full_test_data = pd.read_csv(full_test_data_path)
    actual = list(full_test_data['Label'])
    return (mean_squared_error(actual, predictions), 
            accuracy_score(actual, predictions, normalize=True),
            precision_score(actual, predictions),
            recall_score(actual, predictions),
            f1_score(actual, predictions))

if __name__ == '__main__':
    model_path = 'script/models/xgb_model'
    input_test_data_path = 'script/data/Phishing_Test.csv'
    full_test_data_path = 'script/data/Phishing_Test_Full.csv'
    # test model
    rmse, accuracy, precision, recall, f1 = test(model_path, input_test_data_path, full_test_data_path)
    print("RMSE: %f" % (rmse))
    print("f1: %f" % (f1))
    print("accuracy: %f" % (accuracy))
    print("precision: %f" % (precision))
    print("recall: %f" % (recall))