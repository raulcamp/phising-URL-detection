import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from train_xgb_model import feature_transform

def unfeaturize(data):
    del data['numSD']
    del data['hasPercent']
    del data['special']
    del data['numerical']
    del data['len_url']
    
def test(input_model, input_test_data_path, full_test_data_path): 
    # load model
    model = pickle.load(open(input_model, 'rb'))
    # load and transform test data
    test_data = pd.read_csv(input_test_data_path)
    ids = test_data['Unnamed: 0']
    urls = test_data['URL']
    del test_data['Unnamed: 0']
    del test_data['Label']
    feature_transform(test_data)
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
    
    # unfeaturize(test_data)
    # test_data.insert(0, '',ids )
    # test_data.insert(4, 'URL', urls)
    # test_data.insert(5, 'Label', labels)
    # test_data.to_csv(r'script/data/Phishing_Labels.csv', index=False)

if __name__ == '__main__':
    model_path = 'script/models/xgb_model'
    input_test_data_path = 'script/data/Phishing_Test.csv'
    full_test_data_path = 'script/data/Phishing_Test_Full.csv'
    # test
    rmse, accuracy, precision, recall, f1 = test(model_path, input_test_data_path, full_test_data_path)
    print("RMSE: %f" % (rmse))
    print("f1: %f" % (f1))
    print("accuracy: %f" % (accuracy))
    print("precision: %f" % (precision))
    print("recall: %f" % (recall))