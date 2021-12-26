import pandas as pd
import pickle
from train_xgb_model import feature_transform

model = pickle.load(open('script/xgb_model', 'rb'))

# testing
test_data = pd.read_csv('script/data/Phishing_Test.csv')
ids = test_data['Unnamed: 0']
urls = test_data['URL']
del test_data['Unnamed: 0']
del test_data['Label']

feature_transform(test_data)
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

#test_data.to_csv(r'script/data/Phishing_Labels.csv', index=False)

if __name__ == '__main__':
    pass