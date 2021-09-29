import pandas as pd
from pycaret.classification import predict_model, load_model
def load_data(new_churn_data):
    df= pd.read_csv('new_churn_data.csv')
def make_predictions(df):
    model= load_model('GBC')
    predictions= predict_model('GBC', data=df)
    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'}, inplace=True)
if __name__ == "__main__":
    df = load_data('new_churn_data.csv')
    predictions= make_predictions(df)
    print('predictions:')
    print(predictions)
