from sklearn.model_selection import train_test_split
import pandas as pd
import re
import numpy as np


class Dataset:
    def __init__(self, data_path=None, categorical_features=[]):
        self.data_path = data_path
        self.categorical_features = categorical_features

        if len(categorical_features) < 1:
            raise ValueError('there should be at least on categorical feature')

    def clean_and_encode_data(self):
        data = pd.read_csv(self.data_path)
        data['MINUTE'] = data['CRASH TIME'].apply(lambda x: x.split(':')[1])
        data['MINUTE'] = data['MINUTE'].astype(int)
        data['YEAR'] = (data['CRASH DATE'].apply(lambda x: x.split('-')[0])).astype(int)
        data['MONTH'] = (data['CRASH DATE'].apply(lambda x: x.split('-')[1])).astype(int)
        data['DOW'] = (data['CRASH DATE'].apply(lambda x: pd.Timestamp(x).dayofweek)).astype(int)

        data['LL_X'] = np.cos(data['LATITUDE']) * np.cos(data['LONGITUDE'])
        data['LL_Y'] = np.cos(data['LATITUDE']) * np.sin(data['LONGITUDE'])
        data['LL_Z'] = np.sin(data['LATITUDE'])
        '''ll_scaler = preprocessing.StandardScaler()
        ll_scaler.fit(cleanData[["XLL","YLL", "ZLL"]])
        cleanData[["XLL","YLL", "ZLL"]] = ll_scaler.transform(cleanData[["XLL","YLL", "ZLL"]])'''

        print("Creating ordinal representations of circular features (Month/Year/Weekday/Minute)")
        data['HOUR_SIN'] = np.sin(2 * np.pi * data['HOUR'] / 23)
        data['HOUR_COS'] = np.cos(2 * np.pi * data['HOUR'] / 23)
        data['MONTH_SIN'] = np.sin(2 * np.pi * data['MONTH'] / 11)
        data['MONTH_COS'] = np.cos(2 * np.pi * data['MONTH'] / 11)
        data['DOW_SIN'] = np.sin(2 * np.pi * data['DOW'] / 6)
        data['DOW_COS'] = np.cos(2 * np.pi * data['DOW'] / 6)

        data = data.drop(columns=['CRASH DATE', 'CRASH TIME', 'ZIP CODE', 'geometry', 'NonPed', 'Wind',
                                  'XFrom', 'YFrom', 'XTo', 'YTo', 'TRUCK_ROUTE_TYPE', 'ON STREET NAME',
                                  'MINUTE', 'YEAR', 'Pressure', 'WindSpeed', 'Humidity', 'Dew', 'BOROUGH',
                                  'Temperature', 'LATITUDE', 'LONGITUDE', 'HOUR', 'DOW', 'MONTH'])

        data = data.dropna()

        for categorical_feature in self.categorical_features:
            one_hot_encoded = pd.get_dummies(data[categorical_feature], prefix=categorical_feature)
            data = data.drop(categorical_feature, axis=1)
            data = data.join(one_hot_encoded)

        y = data.SEVERITY
        y = y.replace(2, 0)
        x = data.drop(columns=['SEVERITY'])
        x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=42)

        print(f'Columns: {list(x.columns)}')

        return x, x_test, y, y_test
