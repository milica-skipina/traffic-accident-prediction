import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils import compute_sample_weight
from src.model.utils import *


class XGB(XGBClassifier):
    def __init__(self, model=None, **kwargs):
        self.model = model
        self.params = {**kwargs}

    def fit(self, x, y=None):
        class_weights = compute_sample_weight('balanced', y)
        print(f'Training loop started [{time.asctime(time.localtime())}]')
        self.model = XGBClassifier(**self.params).fit(x, y, sample_weight=class_weights)
        print(f'Training loop ended [{time.asctime(time.localtime())}]')
        return self

    def predict(self, x):
        return self.model.predict(x)

    def feature_importance(self):
        feature_important = self.model.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        data.plot(kind='barh')

    def classification_report(self, x_test, y_test=None):
        create_classification_report(self.model, x_test, y_test)
