from sklearn.ensemble import ExtraTreesClassifier
from src.model.utils import *


class ExtraTree(ExtraTreesClassifier):
    def __init__(self, model=None, **kwargs):
        self.model = model
        self.params = {**kwargs}

    def fit(self, x, y=None):
        print(f'Training loop started [{time.asctime(time.localtime())}]')
        self.model = ExtraTreesClassifier(**self.params).fit(x, y)
        print(f'Training loop ended [{time.asctime(time.localtime())}]')
        return self

    def predict(self, x):
        return self.model.predict_proba(x).argmax(axis=1)

    def feature_importance(self, x):
        plot_importance((self.model.feature_importances_ * 10000).astype(int), x.columns, max_num_features=10)

    def classification_report(self, x_test, y_test=None):
        create_classification_report(self.model, x_test, y_test)
