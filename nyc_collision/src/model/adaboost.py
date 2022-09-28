from sklearn.ensemble import AdaBoostClassifier
from src.model.utils import *
from sklearn.utils import compute_sample_weight


class AdaBoost(AdaBoostClassifier):
    def __init__(self, model=None, **kwargs):
        self.model = model
        self.params = {**kwargs}

    def fit(self, x, y=None):
        class_weights = compute_sample_weight('balanced', y)
        print(f'Training loop started [{time.asctime(time.localtime())}]')
        self.model = AdaBoostClassifier(**self.params).fit(x, y, sample_weight=class_weights)
        print(f'Training loop ended [{time.asctime(time.localtime())}]')
        return self

    def predict(self, x):
        return self.model.predict_proba(x).argmax(axis=1)

    def feature_importance(self, x):
        return plot_importance((self.model.feature_importances_ * 10000).astype(int), x.columns, max_num_features=10)

    def classification_report(self, x_test, y_test=None):
        create_classification_report(self.model, x_test, y_test)
