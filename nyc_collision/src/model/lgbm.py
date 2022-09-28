from lightgbm import plot_importance
from lightgbm.sklearn import LGBMClassifier
from src.model.utils import *


class LGBM(LGBMClassifier):
    def __init__(self, model=None, **kwargs):
        self.model = model
        self.params = {**kwargs}

    def fit(self, x, y=None):
        print(f'Training loop started [{time.asctime(time.localtime())}]')
        self.model = LGBMClassifier(**self.params).fit(x, y)
        print(f'Training loop ended [{time.asctime(time.localtime())}]')
        return self

    def predict(self, x, basic=False):
        return self.model.predict_proba(x).argmax(axis=1)

    def feature_importance(self):
        plot_importance(self.model, max_num_features=10)

    def classification_report(self, x_test, y_test=None):
        prediction = self.predict(x_test)
        accuracy = accuracy_score(y_test, prediction)
        f1 = f1_score(y_test, prediction, average='micro')
        f1_2 = f1_score(y_test, prediction, average='weighted')
        report = classification_report(y_test, prediction)

        print('Accuracy: \n {} \n F1: \n {} \n F1_2: \n {} \n Classification report: \n {}'.format(accuracy, f1, f1_2, report))
