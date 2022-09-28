from model.lgbm import LGBM
from data_loader.data_loader import Dataset
from sklearn.model_selection import GridSearchCV
from lightgbm.sklearn import LGBMClassifier
import joblib
from model.catboost_model import CatBoost
from model.extra_tree import ExtraTree

CATEGORICAL_FEATURES = ['BOROUGH', 'Condition', 'Clouds', 'Day/Night', 'Snow_Priority']

if __name__ == '__main__':

    #model = joblib.load("../models/grid_model.pkl")

    #print(model.cv_results_)
    dataset = Dataset('../data/processed/merged_data_2.csv', CATEGORICAL_FEATURES)
    x, x_test, y, y_test = dataset.clean_and_encode_data()

    '''parameters = {'num_leaves': [28, 31, 40, 65, 70, 100],
                  'max_depth': [-1, 3, 5, 7, 10],
                  'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                  'n_estimators': [30, 40, 50, 75, 100, 120],
                  'random_state': [42]}

    model = LGBMClassifier()
    grid_search = GridSearchCV(model, parameters, return_train_score=True, verbose=1)
    grid_model = grid_search.fit(x, y)'''

    # print(model.classification_report(x_test, y_test))
    # print(model.feature_importances_())

    #model = LGBM(n_estimators=120, num_leaves=110, learning_rate=0.7)
    #model.fit(x, y)
    #print(model.classification_report(x_test, y_test))

    extra = ExtraTree(class_weight='balanced')
    extra.fit(x, y)
