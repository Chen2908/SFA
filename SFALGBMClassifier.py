from SFAClassifier import SFAClassifier
from lightgbm import Dataset, train, Booster
from sklearn.metrics import roc_auc_score as auc
import warnings
import numpy as np


class SFALGBMClassifier(SFAClassifier):

    def __init__(self, ds_name, seed):
        super().__init__(ds_name, seed)
        self.model_name = 'lgbm'

    def objective(self, trial):
        """
        Hyperparameters optimization
        :param trial: the current trial
        :return: the auc score achieved in the trial
        """
        train_x, train_y = self.get_train_data()
        valid_x, valid_y = self.get_test_data()
        dtrain = Dataset(train_x, label=train_y, categorical_feature=self.categories) if self.categories is not None else Dataset(train_x, label=train_y)
        valid_y_np = self.get_y_np(valid_y)

        sub_samples_l, sub_samples_h = self.get_high_low_subsamples()
        col_sample_bytree_l, col_sample_bytree_h = self.get_high_low_col_samples()

        params = {
            'objective': self.get_task(),
            'verbosity': -1,
            'min_gain_to_split': 0.0001,
            'num_classes': self.get_num_classes(),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.03),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 40.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 11),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', sub_samples_l, sub_samples_h),
            'feature_fraction': trial.suggest_uniform('feature_fraction', col_sample_bytree_l, col_sample_bytree_h),
        }
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            bst = train(params, dtrain, num_boost_round=int((10 / (0.01 + params["learning_rate"]) ** 2) / 5))
        probas = bst.predict(valid_x)
        auc_score = auc(valid_y_np, probas, multi_class='ovo') if self.get_n_classes() > 2 else auc(valid_y_np, probas)
        return auc_score

    def train(self, x_train, y_train):
        """
        Initialize LGBM classifier and train it
        :param x_train: train features
        :param y_train: train target
        :return: the trained classifier
        """
        params = self.get_hyper_params()
        dtrain = Dataset(x_train, label=y_train, categorical_feature=self.categories) if self.categories is not None else Dataset(x_train, label=y_train)
        params['verbosity'] = -1
        params['num_classes'] = self.get_num_classes()
        params['objective'] = self.get_task()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = train(params, dtrain, num_boost_round=int((10 / (0.01 + params["learning_rate"]) ** 2) / 5))
        return model

    def predict_proba(self, clf, val_data):
        """
        Return the predicted probability for the given classifier.
        :param clf: LGBM classifier
        :param val_data: data
        :return: val_data's predicted probability
        """
        x_val = val_data[0]
        probs = clf.predict(x_val)
        if self.get_n_classes() == 2:
            probs = np.array([np.array([1 - i, i]) for i in probs])
        return probs

    def get_task(self):
        """
        Return the task based on the amount of classed in the data
        :return: binary if there are two classed and 'multiclass' otherwise
        """
        return 'binary' if self.get_n_classes() == 2 else 'multiclass'

    def save_model(self, clf, path):
        """
        Saved the model in .model format
        :param clf: LGBM classifier
        :param path: path to save the model in
        """
        clf.save_model(path+'.model')

    def get_num_classes(self):
        """Return the number of classes"""
        return 1 if self.get_n_classes() == 2 else self.get_n_classes()

    def load_model(self, path):
        """
        Load the LGBM classifier from the given path
        :param path: path
        :return: LGBM classifier
        """
        booster = Booster(model_file=path + '.model')
        booster.params['objective'] = self.get_task()
        return booster
