from SFAClassifier import SFAClassifier
from xgboost import DMatrix, train, Booster
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc
import numpy as np


class SFAXGBoostClassifier(SFAClassifier):

    def __init__(self,ds_name, seed):
        super().__init__(ds_name, seed)
        self.model_name = 'xgb'

    def objective(self, trial):
        """
        Hyperparameters optimization
        :param trial: the current trial
        :return: the auc score achieved in the trial
        """
        train_x, train_y = self.get_train_data()
        valid_x, valid_y = self.get_test_data()
        if self.get_categories() is not None:
            dtrain = DMatrix(train_x, label=train_y, enable_categorical=True)
            dvalid = DMatrix(valid_x, label=valid_y, enable_categorical=True)
        else:
            dtrain = DMatrix(train_x, label=train_y)
            dvalid = DMatrix(valid_x, label=valid_y)
        valid_y_np = self.get_y_np(valid_y)

        least_freq_class_frec = min(self.class_dist.values())
        max_scale_pos_weight = 1 / min((least_freq_class_frec * self.num_classes), 1)

        sub_samples_l, sub_samples_h = self.get_high_low_subsamples()
        col_sample_bytree_l, col_sample_bytree_h = self.get_high_low_col_samples()

        params = {
            'verbosity': 0,
            'objective': self.get_task(),
            'num_class': 1 if self.get_n_classes() == 2 else self.get_n_classes(),
            'tree_method': 'gpu_hist',
            # L2 regularization weight
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            # controls whether a given node will split based on the expected reduction in loss after the split
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            # sampling ratio for training data
            'subsample': trial.suggest_float('subsample', sub_samples_l, sub_samples_h),
            # sampling according to each tree
            'colsample_bytree': trial.suggest_float('colsample_bytree', col_sample_bytree_l, col_sample_bytree_h),
            # how deeply each tree is allowed to grow during any boosting round
            'max_depth': trial.suggest_int('max_depth', 3, 11, step=2),
            # learning rate
            'eta': trial.suggest_loguniform('eta', 0.001, 0.05),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1,
                                                    max_scale_pos_weight if max_scale_pos_weight < 20 else 20)
        }
        bst = train(params, dtrain, num_boost_round=int((10 / (0.01 + params["eta"]) ** 2) / 5))
        probas = bst.predict(dvalid)
        auc_score = auc(valid_y_np, probas, multi_class='ovo') if self.get_n_classes() > 2 else auc(valid_y_np, probas)
        return auc_score

    def train(self, x_train, y_train):
        """
        Initialize XGBoost classifier and train it
        :param x_train: train features
        :param y_train: train target
        :return: the trained classifier
        """
        params = self.get_hyper_params()
        if self.get_categories() is not None:
            dtrain = DMatrix(x_train, label=y_train, enable_categorical=True)
        else:
            dtrain = DMatrix(x_train, label=y_train)
        params['objective'] = self.get_task()
        params['num_class'] = 1 if self.get_n_classes() == 2 else self.get_n_classes()
        params['verbosity'] = 0
        params['tree_method'] = 'gpu_hist'
        return train(params=params, dtrain=dtrain, num_boost_round=int((10 / (0.01 + params['eta']) ** 2) / 5))

    def predict_proba(self, clf, val_data):
        """
        Return the predicted probability for the given classifier.
        :param clf: XGBoost classifier
        :param val_data: data
        :return: val_data's predicted probability
        """
        x_val, y_val = val_data[0], val_data[1]
        if self.get_categories() is not None:
            dvalid = DMatrix(x_val, label=y_val, enable_categorical=True)
        else:
            dvalid = DMatrix(x_val, label=y_val)
        probs = clf.predict(dvalid)
        if self.get_n_classes() == 2:
            probs = np.array([np.array([1-i, i]) for i in probs])
        return probs

    def get_task(self):
        """
        Return the task based on the amount of classed in the data
        :return: 'binary:logistic' if there are two classed and 'multi:softprob' otherwise
        """
        return 'binary:logistic' if self.get_n_classes() == 2 else 'multi:softprob'

    def save_model(self, clf, path):
        """
        Saved the model in .model format
        :param clf: XGBoost classifier
        :param path: path to save the model in
        """
        clf.save_model(path+'.model')

    def load_model(self, path):
        """
        Load the XGBoost classifier from the given path
        :param path: path
        :return: XGBoost classifier
        """
        bst = Booster()
        bst.load_model(path + '.model')
        return bst

    def get_DMatrix(self, X, y):
        """
        Wrap the dataframe in a DMatrix
        :param X: features
        :param y: target
        :return: data in DMatrix format
        """
        if self.get_categories() is not None:
            return DMatrix(X, label=y, enable_categorical=True)
        else:
            return DMatrix(X, label=y)