import numpy as np
import os
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score as auc
import optuna
import pandas as pd


class SFAClassifier:
    def __init__(self,ds_details, seed, n_folds=10, metric='auc'):
        """
        Initialize class parameters
        :param ds_details: the details of the dataset for this run
        :param seed: the seed used for outer and inner splits of the data
        :param n_folds: amount of folds to use in the k fold, 10 is default
        :param metric: the metric used to measure performance, auc is default
        """
        self.ds_name, self.num_samples, self.num_features, self.num_classes, self.class_dist = ds_details
        self.model_name = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.categories = None
        self.seed = seed
        self.params = None
        self.n_folds = n_folds
        self.metric = metric
        self.len_preds = self.num_classes if self.num_classes > 2 else 1

    '''Getter and setters'''

    def objective(self, trial):
        return 0

    def set_train_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def set_test_data(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_y_test_np(self):
        return self.y_test.to_numpy().reshape(-1)

    def get_y_train_np(self):
        return self.y_train.to_numpy()

    def get_X_test_np(self):
        return self.X_test.to_numpy()

    def get_X_train_np(self):
        return self.X_train.to_numpy()

    def set_hyper_params(self, params):
        self.params = params

    def get_hyper_params(self):
        return self.params

    def get_n_classes(self):
        return self.num_classes

    def get_n_features(self):
        return self.num_features

    @staticmethod
    def get_y_np(y):
        return y.to_numpy().reshape(-1)

    @staticmethod
    def get_X_np(X):
        return X.to_numpy()

    def set_categories(self, categories):
        self.categories = categories

    def get_categories(self):
        return self.categories

    def run_optimization(self, X_train, y_train, X_test, y_test, categories):
        """
        Optimize hyperparameters using optuna:
        @inproceedings{akiba2019optuna,
        title={Optuna: A next-generation hyperparameter optimization framework},
        author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
        booktitle={Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery \& data mining},
        pages={2623--2631},
        year={2019}
        }
        :param X_train: train features
        :param y_train: train target column
        :param X_test: test features
        :param y_test: test target column
        :param categories: indices of categorical columns
        :return: num_trials - the number of trials used to find hyperparameters
        :        best_trial - the details of the trial with the best score
        """
        self.set_train_data(X_train, y_train)
        self.set_test_data(X_test, y_test)
        self.set_categories(categories)
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=15)
        num_trials = len(study.trials)
        best_trial = study.best_trial
        return num_trials, best_trial

    def get_high_low_subsamples(self):
        """
        Define the lower and upper bounds for the instances subsample in reference to the number of instances
        :return: sub_samples_l - lower bound
                 sub_samples_h - upper bound
        """
        if self.num_samples < 5000:
            sub_samples_l, sub_samples_h = 0.7, 0.95
        elif 5000 <= self.num_samples < 100000:
            sub_samples_l, sub_samples_h = 0.5, 0.85
        else:  # > 100000
            sub_samples_l, sub_samples_h = 0.3, 0.85
        return sub_samples_l, sub_samples_h

    def get_high_low_col_samples(self):
        """
        Define the lower and upper bounds for the features subsample in reference to the number of features
        :return: col_sample_bytree_l - lower bound
                 col_sample_bytree_h - upper bound
        """
        if self.num_features < 50:
            col_sample_bytree_l, col_sample_bytree_h = 0.3, 1
        elif 50 <= self.num_features < 500:
            col_sample_bytree_l, col_sample_bytree_h = 0.6, 1
        else:
            col_sample_bytree_l, col_sample_bytree_h = 0.15, 0.8
        return col_sample_bytree_l, col_sample_bytree_h

    def fit(self, X_train, y_train):
        """
        Train the SFA models in two stages and save them.
        :param X_train: train data
        :param y_train: train target
        """
        # Train first-stage model
        val_preds, val_shap_values, shap_time = self.train_first_stage(X_train, y_train)

        # use the OOP predictions and Shapley values to create 3 variations of augmented features
        train_df_shap = pd.DataFrame(val_shap_values, columns=[f'shap_{col}' for col in X_train.columns],
                                     index=X_train.index)
        train_df_preds = pd.DataFrame(val_preds, columns=[f'preds_{i}' for i in range(self.len_preds)], index=X_train.index)

        X_train_ex_p_shap = X_train.join(train_df_shap).join(train_df_preds)  # p-shap
        X_train_ex_shap = X_train.join(train_df_shap)  # shap
        X_train_ex_p = X_train.join(train_df_preds)  # p

        # Train 3 second-stage models
        self.train_second_stage(X_train_ex_p, y_train, 'p')
        self.train_second_stage(X_train_ex_shap, y_train, 'shap')
        self.train_second_stage(X_train_ex_p_shap, y_train, 'p_shap')

    def train_first_stage(self, X_train_val, y_train_val):
        """
        Train the first-stage models (base model) using k-fold cross validation. Save the models in .model form.
        Calculate the OOF predictions and their corresponding SHAP values using TreeExplainer for the second stage.
        @article{lundberg2020local,
        title={From local explanations to global understanding with explainable AI for trees},
        author={Lundberg, Scott M and Erion, Gabriel and Chen, Hugh and DeGrave, Alex and Prutkin, Jordan M and Nair,
         Bala and Katz, Ronit and Himmelfarb, Jonathan and Bansal, Nisha and Lee, Su-In},
        journal={Nature machine intelligence},
        volume={2},
        number={1},
        pages={56--67},
        year={2020},
        publisher={Nature Publishing Group}
        }
        :param X_train_val: train + validation data
        :param y_train_val: train + validation target
        :return:
        """
        kf = StratifiedKFold(n_splits=self.n_folds, random_state=self.seed, shuffle=True)
        y_train_np = self.get_y_np(y_train_val)

        # validation
        val_preds = np.zeros((X_train_val.shape[0], self.len_preds))
        val_all_predicitions = np.zeros(X_train_val.shape[0])
        val_all_probas = np.zeros((X_train_val.shape[0], self.num_classes))
        val_shap_vals = np.zeros(X_train_val.shape)

        for i, (tr_ind, val_ind) in enumerate(kf.split(X_train_val, y_train_val)):
            X_train, y_train, X_val, y_val = X_train_val.iloc[tr_ind], y_train_val.iloc[tr_ind], \
                                             X_train_val.iloc[val_ind], y_train_val.iloc[val_ind]
            # initialize and train the tree-based classifier
            clf = self.train(X_train, y_train)
            # save the trained model
            if not os.path.exists('models'):
                os.makedirs('models', exist_ok=True)
            if not os.path.exists(f'models/{self.ds_name}'):
                os.makedirs(f'models/{self.ds_name}', exist_ok=True)
            if not os.path.exists(f'models/{self.ds_name}/{self.model_name}'):
                os.makedirs(f'models/{self.ds_name}/{self.model_name}', exist_ok=True)
            self.save_model(clf, f'models/{self.ds_name}/{self.model_name}/base_fold_{i}_seed_{self.seed}')
            # predict on validation
            probabilities = self.predict_proba(clf, (X_val, y_val))
            prediction = probabilities.argmax(axis=1)
            val_preds[val_ind, :] = probabilities if self.len_preds > 1 else \
                    probabilities[:, 1].reshape(probabilities.shape[0], 1) # highest_probabilities
            val_all_probas[val_ind, :] = probabilities
            val_all_predicitions[val_ind] = prediction
            # calculate SHAP values for the validation
            clf_ex = shap.TreeExplainer(clf)
            if self.model_name in ['xgb', 'random_forest'] and self.categories is not None:
                dvalid = self.get_DMatrix(X_val, y_val)
                shap_values = clf_ex.shap_values(dvalid, check_additivity=False)
            else:
                shap_values = clf_ex.shap_values(X_val, check_additivity=False)
            val_shap_vals[val_ind] = [shap_values[height_idx][j] for j, height_idx in enumerate(prediction)] if (
                        self.get_n_classes() > 2 or self.model_name == 'lgbm') \
                else shap_values

        # calculate evaluation metric
        if self.metric == 'auc':
            val_score = float('{:.4f}'.format(auc(y_train_np, val_all_probas, multi_class='ovo') if
                                              self.get_n_classes() > 2 else auc(y_train_np, val_all_probas[:, 1])))
        elif self.metric == 'logloss':
            val_score = float('{:.4f}'.format(log_loss(y_train_np, val_all_probas)))
        print('base val score', str(val_score))

        return val_preds, val_shap_vals

    def train_second_stage(self, X_train_ext, y_train, config):
        """
        Train the second-stage model on the augmented features
        :param X_train_ext: train augmented features
        :param y_train: train target
        :param config: augmented data variation (P augmented, SHAP augmented or P+SHAP augmented)
        :return:
        """
        clf = self.train(X_train_ext, y_train)
        self.save_model(clf, f'models/{self.ds_name}/{self.model_name}/meta_{config}_seed_{self.seed}')
        preds = self.predict_proba(clf, (X_train_ext, y_train))
        y_train_np = self.get_y_np(y_train)
        if self.metric == 'auc':
            train_score = float('{:.4f}'.format(auc(y_train_np, preds, multi_class='ovo') if self.get_n_classes() > 2
                                                else auc(y_train_np, preds[:, 1])))
        elif self.metric == 'logloss':
            train_score = float('{:.4f}'.format(log_loss(y_train_np, preds)))
        print((f'train meta score- {config}', str(train_score)))

    def predict(self, X_test, y_test):
        """
        Predict the score for the test set using the trained first-stage and second-stage models
        :param X_test: test features
        :param y_test: test target
        :return: SFA score
        """
        # predict using the first-stage model
        test_preds, test_shap, test_all_probas = self.predict_first_stage(X_test, y_test)

        # use the OOP predictions and Shapley values to create 3 variations of augmented features
        test_df_preds = pd.DataFrame(test_preds, columns=[f'preds_{i}' for i in range(self.len_preds)],
                                     index=X_test.index)
        test_df_shap = pd.DataFrame(test_shap, columns=[f'shap_{col}' for col in X_test.columns],
                                    index=X_test.index)

        X_test_ex_p_shap = X_test.join(test_df_shap).join(test_df_preds)  # p-shap
        X_test_ex_shap = X_test.join(test_df_shap)  # shap
        X_test_ex_p = X_test.join(test_df_preds)  # p

        # predict using the second_stage model
        preds_p = self.predict_second_stage(X_test_ex_p, y_test, 'p')
        preds_shap = self.predict_second_stage(X_test_ex_shap, y_test, 'shap')
        preds_p_shap = self.predict_second_stage(X_test_ex_p_shap, y_test, 'p_shap')

        total_score_mean = self.calc_average_test_score(test_all_probas, [preds_p,
                                                              preds_shap, preds_p_shap], y_test)
        print(f'SFA test score', str(total_score_mean))
        return total_score_mean

    def predict_first_stage(self, X_test, y_test):
        """
        Predict score for the test set using the k first-stage models. For each model - load it and calculate prediction and SHAP values.
        Also calculate and print metric value.
        :param X_test: test features
        :param y_test: test target
        :return: avg_test_preds - the average (probability) prediction for each instance
                 avg_test_all_probas - the average (probability) prediction for each instance of all class if multi class, of the positive class if binary
                 avg_test_shap -  average SHAP values for each instance
        """
        test_preds = np.zeros((self.n_folds, X_test.shape[0], self.len_preds))
        test_all_probas = np.zeros((self.n_folds, X_test.shape[0], self.num_classes))
        test_shap_vals = np.zeros((self.n_folds, X_test.shape[0], X_test.shape[1]))
        all_test_score = np.zeros(self.n_folds)
        y_test_np = self.get_y_np(y_test)

        for i in range(self.n_folds):
            path = f'models/{self.ds_name}/{self.model_name}/base_fold_{i}_seed_{self.seed}'
            clf = self.load_model(path)
            # prediction
            model_preds = self.predict_proba(clf, (X_test, y_test))
            model_prediction = model_preds.argmax(axis=1)
            # SHAP values
            clf_ex = shap.TreeExplainer(clf)
            if self.model_name in ['xgb', 'random_forest'] and self.categories is not None:
                dtest = self.get_DMatrix(X_test, y_test)
                shap_values = clf_ex.shap_values(dtest, check_additivity=False)
            else:
                shap_values = clf_ex.shap_values(X_test, check_additivity=False)
            test_shap_vals[i, :, :] = [shap_values[height_idx][j] for j, height_idx in enumerate(model_prediction)] if (
                    self.get_n_classes() > 2 or self.model_name == 'lgbm') else shap_values
            test_preds[i, :, :] = model_preds if self.len_preds > 1 else \
                    model_preds[:, 1].reshape(model_preds.shape[0], 1)
            test_all_probas[i, :, :] = model_preds
            if self.metric == 'auc':
                fold_score = auc(y_test_np, model_preds, multi_class='ovo') if self.get_n_classes() > 2 else \
                    auc(y_test_np, model_preds[:, 1])
            elif self.metric == 'logloss':
                fold_score = log_loss(y_test_np, model_preds)
            all_test_score[i] = fold_score

        avg_test_score = float('{:.4f}'.format(np.mean(all_test_score, axis=0)))
        print('base test score', str(avg_test_score))
        avg_test_preds = np.mean(test_preds, axis=0)
        avg_test_all_probas = np.mean(test_all_probas, axis=0)
        avg_test_shap = np.mean(test_shap_vals, axis=0)

        return avg_test_preds, avg_test_shap, avg_test_all_probas

    def predict_second_stage(self, X_test, y_test, config):
        """
         Predict score for the test set using the second-stage model according to config (either P augmented model, SHAP augmented model or P+SHAP augmented model).
         Load the model and calculate prediction and metric value.
        :param X_test: test features
        :param y_test: test target
        :param config: the name of the augmented model to load
        :return: probability predictions
        """
        path = f'models/{self.ds_name}/{self.model_name}/meta_{config}_seed_{self.seed}'
        clf = self.load_model(path)
        preds = self.predict_proba(clf, (X_test, y_test))
        y_test_np = self.get_y_np(y_test)
        if self.metric == 'auc':
            second_stage_test_score = float('{:.4f}'.format(auc(y_test_np, preds, multi_class='ovo') if self.get_n_classes() > 2 \
                else auc(y_test, preds[:, 1])))
        elif self.metric == 'logloss':
            second_stage_test_score = float('{:.4f}'.format(log_loss(y_test_np, preds)))
        print(f'{config} second stage score', str(second_stage_test_score))
        return preds

    def calc_average_test_score(self, test_preds_base, list_preds, y_test):
        avg_preds = np.mean([test_preds_base] + list_preds, axis=0)
        y_test_np = self.get_y_np(y_test)
        if self.metric == 'auc':
            total_test_score = float('{:.4f}'.format(auc(y_test_np, avg_preds, multi_class='ovo')
                                                     if self.get_n_classes() > 2 else auc(y_test, avg_preds[:, 1])))
        elif self.metric == 'logloss':
            total_test_score = float('{:.4f}'.format(log_loss(y_test_np, avg_preds)))
        return total_test_score

    '''compare to featuretools and pca augment'''
    # todo: remove for production - experiments only
    def train_other(self, X_train, y_train, other_name):
        """
        Train model using k fold cross validation on augmented features created using a different augmentation method
        :param X_train: augmented train features
        :param y_train: train target
        :param other_name: augmentation method's name
        """
        kf = StratifiedKFold(n_splits=self.n_folds, random_state=self.seed, shuffle=True)
        for i, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            train_x, val_x, train_y, val_y = X_train.iloc[tr_idx], X_train.iloc[val_idx], \
                                             y_train.iloc[tr_idx], y_train.iloc[val_idx]
            clf = self.train(train_x, train_y)
            if not os.path.exists(f'models_{other_name}'):
                os.makedirs(f'models_{other_name}', exist_ok=True)
            if not os.path.exists(f'models_{other_name}/{self.ds_name}'):
                os.makedirs(f'models_{other_name}/{self.ds_name}', exist_ok=True)
            if not os.path.exists(f'models_{other_name}/{self.ds_name}/{self.model_name}'):
                os.makedirs(f'models_{other_name}/{self.ds_name}/{self.model_name}', exist_ok=True)
            self.save_model(clf, f'models_{other_name}/{self.ds_name}/{self.model_name}/fold_{i}_seed_{self.seed}')

    # todo: remove for production - experiments only
    def predict_other(self, X_test, y_test, other_name):
        """
        Predict using the k models trained on augmented features created using a different augmentation method.
        Also calculate and print metric value.
        :param X_test: augmented test features
        :param y_test: test target
        :param other_name:
        :return: other_name: augmentation method's name
        """
        y_test_np = self.get_y_np(y_test)
        test_all_auc = []
        for i in range(self.n_folds):
            path = f'models_{other_name}/{self.ds_name}/{self.model_name}/fold_{i}_seed_{self.seed}'
            clf = self.load_model(path)
            model_preds = self.predict_proba(clf, (X_test, y_test))

            if self.metric == 'auc':
                score = auc(y_test_np, model_preds, multi_class='ovo') if self.get_n_classes() > 2 else \
                    auc(y_test_np, model_preds[:, 1])
            elif self.metric == 'logloss':
                score = log_loss(y_test_np, model_preds)
            test_all_auc.append(score)

            test_score = float('{:.4f}'.format(np.mean(test_all_auc, axis=0)))
            print(f'{other_name} test score', str(test_score))



