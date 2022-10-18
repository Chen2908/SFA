from data_load_automl import *
import os
import pickle
import argparse
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from SFALGBMClassifier import SFALGBMClassifier
from SFARandomForestClassifier import SFARandomForestClassifier
from SFAXGBoostClassifier import SFAXGBoostClassifier
from features_eng import features_tools_extend_X, pca_extend_X


BINARY_DATA_PATH = 'datasets/binary'
MULTI_DATA_PATH = 'datasets/multi_class'
PICKLE_PATH = 'pickle'
BINARY_PICKLE_PATH = PICKLE_PATH + '/binary'
MULTI_PICKLE_PATH = PICKLE_PATH + '/multi_class'
PICKLE_ALL_BINARY_DATASETS_PATH = PICKLE_PATH + '/binary_datasets_idx.pkl'
PICKLE_ALL_MULTI_DATASETS_PATH = PICKLE_PATH + '/multi_datasets_idx.pkl'


def main(args, nepex):
    # extract user arguments
    model_name = args.model_name
    ds_id = args.dataset_id
    task = args.task
    seed = args.seed
    compare = args.compare

    one_hot = True
    categories = None
    print('loading dataset')

    '''load the dataset according to the selected dataset id'''
    if task == 'binary':
        pickle_path = BINARY_PICKLE_PATH
        data_path = BINARY_DATA_PATH
        all_datasets_path = PICKLE_ALL_BINARY_DATASETS_PATH
    elif task == 'multi':
        pickle_path = MULTI_PICKLE_PATH
        data_path = MULTI_DATA_PATH
        all_datasets_path = PICKLE_ALL_MULTI_DATASETS_PATH

    if os.path.exists(all_datasets_path):
        with open(all_datasets_path, 'rb') as ds_by_id_file:
            ds_by_id = pickle.load(ds_by_id_file)
            ds_details = ds_by_id[ds_id]
        # load from pickle
        ds_path_X = pickle_path + f'/{ds_details[0]}_X.pkl'
        ds_path_y = pickle_path + f'/{ds_details[0]}_y.pkl'
        if one_hot:
            ds_path_X_oh = pickle_path + f'/{ds_details[0]}_X_oh.pkl'
            if os.path.exists(ds_path_X_oh):
                ds_path_X = ds_path_X_oh
        ds_path_categories = pickle_path + f'/{ds_details[0]}_categories.pkl'

        with open(ds_path_X, 'rb') as file_X:
            X = pickle.load(file_X)
        with open(ds_path_y, 'rb') as file_y:
            y = pickle.load(file_y)
        if not one_hot:
            if os.path.exists(ds_path_categories):
                with open(ds_path_categories, 'rb') as file_c:
                    categories = pickle.load(file_c)

    else: # In case the dataset's pickle file does not yet exist, create it
        if not os.path.exists(PICKLE_PATH):
            os.makedirs(PICKLE_PATH, exist_ok=True)
        # read from csv and create pickles
        X, y, ds_details = read_all_data_files(all_datasets_path, pickle_path, data_path, ds_id, one_hot)
        print(f'finished datasets load\n')
        if len(ds_details) == 6:
            categories = ds_details[5]
            ds_details = ds_details[:-1]

    print('seed:', str(seed))

    ds_name = ds_details[0]
    print(f'\nDatatset: {ds_name} \n')

    models_inits = {'xgb': SFAXGBoostClassifier(ds_details, seed),
                    'lgbm': SFALGBMClassifier(ds_details, seed),
                    'random_forest': SFARandomForestClassifier(ds_details, seed)}

    PEnTex_clf = models_inits[model_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)

    '''optimize hyperparameters'''
    if model_name != 'random_forest':
        num_trials, best_trial = PEnTex_clf.run_optimization(X_train, y_train, X_test, y_test, categories)
        pre_params = params_dict(best_trial.params)
    else:
        pre_params = PEnTex_clf.get_hyper_params()
        PEnTex_clf.set_categories(categories)

    params_to_set = best_trial.params if model_name != 'random_forest' else pre_params
    PEnTex_clf.set_hyper_params(params_to_set)
    nepex.log_text('hyper_parameters', str(pre_params))
    hyper_opts_df = pd.DataFrame({
        'Seed': [seed],
        'Dataset': [ds_name],
        'Model': [model_name],
        'HyperOpt params': [pre_params]
    })
    if os.path.exists('hyper_opt.csv'):
        hyper_opts_df.to_csv('hyper_opt.csv', index=False, mode='a', header=False)
    else:
        hyper_opts_df.to_csv('hyper_opt.csv', index=False)

    '''Run SFA'''
    # fit two-step models
    PEnTex_clf.fit(X_train, y_train)
    # predict
    PEnTex_clf.predict(X_test, y_test)

    if compare:
        '''compare to FeaturesTools'''
        X_extended_ft = features_tools_extend_X(X, y, ds_name)
        X_train_ft, X_test_ft, y_train_ft, y_test_ft = train_test_split(X_extended_ft, y, random_state=seed, stratify=y)
        PEnTex_clf.train_other(X_train_ft, y_train_ft, 'ft')
        PEnTex_clf.predict_other(X_test_ft, y_test_ft, 'ft')

        '''compare to PCA Augment'''
        X_extended_pca = pca_extend_X(X)
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_extended_pca, y, random_state=seed,
                                                                            stratify=y)
        PEnTex_clf.train_other(X_train_pca, y_train_pca, 'pca')
        PEnTex_clf.predict_other(X_test_pca, y_test_pca, 'pca')


def plot_classes_distribution(ds_name, y, preds):
    if not os.path.exists('plots/'):
        os.makedirs('plots/')
    plt.clf()
    for i in set(y):
        plt.hist(preds[y[0] == i], bins=40, alpha=0.4, label=str(i))
    plt.legend(loc='upper left')
    plt.title(f'class distribution- {ds_name}')
    plt.savefig(f'plots/{ds_name}_class_dist.png')


def params_dict(best_trial_params):
    params = {}
    for key, value in best_trial_params.items():
        print("    {}: {}".format(key, value))
        params[key] = '{:.4f}'.format(value)
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', default='', type=str)
    parser.add_argument('--task', default='', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--compare', default=False, type=bool)
    all_args = parser.parse_args()

    main(all_args)



