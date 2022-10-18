import pandas as pd
import glob
import pickle
import sklearn.preprocessing as preprocessing
from sklearn.impute import IterativeImputer
import numpy as np
import os
import matplotlib.pyplot as plt
import string


def datasets_to_X_y(ds_path):
    """
    This function loads the dataset, handles missing values and categorical features and returns the dataset separated into features and target
    :param ds_path: the dataset path
    :return: ds_name - dataset name
            X - dataset features
            y - dataset target column
            categorical- categorical indices
            X_oh - dataset features after onehot transformation
    """
    ds_name = ds_path.split('/')[2].split('.csv')[0].replace('1', '')
    df = pd.read_csv(ds_path)
    df = label_encode_class(df)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    X = X.replace({' ?': np.nan, '?': np.nan})
    y = y.to_frame()
    categorical, X_oh = [], X
    if ds_name == 'adult':
        X = X.drop('fnlwgt', axis=1)
    if ds_name == 'airlines':
        X = X.drop('Flight', axis=1)
    X_cols = X.columns
    if ds_name in ['airlines', 'adult', 'bank_marketing', 'credit-g']:
        categorical_names = X.select_dtypes(include=['object']).columns.tolist()
        categorical = [i for i, j in enumerate(X_cols) if j in categorical_names]
        for col in categorical_names:
            X[col] = X[col].replace(np.nan, -1).astype('category')
        X_oh = columns_transform(X)
    else:
        for col in X_cols:
            X[col] = X[col].astype('float')
        X = impute_missing_values(X)

    return ds_name, X, y, categorical, X_oh


def impute_missing_values(dataframe):
    """
    This function imputes missing values using multiple imputation
    :param dataframe: original dataset
    :return: temp_dataframe: dataset after imputation
    """
    temp_dataframe = pd.DataFrame.copy(dataframe, deep=True)
    imp = IterativeImputer(max_iter=10, random_state=0)
    num_features = temp_dataframe.columns.tolist()
    for col in num_features:
        imp = imp.fit(temp_dataframe[[col]])
        temp_dataframe[col] = imp.transform(temp_dataframe[[col]])
    return temp_dataframe


def label_encode_class(dataframe):
    """
    This function applies label encoding on the target column
    :param dataframe: original dataframe
    :return: temp_dataframe: dataframe after label encoding
    """
    enc = preprocessing.LabelEncoder()
    temp_dataframe = dataframe.copy()
    class_name = temp_dataframe.columns[-1]
    temp_dataframe[class_name] = enc.fit_transform(temp_dataframe[class_name])
    return temp_dataframe


def columns_transform(dataframe):
    """
    This function applies onehot encoding on categorical columns
    :param dataframe: original dataframe
    :return: temp_dataframe: dataframe after onehot encoding
    """
    new_cols = []
    binary_data = pd.get_dummies(dataframe)
    for col in binary_data.columns:
        new_cols.append(col.translate(col.maketrans('', '', string.punctuation)))
    binary_data.columns = new_cols
    return binary_data


def read_all_data_files(all_pickle_path, file_pickle_path, files_path, ds_id, one_hot):
    """
    This function retrieves all datasets' paths  and send them to the datasets_to_X_y function one by one to be loaded. Then she saves them as pickle files
    :param all_pickle_path: path of the outer pickle folder
    :param file_pickle_path: path to save the pickle files in
    :param files_path: local path where datasets' csv files are located
    :param ds_id: the id of the dataset for this run
    :param one_hot: boolean, where to encode datasets using one hot encoding
    :return: The relevant dataset with id equals to ds_id (x, y and details)
    """
    all_data_idx_name = {}
    metadata = {'ds name': [], '# samples': [], '# features': [], '# classes': [], 'class dist': [], 'id': []}
    data_paths_list = [data_file for data_file in glob.glob(files_path + "/*.csv")]
    if not os.path.exists(file_pickle_path):
        os.mkdir(file_pickle_path)
    for idx, data_file_path in enumerate(data_paths_list):
        ds_name, X, y, categorical, X_oh = datasets_to_X_y(data_file_path)
        ds_path_X = file_pickle_path + f'/{ds_name}_X.pkl'
        ds_path_y = file_pickle_path + f'/{ds_name}_y.pkl'
        with open(ds_path_X, 'wb') as file_X:
            pickle.dump(X, file_X, pickle.HIGHEST_PROTOCOL)
        with open(ds_path_y, 'wb') as file_y:
            pickle.dump(y, file_y, pickle.HIGHEST_PROTOCOL)

        if categorical:
            ds_path_X_oh = file_pickle_path + f'/{ds_name}_X_oh.pkl'
            ds_path_categorical = file_pickle_path + f'/{ds_name}_categories.pkl'
            with open(ds_path_X_oh, 'wb') as file_X_oh:
                pickle.dump(X_oh, file_X_oh, pickle.HIGHEST_PROTOCOL)
            with open(ds_path_categorical, 'wb') as file_categories:
                pickle.dump(categorical, file_categories, pickle.HIGHEST_PROTOCOL)

        metadata['ds name'].append(ds_name)
        metadata['# samples'].append(X.shape[0])
        if categorical and one_hot:
            metadata['# features'].append(X_oh.shape[1])
        else:
            metadata['# features'].append(X.shape[1])
        y_array = y.to_numpy().reshape(-1)
        classes = set(y_array)
        metadata['# classes'].append(len(classes))
        class_dist = {i: np.round(list(y_array).count(i)/X.shape[0], 3) for i in classes}
        metadata['class dist'].append(class_dist)
        metadata['id'].append(idx)
        plot_classes_distribution(ds_name, y_array)
        if categorical and one_hot:
            all_data_idx_name[str(idx)] = [ds_name, X_oh.shape[0], X_oh.shape[1], len(classes), class_dist]
        else:
            all_data_idx_name[str(idx)] = [ds_name, X.shape[0], X.shape[1], len(classes), class_dist]
        if str(idx) == ds_id:
            if categorical and one_hot:
                final_x, final_y, *final_ds_details = X_oh, y, ds_name, X_oh.shape[0], X_oh.shape[1], len(classes), class_dist
            elif categorical:
                final_x, final_y, *final_ds_details = X, y, ds_name, X.shape[0], X.shape[1], len(classes), class_dist, categorical
            else:
                final_x, final_y, *final_ds_details = X, y, ds_name, X.shape[0], X.shape[1], len(classes), class_dist
    with open(all_pickle_path, 'wb') as all_datasets_names:
        pickle.dump(all_data_idx_name, all_datasets_names, pickle.HIGHEST_PROTOCOL)
    metadata_df = pd.DataFrame.from_dict(metadata)
    if os.path.exists('metadata.csv'):
        metadata_df.to_csv('metadata.csv', index=False, mode='a', header=False)
    else:
        metadata_df.to_csv('metadata.csv', index=False)
    return final_x, final_y, final_ds_details


def plot_classes_distribution(ds_name, y,):
    """
    This function plot the dataset distribution of classes
    :param ds_name: dataset name
    :param y: the class column
    """
    if not os.path.exists('plots/'):
        os.makedirs('plots/')
    plt.clf()
    plt.hist(y, bins=40, alpha=0.4)
    plt.xticks(list(set(y)))
    plt.xlabel('classes')
    plt.legend(loc='upper left')
    plt.title(f'class distribution- {ds_name}')
    plt.savefig(f'plots/{ds_name}_class_dist.png')