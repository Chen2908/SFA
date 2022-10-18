import featuretools as ft
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA


"""Other feature augmentation methods"""


def features_tools_extend_X(x, y, dataset_name):
    """
    Calculate augmented features using FeatureTools and use feature selection to select 2n+1 best features
    :param x: original features
    :param y: original target column
    :param dataset_name: dataset name
    :return: 2n+1 X N matrix of augmented features
    """
    copy_x = x.copy(deep=True)
    es = ft.EntitySet(id=dataset_name)
    es.entity_from_dataframe(entity_id='data', dataframe=copy_x,
                             make_index=False, index='index')
    # Run deep feature synthesis with transformation primitives
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='data',
                                          trans_primitives=['multiply_numeric', 'add_numeric', 'modulo_numeric',
                                                            'percentile'])
    feature_matrix.fillna(0, inplace=True)  # the modulo_numeric creates a few nan values, fill them with 0
    x_selectKBest = SelectKBest(mutual_info_classif, k=x.shape[1]*2+1).fit(feature_matrix, y.to_numpy().reshape(-1)) # select 2n+1 features
    mask = x_selectKBest.get_support()
    selected_features = list(feature_matrix.columns[mask])
    x_new = feature_matrix[selected_features]
    return x_new


def pca_extend_X(x):
    """
    Calculate PCA features and add them to the original features
    :param x: original features
    :return: 2n X N matrix of augmented features
    """
    pca = PCA()
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=[col + '_pca' for col in x])
    new_x = x.join(principalDf)
    return new_x

