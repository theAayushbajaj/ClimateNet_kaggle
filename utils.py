# Importing all the required libraries
from IPython.display import display

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, \
                                    StratifiedKFold

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, \
            classification_report, recall_score, precision_recall_curve

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = np.random.seed(0)

import wandb

def draw_missing_data_table(df):
    '''
    Docstring: Returns a datarframe with percent of missing/nan values per feature/column
    
    Parameters:
    ------------
    df: dataframe object
    
    Returns:
    ------------
    Dataframe containing missing value information
    '''
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent of NaNs'])
    return missing_data

def preprocess(train, test, standardize=True):  

    # drop PSL as PS and PSL are highly correlated
    train = train.drop(columns=['PSL'])
    test = test.drop(columns=['PSL'])  
    # drop duplicates
    train.drop_duplicates(inplace=True)

    num_clusters = 6  # Example number, adjust based on your dataset and domain knowledge

    # Apply K-means clustering to the training data
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    train['cluster'] = kmeans.fit_predict(train[['lat', 'lon']])

    # Initialize a list to keep all the scalers
    scalers = []
    # Normalize data within each cluster for the training set
    for cluster in range(num_clusters):
        # Initialize a scaler (standard or min-max based on 'standardize' flag)
        scaler = StandardScaler() if standardize else MinMaxScaler()
        
        # Select features to be scaled, excluding 'lat', 'lon', and 'cluster'
        features_to_scale = [col for col in train.columns if col not in ['lat', 'lon', 'cluster', 'Label']]

        # Fit the scaler on the training data of this cluster
        scaler.fit(train.loc[train['cluster'] == cluster, features_to_scale])
        
        # Apply the scaler to the training data of this cluster
        train.loc[train['cluster'] == cluster, features_to_scale] = scaler.transform(
            train.loc[train['cluster'] == cluster, features_to_scale])
        
        # Store the scaler for later use on the test set
        scalers.append(scaler)

    # Predict the cluster for the test data based on the trained K-means model
    test['cluster'] = kmeans.predict(test[['lat', 'lon']])

    # Normalize test data based on the cluster it belongs to and the corresponding scaler
    for cluster, scaler in zip(range(num_clusters), scalers):
        # Select test data in the current cluster
        test_cluster_data = test[test['cluster'] == cluster]

        # Scale features for the test data in this cluster using the corresponding scaler
        test.loc[test['cluster'] == cluster, features_to_scale] = scaler.transform(
            test_cluster_data[features_to_scale])

    # Dropping 'cluster' column as it is no longer needed after scaling
    train = train.drop(columns=['cluster'])
    test = test.drop(columns=['cluster'])


    # Extract year, month, and day from the 'time' column
    train['year'] = train.index.year
    test['year'] = test.index.year
    train['month'] = train.index.month
    test['month'] = test.index.month
    train['day'] = train.index.day
    test['day'] = test.index.day

    # Splitting the dataset into features (X) and target (y)
    X_train = train.drop(columns=['Label'])
    y_train = train['Label']
    X_test = test

    # Squaring the features in X_normalized
    for col in X_train.drop(['year', 'month', 'day'],axis=1).columns:
        X_train[col + '_squared'] = X_train[col] ** 2

    # Transforming month and day into cyclical features
    X_train['sin_month'] = np.sin(2 * np.pi * X_train['month'] / 12)
    X_train['cos_month'] = np.cos(2 * np.pi * X_train['month'] / 12)
    X_train['sin_day'] = np.sin(2 * np.pi * X_train['day'] / 30)
    X_train['cos_day'] = np.cos(2 * np.pi * X_train['day'] / 30)

    # Dropping the original month and day columns
    X_train = X_train.drop(columns=['month', 'day', 'year'])
    # X = X.drop(columns=['month', 'day'])

    # Squaring the features in X_normalized
    for col in X_test.drop(['year', 'month', 'day'],axis=1).columns:
        X_test[col + '_squared'] = X_test[col] ** 2

    # Transforming month and day into cyclical features
    X_test['sin_month'] = np.sin(2 * np.pi * X_test['month'] / 12)
    X_test['cos_month'] = np.cos(2 * np.pi * X_test['month'] / 12)
    X_test['sin_day'] = np.sin(2 * np.pi * X_test['day'] / 30)
    X_test['cos_day'] = np.cos(2 * np.pi * X_test['day'] / 30)

    # Dropping the original month and day columns
    X_test = X_test.drop(columns=['month', 'day', 'year'])
    # X = X.drop(columns=['month', 'day'])

    
    return X_train, y_train, X_test


def get_classification_report(y_val, y_pred):
    print('1. The F-1 score of the model {}\n'.format(f1_score(y_val, y_pred, average='macro')))
    print('2. The recall score of the model {}\n'.format(recall_score(y_val, y_pred, average='macro')))
    print('3. Classification report \n {} \n'.format(classification_report(y_val, y_pred)))
    print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_val, y_pred)))

def get_class_weights(n,c,y):
    return n/(c*np.bincount(y))

def get_class_weights_dict(n,c,y):
    return {i: n/(c*v) for i, v in enumerate(np.bincount(y))}