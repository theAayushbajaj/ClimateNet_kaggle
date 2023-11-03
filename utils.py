# Importing all the required libraries
from IPython.display import display

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
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
    # drop duplicates
    train.drop_duplicates(inplace=True)

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

    if standardize:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    # set aside a matrix of temporal features
    X_train_temporal = X_train[['year', 'month', 'day']]
    X_test_temporal = X_test[['year', 'month', 'day']]
    # drop the temporal features from X
    X_train = X_train.drop(columns=['year', 'month', 'day'])
    X_test = X_test.drop(columns=['year', 'month', 'day'])

    # Fit scaler on training data and transform both train and test data
    scaler.fit(X_train)  # Fit only on training data
    X_train_normalized = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_normalized = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


    X_train_temporal = X_train_temporal.reset_index(drop=True)
    X_train_normalized = X_train_normalized.reset_index(drop=True)

    # concatenate X_temporal and X_normalized to get X
    X_train = pd.concat([X_train_temporal, X_train_normalized], axis=1)

    X_test_temporal = X_test_temporal.reset_index(drop=True)
    X_test_normalized = X_test_normalized.reset_index(drop=True)

    # concatenate X_temporal and X_normalized to get X
    X_test = pd.concat([X_test_temporal, X_test_normalized], axis=1)


    # Squaring the features in X_normalized
    for col in X_train_normalized.columns:
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
    for col in X_test_normalized.columns:
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