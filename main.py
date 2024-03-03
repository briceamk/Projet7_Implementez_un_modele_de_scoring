""" Point d'entrée.// Importation des librairies et classes"""


# numpy and pandas for data manipulation
import numpy as np
import pandas as pd
import pickle
from tools import missing_values_table, encoding_data, plot_hist, hist_kde_plot, group_age
from tools import features_engineering_poly, log_classification
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, MinMaxScaler
import streamlit as st
import datetime


# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Main function."""
    seed = 123

    directory = "./input/"

    # Training data
    app_train = pd.read_csv(directory + "application_train.csv")
    print('Training data shape: ', app_train.shape)

    # Testing data features
    app_test = pd.read_csv('./input/application_test.csv')
    print('Testing data shape: ', app_test.shape)
    app_test.head()

    """ Exploratory data Analysis"""
    # Distribution of the target Columns
    print(app_train['TARGET'].value_counts())
    app_train['TARGET'].astype(int).plot.hist()

    # Missing values statistics
    missing_values = missing_values_table(app_train)
    missing_values.head(10)

    # Number of each type of column
    app_train.dtypes.value_counts()

    # Number of unique classes in each object column
    app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)

    # Transformation of the data
    app_train, app_test, train_labels = encoding_data(app_train, app_test)

    # Anomalies
    print((app_train['DAYS_BIRTH'] / -365).describe())
    print(app_train['DAYS_EMPLOYED'].describe())

    # Days Employed avant Traitement
    name = 'DAYS_EMPLOYED'
    plot_hist(name, app_train)

    # Traitement
    anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
    non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
    print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
    print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
    print('There are %d anomalous days of employment' % len(anom))

    # Create an anomalous flag column
    app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    # DaysEmployed après Traitement
    plot_hist(name, app_train)

    # DaysEmployed après Traitement sur Test Dataset
    app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
    app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

    # Find correlations with the target and sort
    correlations = app_train.corr()['TARGET'].sort_values()

    # Display correlations
    print('Most Positive Correlations:\n', correlations.tail(10))
    print('\nMost Negative Correlations:\n', correlations.head(10))

    # Find the correlation of the positive days since birth and target
    app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
    app_train['DAYS_BIRTH'].corr(app_train['TARGET'])

    # Find the correlation of the positive days since birth and target
    name = 'DAYS_BIRTH'
    hist_kde_plot(name, app_train)

    # Age information into a separate dataframe
    age_data = app_train[['TARGET', 'DAYS_BIRTH']]
    age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

    # Bin the age data
    age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20, 70, num=11))

    # Appeler la fonction avec votre DataFrame age_data
    group_age(age_data)

    # Extract the EXT_SOURCE variables and show correlations
    ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    ext_data_corrs = ext_data.corr()
    print(ext_data_corrs)

    # Heatmap of correlations
    plt.figure(figsize = (8, 6))
    sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
    plt.title('Correlation Heatmap')

    # iterate through the sources
    for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
        hist_kde_plot(name, app_train)

    # Feature Engineering
    list_train = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']
    list_test = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']

    app_train_poly, app_test_poly = features_engineering_poly(app_train, list_train, app_test, list_test)

    """Exploratory data analysis."""

    model, log_proba, log_accuracy, train, y_train_labels = log_classification(app_train, train_labels, seed)

###################################################################################################

    reg_lin_model = model

    save_dir = './obj_save/'

    # Vérifier si le répertoire existe, sinon le créer
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # # Save
    # with open(save_dir + "reg_lin_model.pkl", "wb") as write_file:
    #     pickle.dump(reg_lin_model, write_file)

    # Load
    with open(save_dir + "reg_lin_model.pkl", "rb") as read_file:
        loaded_model = pickle.load(read_file)

    st.write(loaded_model)

######################################################################


if __name__ == '__main__':
    main()
