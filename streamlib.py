import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression



@st.cache_data(show_spinner="plot amount ...")
def plot_amount(data, col, val, bins=50, label_rotation=True):
    plt.style.use('fivethirtyeight')

    '''use this for ploting the distribution of numercial features'''
    fig, ax = plt.subplots()
    ax.set_title(f"Distribution of {col}")
    sns.histplot(data[col].dropna(), kde=True, bins=bins,stat="density", kde_kws=dict(cut=3), ax=ax)
    ax.set_xlabel(f"{col}")
    ax.set_ylabel("Density")
    plt.axvline(x=val, color='red', linestyle='--')  # Set the y-axis limit

    if label_rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    st.pyplot(fig)


@st.cache_data(show_spinner="plot years ...")
def plot_years(data, col, val, bins=50, label_rotation=True):
    plt.style.use('fivethirtyeight')

    '''use this for plotting the distribution of numerical features'''
    fig, ax = plt.subplots()
    ax.set_title(f"Distribution of {col}")
    sns.histplot(-(data[col].dropna()), kde=True, bins=bins, ax=ax)
    ax.set_xlabel(f"{col} (Years)")
    ax.set_ylabel("Density")

    if label_rotation:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')  # Ajout de l'argument 'ha'

    plt.axvline(x=val, color='red', linestyle='--')  # Set the y-axis limit

    st.pyplot(fig)


@st.cache_data(show_spinner="Post treatment ...")
def post_treatment(df):
    # Create an anomalous flag column
    data = df.copy()

    data['DAYS_EMPLOYED_ANOM'] = data["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    data['DAYS_BIRTH'] = abs(data['DAYS_BIRTH'])

    return pd.DataFrame(data)


@st.cache_resource(show_spinner="get target encoding ...")
def get_target_encoder(data):
    le = LabelEncoder()
    le.fit(data['TARGET'])

    return le


@st.cache_resource(show_spinner="get feature encoding ...")
def pre_encoded_feature(df):
    le = LabelEncoder()
    le_count = 0

    data = df.copy()
    # Extract feature columns
    x_cols = data.columns.tolist()

    # Check if 'TARGET' is in columns
    if 'TARGET' in x_cols:
        x_cols.remove('TARGET')

    for col in x_cols:
        if data[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(data[col].unique())) <= 2:
                #print(col)
                # Train on the training data
                le.fit(data[col])
                #Transform both training and testing data
                data[col] = le.transform(data[col])
                le_count += 1

    return data


@st.cache_data(show_spinner="Encoding data...")
def encode_data(df, _y_encoder):

    data = df.copy()

    if 'TARGET' in data.columns:
        data['TARGET'] = _y_encoder.transform(data['TARGET'])

    # #x_cols = df.select_dtypes('object').columns
    # data = pd.get_dummies(data, columns = data.select_dtypes('object').columns)

    data = pd.get_dummies(data)

    return data


@st.cache_resource(show_spinner="get feature impute ...")
def get_impute_data(data):

    # Median imputation of missing values
    impute = SimpleImputer(strategy='median')

    if 'TARGET' in data.columns:
        data = data.drop('TARGET', axis=1)
    else:
        pass

    # Fit on the training data
    impute.fit(data)

    return impute


@st.cache_data(show_spinner="imputing data ...")
def impute_data(df, _impute):

    data = df.copy()

    if 'TARGET' in data.columns:
        x_columns = data.drop('TARGET', axis=1).columns
    else:
        x_columns = data.columns

    # Median imputation of missing values
    data[x_columns] = _impute.transform(data[x_columns])

    return data


@st.cache_resource(show_spinner="get scaling data ...")
def get_scaling_data(df):

    scaler = MinMaxScaler(feature_range=(0, 1))

    data = df.copy()

    if 'TARGET' in data.columns:
        data = data.drop('TARGET', axis=1)
        # Repeat with the scaler
        scaler.fit(data)
    else:
        # Repeat with the scaler
        scaler.fit(data)

    # Scale each feature to 0-1
    return scaler


@st.cache_data(show_spinner="scaling data ...")
def scaling_data(df, _scaler):

    data = df.copy()

    if 'TARGET' in data.columns:
        y = data['TARGET']
        data = data.drop('TARGET', axis=1)
        x_columns = data.columns
        data[x_columns] = _scaler.transform(data)
        data = pd.concat([data,y], axis=1)
    else:
        # Repeat with the scaler
        x_columns = data.columns
        data[x_columns] = _scaler.transform(data)

    return data


@st.cache_resource(show_spinner="Training of the model ...")
def train_model(data):

    x = data.drop(['TARGET'], axis=1)
    y = data['TARGET']

    # Train on the training data
    model = LogisticRegression(C=0.0001)

    model.fit(x, y)

    return model

@st.cache_resource(show_spinner="Make a prediction ...")
def predict(model, x):

    y_pre = model.predict(x)

    return y_pre
