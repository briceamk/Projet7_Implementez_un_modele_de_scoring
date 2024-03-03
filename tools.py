import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

# imputer for handling missing values
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# Function to calculate missing values by column# Funct
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def encoding_data(train, test):
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in train:
        if train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(train[col].unique())) <= 2:
                # Train on the training data
                le.fit(train[col])
                # Transform both training and testing data
                train[col] = le.transform(train[col])
                test[col] = le.transform(test[col])

                # Keep track of how many columns were label encoded
                le_count += 1

    print('%d columns were label encoded.' % le_count)

    # Onehot encoding of categorical Variables

    a_train = pd.get_dummies(train)
    a_test = pd.get_dummies(test)

    train_labels = train['TARGET']

    # Align the training and testing data, keep only columns present in both dataframes
    train, test = train.align(test, join='inner', axis=1)

    # Add the target back in
    train['TARGET'] = train_labels

    print('Training Features shape after align: ', train.shape)
    print('Testing Features shape after align ', test.shape)

    return a_train, a_test, train_labels

    ################################################################


def plot_hist(name, app_train):
    figure = app_train[name].plot.hist(title=name + ' Histogram')
    figure.set_xlabel(name)
    plt.savefig(name + '.png')


def hist_kde_plot(name, app_train):
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))

    # Set the style of plots
    sns.set(style="whitegrid")

    # KDE plot of loans that were repaid on time
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, name] / 365, label='target == 0', ax=ax[0])

    # KDE plot of loans which were not repaid on time
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, name] / 365, label='target == 1', ax=ax[0])

    # Plot the distribution of ages in years
    ax[1].hist(app_train[name] / 365, edgecolor='k', bins=25)

    # Labeling of plot
    ax[0].set_xlabel('Age (years)')
    ax[0].set_ylabel('Density')
    ax[0].set_title('Distribution of Ages')
    ax[1].set_title('Age of Client')
    ax[1].set_xlabel('Age (years)')
    ax[1].set_ylabel('Count')

    # Show pic
    plt.savefig(name + '.png')


def group_age(age_data):
    # Group by the bin and calculate averages
    age_groups = age_data.groupby('YEARS_BINNED').mean()
    print(age_groups)

    plt.figure(figsize=(8, 8))

    # Graph the age bins and the average of the target as a bar plot
    plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

    # Plot labeling
    plt.xticks(rotation=75)
    plt.xlabel('Age Group (years)')
    plt.ylabel('Failure to Repay (%)')
    plt.title('Failure to Repay by Age Group')


def features_engineering_poly(app_train, list_train, app_test, list_test):
    # Make a new dataframe for polynomial features
    poly_features = app_train[list_train]
    poly_features_test = app_test[list_test]

    # imputer for handling missing values
    imputer = SimpleImputer(strategy='median')

    poly_target = poly_features['TARGET']
    poly_features = poly_features.drop(columns=['TARGET'])

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)

    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)
    print('Polynomial Features shape: ', poly_features.shape)

    # Create a dataframe of the features
    poly_features = pd.DataFrame(poly_features,
                                 columns=poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                 'EXT_SOURCE_3', 'DAYS_BIRTH']))

    # Add in the target
    poly_features['TARGET'] = poly_target

    # Find the correlations with the target
    poly_corrs = poly_features.corr()['TARGET'].sort_values()

    # Display most negative and most positive
    print('\nMost positive Correlation\n', poly_corrs.head(5))
    print('\nMost Negative Correlation\n', poly_corrs.tail(5))

    # Put test features into dataframe
    poly_features_test = pd.DataFrame(poly_features_test,
                                      columns=poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                      'EXT_SOURCE_3', 'DAYS_BIRTH']))

    # Merge polynomial features into training dataframe
    poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
    app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')

    # Merge polnomial features into testing dataframe
    poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
    app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')

    # Align the dataframes
    app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)

    # Print out the new shapes
    print('Training data with polynomial features shape: ', app_train_poly.shape)
    print('Testing data with polynomial features shape:  ', app_test_poly.shape)

    return app_train_poly, app_test_poly


def log_classification(app_train, train_labels, seed):
    # Drop the target from the training data
    if 'TARGET' in app_train:
        train = app_train.drop(columns=['TARGET'])
    else:
        train = app_train.copy()

    train, test, y_train_labels, y_test_labels = train_test_split(train, train_labels, test_size=0.2, random_state=seed)

    # Median imputation of missing values
    imputer = SimpleImputer(strategy='median')

    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit on the training data
    imputer.fit(train)

    # Transform both training and testing data
    train = imputer.transform(train)
    test = imputer.transform(test)

    # Repeat with the scaler
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    print('Training data shape: ', train.shape)
    print('Testing data shape: ', test.shape)

    # Make the model with the specified regularization parameter
    log_reg = LogisticRegression(C=0.0001)

    # Train on the training data
    model = log_reg.fit(train, y_train_labels)

    # Make predictions
    # Make sure to select the second column only
    log_reg_pre_proba = log_reg.predict_proba(test)
    log_reg_accuracy = accuracy_score(log_reg.predict(test), y_test_labels)
    print('y_test_labels', y_test_labels)
    print('accuracy', log_reg_accuracy)

    return model, log_reg_pre_proba, log_reg_accuracy, train, y_train_labels

def data_description(folder):
    '''Check the number of rows, columns, missing values, and duplicates.
       Count the type of columns.
       Memory indication'''

    data_dict = {}
    for file in folder:
        data = pd.read_csv(file, encoding='ISO-8859-1')
        data_dict[file] = {
            'Rows': data.shape[0],
            'Columns': data.shape[1],
            '%NaN': round(data.isna().sum().sum() / data.size * 100, 2),
            '%Duplicate': round(data.duplicated().sum().sum() / data.size * 100, 2),
            'Object Dtype': data.select_dtypes(include=['object']).shape[1],
            'Float Dtype': data.select_dtypes(include=['float']).shape[1],
            'Int Dtype': data.select_dtypes(include=['int']).shape[1],
            'Bool Dtype': data.select_dtypes(include=['bool']).shape[1],
            'Memory (MB)': round(data.memory_usage(deep=True).sum() / 1024**2, 3)
        }

    comparative_table = pd.DataFrame.from_dict(data_dict, orient='index')
    return comparative_table
#################################################################################################

def load_data():
    path = './input/application_train.csv'
    data = pd.read_csv(path)

    return data


def plot_amount(data, col, val, bins=30, label_rotation=True):
    plt.style.use('fivethirtyeight')

    # Plot the distribution of the numerical feature
    fig, ax = plt.subplots()
    ax.set_title(f"Distribution of {col}")
    ax.hist(data[col].dropna(), bins=bins, density=True, alpha=0.7, color='blue', label='Histogram') # Draw a vertical dashed line at the specified value
    ax.axvline(x=val, color='red', linestyle='--', label=f'Value: {val}')

    # Set labels and legend
    ax.set_xlabel(f"{col}")
    ax.set_ylabel("Density")
    ax.legend()

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Optionally rotate x-axis labels
    if label_rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    return fig

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


def post_treatment(df):
    # Create an anomalous flag column
    data = df.copy()

    data['DAYS_EMPLOYED_ANOM'] = data["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    data['DAYS_BIRTH'] = abs(data['DAYS_BIRTH'])

    return pd.DataFrame(data)


def get_target_encoder(data):
    le = LabelEncoder()
    le.fit(data['TARGET'])

    return le


def pre_encoded_feature(df, feature_le_encoded=None):
    """
    Encode les colonnes de type 'object' avec LabelEncoder.

    Args:
        df (DataFrame): Le DataFrame à encoder.
        feature_le_encoded (list, optional): Liste des colonnes à encoder. Si non spécifié,
            toutes les colonnes de type 'object' avec 2 catégories ou moins seront encodées.

    Returns:
        DataFrame: Le DataFrame modifié après encodage.
        list: La liste des colonnes encodées.
        int: Le nombre total d'encodages effectués.
    """
    le = LabelEncoder()
    le_count = 0

    data = df.copy()
    # Extract feature columns
    x_cols = data.columns.tolist()

    if feature_le_encoded is not None:
        # Si feature_le_encoded est spécifié, encoder les colonnes spécifiées
        for col in feature_le_encoded:
            if data[col].dtype == 'object':
                # Si 2 catégories ou moins, encoder avec LabelEncoder
                if len(list(data[col].unique())) <= 2:
                    le.fit(data[col])
                    data[col] = le.transform(data[col])
                    le_count += 1

        return data, feature_le_encoded, le_count

    # Si feature_le_encoded n'est pas spécifié, encoder toutes les colonnes de type 'object'
    feature_le_encoded = []
    # Check if 'TARGET' is in columns
    if 'TARGET' in x_cols:
        x_cols.remove('TARGET')

    if 'SK_ID_CURR' in x_cols:
        x_cols.remove('SK_ID_CURR')

    for col in x_cols:
        if data[col].dtype == 'object':
            # Si 2 catégories ou moins, encoder avec LabelEncoder
            if len(list(data[col].unique())) <= 2:
                feature_le_encoded.append(col)
                le.fit(data[col])
                data[col] = le.transform(data[col])
                le_count += 1

    return data, feature_le_encoded, le_count


def get_encoded_feature(df):

    data = df.select_dtypes('object').copy()
    ohe = OneHotEncoder(sparse=False)

    ohe.fit(data.drop('TARGET', axis=1)) if 'TARGET' in data.columns else ohe.fit(data)

    return ohe


def encode_data_2(df, _y_encoder, _x_encoder):
    data = df.copy()

    # Extract non-object columns
    rest_columns = [name for name in data.columns if name not in data.select_dtypes('object').columns and name != 'TARGET']
    data_rest = data[rest_columns]

    # Extract object columns for one-hot encoding
    data_ohe_columns = data.select_dtypes('object').columns
    one_hot_encod = _x_encoder.transform(data[data_ohe_columns])
    data_ohe = pd.DataFrame(one_hot_encod, columns=_x_encoder.get_feature_names_out(data_ohe_columns))


    # Encoding of Target
    if 'TARGET' in data.columns:
        data['TARGET'] = _y_encoder.transform(data['TARGET'])
        # Concatenate the encoded data
        data_encoded = pd.concat([data['TARGET'], data_rest, data_ohe], axis=1)
    else:
        data_encoded = pd.concat([data_rest, data_ohe], axis=1)

    return data_encoded


def get_impute_data(df):

    # Median imputation of missing values
    impute = SimpleImputer(strategy='median')

    data = df.copy()
    x_columns = [name for name in data.columns if name not in ['SK_ID_CURR', 'TARGET']]

    # Fit on the training data
    impute.fit(data[x_columns])

    return impute


def impute_data(df, _impute):

    data = df.copy()
    x_columns = [name for name in data.columns if name not in ['SK_ID_CURR', 'TARGET']]

    # Median imputation of missing values
    data[x_columns] = _impute.transform(data[x_columns])

    return data


def get_scaling_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = df.copy()
    x_columns = [name for name in data.columns if name not in ['SK_ID_CURR', 'TARGET']]
    # Repeat with the scaler
    scaler.fit(data[x_columns])

    return scaler


def scaling_data(df, _scaler):
    data = df.copy()
    x_columns = [name for name in data.columns if name not in ['SK_ID_CURR', 'TARGET']]
    data[x_columns] = _scaler.transform(data[x_columns])

    return data


def train_model(df):

    data = df.copy()
    x_columns = [name for name in data.columns if name not in ['SK_ID_CURR', 'TARGET']]
    x = data[x_columns]
    y = data['TARGET']

    # Train on the training data
    model = LogisticRegression(C=0.0001)

    model.fit(x, y)

    return model


def predict(model, df):
    data = df.copy()
    x_columns = [name for name in data.columns if name not in ['SK_ID_CURR', 'TARGET']]
    y_pre = model.predict(data[x_columns])

    return y_pre


def draw_gauge(value, max_value):
    # Calcule la proportion de la valeur par rapport à la valeur maximale
    ratio = value / max_value

    # Définit la couleur de la barre en fonction du ratio
    color = (ratio, 1 - ratio, 0)

    # Trace le graphique à barres horizontales
    fig, ax = plt.subplots(figsize=(8, 1))

    # Barre principale
    ax.barh([0], [ratio], color=color, height=1)

    # Barre au milieu (en noir)
    ax.vlines(x=0.5, ymin=0, ymax=1, color='black', linestyle='-', linewidth=1)

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Ajoute un titre à la jauge
    ax.set_title("Niveau de risque du client")


    # Ajoute un texte indiquant si le crédit est accordé ou non
    credit_status = "Crédit Accordé" if ratio < 0.5 else "Crédit Refusé"
    color_status = "green" if ratio < 0.5 else "red"

    # Utilise st.markdown avec du code HTML pour ajuster la couleur du texte
    st.markdown(f'<p style="color:{color_status}">{credit_status}</p>', unsafe_allow_html=True)

    st.pyplot(fig)

    # Exemple d'utilisation


