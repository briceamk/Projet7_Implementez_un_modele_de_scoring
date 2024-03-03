import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tools import encode_data_2, impute_data, scaling_data

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'SK_ID_CURR': [1, 2, 3],
        'Numeric_Column': [10, np.nan, 30],
        'Another_Numeric_Column': [5, 15, np.nan],
        'TARGET': [0, 1, 0],
        'Category_Column': ['A', 'B', 'A']
    })


# Définition de la fixture encoders
@pytest.fixture
def encoders(sample_data):
    y_encoder = LabelEncoder()
    x_encoder = OneHotEncoder(sparse_output=False)

    if 'TARGET' in sample_data.columns:
        y_encoder.fit(sample_data['TARGET'])

    x_encoder.fit(sample_data[['Category_Column']])

    return y_encoder, x_encoder


# Test de la fonction encode_data_2 avec la colonne 'TARGET'
def test_encode_data_2_with_target(sample_data, encoders):
    y_encoder, x_encoder = encoders
    result = encode_data_2(sample_data, y_encoder, x_encoder)

    # Assertions pour vérifier le résultat
    assert result.shape[0] == sample_data.shape[0]
    assert result.shape[1] == len(sample_data.columns) - 1 + len(x_encoder.get_feature_names_out(['Category_Column']))


# Test de la fonction encode_data_2 sans la colonne 'TARGET'
def test_encode_data_2_without_target(sample_data, encoders):
    df_no_target = sample_data.drop('TARGET', axis=1)
    y_encoder, x_encoder = encoders

    result = encode_data_2(df_no_target, y_encoder, x_encoder)

    # Assertions pour vérifier le résultat
    assert result.shape[0] == df_no_target.shape[0]
    assert result.shape[1] == df_no_target.shape[1] - 1 + len(x_encoder.get_feature_names_out(['Category_Column']))


def test_impute_data(sample_data):
    columns = [name for name in sample_data.columns if name not in ['SK_ID_CURR', 'TARGET']]
    # Créer une instance de SimpleImputer pour imputer les valeurs manquantes avec la médiane
    imputer= SimpleImputer(strategy='median').fit(sample_data[columns])

    # Appeler la fonction à tester avec le DataFrame d'exemple et l'imputer
    result = impute_data(sample_data, imputer)

    # Vérifier que les valeurs manquantes ont été correctement imputées
    assert result['Numeric_Column'].isnull().sum() == 0
    assert result['Another_Numeric_Column'].isnull().sum() == 0

    # Vérifier que les autres colonnes n'ont pas été modifiées
    assert result['SK_ID_CURR'].equals(sample_data['SK_ID_CURR'])
    assert result['TARGET'].equals(sample_data['TARGET'])


def test_scaling_data(sample_data):
    columns = [name for name in sample_data.columns if name not in ['SK_ID_CURR', 'TARGET']]
    # Créer une instance de StandardScaler pour le test
    scaler = StandardScaler().fit(sample_data[columns])

    # Appeler la fonction à tester avec le DataFrame d'exemple et le scaler
    result = scaling_data(sample_data, scaler)

    # Vérifier que les colonnes numériques ont été correctement mises à l'échelle
    assert result['Numeric_Column'].mean() == pytest.approx(0, abs=1e-6)
    assert result['Numeric_Column'].std() == pytest.approx(1, abs=1e-6)

    assert result['Another_Numeric_Column'].mean() == pytest.approx(0, abs=1e-6)
    assert result['Another_Numeric_Column'].std() == pytest.approx(1, abs=1e-6)

    # Vérifier que les autres colonnes n'ont pas été modifiées
    assert result['SK_ID_CURR'].equals(sample_data['SK_ID_CURR'])
    assert result['TARGET'].equals(sample_data['TARGET'])
