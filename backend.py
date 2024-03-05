from tools import plot_amount, post_treatment, pre_encoded_feature, impute_data, scaling_data, encode_data_2
from flask import Flask, request, jsonify
from flask_caching import Cache
import pandas as pd
import pickle
# import export
# from pyspark.sql.functions import struct, col

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'FileSystemCache'
app.config['CACHE_DIR'] = 'cache' # path to your server cache folder
app.config['CACHE_THRESHOLD'] = 100000 # number of 'files' before start auto-delete
cache = Cache(app)


@app.route("/", methods=['GET'])
def root_func():
    return "<p>Hello, World!</p>"


def graph_numeric(data, name, value):
    fig = plot_amount(data, col=name, val=value, bins=50, label_rotation=False)
    fig.savefig(f'{name}.png')


def load_process():
    path = 'input/application_test.csv'

    save_dir = 'obj_save/'

    objects_to_save = ["data", "model", "_scaler", "_impute", "_le", "feature_le_encoded", "_ohe"]
    loaded_objects = {}

    if cache.get('load_data'):
        print('Inside the cache..... and we have data')
        return cache.get('load_data')
    else:
        for key in objects_to_save:
            if key != "data":
                with open(save_dir + f"{key}.pkl", "rb") as file:
                    loaded_objects[key] = pickle.load(file)
            else:
                loaded_objects[key] = pd.read_csv(path)
        cache.set('load_data', loaded_objects)
        return loaded_objects


object_loaded = load_process()

df = object_loaded["data"]
model = object_loaded["model"]
_scaler = object_loaded["_scaler"]
_impute = object_loaded["_impute"]
_le = object_loaded["_le"]
_ohe = object_loaded["_ohe"]
feature_le_encoded = object_loaded["feature_le_encoded"]

#
# # # Loading of the model
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# logged_model = 'runs:/2b5cf95663c840a8b5b3b39b7cc51d9e/LGBMClassifier'
# # Load model as a PyFuncModel.
# model = mlflow.sklearn.load_model(logged_model)


def process_2(x, var=None):
    """
    Effectue un traitement sur les données en plusieurs étapes.

    """
    x = post_treatment(x)

    if var is None:
        x, feature, le_count = pre_encoded_feature(x)
    else:
        x, feature, le_count = pre_encoded_feature(x, var)

    # Utilisez l'encodage personnalisé si var est spécifié, sinon utilisez l'encodage par défaut
    x = encode_data_2(x, _le, _ohe)
    x = impute_data(x, _impute)
    x = scaling_data(x, _scaler)

    return x

@app.route("/load_initial_data/v2", methods=['GET'])
def load_initial_data():
    row_select = df.iloc[0, :]
    details = {
        'code_gender': row_select["CODE_GENDER"],
        'occupation_type': row_select['OCCUPATION_TYPE'],
        'name_income_type': row_select['NAME_INCOME_TYPE'],
        'education_type': row_select['NAME_EDUCATION_TYPE'],
        'housing_type': row_select['NAME_HOUSING_TYPE'],
        'amt_credit': row_select["AMT_CREDIT"],
        'amt_income_total': row_select["AMT_INCOME_TOTAL"],
        'amt_annuity': row_select["AMT_ANNUITY"],
        'days_employed': abs(int(row_select["DAYS_EMPLOYED"])),
        'days_birth': int(row_select["DAYS_BIRTH"])
    }

    graph_numeric(df, name="AMT_CREDIT", value=details.get('amt_credit'))
    graph_numeric(df, name="AMT_INCOME_TOTAL",value=details.get('amt_income_total'))
    graph_numeric(df, name="AMT_ANNUITY", value=details.get('amt_annuity'))
    graph_numeric(df, name="DAYS_EMPLOYED", value=details.get('days_employed'))
    graph_numeric(df, name="DAYS_BIRTH", value=details.get('days_birth'))

    # Convertissez le DataFrame en un dictionnaire JSON compatible
    json_data = {'ids': df['SK_ID_CURR'].to_dict(), 'values': details}

    return jsonify(json_data)

@app.route("/load_data/v2/<int:id>", methods=['GET'])
def load_data(id):
    print('Inside load_data....')
    mask = df['SK_ID_CURR'] == id
    row_select = df[mask]
    details = {
        'code_gender': row_select["CODE_GENDER"].values[0],
        'occupation_type': row_select['OCCUPATION_TYPE'].values[0],
        'name_income_type': row_select['NAME_INCOME_TYPE'].values[0],
        'education_type': row_select['NAME_EDUCATION_TYPE'].values[0],
        'housing_type': row_select['NAME_HOUSING_TYPE'].values[0],
        'amt_credit': row_select["AMT_CREDIT"].values[0],
        'amt_income_total': row_select["AMT_INCOME_TOTAL"].values[0],
        'amt_annuity': row_select["AMT_ANNUITY"].values[0],
        'days_employed': abs(int(row_select["DAYS_EMPLOYED"].values[0])),
        'days_birth': int(row_select["DAYS_BIRTH"].values[0])
    }
    graph_numeric(df, name="AMT_CREDIT", value=details.get('amt_credit'))
    graph_numeric(df, name="AMT_INCOME_TOTAL", value=details.get('amt_income_total'))
    graph_numeric(df, name="AMT_ANNUITY", value=details.get('amt_annuity'))
    graph_numeric(df, name="DAYS_EMPLOYED", value=details.get('days_employed'))
    graph_numeric(df, name="DAYS_BIRTH", value=details.get('days_birth'))

    # Convertissez le DataFrame en un dictionnaire JSON compatible
    json_data = {'ids': df['SK_ID_CURR'].to_dict(), 'values': details}

    return jsonify(json_data)



@app.route("/predict", methods=['GET'])
def predict():

    if 'id' not in request.args:
        return 'Error: No id field provided. Please specify an id.'

    idx = int(request.args['id'])
    mask = df['SK_ID_CURR'] == idx
    data = df[mask]
    data = process_2(data, var=feature_le_encoded)

    if len(data) == 0:
        return f'Error: an id.{len(data)}'

    if len(data) >= 1:
        row_data = data.iloc[0]
        row_data_x = row_data.drop(['TARGET','SK_ID_CURR'], errors='ignore')

        # Faites la prédiction (remplacez cela par votre propre logique)
        row_data_x_reshape = row_data_x.values.reshape(1, -1)
        proba_pre = model.predict_proba(row_data_x_reshape)
        y_pre = model.predict(row_data_x_reshape)

        # Convertissez le résultat de la prédiction en un dictionnaire JSON
        prediction_result = {'proba': proba_pre.tolist(), 'prediction': y_pre.tolist()}  # Assurez-vous que y_pre est sérialisable

        # Retournez le résultat sous forme de réponse JSON
        return jsonify(prediction_result)


if __name__ == "__main__":
    app.run(host='localhost', port=3000, debug=True)


# http://localhost:1234/api/get_data_from_id/?id=1

# app.run(host='localhost', port=1234)


# 1- Brut
# 2- Acces au model Ramifié
# 3- Acces au Model par le serveur

######################"
# - Finaliser le Web
# - Mettre à jour les modèles , Creant score metier

# - Charge MLFLOW pour tracking
# - Faire le Flask
# - Faire le GIT
