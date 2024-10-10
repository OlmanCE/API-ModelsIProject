from flask import Flask, request, jsonify, Response
import joblib
import pandas as pd
import json
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/api/*": {"origins": "*"}})
#---------------------------------------------O----------------------------------------------------------
# Cargar el primer modelo (Decision Tree)
decision_tree_model = joblib.load('models/modelo_decision_tree.pkl')

# Cargar el segundo modelo (Random Forest) que acabas de descargar
random_forest_model = joblib.load('models/modelo_bodyfat.pkl')

# Cargar las características de entrada para el primer modelo
with open('models/input_features.json', 'r') as f:
    input_features_decision_tree = json.load(f)

# Características para el modelo de Random Forest (asegúrate de ajustarlas según los datos del nuevo modelo)
input_features_random_forest = ['Abdomen', 'Wrist', 'Weight', 'Forearm', 'Chest', 'Hip', 'Thigh', 'Neck', 'Knee']


# Cargar el tercer modelo entrenado de stroke (Random Forest)
stroke_model = joblib.load('models/stroke_prediction_model.joblib')

# Definir las características de entrada para el modelo de stroke
input_features_stroke = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 
                         'avg_glucose_level', 'bmi', 'work_type_Never_worked', 'work_type_Private', 
                         'work_type_Self-employed', 'work_type_children', 'smoking_status_formerly smoked', 
                         'smoking_status_never smoked', 'smoking_status_smokes']


# Cargar el modelo XGBoost (predicción de precios de casas)
xgb_pipeline_model = joblib.load('models/xgb_pipeline_housing.joblib')

# # Definir las características de entrada del modelo XGBoost
# input_features_housing = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 
#                           'area_per_room', 'total_rooms', 'price_per_area', 'mainroad', 
#                           'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
#                           'prefarea', 'furnishingstatus']

#----------------------------------------C----------------------------------------------------------------
# Cargar el modelo previamente guardado
modeloCirrosis = joblib.load('models/Cirrosis_RF.pkl') 

# Cargar el modelo previamente guardado
modeloTelefonico = joblib.load('models/telefonico_SVM.pkl') 

# Cargar el modelo previamente guardado
modeloVino = joblib.load('models/vino_RF.pkl') 

# Cargar el modelo previamente guardado
modeloHepatitis = joblib.load('models/Hepatitis_RF.pkl') 


#--------------------------------------------------------------------------------------------------------

# Cargar el modelo guardado
model_bitcoin = joblib.load('models/bitcoin.pkl')

# Cargar el modelo guardado
model_Market = joblib.load('models/mercadoSP.pkl')

#--------------------------------------------------------------------------------------------------------



@app.route('/predict_cirrosis', methods=['POST'])
def predictCirrosis():

    STAGE_MAPPING = {
    1: 'El paciente tiene fibrosis leve',
    2: 'El paciente tiene fibrosis moderada',
    3: 'El paciente tiene fibrosis avanzada',
    4: 'El paciente tiene cirrosis'
}

    try:
        # Recibir datos en formato JSON
        data = request.json
        # Los datos deben estar en el mismo orden que las features que utilizaste para entrenar el modelo
        features = np.array([data['N_Days'], data['Status'], data['Age'], data['Ascites'], 
                             data['Hepatomegaly'], data['Spiders'], data['Edema'], 
                             data['Bilirubin'], data['Albumin'], data['Copper'], data['Alk_Phos'], 
                             data['SGOT'], data['Tryglicerides'], data['Platelets'], 
                             data['Prothrombin']])

        # Hacer la predicción
        prediction = modeloCirrosis.predict([features])


        predicted_class = int(prediction[0])

        stage_description = STAGE_MAPPING.get(predicted_class, 'Desconocido')
        # Devolver la predicción como respuesta JSON
        # cuatro posibles opciones
        return jsonify(stage_description)

    except Exception as e:
        return jsonify({'error': str(e)})
    
# Ruta para hacer predicciones
@app.route('/predict_telephony', methods=['POST'])
def predictTelefonico():
    CHURN_MAPPING = {
        0: 'El usuario no va a abandonar la empresa',
        1: 'El usuario va a abandonar la empresa'
    }

    try:
        # Recibir datos en formato JSON
        data = request.json
        # Los datos deben estar en el mismo orden que las features que utilizaste para entrenar el modelo
        features = np.array([data['SeniorCitizen'], data['Partner'], data['Dependents'], data['tenure'], 
                             data['InternetService'], data['OnlineSecurity'], data['OnlineBackup'], 
                             data['DeviceProtection'], data['TechSupport'], data['Contract'], data['PaperlessBilling'], 
                             data['PaymentMethod'], data['MonthlyCharges']])

        # Hacer la predicción
        prediction = modeloTelefonico.predict([features])

        # Obtener el valor entero de la predicción
        predicted_class = int(prediction[0])

        # Mapear la clase predicha a la descripción correspondiente
        churn_description = CHURN_MAPPING.get(predicted_class, 'Desconocido')

        # Devolver la predicción como respuesta JSON
        return jsonify(churn_description)

    except Exception as e:
        return jsonify({'error': str(e)})
    
# Ruta para hacer predicciones
@app.route('/classify_vino', methods=['POST'])
def predictVinoo():

    WINE_QUALITY ={
        0: "La calidad del vino es mala",
        1: "La calidad del vino es buena"

    }
    try:
        # Recibir datos en formato JSON
        data = request.json
        # Los datos deben estar en el mismo orden que las features que utilizaste para entrenar el modelo
        features = np.array([data['volatile acidity'], data['citric acid'], data['chlorides'], data['total sulfur dioxide'], 
                             data['density'], data['sulphates'], data['alcohol']])

        # Hacer la predicción
        prediction = modeloVino.predict([features])

        predicted_class = int(prediction[0])


        wine_description = WINE_QUALITY.get(predicted_class)
        # Devolver la predicción como respuesta JSON
        # 0 Mala 1 Calidad buena
        return jsonify(wine_description)

    except Exception as e:
        return jsonify({'error': str(e)})
    
# Ruta para hacer predicciones
# 0 no tiene 1 si tiene hepatitis
@app.route('/predict_hepatitis', methods=['POST'])
def predictHepatitis():
    HEPATITIS_MAPPING ={
        0 : "El paciente no tiene Hepatitis.",
        1 : "El paciente tiene Hepatitis"
    }

    try:
        # Recibir datos en formato JSON
        data = request.json
        # Los datos deben estar en el mismo orden que las features que utilizaste para entrenar el modelo
        features = np.array([data['Sex'], data['ALB'], data['ALT'], 
                             data['AST'], data['BIL'], data['CHE'],
                             data['CHOL'], data['CREA'], data['GGT']])

        # Hacer la predicción
        prediction = modeloHepatitis.predict([features])

        predicted_class = int(prediction[0])

        hapatitis_description = HEPATITIS_MAPPING.get(predicted_class, 'Desconocido')
        # Devolver la predicción como respuesta JSON
        return jsonify(hapatitis_description)

    except Exception as e:
        return jsonify({'error': str(e)})

#--------------------------------------------------------------------------------------------------------

@app.route('/predict_price_vehicle', methods=['POST'])
def predict_decision_tree():
    try:
        # Obtener los datos JSON del request
        data = request.get_json(force=True)
        
        # Verificar si 'data' es una lista (múltiples entradas) o un diccionario (una sola entrada)
        if isinstance(data, list):
            # Convertir la lista de entradas en un DataFrame
            input_data = pd.DataFrame(data, columns=input_features_decision_tree)
        else:
            # Convertir la entrada única en un DataFrame
            input_data = pd.DataFrame([data], columns=input_features_decision_tree)
        
        # Realizar las predicciones con el Decision Tree
        predictions = decision_tree_model.predict(input_data)
        
    
        # Obtener el precio predicho
        predicted_price = predictions[0]
        
        # Formatear la respuesta
        response_text = f"El precio del vehículo es {float(predicted_price)} dolares"
        
        return jsonify(response_text)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_bodyfat', methods=['POST'])
def predict_random_forest():
    try:
        # Obtener los datos JSON del request
        data = request.get_json(force=True)

        # Verificar si 'data' es una lista (múltiples entradas) o un diccionario (una sola entrada)
        if isinstance(data, list):
            # Convertir la lista de entradas en un DataFrame
            input_data = pd.DataFrame(data, columns=input_features_random_forest)
        else:
            # Convertir la entrada única en un DataFrame
            input_data = pd.DataFrame([data], columns=input_features_random_forest)
        
        # Realizar las predicciones con el modelo Random Forest
        predictions = random_forest_model.predict(input_data)

        # Obtener el precio predicho
        predicted_bodyfat = predictions[0]
        # Formatear la respuesta
        response_text = f"El porcentaje de grasa corporal es {predicted_bodyfat}"
        return jsonify(response_text)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/predict_stroke', methods=['POST'])
def predict_stroke():
    STROKE_MAPPING = {
        0: "El paciente no tiene riesgo de sufrir un ataque cardiovascular",
        1: "El paciente tiene riesgo de sufrir un ataque cardiovascular"
    }
    
    try:
        # Obtener los datos JSON del request
        data = request.get_json(force=True)

        # Verificar si 'data' es una lista (múltiples entradas) o un diccionario (una sola entrada)
        if isinstance(data, list):
            # Convertir la lista de entradas en un DataFrame
            input_data = pd.DataFrame(data, columns=input_features_stroke)
        else:
            # Convertir la entrada única en un DataFrame
            input_data = pd.DataFrame([data], columns=input_features_stroke)
        
        # Realizar las predicciones con el modelo de stroke
        y_pred_proba = stroke_model.predict_proba(input_data)[:, 1]  # Probabilidades de la clase 1 (stroke)
        
        # Aplicar el umbral ajustado de 0.25 para la clase stroke
        threshold = 0.25
        predictions = (y_pred_proba >= threshold).astype(int)

        # Preparar la respuesta
        if len(predictions) == 1:
            stroke_description = STROKE_MAPPING.get(predictions[0], 'Desconocido')
            response = stroke_description
        else:
            response = [STROKE_MAPPING.get(pred, 'Desconocido') for pred in predictions]
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
    try:
        # Obtener los datos JSON del request
        data = request.get_json(force=True)

        # Función para transformar los datos de entrada
        def transform_input(entry):
            # Convertir 1/0 a yes/no
            boolean_fields = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
            for field in boolean_fields:
                entry[field] = 'yes' if entry[field] == 1 else 'no'
            
            # Convertir el estado de amueblado
            furnishing_map = {0: 'unfurnished', 1: 'furnished', 2: 'semi-furnished'}
            entry['furnishingstatus'] = furnishing_map.get(entry['furnishingstatus'], 'unfurnished')
            
            return entry

        # Verificar si 'data' es una lista (múltiples entradas) o un diccionario (una sola entrada)
        if isinstance(data, list):
            input_data = pd.DataFrame([transform_input(entry) for entry in data])
        else:
            input_data = pd.DataFrame([transform_input(data)])

        # Realizar los cálculos de ingeniería de características automáticamente
        input_data['total_rooms'] = input_data['bedrooms'] + input_data['bathrooms']
        input_data['area_per_room'] = input_data['area'] / input_data['total_rooms']

        # Asegurarse de que las columnas estén en el orden correcto y coincidan con las características usadas en el modelo
        input_data = input_data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus', 'area_per_room', 'total_rooms']]

        # Realizar las predicciones con el modelo XGBoost
        predictions = xgb_pipeline_model.predict(input_data)

        # Preparar la respuesta
        if len(predictions) == 1:
            response = f"El precio de la casa es: {float(predictions[0])} dolares"
        else:
            response = f"Precios predecidos {[float(price) for price in predictions]}"

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
#--------------------------------------------------------------------------------------------------------
@app.route('/predict_bitcoin', methods=['POST'])
def predict_predict_bitcoin():
    # Última fecha del conjunto de datos del modelo
    ultima_fecha_modelo = pd.Timestamp('2017-07-31')
    try:
        # Obtener los datos de la solicitud
        data = request.json
        year = int(data.get('year'))
        month = int(data.get('month'))
        day = int(data.get('day'))

        # Crear la fecha con los datos recibidos
        fecha_input = f'{year}-{month:02d}-{day:02d}'
        fecha_dt = pd.Timestamp(fecha_input)  # Convertir a pandas.Timestamp

        # Validar que la fecha de entrada sea mayor a la fecha del modelo
        if fecha_dt <= ultima_fecha_modelo:
            return jsonify({'error': 'La fecha debe ser mayor que la última fecha conocida en el modelo.'}), 400

        # Realizar la predicción
        prediccion = model_bitcoin.predict(fecha_dt)

        # Devolver la predicción
        return jsonify({'prediccion': prediccion[0]})

    except KeyError as e:
        return jsonify({'error': f"KeyError: {e}. Verifica si la fecha existe en los datos del modelo."}), 400
    except TypeError as e:
        return jsonify({'error': f"TypeError: {e}. Verifica el formato de entrada esperado por el modelo."}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/mercadoSP', methods=['POST'])
def predict_mercadoSP():
    # Última fecha del conjunto de datos del modelo
    try:
        ultima_fecha_modelo = pd.Timestamp('2018-02-07')
        # Obtener los datos de la solicitud
        data = request.json
        year = int(data.get('year'))
        month = int(data.get('month'))
        day = int(data.get('day'))

        # Crear la fecha con los datos recibidos
        fecha_input = f'{year}-{month:02d}-{day:02d}'
        fecha_dt = pd.Timestamp(fecha_input)  # Convertir a pandas.Timestamp

        # Validar que la fecha de entrada sea mayor a la fecha del modelo
        if fecha_dt <= ultima_fecha_modelo:
            return jsonify({'error': 'La fecha debe ser mayor que la última fecha conocida en el modelo.'}), 400

        # Realizar la predicción
        prediccion = model_Market.predict(fecha_dt)

        # Devolver la predicción
        return jsonify({'prediccion': prediccion[0]})

    except KeyError as e:
        return jsonify({'error': f"KeyError: {e}. Verifica si la fecha existe en los datos del modelo."}), 400
    except TypeError as e:
        return jsonify({'error': f"TypeError: {e}. Verifica el formato de entrada esperado por el modelo."}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


#--------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run(debug=True)