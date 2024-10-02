from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import numpy as np
app = Flask(__name__)
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

# Definir las características de entrada del modelo XGBoost
input_features_housing = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 
                          'area_per_room', 'total_rooms', 'price_per_area', 'mainroad', 
                          'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                          'prefarea', 'furnishingstatus']

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

@app.route('/predict_cirrosis', methods=['POST'])
def predictCirrosis():
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

        # Devolver la predicción como respuesta JSON
        return jsonify({'Stage': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})
    
# Ruta para hacer predicciones
@app.route('/predict_telephony', methods=['POST'])
def predictTelefonico():
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

        # Devolver la predicción como respuesta JSON
        return jsonify({'Churn': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})
    
# Ruta para hacer predicciones
@app.route('/classify_vino', methods=['POST'])
def predictVinoo():
    try:
        # Recibir datos en formato JSON
        data = request.json
        # Los datos deben estar en el mismo orden que las features que utilizaste para entrenar el modelo
        features = np.array([data['volatile acidity'], data['citric acid'], data['chlorides'], data['total sulfur dioxide'], 
                             data['density'], data['sulphates'], data['alcohol']])

        # Hacer la predicción
        prediction = modeloVino.predict([features])

        # Devolver la predicción como respuesta JSON
        return jsonify({'La calidad del vino es': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})
    
# Ruta para hacer predicciones
@app.route('/predict_hepatitis', methods=['POST'])
def predictHepatitis():
    try:
        # Recibir datos en formato JSON
        data = request.json
        # Los datos deben estar en el mismo orden que las features que utilizaste para entrenar el modelo
        features = np.array([data['Sex'], data['ALB'], data['ALT'], 
                             data['AST'], data['BIL'], data['CHE'],
                             data['CHOL'], data['CREA'], data['GGT']])

        # Hacer la predicción
        prediction = modeloHepatitis.predict([features])

        # Devolver la predicción como respuesta JSON
        return jsonify({'La clasificación de la persona es': int(prediction[0])})

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
        
        # Convertir las predicciones a una lista de Python
        output = {'predicted_prices': predictions.tolist()}
        
        return jsonify(output)
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

        # Convertir las predicciones a una lista de Python
        output = {'predicted_bodyfat': predictions.tolist()}
        
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/predict_stroke', methods=['POST'])
def predict_stroke():
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

        # Convertir las predicciones a una lista de Python
        output = {'predicted_stroke_risk': predictions.tolist()}
        
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
    try:
        # Obtener los datos JSON del request
        data = request.get_json(force=True)

        # Verificar si 'data' es una lista (múltiples entradas) o un diccionario (una sola entrada)
        if isinstance(data, list):
            # Convertir la lista de entradas en un DataFrame
            input_data = pd.DataFrame(data)
        else:
            # Convertir la entrada única en un DataFrame
            input_data = pd.DataFrame([data])

        # Realizar los cálculos de ingeniería de características automáticamente
        input_data['total_rooms'] = input_data['bedrooms'] + input_data['bathrooms']
        input_data['area_per_room'] = input_data['area'] / input_data['total_rooms']

        # Asegurarse de que las columnas estén en el orden correcto y coincidan con las características usadas en el modelo
        input_data = input_data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
                                 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 
                                 'furnishingstatus', 'area_per_room', 'total_rooms']]

        # Realizar las predicciones con el modelo XGBoost
        predictions = xgb_pipeline_model.predict(input_data)

        # Convertir las predicciones a una lista de Python
        output = {'predicted_house_prices': predictions.tolist()}

        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
#--------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)