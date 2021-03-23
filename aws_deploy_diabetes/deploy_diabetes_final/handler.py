import json
import pickle

# 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'

model = pickle.load(open("./RandomForest", "rb"))


def predict(event, context):
    body = {
        "message": "ok",
    }

    params = event["queryStringParameters"]

    Pregnancies = float(params['Pregnancies'])
    glucose = float(params['glucose'])
    BP = float(params['BP'])
    SkinThickness = float(params['SkinThickness'])
    Insulin = float(params['Insulin'])
    BMI = float(params['BMI'])
    DiabetesPedigreeFunction = float(params['DiabetesPedigreeFunction'])
    Age = float(params['Age'])

    input_data = [[Pregnancies, glucose, BP, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]

    pred = model.predict(input_data)[0]

    body['prediction'] = int(pred)
    print(params)
    print("Prediction: ", pred)

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Access-Control-Allow-Origin": "*"
        }
    }

    return response
