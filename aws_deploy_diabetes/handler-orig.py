import json
import pickle




#'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'


model = pickle.load(open("Random_Forest.pkl","rb"))



def predict(event, context):
    body = {
        "message": "ok",
       
    }


    params = event['queryStringParameters']
    
    pregnancies = float(params['pregnancies'])
    glucose = float(params['glucose'])
    BP = float(params['BP'])
    SkinThickness = float(params['SkinThickness'])
    Insulin = float(params['Insulin'])
    BMI = float(params['BMI'])
    DiabetesPedigreeFunction = float(params['DiabetesPedigreeFunction'])
    Age = float(params['Age'])

    input_data = [[pregnancies,glucose,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]

    pred =  model.predict(input_data)[0]

    body['prediction'] = int(pred)
    print(params)
    print("Prediction: ",pred)

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers":{
            "Access-Control-Allow-Origin":"*"
        }
    }

    return response




def do_main():
    event = {
        "queryStringParameters":{
            "pregnancies":1,
             "glucose":12, 
             "BP":120, 
             "SkinThickness":1 , 
             "Insulin":2,
             "BMI":23, 
             "DiabetesPedigreeFunction":21, 
             "Age":54

        }
    }

    res = predict(event,None)
    print(res)



do_main()