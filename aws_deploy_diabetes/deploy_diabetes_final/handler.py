import io
import json
import os
import boto3
from urllib.request import Request, urlopen

runtime=boto3.client("runtime.sagemaker")
SAGEMAKER_ENDPOINT_NAME: os.environ["SAGEMAKER_ENDPOINT_NAME"]

def detectDiabetes(event, context):

    body = {
        "message": "ok",
    }

    params = event["queryStringParameters"]

    Pregnancies = int(params['Pregnancies'])
    glucose = int(params['glucose'])
    BP = int(params['BP'])
    SkinThickness = int(params['SkinThickness'])
    Insulin = int(params['Insulin'])
    BMI = int(params['BMI'])
    DiabetesPedigreeFunction = int(params['DiabetesPedigreeFunction'])
    Age = int(params['Age'])

    input_data = [Pregnancies, glucose, BP, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    body = ",".join([str(item) for item in input_data])

    response = runtime.invoke_endpoint(EndpointName="diabetes-prediction",
                                       ContentType="text/csv",
                                       Body=body.encode("utf-8"))

    result = response["Body"].read().decode("utf-8")
## the .read() method is required; we apply that to the stream

    response = {
        "statusCode": 200,
        "body": json.dumps(round(float(result))),
        "headers": {
            "Access-Control-Allow-Origin": "*"
        }
    }

    return response
