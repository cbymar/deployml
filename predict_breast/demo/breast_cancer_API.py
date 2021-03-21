from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

loaded_model = pickle.load(open("XGBoost_norm", "rb"))

cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst']

"""Home route for app"""
@app.route("/")
def getModel():
    """Home route simply shows a web form for accepting user input"""
    # return str(loaded_model)   # stringified version; shows params.
    return render_template("form.html")


@app.route('/predict', methods=["POST"])
def predict():
    """
    Get data and do the same processing as when we prototyped,
    because we need to normalize based on training data summary stats
    :return:

    """
    data = pd.read_csv('data.csv')
    df = data.drop("Unnamed: 32", axis=1)
    df = data.drop("id", axis=1)

    df.drop(columns=["Unnamed: 32"], inplace=True)
    X = df.drop(labels="diagnosis", axis=1)

    input_data = []

    for col in cols:
        input_data.append(float(request.form[col]))

    df_norm = (input_data - X.mean()) / (X.max() - X.min())

    pred = loaded_model.predict(df_norm)

    if pred == 1:
        return "Prediction : Benign Tumor Found"
    else:
        return "Prediction : Malignant Tumor Found"


if __name__ == '__main__':
    app.run(host='0.0.0.0')
