from flask import Flask, render_template
import joblib
# import pickle
from flask import request
import numpy as np

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html', title='Health App')


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html", title='Diabetes')


def diabetes_collect(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 8:
        loaded_model = joblib.load(r'diabetes_model.pkl')          # pickle.load(open('diabetes_model.pkl', 'rb'))
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/diabetes/predict', methods=["POST"])
def diabetes_predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        # diabetes
        if len(to_predict_list) == 8:
            result = diabetes_collect(to_predict_list, 8)

    if int(result) == 1:
        prediction = "Sorry! There is a chance that you have Diabetes. Please consult the doctor immediately!!"
    else:
        prediction = "You have NO dangerous symptoms of the disease. Cheers!!"

    return render_template("result.html", prediction_text=prediction)


@app.route("/heart")
def heart():
    return render_template("heart.html", title='Heart Disease')


def heart_collect(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 11:
        loaded_model = joblib.load(r'heart_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/heart/predict', methods=["POST"])
def heart_predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        # diabetes
        if len(to_predict_list) == 11:
            result = heart_collect(to_predict_list, 11)

    if int(result) == 1:
        prediction = "Sorry! There is a chance that you have Heart disease. Please consult the doctor immediately!!"
    else:
        prediction = "You have NO dangerous symptoms of heart disease. Cheers!!"

    return render_template("result.html", prediction_text=prediction)


@app.route("/liver")
def liver():
    return render_template("liver.html", title='Liver Disease')


def liver_collect(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 9:
        loaded_model = joblib.load(r'liver_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/liver/predict', methods=["POST"])
def liver_predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        # diabetes
        if len(to_predict_list) == 9:
            result = liver_collect(to_predict_list, 9)

    if int(result) == 1:
        prediction = "Sorry! There is a chance that you have Liver disease. Please consult the doctor immediately!!"
    else:
        prediction = "You have NO dangerous symptoms of liver disease. Cheers!!"

    return render_template("result.html", prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=False)
