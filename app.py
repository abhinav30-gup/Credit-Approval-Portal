from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
import joblib

model = joblib.load("ml_credit.lb")



# 🔁 Mapping values
job_map = {
    "admin": 0, "technician": 1, "services": 2, "management": 3,
    "retired": 4, "blue-collar": 5, "unemployed": 6, "entrepreneur": 7,
    "housemaid": 8, "student": 9, "self-employed": 10
}
marital_map = {"married": 0, "single": 1, "divorced": 2}
education_map = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": 3}
housing_map = {"yes": 1, "no": 0}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/apply")
def apply():
    return render_template("apply.html")

@app.route("/status")
def status():
    return render_template("status.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get and map form values
        age = int(request.form['age'])
        job = job_map[request.form['job']]
        marital = marital_map[request.form['marital']]
        education = education_map[request.form['education']]
        balance = float(request.form['balance'])
        housing = housing_map[request.form['housing']]
        duration = float(request.form['duration'])

        # Predict
        features = np.array([[age, job, marital, education, balance, housing, duration]])
        prediction = model.predict(features)

        result = "✅ Credit Approved!" if prediction[0] == 1 else "❌ Credit Not Approved"
        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
