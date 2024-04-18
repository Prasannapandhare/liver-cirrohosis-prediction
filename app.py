from flask import Flask, render_template, request
import random
import joblib

app = Flask(__name__)
clf_rf = joblib.load('random_forest_model.pkl')  # Load your trained model

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Fetch input values from the form
            age = int(request.form['age'])
            gender = request.form['gender']
            tb = float(request.form['tb'])
            db = float(request.form['db'])
            alkphos = int(request.form['alkphos'])
            sgpt = int(request.form['sgpt'])
            sgot = int(request.form['sgot'])
            tp = float(request.form['tp'])
            alb = float(request.form['alb'])
            ag_ratio = float(request.form['ag_ratio'])

            # Convert Gender to binary (0 for Male, 1 for Female)
            gender_binary = 1 if gender == 'Male' else 0

            # Create feature vector
            user_features = [[age, gender_binary, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]]

            # Predict using the trained model
            prediction = clf_rf.predict(user_features)

            # Return prediction result
            result = "Affected" if prediction[0] == 1 else "Not affected"
            return render_template('result.html', result=result)
        
            # # Randomly choose "Affected" or "Not affected"
            # random_result = random.choice(["Affected", "Not affected"])
            # return render_template('result.html', result=random_result)

    except Exception as e:
        # In case of any exception, handle it and return an error message
        error_message = f"An error occurred during prediction: {str(e)}"
        return render_template('result.html', result=error_message)

if __name__ == '__main__':
    app.run(debug=True)
