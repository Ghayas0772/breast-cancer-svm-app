from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get input values and convert to float
            features = [
                float(request.form['Clump_Thickness']),
                float(request.form['Uniformity_Cell_Size']),
                float(request.form['Uniformity_Cell_Shape']),
                float(request.form['Marginal_Adhesion']),
                float(request.form['Single_Epithelial_Cell_Size']),
                float(request.form['Bare_Nuclei']),
                float(request.form['Bland_Chromatin']),
                float(request.form['Normal_Nucleoli'])
            ]

            # Scale features
            features_scaled = scaler.transform([features])

            # Make prediction
            result = model.predict(features_scaled)[0]

            prediction = 'Malignant' if result == 4 else 'Benign'  # adjust according to your labels

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)



print("Template folder:", app.template_folder)