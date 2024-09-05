from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and polynomial features
model = joblib.load('polynomial_regression_model.pkl')
poly = joblib.load('polynomial_features.pkl')

# Function to filter predictions
def filter_data(data):
    data[data < 0] = 0
    return data.astype(int)

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()

        # Convert input data to a DataFrame
        new_data = pd.DataFrame({
            'Time': [data['Time']],
            'sta': [data['sta']],
            'wether': [data['wether']],
            'week or weekend': [data['week_or_weekend']]
        })

        # Transform input data with polynomial features
        new_data_poly = poly.transform(new_data)

        # Get the prediction
        predictions = model.predict(new_data_poly)

        # Apply filter to the prediction
        filtered_prediction = filter_data(predictions)[0]

        # Return the prediction as a JSON response
        return jsonify({'prediction': int(filtered_prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
