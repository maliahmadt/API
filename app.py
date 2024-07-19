from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    data = request.get_json()

    input_features = [
        {'name': 'gender', 'type': 'str'},
        {'name': 'age', 'type': 'int'},
        {'name': 'hypertension', 'type': 'bool'},
        {'name': 'heart_disease', 'type': 'bool'},
        {'name': 'smoking_history', 'type': 'str'},
        {'name': 'bmi', 'type': 'float'},
        {'name': 'HbA1c_level', 'type': 'float'},
        {'name': 'blood_glucose_level', 'type': 'float'}
    ]

    type_mapping = {
        'str': str,
        'int': int,
        'bool': bool,
        'float': float
    }

    # Validate the input data against the input_features schema
    for feature in input_features:
        if feature['name'] not in data:
            return jsonify({'error': f"Missing required feature: {feature['name']}"})
        if not isinstance(data[feature['name']], type_mapping[feature['type']]):
            return jsonify({'error': f"Invalid type for feature: {feature['name']}"})

    # Create a Pandas DataFrame from the input data
    input_df = pd.DataFrame([data])

    # One-hot encode the gender column
    input_df['gender_male'] = input_df['gender'].apply(lambda x: x == 'male')
    input_df['gender_other'] = input_df['gender'].apply(lambda x: x != 'male' and x != 'female')
    input_df.drop('gender', axis=1, inplace=True)

    # One-hot encode the smoking_history column
    smoking_categories = ['current', 'ever', 'former', 'never', 'not current']
    for category in smoking_categories:
        input_df[f'smoking_{category}'] = input_df['smoking_history'].apply(lambda x: x == category)
    input_df.drop('smoking_history', axis=1, inplace=True)

    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)

    print(input_scaled)

    model = joblib.load('diabetes_model.pkl')

    prediction = model.predict(input_scaled)

    return jsonify({'diabetes': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)