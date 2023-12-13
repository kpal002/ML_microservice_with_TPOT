import pandas as pd
from flask import Flask, request, jsonify
from tpot_service import train_and_evaluate_model
from sklearn.model_selection import train_test_split

# Import the preprocessing module
from data_preprocessing import create_preprocessor

# Initialize the Flask application
app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Extract JSON data from the request
        request_data = request.get_json()

        # Retrieve the path to the dataset, feature column names, and target column name
        data_file_path = request_data.get('data_file_path')
        features = request_data.get('features')
        target = request_data.get('target')

        # Validate the presence of required data
        if not data_file_path or not features or not target:
            return jsonify({'error': 'Invalid data provided'}), 400

        # Read the dataset from the specified file path into a pandas DataFrame
        data = pd.read_csv(data_file_path)

        # Check for missing columns in the DataFrame
        missing_cols = [col for col in features + [target] if col not in data.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns in the DataFrame: {missing_cols}'}), 400

        # Create the preprocessor using the data and the specified target column
        preprocessor = create_preprocessor(data, target)

        # Separate the features and target from the dataset
        X = data[features]
        y = data[target]

        # Apply preprocessing to the feature data
        X_preprocessed = preprocessor.fit_transform(X)
        print(X_preprocessed.shape)  # Optional: print the shape of preprocessed data

        # Train and evaluate the model using the preprocessed data
        score, model_details = train_and_evaluate_model(X_preprocessed, y)

        # Return the model's performance score and details as a JSON response
        return jsonify({'score': score, 'model_details': model_details})

    except Exception as e:
        # Return an error message if an exception occurs
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
