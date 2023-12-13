# ML microservice with TPOT
### Overview

This microservice is designed to automate machine learning model training using TPOT, an automated machine learning tool. It accepts dataset details through a REST API, performs data preprocessing, trains a machine learning model using TPOT, and returns the model's performance metrics and details.

### Interacting with the Microservice
Endpoint: Train Model

Description: Trains a machine learning model using the provided dataset.

```
{
    "data_file_path": "path/to/dataset.csv",
    "features": ["column1", "column2", ...],
    "target": "target_column"
}
```
- 'data_file_path': The file path of the dataset (CSV format).
- 'features': List of column names to be used as features.
- 'target': The name of the target variable column.

#### Example Request

```
import requests

url = 'http://localhost:5000/train'
data = {
    "data_file_path": "/path/to/dataset.csv",
    "features": ["age", "blood_pressure"],
    "target": "heart_disease"
}

response = requests.post(url, json=data)
print(response.json())
```

### Internal Documentation

#### Code Structure

- 'app.py': The main Flask application file. Defines the API endpoints and integrates other modules.
- 'data_preprocessing.py': Contains functions for data preprocessing, including handling of numerical and categorical data.
- 'tpot_service.py': Encapsulates the TPOT training and evaluation logic.

### Extending the Service

#### Adding New Endpoints

To add a new endpoint:

- Define a new route in app.py.
- Implement the corresponding function to handle the request.

#### Modifying Preprocessing Steps

Preprocessing steps can be modified in data_preprocessing.py:

- Update or add new preprocessing steps in the create_preprocessor function.
- Ensure the changes are compatible with the TPOT training process.

#### Updating TPOT Configuration

To change the TPOT model training configuration:

- Modify the TPOTClassifier initialization parameters in tpot_service.py.
- Adjust any other relevant model training settings in this file.
