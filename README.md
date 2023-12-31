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

### Example output

Below is an example of the response provided by the service:
```
{
    "model_details": "Pipeline(steps=[('kneighborsclassifier', KNeighborsClassifier(n_neighbors=9, p=1, weights='distance'))])",
    "score": {
        "accuracy": 0.8858695652173914,
        "confusion_matrix": [[66, 11], [10, 97]],
        "f1_score": 0.8825353397172824,
        "precision": 0.8832846003898636,
        "recall": 0.8818424566088117
    }
}
```

#### Interpretation of the Response:

- model_details: Provides the details of the trained machine learning pipeline. In this example, a K-Nearest Neighbors classifier was identified as the optimal model.
- score: A collection of various performance metrics evaluated on the test dataset:
1. accuracy: The proportion of correctly predicted instances.
2. confusion_matrix: A matrix showing the counts of true negative, false positive, false negative, and true positive predictions.
3. f1_score: The harmonic mean of precision and recall.
4. precision: The ratio of correctly predicted positive observations to the total predicted positives.
5. recall: The ratio of correctly predicted positive observations to all actual positives.
