# ML microservice with TPOT
### Overview

This microservice is designed to automate machine learning model training using TPOT, an automated machine learning tool. It accepts dataset details through a REST API, performs data preprocessing, trains a machine learning model using TPOT, and returns the model's performance metrics and details.

### Interacting with the Microservice
Endpoint: Train Model

URL: /train
Method: POST
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
