from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_and_evaluate_model(X, y):
    """
    Train and evaluate a machine learning model using TPOT.

    Parameters:
    - X (pd.DataFrame): The preprocessed feature data.
    - y (pd.Series): The target variable.

    Returns:
    - dict: A dictionary containing various evaluation scores of the model (accuracy, precision, recall, f1_score, confusion_matrix).
    - str: A string representation of the trained TPOT pipeline.
    """

    # Split the dataset into training and testing sets with a test size of 20%.
    # The random_state ensures reproducibility of the results.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the TPOTClassifier with specified configurations.
    # 'generations' and 'population_size' control the genetic programming process,
    # while 'verbosity=2' enables progress logging to the console.
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)

    # Fit the TPOT classifier on the training data.
    tpot.fit(X_train, y_train)

    # Make predictions on the test set.
    predictions = tpot.predict(X_test)

    # Calculate and compile various evaluation metrics to assess the performance of the model.
    scores = {
        'accuracy': accuracy_score(y_test, predictions),  # Accuracy score
        'precision': precision_score(y_test, predictions, average='macro'),  # Precision score (macro-averaged)
        'recall': recall_score(y_test, predictions, average='macro'),  # Recall score (macro-averaged)
        'f1_score': f1_score(y_test, predictions, average='macro'),  # F1 score (macro-averaged)
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist()  # Confusion matrix converted to list
    }

    # Retrieve the trained pipeline as a string for inspection or further analysis.
    model_details = str(tpot.fitted_pipeline_)

    return scores, model_details
