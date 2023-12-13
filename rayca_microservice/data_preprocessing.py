from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_preprocessor(data, target_col):
    """
    Creates a preprocessor pipeline for a given dataset.

    Parameters:
    - data (pd.DataFrame): The dataset for which the preprocessing pipeline is to be created.
    - target_col (str): The name of the target column to be excluded from preprocessing.

    Returns:
    - ColumnTransformer: A preprocessor object that transforms the data by applying specified
      transformations to numerical and categorical columns.
    """

    # Exclude the target column from the feature set to ensure it is not transformed.
    feature_data = data.drop(columns=[target_col], errors='ignore')

    # Identify numerical columns as those with int64 or float64 data types.
    numerical_cols = feature_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Identify categorical columns as those with object or boolean data types.
    categorical_cols = feature_data.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Define a pipeline for numerical features:
    # 'imputer' fills missing values with the median value of the column.
    # 'scaler' standardizes features by removing the mean and scaling to unit variance.
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define a pipeline for categorical features:
    # 'imputer' fills missing values with the most frequent value of the column.
    # 'onehot' applies one-hot encoding to convert categorical variables into a form that
    # could be provided to ML algorithms.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle the preprocessing steps into a single ColumnTransformer.
    # This ensures that the appropriate transformations are applied to each column type.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return preprocessor
