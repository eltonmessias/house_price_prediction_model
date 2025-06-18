import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step
from zenml.client import Client



experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model


model = Model(
    name="house_price_predictor",
    version=None,
    licence="Apache-2.0",
    description="Price prediction model for houses.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
        X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:

    """
    Builds and trains a Linear Regression model using scikit-learn wrapped in a pipeline

    return:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the Linear Regression model.
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas dataframe")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas series")

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Define preprocessing for categorical and numerical features
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Define the model training pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

    #Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run() # Start a new MLflow run if there isn't one active

    try:
        # Enable autologging for scikit-learn to automatically capture model metrics, parameters, and artifacts
        mlflow.sklearn.autolog()

        logging.info("Building and training the linear Regression model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        #Log the columns that the model expects
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformer_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(
            X_train[categorical_cols]
        )
        expected_columns = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )
        logging.info(f"Model expects the following columns: {expected_columns}.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        #End the MLflow
        mlflow.end_run()

    return pipeline