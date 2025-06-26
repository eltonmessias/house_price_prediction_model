import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy
from zenml import step



@step
def model_evaluator_step(
        trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
)-> tuple[dict, float]:

    """
    :param trained_model:  The trained pipeline containing the model and preprocessing steps
    :param X_test:  The test data features
    :param y_test: The test data labels/target

    Return:
    dict: A dictionary containing evaluation metrics.
    """

    # Ensure the inputs are the correct type
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas dataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas series.")

    logging.info("Applying the same preprocessing to the test data.")

    # Apply the preprocessing and model prediction
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    #Initialize the evaluator with the regression strategy
    evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())

    # Perform the evaluation
    evaluation_metrics = evaluator.evaluate(
        trained_model.named_steps["model"], X_test_processed, y_test
    )

    # Ensure that the evaluation metrics are returned as a dictionary
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    mse = evaluation_metrics.get("Mean Squared Error", None)
    return evaluation_metrics, mse
