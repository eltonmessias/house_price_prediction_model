import logging
from abc import ABC, abstractmethod



import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelEvaluatorStrategy(ABC):
    @abstractmethod
    def evaluate_model(
            self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        pass

class RegressionModelEvaluationStrategy(ModelEvaluatorStrategy):
    def evaluate_model(
            self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:

        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Predicting evaluation metrics.")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"Mean Squared Error": mse, "R2 Score": r2}

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics

class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluatorStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluatorStrategy):

        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)