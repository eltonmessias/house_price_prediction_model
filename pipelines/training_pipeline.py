from pyexpat import features
from zenml import Model, pipeline, step

from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_spitter_step
from steps.feature_eingineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step


@pipeline(
    model=Model(
        name="house_price_predictor",
    ),
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    raw_data = data_ingestion_step(
        file_path="G:/Projects/Data-Analysis/house_price_predictor/data/raw/archive.zip"
    )

    filled_data = handle_missing_values_step(raw_data)


    engineered_data = feature_engineering_step(
        filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )

    clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")

    X_train, X_test, y_train, y_test = data_spitter_step(clean_data, target_column="SalePrice")


    model = model_building_step(X_train=X_train, y_train=y_train)

    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model



if __name__ == "__main__":
    ml_pipeline()