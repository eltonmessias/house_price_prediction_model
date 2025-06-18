from zenml import Model, pipeline, step



@pipeline(
    model=Model(
        name="house_price_predictor",
    ),
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    f