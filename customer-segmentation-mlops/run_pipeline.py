from pipelines.training_pipeline import train_pipeline
from steps.data_ingestion import ingest_data
from steps.data_transformation import transform_data
from steps.model_training import train_model
from steps.model_evaluation import evaluate_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == '__main__':
    train = train_pipeline(ingest_data, transform_data, train_model, evaluate_model)
    train.run()
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

