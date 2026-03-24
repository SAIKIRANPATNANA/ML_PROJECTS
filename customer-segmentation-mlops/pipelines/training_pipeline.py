from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline(ingest_data,transform_data,train_model,evaluate_model):
    data = ingest_data()
    x_train,x_test,y_train,y_test = transform_data(data)
    trained_model = train_model(x_train,y_train,x_test,y_test)
    mse,rmse,mae,r2_score = evaluate_model(trained_model,x_test,y_test)
