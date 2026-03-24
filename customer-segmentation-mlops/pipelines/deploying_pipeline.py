# import json
# import os
# import numpy as np 
# import pandas as pd 
# # from materializer.custom_materializer import cs_materializer
# from steps.data_ingestion import ingest_data
# from steps.data_transformation import transform_data 
# from steps.model_training import train_model
# from steps.model_evaluation import evaluate_model
# from zenml import pipeline,step 
# from zenml.config import DockerSettings
# from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
# from zenml.integrations.constants import MLFLOW,TENSORFLOW
# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
# from zenml.integrations.mlflow.services import MLFlowDeploymentService
# from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# from zenml.steps import BaseParameters, Output
# from utils import get_test_data
# docker_settings = DockerSettings(required_integrations=[MLFLOW])

# @step(enable_cache=False)
# def dynamic_importer()->str:
#     data = get_test_data()
#     return data

# class DeploymentTriggerConfig(BaseParameters):
#     min_accuracy: float = 0.0

# @step 
# def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig)->bool:
#     return accuracy >= config.min_accuracy

# class MLFlowDeploymentLoaderStepParameters(BaseParameters):
#     pipeline_name: str
#     pipeline_step_name: str
#     running: bool = True 

# @step(enable_cache=False)
# def prediction_service_loader(pipeline_name:str,pipeline_step_name:str,running: bool= True, model_name: str="model")->MLFlowDeploymentService:
#     model_deployer = MLFlowModelDeployer.get_active_model_deployer()
#     existing_services = model_deployer.find_model_server(pipeline_name=pipeline_name,pipeline_step_name=pipeline_step_name,model_name=model_name,running=running)
#     if not existing_services:
#         raise RuntimeError(
#             f"No MLflow prediction service deployed by the "
#             f"{pipeline_step_name} step in the {pipeline_name} "
#             f"pipeline for the '{model_name}' model is currently "
#             f"running.")
#     print(existing_services)
#     print(type(existing_services))
#     return existing_services

# @step
# def predict(service: MLFlowDeploymentService, data: np.ndarray)->np.ndarray:
#     service.start(timeout=10)
#     data = json.loads(data)
#     data.pop("columns")
#     data.pop("index")
#     columns_for_data = [
#         "payment_sequential",
#         "payment_installments",
#         "payment_value",
#         "price",
#         "freight_value",
#         "product_name_lenght",
#         "product_description_lenght",
#         "product_photos_qty",
#         "product_weight_g",
#         "product_length_cm",
#         "product_height_cm",
#         "product_width_cm",
#     ]
#     data = pd.DataFrame(data['data'], columns=columns_for_data)
#     json_list = json.loads(json.dumps(list(data.T.to_dict().values())))
#     data = np.array(json_list)
#     prediction = service.predict(data)
#     return prediction

# # @pipeline(enable_cache=True, settings={'docker':docker_settings}) 
# # def continuous_deploy_pipeline(min_accuracy: float=0.0,workers: int=1, timeout: int=DEFAULT_SERVICE_START_STOP_TIMEOUT):
# #     df = ingest_data()
# #     x_train,x_test,y_train,y_test = transform_data(df)
# #     trained_model = train_model(x_train,y_train,x_test,y_test)
# #     mse,rmse,mae,r2_score = evaluate_model(trained_model,x_test,y_test)
# #     deployment_decision = deployment_trigger(accuracy=mse)
# #     mlflow_model_deployer_step(model=trained_model,deploy_decision=deployment_decision,workers=workers,timeout=timeout)
# @pipeline(enable_cache=True, settings={'docker': docker_settings}) 
# def continuous_deploy_pipeline(min_accuracy: float = 0.0, workers: int = 1, timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
#     df = ingest_data()
#     x_train, x_test, y_train, y_test = transform_data(df)
#     trained_model = train_model(x_train, y_train, x_test, y_test)
#     mse, rmse, mae, r2_score = evaluate_model(trained_model, x_test, y_test)
#     deployment_decision = deployment_trigger(accuracy=mse)

#     # Log inputs before calling mlflow_model_deployer_step
#     logger.info(f"Inputs for mlflow_model_deployer_step:")
#     logger.info(f"trained_model: {trained_model}")
#     logger.info(f"deployment_decision: {deployment_decision}")
#     print(f"Inputs for mlflow_model_deployer_step:")
#     print(f"trained_model: {trained_model}")
#     print(f"deployment_decision: {deployment_decision}")
#     mlflow_model_deployer_step(model=trained_model, deploy_decision=deployment_decision, workers=workers, timeout=timeout)

# @pipeline(enable_cache=False, settings={'docker':docker_settings})
# def infer_pipeline(pipeline_name: str, pipeline_step_name: str):
#     batch_data = dynamic_importer()
#     model_deployment_service = prediction_service_loader(pipeline_name = pipeline_name, pipeline_step_name=pipeline_step_name, running=False)
#     predict(service=model_deployment_service,data=batch_data)




import json

# from .utils import get_data_for_test
import os

import numpy as np
import pandas as pd
# from materializer.custom_materializer import cs_materializer
from steps.data_ingestion import ingest_data
from steps.model_evaluation import evaluate_model
from steps.model_training import train_model
from steps.data_transformation import transform_data
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from utils import get_test_data

docker_settings = DockerSettings(required_integrations=[MLFLOW])
import pandas as pd

# import os


# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
#     MLFlowModelDeployer,
# )
# from zenml.integrations.mlflow.services import MLFlowDeploymentService
# from zenml.pipelines import pipeline
# from zenml.steps import BaseParameters, Output, step


requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data


class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float = 0.0


@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy > config.min_accuracy


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    pipeline_step_name: str
    running: bool = True


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]


@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Link all the steps artifacts together
    df = ingest_data()
    x_train, x_test, y_train, y_test = transform_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    mse,rmse,mae,r2_score = evaluate_model(model, x_test, y_test)
    deployment_decision = deployment_trigger(accuracy=mse)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_cle: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)
