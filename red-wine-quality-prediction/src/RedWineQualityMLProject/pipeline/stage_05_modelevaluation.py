from RedWineQualityMLProject.config.configuration import ConfigurationManager
from RedWineQualityMLProject.components.model_evaluation import ModelEvaluation
from RedWineQualityMLProject import logger
STAGE_NAME = "Model Evaluation Stage"
class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_modelEvaluation_config()
            model_evalution = ModelEvaluation(config=model_evaluation_config)
            model_evalution.evaluate()
        except Exception as e:
            raise e

        
if __name__ == '__main__':
    try:
        logger.info(f">>>>stage {STAGE_NAME} started <<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>stage {STAGE_NAME} completed <<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        raise e