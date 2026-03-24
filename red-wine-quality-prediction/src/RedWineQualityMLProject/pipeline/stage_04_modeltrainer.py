from RedWineQualityMLProject.config.configuration import ConfigurationManager
from RedWineQualityMLProject.components.model_trainer import ModelTrainer
from RedWineQualityMLProject import logger
from pathlib import Path
STAGE_NAME = "Model Training Stage"
class ModelTrainerPipeline:
    def __init__(self):
        pass
    def main(self):
        try: 
            config = ConfigurationManager()
            model_trainer_config = config.get_modelTraniner_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
        except Exception as e:
            raise e
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>stage {STAGE_NAME} started <<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>stage {STAGE_NAME} completed <<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        raise e