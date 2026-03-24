from RedWineQualityMLProject.config.configuration import ConfigurationManager
from RedWineQualityMLProject.components.data_validation import DataValidation
from RedWineQualityMLProject import logger
STAGE_NAME = "Data Validation Stage"
class DataValidationPipeline:
    def __init__(self):
        pass
    def main(self):
        try: 
            config = ConfigurationManager()
            dataValidation_config = config.get_dataValidation_config()
            dataValidation = DataValidation(config=dataValidation_config)
            dataValidation.validate_all_cols()
        except Exception as e:
            raise e
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>stage {STAGE_NAME} started <<<<")
        obj = DataValidationPipeline()
        obj.main()
        logger.info(f">>>>stage {STAGE_NAME} completed <<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        raise e
