from RedWineQualityMLProject.config.configuration import ConfigurationManager
from RedWineQualityMLProject.components.data_transformation import DataTransformation
from RedWineQualityMLProject import logger
from pathlib import Path
STAGE_NAME = "Data Transformation Stage"
class DataTransformationPipeline:
    def __init__(self):
        pass
    def main(self):
        try: 
            with open(Path('artifacts/data_validation/status.txt'), 'r') as f:
                status = f.read().split()[-1]
            if(status=='True'):
                config = ConfigurationManager()
                dataTransformation_config = config.get_dataTransformation_config()
                DataTransformtion = DataTransformation(config=dataTransformation_config)
                DataTransformtion.train_test_splitting()
            else:
                raise Exception('Data Schema in invalid')
        except Exception as e:
            raise e
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>stage {STAGE_NAME} started <<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>stage {STAGE_NAME} completed <<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        raise e