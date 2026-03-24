from RedWineQualityMLProject.config.configuration import ConfigurationManager
from RedWineQualityMLProject.components.data_ingestion import DataIngestion
from RedWineQualityMLProject import logger

STAGE_NAME = "Data Ingestion Stage"
class DataIngestionPipeline:
    def __init__(self):
        pass
    def main(self):
        try: 
            config = ConfigurationManager()
            dataIngestion_config = config.get_dataIngestion_config()
            dataIngestion = DataIngestion(config=dataIngestion_config)
            dataIngestion.download_file()
            dataIngestion.extract_zip_file()
        except Exception as e:
            raise e
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>stage {STAGE_NAME} started <<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>stage {STAGE_NAME} completed <<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        raise e
