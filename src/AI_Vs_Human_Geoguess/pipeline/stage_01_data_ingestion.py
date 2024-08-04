from AI_Vs_Human_Geoguess.config.configuration import ConfigurationManager
from AI_Vs_Human_Geoguess.components.data_ingestion import DataIngestion
from AI_Vs_Human_Geoguess import logger

STAGE_NAME='Data Ingestion Stage'

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        dataingestion=DataIngestion(config=data_ingestion_config)
        dataingestion.download_file()
        dataingestion.extract_zip_file()

if __name__=='__main__':
    try:
        logger.info(f">>>>>>stage {STAGE_NAME} started<<<<<<")
        obj=DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>stage {STAGE_NAME} completed<<<<<<")
    except Exception as e:
        logger.exception(e)