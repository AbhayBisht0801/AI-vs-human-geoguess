from AI_Vs_Human_Geoguess import logger
from AI_Vs_Human_Geoguess.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from AI_Vs_Human_Geoguess.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
STAGE_NAME='Data Ingestion Stage'
try:
    logger.info(f">>>>>>stage {STAGE_NAME} started<<<<<<")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>stage {STAGE_NAME} completed<<<<<<")
except Exception as e:
    logger.exception(e)
STAGE_NAME='PrepareBase Model Stage'
try:
    logger.info(f">>>>>>stage {STAGE_NAME} started<<<<<<")
    obj=PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>stage {STAGE_NAME} completed<<<<<<")
except Exception as e:
    logger.exception(e)