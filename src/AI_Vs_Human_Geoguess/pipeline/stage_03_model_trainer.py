from AI_Vs_Human_Geoguess.components.model_trainer import Training
from AI_Vs_Human_Geoguess.config.configuration import ConfigurationManager
from AI_Vs_Human_Geoguess import logger
STAGE_NAME='Model traning'
class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        training_config=config.get_training_config()
        training=Training(config=training_config)
        training.get_base_model()
        training.train_validation_generator()
        training.train()
    
if __name__=='__main__':
    try:
        logger.info(f">>>>>>stage {STAGE_NAME} started<<<<<<")
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>stage {STAGE_NAME} completed<<<<<<")
    except Exception as e:
        logger.exception(e)