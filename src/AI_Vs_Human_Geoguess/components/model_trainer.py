import tensorflow as tf
import keras
import os
import pandas as pd
from AI_Vs_Human_Geoguess.config.configuration import PrepareTrainingConfig
import os
import zipfile
import gdown
from  pathlib import Path
from AI_Vs_Human_Geoguess import logger
from AI_Vs_Human_Geoguess.utils.common import get_size
class Training:
    def __init__(self,config:PrepareTrainingConfig):
        self.config=config
    def get_base_model(self):
        self.model=tf.keras.models.load_model(
            self.config.updated_base_model
        ) 
        self.data=pd.read_csv(os.path.join(self.config.training_data,'coords.csv'))
    def train_validation_generator(self):
        datagenerator_kwargs=dict(
            rescale=1/255,
            validation_split=0.2
        )
        dataflow_kwargs=dict(
            dataframe=pd.read_csv(os.path.join(self.config.training_data,'coords.csv')),
            directory=self.config.training_data,
            x_col='image',
             y_col=['normalized_lat', 'normalized_long'],  # Specify that this subset is for training
            target_size=(180, 180),
            batch_size=32,
            class_mode='raw'
            
        )
        valid_datagenerator=tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator=valid_datagenerator.flow_from_dataframe(
            
            subset='validation',
            **dataflow_kwargs  # Specify that this subset is for training
            

        )
        training_generator=valid_datagenerator
        self.train_generator=training_generator.flow_from_dataframe(
            subset='training',
            ** dataflow_kwargs
        )
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        self.model.summary()
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=tf.keras.callbacks.ModelCheckpoint(self.config.trained_model_path, save_best_only=True, monitor='val_loss', mode='min')
            




        )

       
    


