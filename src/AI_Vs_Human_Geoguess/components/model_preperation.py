from AI_Vs_Human_Geoguess.config.configuration import PrepareBaseModelConfig
import os
import zipfile
import gdown
from AI_Vs_Human_Geoguess import logger
from AI_Vs_Human_Geoguess.utils.common import get_size
from  pathlib import Path

import pandas as pd
import tensorflow as tf
class PrepareBaseModel:
    def __init__(self,config=PrepareBaseModelConfig):
        self.config=config
    def get_base_model(self):
        self.model=tf.keras.applications.vgg16.VGG16(
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            input_shape=self.config.params_image_size
        )
        self.save_model(path=self.config.base_model_path,model=self.model)
    @staticmethod
    def save_model(path:Path,model:tf.keras.Model):
        model.save(path)
    @staticmethod
    def  _prepare_full_model(model,output_nueron,freeze_all,freeze_till):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till>0):
            for layer in model.layers[:freeze_all]:
                model.trainable=False
        flatten_in=tf.keras.layers.Flatten()(model.output)
        Dense1 = tf.keras.layers.Dense(512, activation='relu')(flatten_in)
        normalize1 = tf.keras.layers.BatchNormalization()(Dense1)
        dropout1 = tf.keras.layers.Dropout(0.5)(normalize1)
        Dense2 = tf.keras.layers.Dense(256, activation='relu')(dropout1)
        normalize2 = tf.keras.layers.BatchNormalization()(Dense2)
        dropout2 = tf.keras.layers.Dropout(0.4)(normalize2)
        Dense3 = tf.keras.layers.Dense(128, activation='relu')(dropout2)
        normalize3 = tf.keras.layers.BatchNormalization()(Dense3)
        dropout3 = tf.keras.layers.Dropout(0.4)(normalize3)
        Dense4 = tf.keras.layers.Dense(64, activation='relu')(dropout3)
        normalize4 = tf.keras.layers.BatchNormalization()(Dense4)
        dropout4 = tf.keras.layers.Dropout(0.3)(normalize4)
        Dense5 = tf.keras.layers.Dense(32,activation='relu')(dropout4)
        normalize5 = tf.keras.layers.BatchNormalization()(Dense5)
        dropout5 = tf.keras.layers.Dropout(0.3)(normalize5)
        Dense6 = tf.keras.layers.Dense(16, activation='relu')(dropout5)
        normalize6 = tf.keras.layers.BatchNormalization()(Dense6)
        dropout6 = tf.keras.layers.Dropout(0.2)(normalize6)
        outputs = tf.keras.layers.Dense(units=output_nueron, activation='sigmoid')(dropout6)
        full_model=tf.keras.models.Model(
            inputs=model.input,
            outputs=outputs
        )
        full_model.compile(optimizer='adam',loss='mse')
        full_model.summary()
        return full_model
    def update_base_model(self):
        self.full_model=self._prepare_full_model(
            model=self.model,
            output_nueron=self.config.params_output_nueron,
            freeze_all=True,
            freeze_till=None
            
        )
        self.save_model(path=self.config.updated_base_model_path,model=self.full_model)

