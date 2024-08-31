import os
from box.exceptions import BoxValueError
import yaml
from AI_Vs_Human_Geoguess import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import random
import pandas as pd
import cv2 as cv
import numpy as np
from keras.models import load_model


paths='artifacts\data_ingestion\dataset'
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
def normalize_lat(lat):
    lat_norm = (lat + 90) / 180

    return lat_norm
def normalize_long(lon):
  lon_norm = (lon + 180) / 360
  return lon_norm
def denormalize_lat(lat_norm):
  lat = (lat_norm * 180) - 90

  return lat
def denormalize_lon(lon_norm):
  lon = (lon_norm * 360) - 180
  return lon
def generate_image(paths='artifacts\data_ingestion\dataset'):
    path=paths
    files=os.listdir(paths)
    images=[i for i in files if i.endswith('.png')  ]
    return random.choice(images)
def actual_coords(path):
    index=path.split('.')[0]
    df=pd.read_csv('artifacts\data_ingestion\dataset\coords.csv')
    data=df.iloc[int(index),:]
    lat,lon=data['latitude'],data['longitude']
    
    return lat,lon

def model_predictions(img):
    image=cv.imread(os.path.join('artifacts\data_ingestion\dataset',img))
    resized_img=cv.resize(image,(180,180))
    image=resized_img/255
    image=np.expand_dims(image,axis=0)
    print(image.shape)
    model=load_model('artifacts\\training\\model.keras')
    result=model.predict(image)
    return result[0][0],result[0][1]
def deg_to_rad(degrees):
   return degrees*(np.pi/180)
def dist(lat1,lon1,lat2,lon2):
   R=6371 
   d_lat=deg_to_rad(lat2-lat1)
   d_lon=deg_to_rad(lon2-lon1)
   a=np.sin(d_lat/2)**2+ np.cos(deg_to_rad(lat1))* np.cos(deg_to_rad(lat2))*np.sin(d_lon/2)**2
   c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
   return R*c
def random_trash_talk():
   trash_talk_statement=['Better luck next time kid'," I don't want a sparing partner do better next time","I almost feel bad for how easy that was... almost.",
    "Guess you missed the tutorial on how to win!",
    "That game was like taking candy from a baby—except the baby had more fight!",
    "Don’t worry, losing builds character… and it looks like you’re getting plenty of both." ]
   return random.choice(trash_talk_statement)