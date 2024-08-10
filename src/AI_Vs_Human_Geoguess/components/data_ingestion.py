import os
import zipfile
import gdown
from AI_Vs_Human_Geoguess import logger
from AI_Vs_Human_Geoguess.utils.common import get_size,normalize_lat,normalize_long
import pandas as pd
from AI_Vs_Human_Geoguess.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config
    def download_file(self)->str:
        try:
            dataset_url=self.config.source_URL
            zip_download_dir=self.config.local_data_file
            os.makedirs('artifacts/data_ingestion',exist_ok=True)
            logger.info(f'Downloading data from {dataset_url} into file {zip_download_dir}')
            file_id=dataset_url.split('/')[-2]
            prefix='http://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)
            logger.info(f'Download data from {dataset_url} into file {zip_download_dir}')
        except Exception as e:
            raise e
    def extract_zip_file(self):
        unzip_path=self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)
    def preprocess_data(self):
        data=pd.read_csv(os.path.join(self.config.unzip_dir,'dataset\coords.csv'),header=None)
        data.rename(columns={0:'latitude',1:'longitude'},inplace=True)
        image=[]
        for i in (os.listdir(os.path.join(self.config.unzip_dir,'dataset'))):
            if i.endswith('.png'):
                image.append(i)
        
        data['image']=np.array(sorted(image,key=lambda x:int(x.split('.')[0])))
        data.rename(columns={0:'latitude',1:'longitude'},inplace=True)
        data['normalized_lat']=data['latitude'].apply(normalize_lat)
        data['normalized_long']=data['longitude'].apply(normalize_long)
        data.to_csv(os.path.join(self.config.unzip_dir,'dataset\coords.csv'))
        
   
