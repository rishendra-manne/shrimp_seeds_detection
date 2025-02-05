from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import cv2
import shutil
import tensorflow as tf
import requests
import zipfile
import json
import os
import sys

@dataclass
class IngestionConfig:
    train_data_path=os.path.join('artifacts','dataset','train_data')
    test_data_path=os.path.join('artifacts','dataset','test_data')
    valid_data_path=os.path.join('artifacts','dataset','valid_data')
    output_dir=os.path.join(r'C:\Users\mrish\PycharmProjects\PrawnSeed\src\artifacts','dataset')
    dataset_url = "https://universe.roboflow.com/ds/OoffgDRtBF?key=nHXCifHwcG"
    zip_path = "roboflow.zip"
    annotation_file=output_dir+'annotations.coco.json'

class DataIngestion:

    def __init__(self,train_split):
        self.ingestion_config=IngestionConfig()
        self.train_split=train_split

    def get_data(self,url):
        """

        :param url: url of the data
        :return: fetches the data from the url
        """
        try:
            logging.info("data fetching started")
            response=requests.get(url,stream=True)
            if response.status_code==200:
                with open (self.ingestion_config.zip_path,'wb') as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
            else:
                logging.info(f"failed to fetch the data due to {response.status_code}")
                raise Exception(f"failed to download dataset {response.status_code}")
        except Exception as e:
            raise CustomException(e,sys)

    def extract_data(self):
        """

        :return: extracts the zipped data fetched from the url to artifacts folder
        """
        logging.info("data extraction has stared")
        try:
            with zipfile.ZipFile(self.ingestion_config.zip_path, "r") as zip_ref:
                zip_ref.extractall(self.ingestion_config.output_dir)
            print(f"✅ Dataset extracted to: {self.ingestion_config.output_dir}")

            # Step 3: Clean up (Remove the zip file)
            os.remove(self.ingestion_config.zip_path)
        except Exception as e:
            logging.info("logging has failed")
            raise CustomException(e,sys)

dataset_url = "https://universe.roboflow.com/ds/OoffgDRtBF?key=nHXCifHwcG"
dataing=DataIngestion(0.8)
dataing.get_data(dataset_url)





