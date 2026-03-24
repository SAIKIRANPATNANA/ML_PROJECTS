import pandas as pd
from sklearn.model_selection import train_test_split
from RedWineQualityMLProject import logger
import os
from RedWineQualityMLProject.config.configuration import DataTransformationConfig

class DataTransformation:
    def __init__(self,config=DataTransformationConfig):
        self.config = config
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)
        train,test = train_test_split(data, test_size=0.25, random_state=42)
        # std_scaler = StandardScaler()
        # train.iloc[:,:-1] = std_scaler.fit_transform(train.iloc[:,:-1])
        train.to_csv(os.path.join(self.config.root_dir[0], 'train.csv'), index = False)
        test.to_csv(os.path.join(self.config.root_dir[0], 'test.csv'), index = False)
        logger.info('splitted train and test data')
        logger.info(train.shape)
        logger.info(test.shape)
        print(train.shape)
        print(test.shape)
    