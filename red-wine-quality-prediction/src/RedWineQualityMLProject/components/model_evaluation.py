import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from RedWineQualityMLProject.utils.common import save_json
from RedWineQualityMLProject import logger
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from RedWineQualityMLProject.config.configuration import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config = config
    def calc_metrics(self,y_test,y_test_pred):
        r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        return r2, mae, rmse
    def evaluate(self):
        with open(self.config.model_path,'rb') as f:
            model = pkl.load(f)
        with open(self.config.scaler_path,'rb') as f:
            scaler = pkl.load(f)
        test_data = pd.read_csv(self.config.test_data_path)
        x_test = test_data.drop(self.config.target_column, axis = 1)
        x_test = scaler.transform(x_test)
        y_test = test_data[self.config.target_column]
        y_test_pred = model.predict(x_test)
        r2, mae, rmse = self.calc_metrics(y_test,y_test_pred)
        logger.info('Model Evaluation Completed Successfully')
        logger.info(f'Metrics are r2 = {r2}, mae={mae}, rmse={rmse}')
        scores = {'r2':r2,'mae': mae, 'mse': rmse}
        save_json(path= Path(self.config.metrics_file_name), data=scores) 


        


