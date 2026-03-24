import pandas as pd
import pickle as pkl
import numpy as np
from pathlib import Path

class PredictionPineline:
    def __init__(self):
        with open(Path('artifacts/model_trainer/model.pkl'), 'rb') as f:
            self.model = pkl.load(f)
        with open(Path('artifacts/model_trainer/scaler.pkl'), 'rb') as f:
            self.scaler = pkl.load(f)
    def predict(self,data):
        data = self.scaler.transform(data)
        pred = self.model.predict(data)
        return pred