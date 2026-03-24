import os
import pickle as pkl 
import warnings as warn
warn.filterwarnings('ignore')
from src.CreditCardFraudDetectionMLProject.logger import logging
from src.CreditCardFraudDetectionMLProject.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def save_object(obj,file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as f:
            pkl.dump(obj,f)
    except CustomException as e:
        raise (e,sys)

def load_object(file_path):
    try: 
        with open(file_path, 'rb') as f:
            obj = pkl.load(f)
            return obj 
    except CustomException as e:
        raise (e,sys)

def plot_auc_roc_curve(y_test, y_pred, plot_path):
    try:
        plt.clf()
        plt.figure()
        fpr, tpr, _ = roc_curve(y_test,y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(plot_path)
        return 
    except CustomException as e:
        return (e,sys)

def plot_confusion_matrix(y_test, y_pred, plot_path):
    try:
        plt.clf()
        plt.figure()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(plot_path)
        return 
    except CustomException as e:
        raise (e,sys)

def evaluate_metrics(y_test, y_pred):
    try: 
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        return accuracy, precision, recall, f1
    except CustomException as e:
        raise (e,sys)


    