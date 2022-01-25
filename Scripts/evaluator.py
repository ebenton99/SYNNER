import os
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

def evaluation(refTrainFile, refTestFile, baseFile):
    #Get reference and synthetic data
    refTrainData = np.loadtxt(open(refTrainFile, "rb"), delimiter = ",", skiprows = 1)
    refTestData = np.loadtxt(open(refTestFile, "rb"), delimiter = ",", skiprows = 1)
    baseData = np.loadtxt(open(baseFile, "rb"), delimiter = ",", skiprows = 1)
    
    refTrainX = refTrainData[:,:-1]
    refTrainY = refTrainData[:,-1]
    refTestX = refTestData[:,:-1]
    refTestY = refTestData[:,-1]
    baseTrainX = baseData[:,:-1]
    baseTrainY = baseData[:,-1]
    
    #Logistic Regression Models
    refLogModel = LogisticRegression(max_iter=1000, verbose=0).fit(refTrainX, refTrainY)
    baseLogModel = LogisticRegression(max_iter=1000, verbose=0).fit(baseTrainX, baseTrainY)
    yRefLogPreds = refLogModel.predict(refTestX)
    yBaseLogPreds = baseLogModel.predict(refTestX)
    
    print("---Logistic Regression Models---")
    print("Reference Model Metrics:  ")
    print("F-Beta Score: " + str(fbeta_score(refTestY, yRefLogPreds, average='macro', beta=0.5)))
    print("Accuracy: " + str(accuracy_score(refTestY, yRefLogPreds)))
    print("Precision: " + str(precision_score(refTestY, yRefLogPreds, average='macro', zero_division=0)))
    print("Recall: " + str(recall_score(refTestY, yRefLogPreds, average='macro', zero_division=0)))
    p, r, t = precision_recall_curve(refTestY, yRefLogPreds)
    print("AUPRC: " + str(auc(r, p)))
    print()
    
    print("Baseline Model Metrics: ")
    print("F-Beta Score: " + str(fbeta_score(refTestY, yBaseLogPreds, average='macro', beta=0.5)))
    print("Accuracy: " + str(accuracy_score(refTestY, yBaseLogPreds)))
    print("Precision: " + str(precision_score(refTestY, yBaseLogPreds, average='macro', zero_division=0)))
    print("Recall: " + str(recall_score(refTestY, yBaseLogPreds, average='macro', zero_division=0)))
    p, r, t = precision_recall_curve(refTestY, yBaseLogPreds)
    print("AUPRC: " + str(auc(r, p)))
    print()
    
    #Random Forest Models
    refRFModel = RandomForestClassifier(n_estimators=200, verbose=0).fit(refTrainX, refTrainY)
    baseRFModel = RandomForestClassifier(n_estimators=200, verbose=0).fit(baseTrainX, baseTrainY)
    yRefRFPreds = refRFModel.predict(refTestX)
    yBaseRFPreds = baseRFModel.predict(refTestX)
    
    print("---Random Forest Models---")
    print("Reference Model Metrics:  ")
    print("F-Beta Score: " + str(fbeta_score(refTestY, yRefRFPreds, average='macro', beta=0.5)))
    print("Accuracy: " + str(accuracy_score(refTestY, yRefRFPreds)))
    print("Precision: " + str(precision_score(refTestY, yRefRFPreds, average='macro', zero_division=0)))
    print("Recall: " + str(recall_score(refTestY, yRefRFPreds, average='macro', zero_division=0)))
    p, r, t = precision_recall_curve(refTestY, yRefRFPreds)
    print("AUPRC: " + str(auc(r, p)))
    print()
    
    print("Baseline Model Metrics: ")
    print("F-Beta Score: " + str(fbeta_score(refTestY, yBaseRFPreds, average='macro', beta=0.5)))
    print("Accuracy: " + str(accuracy_score(refTestY, yBaseRFPreds)))
    print("Precision: " + str(precision_score(refTestY, yBaseRFPreds, average='macro', zero_division=0)))
    print("Recall: " + str(recall_score(refTestY, yBaseRFPreds, average='macro', zero_division=0)))
    p, r, t = precision_recall_curve(refTestY, yBaseRFPreds)
    print("AUPRC: " + str(auc(r, p)))

def main():
    datasets = ["breast_cancer_wisconsin", "census_income_data", "credit_card", "dataset_diabetes", "kaggle_banking_dataset"]
    refTrainFile = "../Datasets/" + datasets[2] + "/train.csv"
    refTestFile = "../Datasets/" + datasets[2] + "/test.csv"
    baseFile = "../Datasets/" + datasets[2] + "/baseline.csv"
    
    evaluation(refTrainFile, refTestFile, baseFile)
    
if __name__ == "__main__":
    main()