import os
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#def splitData(data):
    

def main():
    referenceFile = "../Datasets/UCI_Credit_Card.csv"
    baseFile = "./baseline.csv"
    
    #Get reference and synthetic data
    referenceData = np.loadtxt(open(referenceFile, "rb"), delimiter = ",", skiprows = 1)
    baseData = np.loadtxt(open(baseFile, "rb"), delimiter = ",", skiprows = 1)
    
    refX = referenceData[:,:-1]
    refY = referenceData[:,-1]
    baseX = baseData[:,:-1]
    baseY = baseData[:,-1]
    
    #Split data
    refTrainX, refTestX, refTrainY, refTestY = train_test_split(refX, refY, test_size=0.2)
    baseTrainX, baseTestX, baseTrainY, baseTestY = train_test_split(baseX, baseY, test_size=0.2)
    
    #Train and test model
    model = LogisticRegression(max_iter=1000, verbose=0, class_weight='balanced').fit(refTrainX, refTrainY)
    yRefPreds = model.predict(refTestX)
    yBasePreds = model.predict(baseTestX)
    
    print("Reference Data Test Scores:  ")
    print("F-Beta Score: " + str(fbeta_score(refTestY, yRefPreds, average='weighted', beta=0.5)))
    print("Accuracy: " + str(accuracy_score(refTestY, yRefPreds)))
    print("Precision: " + str(precision_score(refTestY, yRefPreds, average='weighted', zero_division=1)))
    print("Recall: " + str(recall_score(refTestY, yRefPreds, average='weighted', zero_division=1)))
    print("ROC/AUC: " + str(roc_auc_score(refTestY, yRefPreds)))
    print()
    
    print("Baseline Data Metrics: ")
    print("F-Beta Score: " + str(fbeta_score(baseTestY, yBasePreds, average='weighted', beta=0.5)))
    print("Accuracy: " + str(accuracy_score(baseTestY, yBasePreds)))
    print("Precision: " + str(precision_score(baseTestY, yBasePreds, average='weighted', zero_division=1)))
    print("Recall: " + str(recall_score(baseTestY, yBasePreds, average='weighted', zero_division=1)))
    print("ROC/AUC: " + str(roc_auc_score(baseTestY, yBasePreds)))
    
if __name__ == "__main__":
    main()