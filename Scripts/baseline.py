import random
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

#Discrete values
def disc_dist(x, numVals):
    classes = np.unique(x)
    
    n = []
    #Get probability of each value in x
    for c in classes:
        n.append(np.count_nonzero(x == c) / len(x))
    
    #Randomly choose discrete value based on given probabilities
    synth_x = np.random.choice(classes, numVals, p = n)
    return synth_x

#Continuous values
def cont_dist(x, isInt, numVals):
    #use matplotlib histogram to discretize values into bins and frequencies
    n, bins, patches = plt.hist(x)
    
    #convert count of each bin into probabilities
    for i in range(0, len(n)):
        n[i] = n[i] / len(x)
    
    #Creates list of indices of bins list to choose from
    choices = list(range(1, len(bins)))
    
    synth_x = []
    
    #Generate values by choosing random bin and then generating random value within that bin
    for i in range(0, numVals):
        c = np.random.choice(choices, 1, p = n)
        if isInt:
            synth_x.append(round(random.uniform(bins[c - 1], bins[c])[0]))
        else:
            synth_x.append(random.uniform(bins[c - 1], bins[c])[0])
    
    return synth_x

def main():
    folder = "../Datasets/kaggle_banking_dataset"
    fName = folder + "/train.csv"
    
    #Get data and headers from data file
    data = np.loadtxt(open(fName, "rb"), delimiter = ",", skiprows = 1)
    with open(fName) as f:
        reader = csv.DictReader(f)
        csvDict = dict(list(reader)[0])
        headers = ','.join(list(csvDict.keys()))
    
    #List of types of each feature in dataset (0 = discrete, 1 = continuous)
    #CC_default_types = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0] #Any way to automate this?
    
    banking_target_types = [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0]
    banking_target_isInt = [1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1]
    numVals = len(data)
    
    synth_data = []
    for i in range(0, len(banking_target_types)):
        if banking_target_types[i] == 0:
            synth_data.append(disc_dist(data[:,i], numVals))
        else:
            if banking_target_isInt[i] == 1:
                synth_data.append(cont_dist(data[:,i], True, numVals))
            else:
                synth_data.append(cont_dist(data[:,i], False, numVals))
    
    #Transpose synthetic data so that dimensions are correct
    synth_data = np.transpose(synth_data)
    
    np.savetxt((folder + "/baseline.csv"), synth_data, delimiter=",", header=headers)

if __name__ == "__main__":
    main()