import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

original_dim = 17
intermediate_dim = 12
latent_dim = 1

inputs = keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
#initializer = keras.initializers.Zeros()
z_log_sigma = layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon


def main():
    folder = "../Datasets/kaggle_banking_dataset"
    fName = folder + "/train.csv"
    
    #Get data and headers from data file
    data = np.loadtxt(open(fName, "rb"), delimiter = ",", skiprows = 1)
    with open(fName) as f:
        reader = csv.DictReader(f)
        csvDict = dict(list(reader)[0])
        headers = ','.join(list(csvDict.keys()))
    
    #y = data[:,-1]
    #print(y)
    #data = data[:,:-1]
    dataMean = np.mean(data, axis=0)
    dataStd = np.std(data, axis=0)
    data = (data - dataMean) / dataStd
    #data = data.tolist()
    
    #for i in range(0, len(data)):
    #    data[i].append(y[i])
    
    z = layers.Lambda(sampling)([z_mean, z_log_sigma])
    
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='softsign')(latent_inputs)
    x = layers.Dense(6, activation='softsign')(x)
    x = layers.Dense(intermediate_dim, activation='softsign')(x)
    outputs = layers.Dense(original_dim)(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name='vae')
    
    opt = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, clipnorm=1.0)
    vae.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt)
    
    #xTrain, xTest = train_test_split(data, test_size=0.2)
    xTrain, xVal = train_test_split(data, test_size=0.2)
    
    vae.fit(xTrain, xTrain,
        epochs=70,
        batch_size=16,
        validation_data=(xVal, xVal))
    
    preds = vae.predict(data)
    preds = (preds * dataStd) + dataMean
    np.savetxt((folder + "/vae.csv"), preds, delimiter=",", header=headers)
    
if __name__ == "__main__":
    main()