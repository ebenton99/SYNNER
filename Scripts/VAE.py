import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

original_dim = 24
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
    fName = "../Datasets/UCI_Credit_Card.csv"
    #Get data and headers from data file
    data = np.loadtxt(open(fName, "rb"), delimiter = ",", skiprows = 1)
    with open(fName) as f:
        reader = csv.DictReader(f)
        csvDict = dict(list(reader)[0])
        headers = ','.join(list(csvDict.keys()))
    
    y = data[:,-1]
    #print(y)
    data = data[:,:-1]
    dataMean = np.mean(data, axis=0)
    dataStd = np.std(data, axis=0)
    data = (data - dataMean) / dataStd
    data = data.tolist()
    
    for i in range(0, len(data)):
        data[i].append(y[i])
    
    z = layers.Lambda(sampling)([z_mean, z_log_sigma])
    
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name='vae_mlp')
    
    '''
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    '''
    opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, clipnorm=1.0)
    vae.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt)
    
    xTrain, xTest = train_test_split(data, test_size=0.2)
    xTrain, xVal = train_test_split(data, test_size=0.2)
    
    vae.fit(xTrain, xTrain,
        epochs=15,
        batch_size=16,
        validation_data=(xVal, xVal))
    
    preds = vae.predict(xTest)
    print(preds)
    
if __name__ == "__main__":
    main()