from keras.layers import Input, Dense
from keras.models import Model
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras import regularizers


# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input
# encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
# decoded = Dense(784, activation='sigmoid')(encoded)

# # add a Dense layer with a L1 activity regularizer
# encoded = Dense(encoding_dim, activation='relu',
#                 activity_regularizer=regularizers.l1(10e-5))(input_img)
# decoded = Dense(784, activation='sigmoid')(encoded)

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded3 = Dense(32, activation='relu')(encoded) 

decoded1 = Dense(64, activation='relu')(encoded3)
decoded2 = Dense(128, activation='relu')(decoded1)
decoded3 = Dense(784, activation='sigmoid')(decoded2)


# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded3)

# this model maps an input to its encoded representation
encoded_input = Input(shape=(encoding_dim,))
# encoded_output = Input(shape=(64,))

encoder = Model(input_img, encoded3)
# decoder = autoencoder.layers[-3:]


# print(decoder.summary())
# decoder = Model(encoded_input, autoencoder.layers[-3:-1])

# create a placeholder for an encoded (32-dimensional) input
# retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# create the decoder model
# decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set

latent_vector = Input(shape=(32,))
decoder1 = Dense(64, activation='relu')(latent_vector)
decoder2 = Dense(128, activation='relu')(decoder1)
decoder3 = Dense(784, activation='sigmoid')(decoder2)

decoder = Model(latent_vector, decoder3)
decoder.compile(optimizer='adadelta', loss='binary_crossentropy')


decoder.layers[-3].set_weights(autoencoder.layers[-3].get_weights())
decoder.layers[-2].set_weights(autoencoder.layers[-2].get_weights())
decoder.layers[-1].set_weights(autoencoder.layers[-1].get_weights())

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

