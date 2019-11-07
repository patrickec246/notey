'''
GAN models for audio learning
'''

import librosa

from keras.layers import *
from keras.models import *
from keras.optimizers import *

import keras.backend as K

temporal_kernel_sizes = [
        (130, 1), # ~ .75s of 1 octave
        (87, 1),  # ~ .50s of 1 octave
        (43, 1),  # ~ .25s of 1 octave
        ]

spectral_kernel_sizes = [
        (20, 4), # ~ .10s of 4 octaves
        (20, 4), # ~ .10s of 4 octaves
        (20, 2),  # ~ .10s of 2 octaves
        (20, 2)   # ~ .10s of 2 octaves
        ]

class Gan(object):
    def __init__(self, shape, latent_mode='linear', latent_dim=100, gen_weights=None, disc_weights=None):
        self.shape = shape
        self.latent_dim = latent_dim

        if latent_mode == 'projection':
            self.gen = keras.load_model(gen_weights) if gen_weights is not None else self.projection_generator()
        else:
            self.gen = keras.load_model(gen_weights) if gen_weights is not None else self.generator()

        self.disc = keras.load_model(disc_weights) if disc_weights is not None else self.discriminator()

        optimizer = Adam(.0002, .5)
        self.disc.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        z = Input(shape=(self.latent_dim,))
        gen_img = self.gen(z)
        self.disc.trainable = False

        self.combined = Model(z, self.disc(gen_img))
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def train(self, batch_size, real_song_data):
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = self.gen.predict(noise, 1)

        d_loss_real = self.disc.train_on_batch(real_song_data, np.ones((batch_size, 1)))
        d_loss_fake = self.disc.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

        return d_loss_real, d_loss_fake, g_loss

    def pred(self):
        return self.gen.predict(np.random.normal(0, 1, (1, 100)))

    def projection_generator(self):
        inp = Input(shape=self.shape)

        return Model(inp, model)

    def generator(self):
        def dense_encoded_space(x, num, prelu=True):
            y = Dense(num)(x)
            y = BatchNormalization(momentum=0.8)(y)
            if prelu:
                y = PReLU()(y)
            y = Dropout(rate=.25)(y)
            return y

        inp = Input(shape=(self.latent_dim,))

        model = dense_encoded_space(inp, 128)
        model = dense_encoded_space(model, 256)
        model = dense_encoded_space(model, 512)
        model = dense_encoded_space(model, np.prod(self.shape), prelu=False)

        model = Reshape(target_shape=self.shape)(model)
        
        # convolve once, then do a reduction convolution
        model = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
        model = PReLU()(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dropout(rate=.1)(model)

        model = Conv2D(12, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Activation('sigmoid')(model)
        
        return Model(inp, model)

    def discriminator(self):
        def spectro_temporal_layer(x, layers=64, kernel_size=(3, 3)):
            y = Conv2D(layers, kernel_size=kernel_size, strides=(1, 1), padding='same')(x)
            y = BatchNormalization(momentum=0.8)(y)
            y = PReLU()(y)
            y = Dropout(rate=.25)(y)
            return y

        # spill layer
        inp = Input(shape=self.shape)
        model = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inp)
        model = BatchNormalization(momentum=0.8)(model)
        model = PReLU()(model)

        # interweave spectro-temporal layers
        for i in range(len(temporal_kernel_sizes) + len(spectral_kernel_sizes)):
            kernel_size = spectral_kernel_sizes[int(i/2)] if i % 2 == 0 else temporal_kernel_sizes[int(i/2)]
            model = spectro_temporal_layer(model, layers=(i+1)*32, kernel_size=kernel_size)

        # + 3 pooling layers
        model = MaxPooling2D()(model)
        model = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = PReLU()(model)

        model = MaxPooling2D()(model)
        model = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = PReLU()(model)

        model = MaxPooling2D()(model)
        model = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = PReLU()(model)

        # flatten, condense
        model = Flatten()(model)
    
        model = Dense(128)(model)
        model = PReLU()(model)
        model = Dropout(rate=.25)(model)

        # tell us if we won
        model = Dense(1, activation='softmax')(model)
        
        return Model(inp, model)
