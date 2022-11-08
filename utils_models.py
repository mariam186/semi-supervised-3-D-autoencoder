#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 20:29:03 2021

@author: marzab
"""
import numpy as np
from keras.layers import  Input, Dense
from keras.models import Model



from keras.layers import Conv3D, UpSampling3D, AveragePooling3D,Dropout,LeakyReLU
from keras.layers import Flatten, Reshape, GaussianNoise,concatenate,BatchNormalization
#%%

# Encoder
def make_encoder_model(n_features, h_dim, z_dim,n_class,sex=2,stddev =0.1,dropout_level=.1):
    """Creates the encoder."""
    inputs = Input(shape=n_features)
    x = GaussianNoise(stddev = stddev)(inputs)
    
    for n_neurons_layer in h_dim:
#        x = Conv3D(n_neurons_layer)(x)
#        
        
        x=Conv3D(n_neurons_layer, 3, padding='same')(x) 
        x = LeakyReLU()(x)
        x = AveragePooling3D(2, padding='same')(x)  # Layer 1
        x=BatchNormalization()(x)
        x=Dropout(dropout_level)(x)
        
    x = Flatten()(x) 
    #Latent = Dense(z_dim)(x) 
    Latent = Dense(z_dim,activation='linear')(x)
    Observed_age =Dense(n_class-sex, activation='softmax', name='Observed_variables_age')(x)
    Observed_sex =Dense(sex, activation='softmax', name='Observed_variables_sex')(x)
    Observed=concatenate([Observed_age, Observed_sex], axis=-1)
    model = Model(inputs=inputs, outputs=[Latent, Observed], name='Encoder')
    return model



# Decoder
def make_decoder_model(z_dim, h_dim,n_class,latent_dimx=[7,8,7,8],dropout_level=.1):
    """Creates the decoder."""
    
    input_latent = Input(shape=(z_dim,), name='Latent_variables')
    input_observed=Input(shape=(n_class,), name='Observed_variables')
    
    x = concatenate([input_latent, input_observed], axis=-1)
    x = Dense(np.prod(latent_dimx))(x)
    x = LeakyReLU()(x)
    x=Reshape(latent_dimx)(x)
    
    for n_neurons_layer in h_dim:
        x=Conv3D(n_neurons_layer, 3, padding='same')(x)
        x = LeakyReLU()(x)
        x = UpSampling3D(2)(x)
        x=BatchNormalization()(x)
        x=Dropout(dropout_level)(x)
        


    reconstruction = Conv3D(1,3,padding='same', activation='linear')(x)
    model = Model(inputs=[input_latent, input_observed], outputs=reconstruction)
    return model

# Encoder
def make_encoder_model_reg(n_features, h_dim, z_dim,n_class=3,sex=2,stddev =0.1,dropout_level=.1):
    """Creates the encoder."""
    inputs = Input(shape=n_features)
    x = GaussianNoise(stddev = stddev)(inputs)
    
    for n_neurons_layer in h_dim:
#        x = Conv3D(n_neurons_layer)(x)
#        
        
        x=Conv3D(n_neurons_layer, 3, padding='same')(x) 
        x = LeakyReLU()(x)
        x = AveragePooling3D(2, padding='same')(x)  # Layer 1
        x=BatchNormalization()(x)
        x=Dropout(dropout_level)(x)
        
    x = Flatten()(x) 
    #Latent = Dense(z_dim)(x) 
    Latent = Dense(z_dim,activation='linear')(x)
    Observed_age =Dense(n_class-sex, activation='linear', name='Observed_variables_age')(x)
    Observed_sex =Dense(sex, activation='softmax', name='Observed_variables_sex')(x)
    Observed=concatenate([Observed_age, Observed_sex], axis=-1)
    model = Model(inputs=inputs, outputs=[Latent, Observed], name='Encoder')
    return model

#%%
# Encoder
def make_encoder_model_joint(n_features, h_dim, z_dim,stddev =0.1,dropout_level=.1):
    """Creates the encoder."""
    inputs = Input(shape=n_features)
    x = GaussianNoise(stddev = stddev)(inputs)
    
    for n_neurons_layer in h_dim:
#        x = Conv3D(n_neurons_layer)(x)
#        
        
        x=Conv3D(n_neurons_layer, 3, padding='same')(x) 
        x = LeakyReLU()(x)
        x = AveragePooling3D(2, padding='same')(x)  # Layer 1
        x=BatchNormalization()(x)
        x=Dropout(dropout_level)(x)
        
    x = Flatten()(x) 
    #Latent = Dense(z_dim)(x) 
    Latent = Dense(z_dim,activation='linear')(x)
    Observed_age =Dense(1, activation='linear', name='Observed_variables_age')(Latent)
    Observed_sex =Dense(1, activation='sigmoid', name='Observed_variables_sex')(Latent)
    Observed=concatenate([Observed_age, Observed_sex], axis=-1)
    model = Model(inputs=inputs, outputs=[Latent, Observed], name='Encoder')
    return model



# Decoder
def make_decoder_model_joint(z_dim, h_dim,latent_dimx=[7,8,7,8],dropout_level=.1):
    """Creates the decoder."""
    
    input_latent = Input(shape=(z_dim,), name='Latent_variables')
    #input_observed=Input(shape=(n_class,), name='Observed_variables')
    
    #x = concatenate([input_latent, input_observed], axis=-1)
    x = Dense(np.prod(latent_dimx))(input_latent)
    x = LeakyReLU()(x)
    x=Reshape(latent_dimx)(x)
    
    for n_neurons_layer in h_dim:
        x=Conv3D(n_neurons_layer, 3, padding='same')(x)
        x = LeakyReLU()(x)
        x = UpSampling3D(2)(x)
        x=BatchNormalization()(x)
        x=Dropout(dropout_level)(x)
        


    reconstruction = Conv3D(1,3,padding='same', activation='linear')(x)
    model = Model(inputs=input_latent, outputs=reconstruction)
    return model
#%%
#%%

# Encoder
def make_encoder(n_features, h_dim, z_dim,stddev =0.1,dropout_level=.1):
    """Creates the encoder."""
    inputs = Input(shape=n_features)
    x = GaussianNoise(stddev = stddev)(inputs)
    
    for n_neurons_layer in h_dim:
#        x = Conv3D(n_neurons_layer)(x)
#        
        
        x=Conv3D(n_neurons_layer, 3, padding='same')(x) 
        x = LeakyReLU()(x)
        x = AveragePooling3D(2, padding='same')(x)  # Layer 1
        x=BatchNormalization()(x)
        x=Dropout(dropout_level)(x)
        
    x = Flatten()(x) 
    #Latent = Dense(z_dim)(x) 
    Latent = Dense(z_dim,activation='linear')(x)

    model = Model(inputs=inputs, outputs=Latent, name='Encoder')
    return model



# Decoder
def make_decoder(z_dim, h_dim,latent_dimx=[7,8,7,8],dropout_level=.1):
    """Creates the decoder."""
    
    input_latent = Input(shape=(z_dim,), name='Latent_variables')

    x = Dense(np.prod(latent_dimx))(input_latent)
    x = LeakyReLU()(x)
    x=Reshape(latent_dimx)(x)
    
    for n_neurons_layer in h_dim:
        x=Conv3D(n_neurons_layer, 3, padding='same')(x)
        x = LeakyReLU()(x)
        x = UpSampling3D(2)(x)
        x=BatchNormalization()(x)
        x=Dropout(dropout_level)(x)
        


    reconstruction = Conv3D(1,3,padding='same', activation='linear')(x)
    model = Model(inputs=input_latent, outputs=reconstruction)
    return model