#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:04:58 2019
@author: marzab
"""
import numpy as np
import pickle

from keras.callbacks  import EarlyStopping
from keras.optimizers import RMSprop

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras import backend as K


from keras.layers import Conv3D, UpSampling3D, AveragePooling3D,Dropout
from keras.layers import Flatten, Reshape, GaussianNoise

from keras.layers.normalization import BatchNormalization

#%% Usage:
#  X_train: training data (nsubject * xdim * ydim * zdim* 1), your 4D nii dataset (e.g. T1). Need to be converted to tensor form
#  Note that the shape of image should be competible with structure of your AE
#  encoder_filters: The number of filters (nodes) in each encoder layer. (e.g. encoder_filters= [36,16,8])
#  decoder_filters: The number of filters (nodes) in each decoder layer. (e.g. decoder_filters= [8,16,36])
#  droput_leve: indicates the droput rate [0, 1] e.g dropout_level=0.3
#  input_shape: the shape of your nii file. For example if you have 1000 images 
## and your image has this dimention (56,64,56) then your input_shape would be (56,64,56,1) 


#%%

def autoencoder(input_shape,encoder_filters,decoder_filters,dense_layer=0,dropout_level=0.2,Gaussian_noise=1,activation='relu',last_layer_activation='linear',kernel_initializer='he_normal',dropout_flag=False,stddev=.1):
#Encoder 
    if dropout_flag:
        print('Drop-out level:',dropout_level)
    input_img = Input(input_shape) 
    if Gaussian_noise:
        x = GaussianNoise(stddev = stddev)(input_img)
        print("GaussianNoise,",stddev)
    else:
        x = input_img
    
    #en_layer=len(encoder_filters)  
    
    x=Conv3D(encoder_filters[0], 3, padding='same', activation=activation,kernel_initializer=kernel_initializer)(x) 
    x = AveragePooling3D((2, 2, 2), padding='same')(x)  # Layer 1
    x=BatchNormalization()(x)
        
    for l in range(len(encoder_filters) -1):

        x=Conv3D(encoder_filters[l+1], 3, padding='same', activation=activation,kernel_initializer=kernel_initializer)(x) 
        x = AveragePooling3D((2, 2, 2), padding='same')(x)  # Layer 2
        x=BatchNormalization()(x)
        
        if dropout_flag:
            x = Lambda(lambda x: K.dropout(x, level=dropout_level))(x) 
        elif dropout_level>0:
            x=Dropout(dropout_level)(x)
            
    if dense_layer:               

    
        latent_dim=K.int_shape(x)
        x = Flatten()(x) 
        x = Dense(dense_layer, kernel_initializer=kernel_initializer, activation=activation)(x)  
        #Encoder with FC layer
        encoder = Model(input_img,x, name='encoder')
        encoder.summary()
        
        latent_inputs = Input(shape=(dense_layer,), name='decoder_input')
        #Decoder Input
        x = Dense(np.prod(latent_dim[1:]), kernel_initializer=kernel_initializer, activation='relu')(latent_inputs) 
        x=Reshape(latent_dim[1:])(x)
        
        x=Conv3D(decoder_filters[0], 3, padding='same', activation='relu',kernel_initializer=kernel_initializer)(x)
        x = UpSampling3D(2)(x)
        x=BatchNormalization()(x)
        print("FC")
    else:
        encoder = Model(input_img,x, name='encoder')
        encoder.summary()
        #Decoder Input
        latent_dim=K.int_shape(x)
        latent_inputs = Input(latent_dim[1:])
        
        x=Conv3D(decoder_filters[0], 3, padding='same', activation='relu',kernel_initializer=kernel_initializer)(latent_inputs)
        x=BatchNormalization()(x)
        x = UpSampling3D(2)(x)
        print("FConv")

    
    
    for l in range(len(decoder_filters)-1):
        
        x=Conv3D(decoder_filters[l+1], 3, padding='same', activation=activation,kernel_initializer=kernel_initializer)(x) 
        x = UpSampling3D(2)(x)  # Layer 2
        x=BatchNormalization()(x)
    
        if dropout_flag:
            x = Lambda(lambda x: K.dropout(x, level=dropout_level))(x) 
        elif dropout_level>0:
            x=Dropout(dropout_level)(x)   
        
    
    outputs= Conv3D(1,3, padding='same', activation=last_layer_activation,kernel_initializer=kernel_initializer)(x)
    #Decoder 
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    ae = Model(input_img, decoder(encoder(input_img)), name='ae_mlp')
    ae.summary()
    

    return ae,encoder,decoder
#%% Training the model 
#==============================================================================    
    
def train_model(X_train,encoder_filters,decoder_filters,save_path,loss='mean_squared_error',opt=RMSprop(lr = 0.001),epochs=20,batch_size=10,validation_split=.05,dense_layer=10,dropout_level=0.2,Gaussian_noise=1,activation='relu',last_layer_activation='linear',kernel_initializer='he_normal',dropout_flag=False,stddev=0.1,patience=8):

    
    input_shape = X_train.shape
    if input_shape[len(input_shape)-1]>1:
        X_train=X_train.reshape(X_train.shape+(1,))
        input_shape=X_train.shape
    print(input_shape)

    model,encoder,decoder=autoencoder(input_shape[1:],encoder_filters,decoder_filters,dense_layer=dense_layer,dropout_level=dropout_level,Gaussian_noise=Gaussian_noise,activation=activation,last_layer_activation=last_layer_activation,kernel_initializer=kernel_initializer,dropout_flag=dropout_flag,stddev=stddev)
    
    opt =opt
    print("============================")
# Here you can change loss function.    
    model.compile(optimizer=opt, loss=loss)
    ##################################Training the Network#########################

    earlyStopping = EarlyStopping(monitor='val_loss',patience=patience,verbose=1, mode='auto')
    history=model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                   shuffle=True, validation_split=validation_split, verbose=1,callbacks=[earlyStopping])

 
    model.save(save_path+'model.h5')
    encoder.save(save_path+'model_encoder.h5')
    decoder.save(save_path+'model_decoder.h5')
    pickle.dump(history.history, open( save_path+"history.p", "wb" ) )
    

    return model,encoder,decoder,history
© 2021 GitHub, Inc.