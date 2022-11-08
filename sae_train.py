#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:43:30 2019

@author: marzab
"""
import os
import time

import numpy as np
import pickle

from sklearn.preprocessing import RobustScaler
from utils import copes_dict,load_dataset


import tensorflow as tf
import pandas as pd
import gc
from utils_models import make_decoder_model_joint,make_encoder_model_joint
#%%

#print(K.tensorflow_backend._get_available_gpus())
print(tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
mem=tf.config.experimental.get_memory_info('GPU:0')
   

#%%
n_epochs=2000
model_name='model_0'
i_kfold=0
kfold_name='kf_{:02d}'.format(i_kfold)

PROJECT_ROOT = 'PROJECT ROOT'
data_path='DATA PATH'
ids_path='IDS ifold'


out_dir=os.path.join(PROJECT_ROOT,'JOINT',model_name,kfold_name)
model_dir=os.path.join(out_dir,'models')
scaler_dir=os.path.join(out_dir,'objects')

#% creating output repositories
os.makedirs(out_dir,exist_ok=True)
os.makedirs(model_dir,exist_ok=True)
os.makedirs(scaler_dir,exist_ok=True)

#%%
#------------------------------------------------------------------------------------------------------
# loading data
time1=time.time()
x,HCP_table=load_dataset(data_path,ids_path,copes_dict)
time2=time.time()
print("================================================================================")
print('data loaded time:',time2-time1)
shape=x.shape
x=x.reshape((shape[0],-1))

#%%
#scaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x) 
del x
gc.collect()
pickle.dump(scaler, open( scaler_dir+'/scaler.p', 'wb' ) )

print('data normalized')
x_train=np.reshape(x_train,shape) 
 #%%
print('Model_name:',model_name)
print('ikfold: ',i_kfold)
#-----------------------------------------------------------------------------------------------------
#%% AGE and SEX MATRIX
nIDPs_path='/project/3022027.01/DL_2020/data/HCP1200/info/demographic.csv'
nIDPs=pd.read_csv(nIDPs_path,index_col=0)
nIDPs_data=nIDPs.loc[HCP_table['Subject']]
age=nIDPs_data['Age'].values[:, np.newaxis].astype('float32')
sex=nIDPs_data['Gender'].values[:, np.newaxis].astype('float32')

y_data= np.concatenate((age, sex), axis=1).astype('float32')
#%%
batch_size = 10
n_samples = x_train.shape[0]
n_samplesx=10000
x_train=x_train.reshape(x_train.shape+(1,))
input_shape=x_train.shape
#%%
# Create the dataset iterator
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_data))
train_dataset = train_dataset.shuffle(buffer_size=n_samplesx)
train_dataset = train_dataset.batch(batch_size)
print(x_train.shape)
del x_train
#%%
n_features = input_shape[1:]

h1_dim = [16,12,8]
h2_dim=[8,12,16]
z_dim = 100
base_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(base_lr,decay_steps=100000,decay_rate=0.96,staircase=True)
#%%model params#Network_specifications
P=pd.DataFrame()
P['dense_layer']=z_dim
P['local_model_name']=model_name
P['local_shape']=shape
P['last_layer_activation']='linear' 
P['loss']='mean_squared_error'
P['lr']=base_lr
P['opt']='Adam'
# LOSS Multipliers
lambda_ = .05
P['lambda']=lambda_
  
#%%
encoder = make_encoder_model_joint(n_features, h1_dim, z_dim)
decoder = make_decoder_model_joint(z_dim , h2_dim)
#%%
# Loss functions
import pandas as pd
df = pd.DataFrame(columns=['epoch','ETA','Reconstruction cost','Supervised cost','sex_acc','age_mae'])
df_cross_validation=pd.DataFrame(columns=['epoch','ETA','Reconstruction cost','Supervised cost','sex_acc','age_mae'])

#%%

#evalution_metrics - not minimizing during the training 
mae_age_fn=tf.keras.metrics.MeanAbsoluteError()
acc_sex_fn = tf.keras.metrics.BinaryAccuracy()
#reconst loss
mse_loss_fn = tf.keras.losses.MeanSquaredError()
# Supervised loss
mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
bin_loss_fn= tf.keras.losses.BinaryCrossentropy(from_logits=False)
#%%# Define optimizers
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

###########################################################################
#%%
# Training
# Training step
@tf.function

def train_on_batch(batch_x, batch_y):
    with tf.GradientTape() as tape:
        acc_sex_fn.reset_states()
        mae_age_fn.reset_states()
        # Inference
        batch_latent, batch_observed = encoder(batch_x)
        batch_reconstruction = decoder(batch_latent)

        # Loss functions
        recon_loss = mse_loss_fn(batch_x, batch_reconstruction)
        age_loss = mae_loss_fn(batch_y[:,:1], batch_observed[:,:1])
        sex_loss=bin_loss_fn(batch_y[:,1:], batch_observed[:,1:])
        
        supervised_loss=age_loss +sex_loss
        ae_loss = lambda_*recon_loss + (1-lambda_)*supervised_loss        
        
        acc_sex=acc_sex_fn(batch_y[:,1:], batch_observed[:,1:])
        mae_age=mae_age_fn(batch_y[:,:1], batch_observed[:,:1])
    gradients = tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return recon_loss, supervised_loss, acc_sex,mae_age

#%% cross_validation whitin the batch
def test_on_batch(batch_tx, batch_ty):
    acc_sex_fn.reset_states()
    mae_age_fn.reset_states()
    # Inference
    batch_latent, batch_observed = encoder(batch_tx)
    batch_reconstruction = decoder(batch_latent)

    # Loss functions
    recon_loss = mse_loss_fn(batch_tx, batch_reconstruction)
    age_loss = mae_loss_fn(batch_ty[:,:1], batch_observed[:,:1])
    sex_loss=bin_loss_fn(batch_ty[:,1:], batch_observed[:,1:])
    
    supervised_loss=age_loss +sex_loss       
    
    acc_sex=acc_sex_fn(batch_ty[:,1:], batch_observed[:,1:])
    mae_age=mae_age_fn(batch_ty[:,:1], batch_observed[:,:1])
    
    return recon_loss, supervised_loss, acc_sex,mae_age
#%%
number_batch_cv=n_samples/batch_size-50 # the last 50 batchs will be used as CV (it's fixed batches through iterations)
# Training loop

start = time.time()
for epoch in range(n_epochs):
    
    # Functions to calculate epoch's mean performance
    epoch_recon_loss_avg = tf.metrics.Mean()
    epoch_supervised_loss_avg = tf.metrics.Mean()    

    epoch_acc_sex_avg=tf.metrics.Mean()
    epoch_mae_age_avg=tf.metrics.Mean()
    # Functions to calculate epoch's mean performance in TEST
    epoch_recon_loss_test_avg = tf.metrics.Mean()
    epoch_supervised_loss_test_avg = tf.metrics.Mean()

    epoch_acc_sex_test_avg=tf.metrics.Mean()
    epoch_mae_age_test_avg=tf.metrics.Mean()
    

    counter=0
    for batch, (batch_x, batch_y) in enumerate(train_dataset):

        if counter < number_batch_cv:
            recon_loss, supervised_loss, acc_sex,mae_age = train_on_batch(batch_x, batch_y)

            epoch_recon_loss_avg(recon_loss)
            epoch_supervised_loss_avg(supervised_loss)

            epoch_acc_sex_avg(acc_sex)
            epoch_mae_age_avg(mae_age)
        
            epoch_time = time.time() - start
    

        else:
            recon_loss_test, supervised_loss_test,acc_sex_test,mae_age_test = test_on_batch(batch_x, batch_y)
#
            epoch_recon_loss_test_avg(recon_loss_test)
            epoch_supervised_loss_test_avg(supervised_loss_test)

            epoch_acc_sex_test_avg(acc_sex_test)
            epoch_mae_age_test_avg(mae_age_test)



        counter=counter+1
    print('{:3d}: {:.2f}s ETA: {:.2f}s  Reconstruction cost: {:.4f}  Supervised cost: {:.4f}   sex_acc:{:.2f} age_mae: {:.4f}'
          .format(epoch + 1, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_recon_loss_avg.result(),
                  epoch_supervised_loss_avg.result(),
                  epoch_acc_sex_avg.result(),
                  epoch_mae_age_avg.result()))
    df.loc[epoch,['epoch']]=epoch + 1
    df.loc[epoch,['ETA']]=epoch_time * (n_epochs - epoch)
    df.loc[epoch,['Reconstruction cost']]=epoch_recon_loss_avg.result()
    df.loc[epoch,['Supervised cost']]=epoch_supervised_loss_avg.result()
    df.loc[epoch,['sex_acc']]=epoch_acc_sex_avg.result()
    df.loc[epoch,['age_mae']]=epoch_mae_age_avg.result()

    
    print('{:3d}: {:.2f}s ETA: {:.2f}s  Reconstruction CV: {:.4f}  Supervised cost CV: {:.4f}  sex_acc:{:.2f} age_mae: {:.4f}'
          .format(epoch + 1, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_recon_loss_test_avg.result(),
                  epoch_supervised_loss_test_avg.result(),
                  epoch_acc_sex_test_avg.result(),
                  epoch_mae_age_test_avg.result()))
    df_cross_validation.loc[epoch,['epoch']]=epoch + 1
    df_cross_validation.loc[epoch,['ETA']]=epoch_time * (n_epochs - epoch)
    df_cross_validation.loc[epoch,['Reconstruction cost']]=epoch_recon_loss_test_avg.result()
    df_cross_validation.loc[epoch,['Supervised cost']]=epoch_supervised_loss_test_avg.result()
    df_cross_validation.loc[epoch,['sex_acc']]=epoch_acc_sex_test_avg.result()
    df_cross_validation.loc[epoch,['age_mae']]=epoch_mae_age_test_avg.result()
    
    print('================================================================================================')
   
 #%%
df_cross_validation.to_csv(model_dir+'/loss_cv.csv')
df.to_csv(model_dir+'/loss_train.csv')
P.to_csv(model_dir+'/model_parameters.csv')
encoder.save(model_dir +'/model_encoder.h5')
decoder.save(model_dir+'/model_decoder.h5')
#%%
HCP_table.to_csv(out_dir+'/train_table.csv')


time3=time.time()
print('overall proccesing time :',time3-start)
   
