#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:32:45 2020

@author: marzab
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from nilearn.datasets import fetch_atlas_harvard_oxford
#dictionaries
cr56=[3,59,5,69,0,56]
copes_dict={'tfMRI_EMOTION':6,'tfMRI_GAMBLING':6,'tfMRI_LANGUAGE':6,'tfMRI_RELATIONAL':6,'tfMRI_SOCIAL':6,'tfMRI_MOTOR':26,'tfMRI_WM':30}
#---------------------------------------------------------------------------------------------------------
def cope_name():
    copes_path='/copes.csv'
    copes_df=pd.read_csv(copes_path)
    copes_df=copes_df.set_index(['task','CopeNumber'])
    return copes_df
copes_name=cope_name()
#copes_dict={'tfMRI_EMOTION':1}
#---------------------------------------------------------------------------------------------------------
def params_table(params,save_path):
    var=np.array(list(params.values()))
    
    params_table=pd.DataFrame(data=var[np.newaxis,:],columns=list(params.keys()))
    params_table.to_csv(save_path+'parameters.csv')
    return

def mkdir_if(model_path):
    if not os.path.isdir(str(model_path)):
        model_path.mkdir()
    return
#-----------------------------------------------------------------------------------------------------------
def mask(mask_name='sub-maxprob-thr25-2mm',downsample=True,crop=False,version_=True):
    mask=fetch_atlas_harvard_oxford(mask_name)
    if version_:
        img=mask.maps
    else:
        img = nib.load(mask.maps)
    
    if downsample:
        img = image.resample_img(img, target_affine = np.eye(3)*3)
    mask_img=img.get_data()
    if crop:
        mask_img=mask_img[cr56[0]:cr56[1],cr56[2]:cr56[3],cr56[4]:cr56[5]]
    maps=np.reshape(mask_img,(np.prod(mask_img.shape),))
    not_atlas_index=np.where(maps<0.5)[0]
    return not_atlas_index

def load_single_tfMRI(data_path,Subject,task,cope,downsample=True):
    data=[]
    subs=[]
    for sub in Subject:
        try:
            img = nib.load(data_path+ np.str(sub)+ '/MNINonLinear/Results/'+ task+'/'+task+'_hp200_s4_level2vol.feat/cope'+np.str(cope)+'.feat/stats/cope1.nii.gz')
            if downsample:
                img = image.resample_img(img, target_affine = np.eye(3)*3)
            data.append(np.expand_dims(np.float32(img.get_data()), 0))
            subs.append(np.int64(sub))
        except:
            #print('No preprocessed data for subject: %s' %(sub))
            continue;  

    HCP_table=pd.DataFrame()
    HCP_table['Subject']=subs
    HCP_table['task']=np.tile(task,len(subs))
    HCP_table['cope']=np.tile(cope,len(subs))
    data = np.concatenate(data,axis=0)
    return data,HCP_table



def load_tfMRIs_HCP(data_path,ids_path, copes_dict, downsample=True,disply_task=False):
    id_df=pd.read_csv(ids_path)
    ids=id_df['Subject'].values

    data_ = []
    HCP_table = pd.DataFrame(columns=['Subject','task','cope'])
    
    for task in copes_dict.keys():
        if disply_task:
            print (task)
        for cope in range(1,copes_dict[task]+1):
            MRI_data,sub_df=load_single_tfMRI(data_path,ids,task,cope,downsample=downsample) 
            HCP_table=pd.concat([HCP_table,sub_df],axis=0)

            data_.append(MRI_data)

    data_ = np.concatenate(data_,axis=0)
    return data_,HCP_table

def load_dataset(data_path,ids_path,copes_dict,downsample=True):
    data,HCP_table=load_tfMRIs_HCP(data_path,ids_path, copes_dict, downsample=True)
    shape=data.shape
    data=data.reshape((shape[0],-1))
    not_atlas_index=mask(downsample=downsample)
    data[:,not_atlas_index]=0
    data=np.reshape(data,shape)    
    data=data[:,cr56[0]:cr56[1],cr56[2]:cr56[3],cr56[4]:cr56[5]]
    HCP_table.reset_index(drop=True)
    return data,HCP_table
    




