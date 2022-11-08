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
from scipy.stats.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

#dictionaries
cr56=[3,59,5,69,0,56]
#copes_dict={'tfMRI_EMOTION':5}
#---------------------------------------------------------------------------------------------------------

#copes_dict={'tfMRI_EMOTION':1}
def nIDP_mapping(nm_file_dir,out_dir,features=['UMAP1','UMAP2']):
    n=len(features)
    IDP=pd.read_csv(nm_file_dir)
    nIDP_df=pd.read_csv('nIDPs.csv',index_col=0)
    data_dictionary=pd.read_csv('data_dictionary.csv',index_col=0) 

    X=pd.merge(IDP,nIDP_df,how='inner',on='Subject')
    nIDPs=nIDP_df.columns[1:]
    R=[]
    P=pd.DataFrame()
    for nIDP in nIDPs:
        #checking the NAN values.
        columns=features+[nIDP]
        
        df=X[columns]
        df=df.dropna()
        if len(df[nIDP]):
            r,p=spearmanr(df[features],df[nIDP])
            pvalue=pd.DataFrame(columns=['FieldID-instance'],data=[nIDP])
            
            pvalue[features]=[-np.log10(p[n,:n])]
            P=pd.concat([P,pvalue])

    Pvalue=pd.merge(data_dictionary[['FieldID-instance','nIDP_category','Field']],P,how='inner',on='FieldID-instance')
    Pvalue=Pvalue.reset_index()
    Pvalue=Pvalue.rename(columns={'index':'nIDPs'})
    Pvalue=Pvalue.drop(Pvalue[Pvalue['nIDP_category']=='age_sex_site'].index)

    for idp in features:
        fig=sns.scatterplot(data=Pvalue, x='nIDPs',y=idp, hue='nIDP_category')
        fig.axhline(-np.log10(.05/len(P)),color='k')
        #if ylim:
    
            
        plt.legend(bbox_to_anchor=(1.1,1.05))
        plt.title(' p-value of corrlation ('+idp+',nIDP)')
        plt.savefig(out_dir+idp+'.png',dpi=300,bbox_inches='tight')
        plt.show()
        plt.close()

def cope_name():
    copes_path='/project/3022027.01/DL_2020/data/UKB/info/copes.csv'
    copes_df=pd.read_csv(copes_path)
    copes_df=copes_df.set_index(['task','CopeNumber'])
    return copes_df
copes_name=cope_name()
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

            img = nib.load(data_path+ np.str(sub)+ '/fMRI/tfMRI.feat/stats/cope'+np.str(cope)+'_standard.nii.gz')
            if downsample:
                img = image.resample_img(img, target_affine = np.eye(3)*3)
            data.append(np.expand_dims(np.float32(img.get_data()), 0))
            subs.append(np.int64(sub))
        except:
            #print('No preprocessed data for subject: %s' %(sub))
            continue;  

    UKB_table=pd.DataFrame()
    UKB_table['Subject']=subs
    UKB_table['task']=np.tile(task,len(subs))
    UKB_table['cope']=np.tile(cope,len(subs))
    data = np.concatenate(data,axis=0)
    return data,UKB_table



def load_tfMRIs_UKB(data_path,ids_path, copes_dict, downsample=True,disply_task=False):
    id_df=pd.read_csv(ids_path)
    ids=id_df['Subject'].values

    data_ = []
    UKB_table = pd.DataFrame(columns=['Subject','task','cope'])
    
    for task in copes_dict.keys():
        if disply_task:
            print (task)
        for cope in copes_dict[task]:
            MRI_data,sub_df=load_single_tfMRI(data_path,ids,task,cope,downsample=downsample)            
            UKB_table=UKB_table.append(sub_df, ignore_index=True)
            data_.append(MRI_data)

    data_ = np.concatenate(data_,axis=0)
    return data_,UKB_table

def load_dataset(data_path,ids_path,copes_dict,downsample=True):
    data,UKB_table=load_tfMRIs_UKB(data_path,ids_path, copes_dict, downsample=True)
    shape=data.shape
    data=data.reshape((shape[0],-1))
    not_atlas_index=mask(downsample=downsample)
    data[:,not_atlas_index]=0
    data=np.reshape(data,shape)    
    data=data[:,cr56[0]:cr56[1],cr56[2]:cr56[3],cr56[4]:cr56[5]]
    return data,UKB_table
    
#-------------------------------------------------------------------------------------------------




