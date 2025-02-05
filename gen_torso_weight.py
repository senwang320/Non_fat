#%%
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


# load excel-like scanlist
df = pd.read_excel('./data/scan_list_120kVp[CAP]_2024_0121_20more.xlsx')

# build basenames from excel data
caselist = ['SHC2.{}.{}.{}'.format(df['Exam Number'][ii], df['Series Number'][ii], df['Scan Number'][ii]) 
                    for ii in range(len(df['Exam Number']))] 

vol_dir     = '/data1/AEC_SharedFiles/Stanford_Data/Exams_niis/dose_niis'
seg_out_dir = '/data1/AEC_SharedFiles/Stanford_Data/Segmentations/TS_dose_segs'

wt_data     = []

for cc in range(len(caselist)):
# for cc in range(5):
    
    print('Processing {}/{}'.format(cc+1, len(caselist)), caselist[cc])
    
    casename    = '{}_doseims'.format(caselist[cc])

    nii_data    = nib.load(os.path.join(vol_dir, casename+'.nii.gz'))
    vol_data    = nii_data.get_fdata()

    header      = nii_data.header
    vox_dim     = header.get_zooms()

    # load bone segmentation
    nii_data    = nib.load(os.path.join(seg_out_dir, casename, '00_merged_bone.nii.gz'))
    bone_M   = nii_data.get_fdata().astype(int)

    # extract soft-tissue region (TODO: exclude table!!)
    sft_M = (vol_data>-800) & (vol_data<1500)
    sft_M[bone_M>0] = 0

    # plt.figure
    # plt.subplot(1,3,1), plt.imshow(vol_data[:,:,200], vmin=-150, vmax=250, cmap='gray'), plt.axis('off')
    # plt.subplot(1,3,2), plt.imshow(sft_M[:,:,200]), plt.axis('off')
    # plt.subplot(1,3,3), plt.imshow(bone_M[:,:,200]), plt.axis('off')

    # body density calculation
    sft_rho = np.clip((vol_data[sft_M]/1000. + 1.), a_min=0, a_max=1.1)
    sft_w   = (1/1000)*np.sum(sft_rho)*(vox_dim[0]*vox_dim[1]*vox_dim[2]/1000)

    bone_rho = np.clip((vol_data[bone_M>0]/1000. + 1.), a_min=0, a_max=1.1) + np.clip((vol_data[bone_M>0]-100)/3195, a_min=0, a_max=None)
    bone_w   =  (1/1000)*np.sum(bone_rho)*(vox_dim[0]*vox_dim[1]*vox_dim[2]/1000)

    torso_w  = sft_w + bone_w

    this_segwts   = []
    seg_fnames    = ['subcutaneous_fat', 'torso_fat', 'skeletal_muscle']

    for ii in range(len(seg_fnames)):
        nii_data    = nib.load(os.path.join(seg_out_dir, casename, seg_fnames[ii]+'.nii.gz'))
        fat_M       = nii_data.get_fdata()

        header      = nii_data.header
        vox_dim     = header.get_zooms()

        this_fat_w  = np.sum(np.clip((vol_data[fat_M>0]/1000. + 1.), a_min=0, a_max=1.1))*(vox_dim[0]*vox_dim[1]*vox_dim[2]/1000)
        this_fat_w  = this_fat_w*(1/1000) # g --> kg

        # print(caselist[cc], nii_fnames[ii], '{:.2f} kg'.format(this_fat_w))
        this_segwts.append(this_fat_w)
    
    wts = [torso_w, torso_w-this_segwts[0]-this_segwts[1]] + this_segwts

    wt_data.append([caselist[cc]]+wts)

wts_fnames = ['casename', 'torso_wt', 'lean_torso_wt'] + seg_fnames

# Convert the list to a DataFrame
df = pd.DataFrame(wt_data, columns=wts_fnames)
df.to_excel("./data/nonfat_torso.xlsx", index=False)

# %%
