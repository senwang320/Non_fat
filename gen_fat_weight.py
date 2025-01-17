
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


seg_out_dir = '/data1/AEC_SharedFiles/Stanford_Data/Segmentations/TS_dose_segs'
fat_density = 0.92 # g/cm^3
nii_fnames  = ['subcutaneous_fat', 'torso_fat']

cols        = ['casename'] + nii_fnames
fat_data    = []

for cc in range(len(caselist)):
# for cc in range(10):
    casefolder  = '{}_doseims'.format(caselist[cc])
    
    this_fats   = []
    for ii in range(len(nii_fnames)):
        nii_data    = nib.load(os.path.join(seg_out_dir, casefolder, nii_fnames[ii]+'.nii.gz'))
        vol_data    = nii_data.get_fdata()

        header      = nii_data.header
        vox_dim     = header.get_zooms()

        this_fat_w  = np.sum(vol_data)*(vox_dim[0]*vox_dim[1]*vox_dim[2]/1000)*fat_density
        this_fat_w  = this_fat_w*(1/1000) # g --> kg

        print(caselist[cc], nii_fnames[ii], '{:.2f} kg'.format(this_fat_w))
        this_fats.append(this_fat_w)
        
    fat_data.append([caselist[cc]]+this_fats)
    

# Convert the list to a DataFrame
df = pd.DataFrame(fat_data, columns=cols)
df.to_excel("./data/fat_list.xlsx", index=False)

# %%
