import os

import numpy as np

import torch
from torch.utils.data import Dataset

import h5py
import scipy.io as sio
import skimage.transform as sktrans

import pandas as pd

class DirectDataset(Dataset):
    def __init__(self, h5_dir, scanlist, tgtlist,
                 casenum=None, imsize=None, scouts_range=[0, 800]):
        
        # load excel-like scanlist
        df = pd.read_excel(scanlist)
        
        # build basenames from excel data
        self.scanlist = ['SHC2.{}.{}.{}'.format(df['Exam Number'][ii], df['Series Number'][ii], df['Scan Number'][ii]) 
                         for ii in range(len(df['Exam Number']))] 
        # dataframe for the whole set
        self.df = df
        
        # read fitting target from data
        data          = pd.read_excel(tgtlist)
        # Build a dictionary using column 0 as the keys and the rest of the row as values
        dict_from_df  = data.set_index(data.columns[0]).to_dict(orient='index')
        self.tgt_dict = dict_from_df
        
        self.casenum    = casenum    # num of cases used in current dataset
        if self.casenum is not None and self.casenum not in ['all']:
            self.scanlist = self.scanlist[:self.casenum]
        
        self.h5_dir     = h5_dir
        self.imsize     = imsize
        
        # value of scouts for global normialzation because the measurements have physical meaning
        self.scouts_range = scouts_range     
        
    def __len__(self) -> int:
        return len(self.scanlist)
    
    def __getitem__(self, index):
        
        scan_fn = self.scanlist[index]
        # load scouts from h5 datasets
        f  = h5py.File(os.path.join(self.h5_dir, '{}.h5'.format(scan_fn)), 'r')
        sF = f['scout_frontal'][()]
        sL = f['scout_lateral'][()]
        scan_I  = f['scan_range'][()]
        
        f.close()
        
        this_tgt = self.tgt_dict[scan_fn]
        
        # preprocessing for images and emts
        sF = np.clip(sF, a_min=0, a_max=None)
        sL = np.clip(sL, a_min=0, a_max=None)
        
        # global normalization for scout images since its value has physical meaning (the line integral)
        sF = (sF*scan_I)/max(self.scouts_range)
        sL = (sL*scan_I)/max(self.scouts_range)
        
        if self.imsize is not None:
            sF    = sktrans.resize(sF, (self.imsize, self.imsize))
            sL    = sktrans.resize(sL, (self.imsize, self.imsize))
            # emt_M = sktrans.resize(emt_M, (self.imsize, self.imsize))
        sF, sL = sF.astype(np.float32), sL.astype(np.float32)
       
        result_dict = {
            'index'     : index, 
            'scan_fn'   : scan_fn, 
            'sF'        : np.expand_dims(sF, axis=0), 
            'sL'        : np.expand_dims(sL, axis=0),
            }
        
        # merge target into result
        result_dict = result_dict | this_tgt
        
        return result_dict

   