#%%
import os
import torch

import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio

from torch.utils.data import DataLoader, random_split

from utils.direct_dataset import DirectDataset
from utils.basis_net import basis_net
import json

wts_dir     = './wts_20more/Weight/'

config_dir  = os.path.join(wts_dir, 'this_cfg.json')
this_cfg    = json.load(open(config_dir, 'r'))

h5_dir      = this_cfg['h5_dir'] #'/data1/Maria_Sen_SharedFiles/Stanford_Data/h5s_dosenoise'

wt_data = torch.load(os.path.join(wts_dir, 'model_bestval.pt'))

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model  = basis_net(ft_arc=this_cfg['ft_arc'], n_b=1, pretrained=True)
model.load_state_dict(wt_data['weights'])
model.eval()

valset   = DirectDataset(h5_dir=this_cfg['h5_dir'], scanlist=this_cfg['test_file'], tgtlist='./data/nonfat_reference.xlsx',
                         casenum=this_cfg['test_num'], imsize=None, scouts_range=this_cfg['scouts_range'])

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

loader_args  = dict(batch_size=1, num_workers=1, pin_memory=True) 
val_loader   = DataLoader(valset, shuffle=False, **loader_args)

model.to(device)

#%%

NORMX    = this_cfg['NORMX']
outdata  = []
for bb, batch in enumerate(val_loader):
    
    # batch, nBPs = batch
    scanNs, sF, sL = batch['scan_fn'], batch['sF'], batch['sL']
    refs           = batch['Weight'].float()

    # imgs
    sF, sL  = sF.to(device), sL.to(device)
    # refs    = refs.to(device)/NORMKG

    scan_fn     = batch['scan_fn'][0]

    x = model(sF, sL).squeeze()
    print('{}/{}, {}...'.format(bb, len(val_loader), batch['scan_fn']), 'REF={:.2f} kg'.format(refs.squeeze().numpy()), 'PRED={:.2f} kg'.format(x.detach().cpu().numpy()*NORMX))
    
    outdata.append([scan_fn, refs.squeeze().item(), x.detach().cpu().numpy()*NORMX])
    # out_fname = os.path.join(nn_para_save_dir, '{}_nn_{}.mat'.format(scan_fn[0], fixed_emt))
    # sio.savemat(out_fname, {'x':para_nn.detach().cpu().numpy().squeeze()})
# %%
for ii in range(len(outdata)):
    print(outdata[ii][0], outdata[ii][1], outdata[ii][2])
# %%
