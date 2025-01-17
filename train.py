#%%
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split

from utils.direct_dataset import DirectDataset
from utils.basis_net import basis_net, dct_basis_layer

from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models

import argparse
import json

cfg_fname       = './cfgs/nonfat_cfg.json'
this_cfg        = json.load(open(cfg_fname, 'r'))

n_epoch  = 1000

print(this_cfg)

trainset = DirectDataset(h5_dir=this_cfg['h5_dir'], scanlist=this_cfg['train_file'], tgtlist='./data/fat_list.xlsx',
                         casenum=this_cfg['train_num'], imsize=None, scouts_range=this_cfg['scouts_range'])

valset   = DirectDataset(h5_dir=this_cfg['h5_dir'], scanlist=this_cfg['val_file'], tgtlist='./data/fat_list.xlsx',
                         casenum=this_cfg['val_num'], imsize=None, scouts_range=this_cfg['scouts_range'])

# trainset = DirectDataset(h5_dir=this_cfg['h5_dir'], scanlist=this_cfg['train_file'], tgtlist='./data/fat_list.xlsx',
#                          casenum=4, imsize=None, scouts_range=this_cfg['scouts_range'])

# valset   = DirectDataset(h5_dir=this_cfg['h5_dir'], scanlist=this_cfg['val_file'], tgtlist='./data/fat_list.xlsx',
#                          casenum=4, imsize=None, scouts_range=this_cfg['scouts_range'])

# init writer of tensorboard
this_cfg['basename'] = 'tmp'
logdir = os.path.join("runs", this_cfg['basename'])
writer = SummaryWriter(logdir)

#%%
loader_args  = dict(batch_size=4, num_workers=2, pin_memory=True)

loaders = {
    'train'  : DataLoader(trainset, shuffle=True, **loader_args),
    'val'    : DataLoader(valset, shuffle=True, **loader_args),
}

# tt = next(iter(train_loader))

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model  = basis_net(ft_arc=this_cfg['ft_arc'], n_b=1, pretrained=True)
model  = model.to(device)

criterion = nn.MSELoss()

learning_rate = 1e-4
weight_decay  = 1e-8
momentum      = 0.999

optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
skdler    = optim.lr_scheduler.StepLR(optimizer, step_size=n_epoch/2, gamma=0.1)

result_dir = os.path.join(this_cfg['save_dir'], this_cfg['basename'])

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

this_cfg['result_dir'] = result_dir

save_cfg_dir           = os.path.join(result_dir, 'this_cfg.json')
best_model_params_path = os.path.join(result_dir, 'model_bestval.pt')


best_val  = 1e9
NORMKG    = 30

for epoch in range(n_epoch):

    trainval_dict = {}
    for this_phase in ['train', 'val']:
    # for this_phase in ['train']:
        
        if this_phase == 'train':
            model.train()
        else:
            model.eval()
        
        epoch_loss        = 0
        
        for bix, batch in enumerate(loaders[this_phase]):
            
            # batch, nBPs = batch
            scanNs, sF, sL = batch['scan_fn'], batch['sF'], batch['sL']
            refs           = batch['subcutaneous_fat'].float()
            
            # imgs
            sF, sL  = sF.to(device), sL.to(device)
            refs    = refs.to(device)/NORMKG
            
            optimizer.zero_grad()
            
            # forward
            # track history if only in train
            with torch.set_grad_enabled(this_phase == 'train'):
                # x  = model(sF, sL)
                x = model(sF, sL).squeeze()
                loss = criterion(x, refs)
                
                if this_phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            # ...log the running loss
            writer.add_scalar('minibatch/{}'.format(this_phase), loss,
                            epoch * len(loaders[this_phase]) + bix)
            
            epoch_loss        += loss.item()
        
        epoch_loss        = epoch_loss/len(loaders[this_phase])
        
        print('[{}] {}/{} epoch, {} batchs: loss={:.4f}, lr={}'.format(this_phase, epoch, n_epoch, len(loaders[this_phase]), epoch_loss,
                                                                                                 optimizer.param_groups[0]['lr']
                                                                                                 ))
        
        trainval_dict[this_phase] = epoch_loss
        # deep copy the model
        if this_phase == 'val' and epoch_loss < best_val:
            best_val = epoch_loss
            torch.save({'weights':model.state_dict(), 'epoch':epoch}, best_model_params_path)
            this_cfg['best_epoch'] = epoch
    
    skdler.step()
    writer.add_scalars('trainval', trainval_dict, epoch)

writer.close()

# write this cfg with some logs
with open(save_cfg_dir, 'w') as outfile:
    outfile.write(json.dumps(this_cfg, indent=4))

