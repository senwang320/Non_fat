#%%
import json
import matplotlib.pyplot as plt

from utils.direct_dataset import DirectDataset

cfg_fname       = './cfgs/nonfat_cfg.json'
this_cfg        = json.load(open(cfg_fname, 'r'))

print(this_cfg)

trainset = DirectDataset(h5_dir=this_cfg['h5_dir'], scanlist=this_cfg['train_file'], tgtlist='./data/fat_list.xlsx',
                         casenum=this_cfg['train_num'], imsize=None, scouts_range=this_cfg['scouts_range'])


# %%
tx = trainset[2]
sF = tx['sF'].squeeze()
sL = tx['sL'].squeeze()
plt.figure()
plt.subplot(1,2,1), plt.imshow(sF, cmap='gray'), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(sL, cmap='gray'), plt.axis('off')
# %%
