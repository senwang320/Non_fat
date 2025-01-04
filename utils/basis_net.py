
import torch
import torch.nn as nn
import torchvision.models as models

import scipy.io as sio


# DCT basis layer, build DCT basis from coefficients
class dct_basis_layer(nn.Module):
    def __init__(self, _basis_dir, device='cpu'):
        super().__init__()
        
        bs_data    = sio.loadmat(_basis_dir)
        bmat       = bs_data['basis_EMT']
        N, H, W    = bmat.shape # number of bases, Height, Width
        
        bmat = torch.from_numpy(bmat.reshape(N, H*W)).float()
        # self.bmat = nn.Parameter(bmat)
        self.bmat = bmat.to(device) # avoid update this layer
        
    def forward(self, x):
        
        return torch.matmul(x, self.bmat)


def build_ft_net(_arc='resnet34', _in_ch=1, pretrained=True):
    
    if _arc == 'resnet34':
        ft_net  = models.resnet34(pretrained=pretrained)
    elif _arc == 'resnet50':
        ft_net  = models.resnet50(pretrained=pretrained)
    elif _arc == 'resnet18':
        ft_net = models.resnet18(pretrained=pretrained)
    else:
        raise ValueError('arc must in [resnet34, resnet50]')
    
    # feature extractor only, and change the # of input channels
    modules = list(ft_net.children())[:-1]
    modules = [nn.Conv2d(_in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)] + modules[1:]
    
    ft_net=nn.Sequential(*modules)
        
    return ft_net


# basis net, predict coefficients of bases for EMTs from scouts

class basis_net(nn.Module):
    def __init__(self, ft_arc='resnet34', basis_dir=None, pretrained=True, n_b=27):
        super().__init__()

        self.ft_net1 = build_ft_net(ft_arc, pretrained)
        self.ft_net2 = build_ft_net(ft_arc, pretrained)
        
        if ft_arc in ['resnet18', 'resnet34']:
            s_ft = 512
        elif ft_arc in ['resnet50']:
            s_ft = 2048
        else:
            raise ValueError('feature arc error')
        
        self.fc1 = nn.Linear(s_ft*2, 128)
        self.fc2 = nn.Linear(128, n_b)  

        self.n_b = n_b  # number of basis
        
        self.basis_dir = basis_dir
    
    def init_basis_layer(self, device):
        if self.basis_dir is not None:
            self.basis_layer = dct_basis_layer(self.basis_dir, device)
    
    # generate reference EMTs
    def refemt_generator(self, x, reshape_2D=False):
        
        emt = self.basis_layer(x)
        
        if reshape_2D:
            emt = emt.detach().cpu().numpy().squeeze().reshape(emt.shape[0], 750, 530)
        
        return emt
    
    # generate emts during inference
    def nnemt_generator(self, s1, s2, reshape_2D=True):
        x, emt = self.forward(s1, s2)
        if reshape_2D:
            emt = emt.detach().cpu().numpy().squeeze().reshape(emt.shape[0], 750, 530)
            
        return emt
    
    def get_basis_paras(self, s1, s2):
        x, _ = self.forward(s1, s2)
        
        return x
    
    def forward(self, s1, s2):
        
        x1 = self.ft_net1(s1)  # feature extraction from scout-1
        x2 = self.ft_net2(s2)  # feature extraction from scout-2

        # concatenate features
        x  = torch.cat((torch.flatten(x1,1), torch.flatten(x2,1)), dim=1)
        
        # linear layer to basis coefficients
        x  = self.fc1(x)
        x  = self.fc2(x)
        
        # build basis from coefficients
        if self.basis_dir is not None:
            e  = self.basis_layer(x)
            x  = [x, e]
        
        return x
        


