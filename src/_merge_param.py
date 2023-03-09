from collections import OrderedDict
import torch

def merge_param(P : list((int,int,OrderedDict([(str, torch.Tensor),])),), device):
    """
    merge and mean parameters from other nodes.
    Device of Tensor can be different, so we need to move the parameters to the device of the first parameter.
    input : list(tuple(int,int,OrderedDict([(str, torch.Tensor),])),)
    output : OrderedDict([(str, torch.Tensor),])
    """
    par = P[0][2]
    #print(par.keys())
    for p in P:
        print(p[1],end=' ')
    print()
    for i in range(1,len(P)):
        for key in par.keys():
            par[key].add_(P[i][2][key].to(torch.device(device)))
    for key in par.keys():
        par[key] = par[key]/len(P)
        #par[key] = torch.divide(par[key],len(P))
    #par = par.to(torch.device(device))
    return par