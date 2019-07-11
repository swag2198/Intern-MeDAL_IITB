import torch

#Does Global Average Pooling. Converts (N,C,H,W) to (N,C)

def get_512_batch(x):
    return x.mean((2,3), keepdim = True).reshape(-1,512)

