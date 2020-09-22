# -*- coding: utf-8 -*-
import torch
from scipy.signal import fftconvolve, convolve2d
from scipy.fftpack import dct, idct
from scipy.ndimage.filters import convolve
import numpy as np
import h5py
import os

def idct2(x):
    return idct(idct(x, norm='ortho').T, norm='ortho').T

def im2col(A,B):

    # Parameters
    M,N = A.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take (A,start_idx.ravel()[:,None] + offset_idx.ravel())
    return out

def wiener_filter(Cover):
    # Compute the wiener filtering the same way matlab does (wiener2)
    # One bug is removed: the convolution used in the wiener2 function does not
    # Use zero padding here.
    lp_filt = [[0.25 , 0.25, 0.0],[0.25, 0.25, 0.0],[0.0, 0.0, 0.0]]
    Cover = Cover.cpu().numpy()
    Loc_mean = convolve2d(Cover,lp_filt,'same')
    Loc_sigma = convolve2d(Cover**2,lp_filt,'same')-Loc_mean**2

    sigma_mean = np.average(Loc_sigma)

    WienerResidual = np.zeros(Cover.shape)
    Wiener_1st = np.zeros(Cover.shape)
    Wiener_1st[Loc_sigma!=0] = Cover[Loc_sigma!=0] - Loc_mean[Loc_sigma!=0]
    Wiener_2nd = np.zeros(Cover.shape)
    Wiener_2nd[Loc_sigma>sigma_mean] =-((Loc_sigma[Loc_sigma>sigma_mean]-sigma_mean)/Loc_sigma[Loc_sigma>sigma_mean])*(Cover[Loc_sigma>sigma_mean]-Loc_mean[Loc_sigma>sigma_mean])

    WienerResidual = Wiener_1st + Wiener_2nd
    return WienerResidual

def variance_estimation_2D(Image, BlockSize, Degree):
    # Estimation of the pixels' variance based on a 2D-DCT (trigonometric polynomial) model

    if BlockSize%2==0:
        raise ValueError('The block dimensions should be odd!!')
    if Degree > BlockSize:
        raise ValueError('Number of basis vectors exceeds block dimension!!')
    # number of parameters per block
    q = Degree*(Degree+1)//2

    # Build G matirx
    BaseMat = np.zeros((BlockSize,BlockSize))
    BaseMat[0,0] = 1
    G = np.zeros((BlockSize**2,q))
    k = 0
    for xShift in range(Degree):
        for yShift in range(Degree - xShift):
            G[:,k] = np.reshape( idct2( np.roll(np.roll(BaseMat,xShift,axis=0),yShift,axis=1)) , BlockSize**2 , 1)
            k=k+1

    # Estimate the variance
    PadSize = [BlockSize//2,BlockSize//2]
    I2C = im2col(np.pad(Image,PadSize,'symmetric'),(BlockSize,BlockSize))
    PGorth = np.eye(BlockSize**2) - np.dot(G,np.dot(np.linalg.pinv(np.dot(G.T,G)),G.T))
    EstimatedVariance = np.reshape(np.sum(np.dot(PGorth,I2C)**2,axis=0)/(BlockSize**2 - q),Image.shape)
    return EstimatedVariance

def find_local_var(cover_images, steg_images, torch_device):
    batch_size = cover_images.shape[0]
    local_variance = torch.zeros(steg_images.shape)
    for j in range(batch_size):
        for i in range(steg_images.shape[-1]):
            wiener_residual = wiener_filter(cover_images[j,:,:,i])
            channel_variance = variance_estimation_2D(wiener_residual,9,9)
            channel_variance[channel_variance< 0.01] = 0.01
            channel_variance = torch.tensor(channel_variance)
            local_variance[j,:,:,i] = channel_variance
    local_variance = local_variance.to(torch_device)
    return(local_variance)
