"""
Tools to simulate embedding using HILL steganographic algorithm
"""
import numpy as np
from scipy.signal import fftconvolve, convolve2d
import os, sys
from cv2 import imread

F = 4 * np.array([[-0.25, 0.5, -0.25],
             [0.5, -1, 0.5],
             [-0.25, 0.5, -0.25]])

WET_COST = 10 ** 10

def HILL(cover) :
    # Get embedding costs
    # initialization

    # compute residual
    R = convolve2d(cover, F, mode = 'same', boundary = 'symm')

    # compute suitability
    # Here we can use convolve2d because filters have odd dimensions
    xi = convolve2d(abs(R), (1.0/9.0)*np.ones((3, 3)), mode = 'same', boundary = 'symm')
    # compute embedding costs \rho
    rho = np.zeros((np.shape(xi)))
    xi[xi == 0] = np.inf
    inv_xi = 1/xi
    inv_xi[xi == np.inf] = np.inf
    rho = convolve2d(inv_xi, (1.0/225.0)*np.ones((15, 15)), mode = 'same', boundary = 'symm')
    # adjust embedding costs
    rho[rho > WET_COST] = WET_COST # threshold on the costs
    rho[np.isnan(rho)] = WET_COST # Check if all elements are numbers

    return rho

def do_process_color(cover):
    cover_image = cover

    b, k, l, d =cover_image.shape
    stegop1 = np.zeros((b, k, l, d))
    stegom1 = np.zeros((b, k, l, d))
    for batch_nb in range(b):
        for c in range(d):
            # Compute embedding costs rho
            canal_image = cover_image[batch_nb,:,:,c]
            rho = HILL(canal_image)

            rhoP1 = np.copy(rho)
            rhoM1 = np.copy(rho)
            #rhoP1[canal_image == 255] = WET_COST # Do not embed +1 if the pixel has max value
            #rhoM1[canal_image == 0] = WET_COST # Do not embed -1 if the pixel has min value

            stegop1[batch_nb, :,:,c] = rhoP1
            stegom1[batch_nb, :,:,c] = rhoM1

    return stegop1#, stegom1#stego, p_change_P1, p_change_M1
