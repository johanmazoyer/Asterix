# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:32:39 2021

@author: sjuillar
"""
import numpy as np
import tools as tls
from pzernike import zernike, pmap

# Tailles
wphi = 32
lphi = 32
ech  = 4

# Pupille
r = 14


# Minimiation
gtol = 1e-3

# Perturbation
varb = 0.2

hypp = 1

# Inits
w   = wphi*ech
l   = lphi*ech
P = tls.circle(wphi,lphi,r)

# Creation d'une carte de defocalisation
[Ro,Theta] = pmap(wphi,lphi)
defoc = zernike(Ro,Theta,4)
