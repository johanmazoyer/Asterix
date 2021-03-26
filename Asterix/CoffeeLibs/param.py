# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:32:39 2021

@author: sjuillar
"""
import numpy as np
import CoffeeLibs.tools as tls

# Tailles
wphi = 32
lphi = 32
ech  = 4

# Pupille
r = 16

# Facteur pour la defocalisation
defoc_factor = 2*np.pi

# Minimiation
gtol = 1e-7

# Perturbation
varb = 1e-5


# Inits
w   = wphi*ech
l   = lphi*ech
P = tls.circle(wphi,lphi,r)