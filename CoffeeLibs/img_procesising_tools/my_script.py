# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:38:56 2021

@author: sjuillar
"""

import process_img as do

# %% Set folder names

imgs_de_base  = "20210614\\"
imgs_moyenne  = 'img_moy\\'

dark_de_base  = "Dark\\"
dark_moyenne  = 'dark_moy\\'

imgs_undarked = 'imgs_undarked\\'

crop_size = 150
imgs_crop  = 'img_crop\\'

imgs_norm  = 'imgs\\'
weight=1e10  # Ponderation en plus pour pas avoir des valeur trop extremes

# %% Do stuff

# NB : 
# v1 -> of fits in given folder
# v2 -> of fits in folders in given folder

do.moy_fits_v1(dark_de_base,dark_moyenne)
do.moy_fits_v2(imgs_de_base,imgs_moyenne)

do.minus_dark(imgs_moyenne,dark_moyenne,imgs_undarked)

do.centrage_v1(imgs_undarked,imgs_crop,crop_size)

do.normliz_v1(imgs_crop,imgs_norm,weight)



