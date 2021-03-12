import numpy as np

import Asterix.propagation_functions as prop
import Asterix.phase_amplitude_functions as phafun
import Asterix.fits_functions as useful
import Asterix.processing_functions as proc

dim_pup = 200
prad = 100

pup = phafun.roundpupil(dim_pup,prad)
print(np.mean(pup[np.where(pup != 0)])) 


useful.quickfits(pup)

psf = prop.mft(pup, 2*prad, 2000, 400, inv= -1 )

pup_back = np.abs(prop.mft( psf, 2000,2*prad, 400, inv = 1))
useful.quickfits(np.abs(pup_back))

print(np.mean(pup_back[np.where(pup_back != 0)])) 


# puppad = proc.crop_or_pad_image(pup, 4* dim_pup)
# psf =  proc.crop_or_pad_image(np.abs(np.fft.fftshift(np.fft.fft2(
#                 np.fft.fftshift(puppad))))**2, 1000)
# useful.quickfits(psf)




