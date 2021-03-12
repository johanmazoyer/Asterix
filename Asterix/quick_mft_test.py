import numpy as np

import Asterix.propagation_functions as prop
import Asterix.phase_amplitude_functions as phafun
import Asterix.fits_functions as useful
import Asterix.processing_functions as proc

dim_pup = 200
prad = 100

pup = phafun.roundpupil(dim_pup,prad)
print(np.mean(pup[np.where(pup != 0)])) 


useful.quickfits(pup,name='pup')

efmft = prop.mft(pup, 2*prad, 4* dim_pup, dim_pup, inverse=False, X_offset_output = 1/2, Y_offset_output = 1/2 )
psfmft = np.abs(efmft)**2

useful.quickfits(np.abs(psfmft),name='psf_mft')

pup_back = prop.mft( efmft, 4* dim_pup,2*prad, dim_pup, inverse=True, X_offset_input = 1/2, Y_offset_input = 1/2 )
useful.quickfits(np.angle(pup_back),name='pupback_mft_phase')
useful.quickfits(np.real(pup_back),name='pupback_mft')

print(np.mean(pup_back[np.where(pup_back != 0)])) 


puppad = proc.crop_or_pad_image(pup, 4* dim_pup)
efffft =  np.fft.fftshift(np.fft.fft2(
                np.fft.fftshift(puppad)))
psffft = np.abs(efffft)**2
useful.quickfits(psffft, name='psf_fft')

pup_back = proc.crop_or_pad_image(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(
                np.fft.fftshift(puppad)))), dim_pup)



useful.quickfits(np.real(pup_back),  name='pupback_fft')






