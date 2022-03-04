import os
import numpy as np

from astropy.io import fits
import Asterix.WSC_functions as wsc



matrix_directory = '/Users/jmazoyer/GitProjects/my_projects/Asterix/Interaction_Matrices/'

Gmatrix_name = 'DirectMatrixSP_EFCampl17.0_DM3_z0_Nact1024_actuatorBasis656_dimEstim158_dimPP332_wl783_resFP7.6_dimFP400_RoundPup83_Apod_ClearPlane_vortex_charge2_LS_RoundPup81.fits'

Nbmodes = 330
regularization='tikhonov' # 'truncation' or 'tikhonov'

name_active_DM = ['DM3'], 

def quick_thd_invert(Nbmodes,Gmatrix_name,name_active_DM, matrix_directory,regularization):
    
    Gmatrix = fits.getdata( os.path.join(matrix_directory , Gmatrix_name))


    _, _, invertGDH = wsc.invertSVD(Gmatrix,
                                    Nbmodes,
                                    goal="c",
                                    regul=regularization,
                                    visu=True,
                                    filename_visu=matrix_directory +
                                    "SVD_Modes" + str(Nbmodes) +
                                    ".png")

    if 'DM1' in name_active_DM:
        invertGDH_DM1 = invertGDH[:testbed.DM1.basis_size]

        EFCmatrix_DM1 = np.transpose(
            np.dot(np.transpose(testbed.DM1.basis), invertGDH_DM1))
        fits.writeto(matrix_directory + "Matrix_control_EFC_DM1.fits",
                        EFCmatrix_DM1.astype(np.float32),
                        overwrite=True)
        if 'DM3' in name_active_DM:
            invertGDH_DM3 = invertGDH[testbed.DM1.basis_size:]
            EFCmatrix_DM3 = np.transpose(
                np.dot(np.transpose(testbed.DM3.basis), invertGDH_DM3))
            fits.writeto(matrix_directory +
                            "Matrix_control_EFC_DM3.fits",
                            EFCmatrix_DM3.astype(np.float32),
                            overwrite=True)
    if 'DM3' in name_active_DM:
        invertGDH_DM3 = invertGDH
        EFCmatrix_DM3 = np.transpose(
            np.dot(np.transpose(testbed.DM3.basis), invertGDH_DM3))
        fits.writeto(matrix_directory + "Matrix_control_EFC_DM3.fits",
                        EFCmatrix_DM3.astype(np.float32),
                        overwrite=True)
    else:
        raise Exception("No active DMs")

def quick_thd_mask(Nbmodes,Gmatrix_name,matrix_directory,regularization):


    fits.writeto(realtestbed_dir + "DH_mask.fits",
                    self.MaskEstim.astype(np.float32),
                    overwrite=True)
    fits.writeto(realtestbed_dir + "DH_mask_where_x_y.fits",
                    np.array(np.where(self.MaskEstim == 1)).astype(
                        np.float32),
                    overwrite=True)
