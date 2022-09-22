# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import sys
import os
import time
import numpy as np

from astropy.io import fits
from .wf_control_functions import invert_svd

def THD_quick_invert(Nbmodes, name_active_DM, matrix_directory, regularization):
    """ --------------------------------------------------
        This code invert the matrix just in the case of THD testbed
        The goal is to be able to invert the matrix directly on the RTC to be 
        able to do it during correction

        Need the following fits (automatically created in corrector.py is onbench = True): 
            if DM1 only is active: 
                - Base_Matrix_DM1.fits  
                - Direct_Matrix_DM1only.fits
            if DM3 only is active: 
                -Base_Matrix_DM3.fits 
                - Direct_Matrix_DM3only.fits
            if both DM1 and DM3 are active: 
                - Base_Matrix_DM1.fits 
                - Base_Matrix_DM1.fits 
                - Direct_Matrix_2DM.fits
        
        Save the following fits (that can be read by the testbed): 
            if DM1 only is active: 
                - Matrix_control_EFC_DM1.fits  
            if DM3 only is active: 
                -Matrix_control_EFC_DM3.fits 
            if both DM1 and DM3 are active: 
                - Matrix_control_EFC_DM1.fits 
                - Matrix_control_EFC_DM3.fits 

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        
        Nbmodes: int
            threshold to cut the singular values
        
        name_active_DM : int
            Simple code to identify which DMs are active
            1, 3 or 13 depending on the DM you want to access

        matrix_directory : str
                        Directory where Direct matrices are read and Inverse matrices are saved
                        Careful Windows and MacOS do not write path the same way
        
        regularization : string, default 'truncation'
            if 'truncation': when goal is set to 'c', the modes with the highest inverse
                            singular values are truncated
            if 'tikhonov': when goal is set to 'c', the modes with the highest inverse
                            singular values are smoothed (low pass filter)

        Returns
        ------
        None
        
        
        -------------------------------------------------- """

    if name_active_DM in (13, 31):
        Gmatrix = fits.getdata(os.path.join(matrix_directory, "Direct_Matrix_2DM.fits"))
        DM1_basis = fits.getdata(os.path.join(matrix_directory, "Base_Matrix_DM1.fits"))
        DM3_basis = fits.getdata(os.path.join(matrix_directory, "Base_Matrix_DM3.fits"))

    elif name_active_DM == 1:
        Gmatrix = fits.getdata(os.path.join(matrix_directory, "Direct_Matrix_DM1only.fits"))
        DM1_basis = fits.getdata(os.path.join(matrix_directory, "Base_Matrix_DM1.fits"))

    elif name_active_DM == 3:
        Gmatrix = fits.getdata(os.path.join(matrix_directory, "Direct_Matrix_DM3only.fits"))
        DM3_basis = fits.getdata(os.path.join(matrix_directory, "Base_Matrix_DM3.fits"))

    else:
        raise Exception("No active DMs")

    _, _, invertGDH = invert_svd(Gmatrix,
                                     Nbmodes,
                                     goal="c",
                                     regul=regularization,
                                     visu=True,
                                     filename_visu=os.path.join(matrix_directory,
                                                               "SVD_Modes" + str(Nbmodes) + ".png"))

    if name_active_DM in (13, 31):
        DM1_basis_size = DM1_basis.shape[0]

        invertGDH_DM1 = invertGDH[:DM1_basis_size]
        EFCmatrix_DM1 = np.transpose(np.dot(np.transpose(DM1_basis), invertGDH_DM1))
        fits.writeto(os.path.join(matrix_directory, "Matrix_control_EFC_DM1.fits"),
                     EFCmatrix_DM1.astype(np.float32),
                     overwrite=True)

        invertGDH_DM3 = invertGDH[DM1_basis_size:]
        EFCmatrix_DM3 = np.transpose(np.dot(np.transpose(DM3_basis), invertGDH_DM3))
        fits.writeto(os.path.join(matrix_directory, "Matrix_control_EFC_DM3.fits"),
                     EFCmatrix_DM3.astype(np.float32),
                     overwrite=True)
    elif name_active_DM == 1:
        invertGDH_DM1 = invertGDH
        EFCmatrix_DM1 = np.transpose(np.dot(np.transpose(DM1_basis), invertGDH_DM1))
        fits.writeto(os.path.join(matrix_directory, "Matrix_control_EFC_DM1.fits"),
                     EFCmatrix_DM1.astype(np.float32),
                     overwrite=True)
    elif name_active_DM == 3:
        invertGDH_DM3 = invertGDH
        EFCmatrix_DM3 = np.transpose(np.dot(np.transpose(DM3_basis), invertGDH_DM3))
        fits.writeto(os.path.join(matrix_directory, "Matrix_control_EFC_DM3.fits"),
                     EFCmatrix_DM3.astype(np.float32),
                     overwrite=True)
    else:
        raise Exception("No active DMs")


if __name__ == '__main__':

    # matrix_directory = '/Users/jmazoyer/GitProjects/my_projects/Asterix/Labview/'
    matrix_directory = r'C:\Users\LESIA-BAT\Desktop\Labview_IDL_routines\Labview_routines\ITHD_v40\DATA\EFC'
    regularization = 'tikhonov'  # 'truncation' or 'tikhonov'

    if len(sys.argv) == 1:
        # we run this code without any argument
        Nbmodes = 610
        name_active_DM = 13  # 1, 3 or 13 depending on the DM you want to access

    else:
        # We run this code with argument. Example
        # python THD_quick_invert.py 650 13
        Nbmodes = int(sys.argv[1])  # number of mode in the inversion
        name_active_DM = int(sys.argv[2])  # 1, 3 or 13 depending on the DM you want to access

    start_time = time.time()
    THD_quick_invert(Nbmodes, name_active_DM, matrix_directory, regularization)
    print("time to invert matrix ", round(time.time() - start_time, 2))
