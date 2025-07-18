import sys
import os
import time
import numpy as np
from astropy.io import fits

from Asterix.utils import invert_svd


def THD_quick_invert(Nbmodes, name_active_DM, matrix_directory, regularization, number_wl_in_matrix=1, silence=False):
    """This code invert the matrix just in the case of THD testbed.

    The goal is to be able to invert the matrix directly on the RTC to be able to do it during correction.
    Needs the following fits files (automatically created in corrector.py if onbench == True):
        if DM1 only is active:
            - Base_Matrix_DM1.fits
            - Direct_Matrix_DM1only.fits
        if DM2 only is active:
            -Base_Matrix_DM2.fits
            - Direct_Matrix_DM2only.fits
        if both DM1 and DM2 are active:
            - Base_Matrix_DM1.fits
            - Base_Matrix_DM1.fits
            - Direct_Matrix_2DM.fits

    Save the following fits files (that can be read by the testbed RTC):
        if DM1 only is active:
            - Matrix_control_EFC_DM1.fits
        if DM2 only is active:
            -Matrix_control_EFC_DM2.fits
        if both DM1 and DM2 are active:
            - Matrix_control_EFC_DM1.fits
            - Matrix_control_EFC_DM2.fits

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
    number_wl_in_matrix : int, default 1
        number of wavelength in the direct matrix
    silence : boolean, default False.
        Whether to silence print outputs.
    """

    if name_active_DM in (13, 31):
        Gmatrix = fits.getdata(os.path.join(matrix_directory, f"Direct_Matrix_2DM_{number_wl_in_matrix}wl.fits"))
        DM1_basis = fits.getdata(os.path.join(matrix_directory, "Base_Matrix_DM1.fits"))
        DM2_basis = fits.getdata(os.path.join(matrix_directory, "Base_Matrix_DM2.fits"))
        headerdm1 = fits.getheader(os.path.join(matrix_directory, "Base_Matrix_DM1.fits"))
        headerdm2 = fits.getheader(os.path.join(matrix_directory, "Base_Matrix_DM2.fits"))

    elif name_active_DM == 1:
        Gmatrix = fits.getdata(os.path.join(matrix_directory, f"Direct_Matrix_DM1only_{number_wl_in_matrix}wl.fits"))
        DM1_basis = fits.getdata(os.path.join(matrix_directory, "Base_Matrix_DM1.fits"))
        headerdm1 = fits.getheader(os.path.join(matrix_directory, "Base_Matrix_DM1.fits"))

    elif name_active_DM == 3:
        Gmatrix = fits.getdata(os.path.join(matrix_directory, f"Direct_Matrix_DM2only_{number_wl_in_matrix}wl.fits"))
        DM2_basis = fits.getdata(os.path.join(matrix_directory, "Base_Matrix_DM2.fits"))
        headerdm2 = fits.getheader(os.path.join(matrix_directory, "Base_Matrix_DM2.fits"))

    else:
        raise ValueError("No active DMs")

    _, _, invertGDH = invert_svd(Gmatrix,
                                 Nbmodes,
                                 goal="c",
                                 regul=regularization,
                                 filename_eigen=os.path.join(matrix_directory, "SVD_Modes" + str(Nbmodes) + ".png"),
                                 silence=silence)

    if name_active_DM in (13, 31):
        DM1_basis_size = DM1_basis.shape[0]

        invertGDH_DM1 = invertGDH[:DM1_basis_size]
        EFCmatrix_DM1 = np.transpose(np.dot(np.transpose(DM1_basis), invertGDH_DM1))
        fits.writeto(os.path.join(matrix_directory, "Matrix_control_EFC_DM1.fits"),
                     EFCmatrix_DM1.astype(np.float32),
                     headerdm1,
                     overwrite=True)

        invertGDH_DM2 = invertGDH[DM1_basis_size:]
        EFCmatrix_DM2 = np.transpose(np.dot(np.transpose(DM2_basis), invertGDH_DM2))
        fits.writeto(os.path.join(matrix_directory, "Matrix_control_EFC_DM2.fits"),
                     EFCmatrix_DM2.astype(np.float32),
                     headerdm2,
                     overwrite=True)
    elif name_active_DM == 1:
        invertGDH_DM1 = invertGDH
        EFCmatrix_DM1 = np.transpose(np.dot(np.transpose(DM1_basis), invertGDH_DM1))
        fits.writeto(os.path.join(matrix_directory, "Matrix_control_EFC_DM1.fits"),
                     EFCmatrix_DM1.astype(np.float32),
                     headerdm1,
                     overwrite=True)
    elif name_active_DM == 3:
        invertGDH_DM2 = invertGDH
        EFCmatrix_DM2 = np.transpose(np.dot(np.transpose(DM2_basis), invertGDH_DM2))
        fits.writeto(os.path.join(matrix_directory, "Matrix_control_EFC_DM2.fits"),
                     EFCmatrix_DM2.astype(np.float32),
                     headerdm2,
                     overwrite=True)
    else:
        raise ValueError("No active DMs")


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
