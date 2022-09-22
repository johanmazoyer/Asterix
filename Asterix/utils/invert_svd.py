
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def invert_svd(matrix_to_invert, cut, goal="e", regul="truncation", visu=False, filename_visu=None):
    """ --------------------------------------------------
    Invert a matrix after a Singular Value Decomposition
    https://en.wikipedia.org/wiki/Singular_value_decomposition
    The inversion can be regularized. We return the inverse, the singular values, 
    their inverse and the pseudo inverse.

    AUTHOR : Axel Potier

    Parameters
    ----------

    matrix_to_invert : numpy array
                        The matrix to invert
    cut : int 
         threshold to cut the singular values
    goal : string, default 'e'
            if 'e': the cut set the inverse singular value not to exceed
            if 'c': the cut set the number of modes to take into account
                            (keep the lowest inverse singular values)
    regul : string, default 'truncation'
            if 'truncation': when goal is set to 'c', the modes with the highest inverse
                            singular values are truncated
            if 'tikhonov': when goal is set to 'c', the modes with the highest inverse
                            singular values are smoothed (low pass filter)
    visu : boolean, default False
            if True, plot and save the crescent inverse singular values,
                            before regularization

    Returns
    ------
    np.diag(InvS) : 2D numpy array
        Inverse eigenvalues of the input matrix in a diagonal matrix
    np.diag(InvS_truncated) : 2D numpy array 
        Inverse eigenvalues of the input matrix after regularization in a diagonal matrix
    pseudoinverse :  2D numpy array
        Regularized inverse of the input matrix



    -------------------------------------------------- """
    U, s, V = np.linalg.svd(matrix_to_invert, full_matrices=False)
    #print(np.max(np.abs(U @ np.diag(s) @ V)))

    S = np.diag(s)
    InvS = np.linalg.inv(S)
    InvS_truncated = np.linalg.inv(S)
    # print(InvS)
    if visu == True:
        plt.figure()
        plt.clf
        plt.plot(np.diag(InvS), "r.")
        plt.yscale("log")
        plt.savefig(filename_visu)
        plt.close()

    if goal == "e":
        InvS_truncated[np.where(InvS_truncated > cut)] = 0
        pseudoinverse = np.dot(np.dot(np.transpose(V), InvS_truncated), np.transpose(U))

    if goal == "c":
        if regul == "truncation":
            InvS_truncated[cut:] = 0
        if regul == "tikhonov":
            InvS_truncated = np.diag(s / (s**2 + s[cut]**2))
            if visu == True:
                plt.ion()
                plt.plot(np.diag(InvS_truncated), "b.")
                plt.yscale("log")
                plt.show()
                plt.pause(2)
                plt.close()
                plt.ioff()
        pseudoinverse = np.dot(np.dot(np.transpose(V), InvS_truncated), np.transpose(U))

    return [np.diag(InvS), np.diag(InvS_truncated), pseudoinverse]
