import matplotlib
from IPython import get_ipython
if get_ipython() is None:  # this matplotlib option is just in non-notebook case
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def invert_svd(matrix_to_invert, cut, goal="e", regul="truncation", filename_eigen=None, silence=False):
    """Invert a matrix after a Singular Value Decomposition
    https://en.wikipedia.org/wiki/Singular_value_decomposition The inversion
    can be regularized. We return the inverse, the singular values, their
    inverse and the pseudo inverse.

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
    filename_eigen : string, default None
            if not None, plot and save the crescent inverse singular values,
                            before regularization
    silence : boolean, default False.
        Whether to silence print/show outputs

    Returns
    --------
    np.diag(InvS) : 2D numpy array
        Inverse eigenvalues of the input matrix in a diagonal matrix
    np.diag(InvS_truncated) : 2D numpy array
        Inverse eigenvalues of the input matrix after regularization in a diagonal matrix
    pseudoinverse :  2D numpy array
        Regularized inverse of the input matrix
    """
    U, s, V = np.linalg.svd(matrix_to_invert, full_matrices=False)
    # print(np.max(np.abs(U @ np.diag(s) @ V)))

    S = np.diag(s)
    InvS = np.linalg.inv(S)
    InvS_truncated = np.linalg.inv(S)
    # print(InvS)
    if filename_eigen is not None:
        plt.plot(np.diag(InvS), "r.")
        plt.yscale("log")
        plt.savefig(filename_eigen)
        plt.close()
    if goal == "e":
        InvS_truncated[np.where(InvS_truncated > cut)] = 0
        pseudoinverse = np.dot(np.dot(np.transpose(V), InvS_truncated), np.transpose(U))

    if goal == "c":
        if regul == "truncation":
            InvS_truncated[cut:] = 0
        if regul == "tikhonov":
            InvS_truncated = np.diag(s / (s**2 + s[cut]**2))
            if not silence:
                plt.ion()
                plt.plot(np.diag(InvS_truncated), "b.")
                plt.yscale("log")
                plt.show()
                plt.pause(2)
                plt.close()
                plt.ioff()
        pseudoinverse = np.dot(np.dot(np.transpose(V), InvS_truncated), np.transpose(U))

    return [np.diag(InvS), np.diag(InvS_truncated), pseudoinverse]
