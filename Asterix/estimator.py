import os
import datetime
import numpy as np
import scipy.ndimage as nd
from astropy.io import fits
import skimage.transform

import Asterix.propagation_functions as prop
import Asterix.processing_functions as proc
import Asterix.phase_amplitude_functions as phase_ampl
import Asterix.fits_functions as useful
import Asterix.Optical_System_functions as OptSy


##############################################
##############################################
### Optical_System
class Estimator:
    """ --------------------------------------------------
    Super Class Estimator allows you to define a WF estimator. it takes as parameter:
                - the testbed structure
                - the estimation parameters

        It must contains 2 functions at least:
        - an initialization (e.g. PW matrix) Estimator.__init__()
        The initialization will probaby require previous initialization of
        the testbed

        - an estimatation function itself with parameters
                - the entrance EF
                - DM voltage
                - the WL
        Estimation = Estimator.estimate(entrance EF, DM voltage, WL)

        It returns the estimation as a 2D array. In all generality, it can be pupil or focal plane,
        complex or real with keywords (Estim.is_focal_plane = True, Estim.is_complex = True)
        to explain the form of the output and potentially prevent wronfull combination of
        estim + correc.


    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self, Estimationconfig, testbed):
        """ --------------------------------------------------
        Initialize the estimator.


        Parameters
        ----------
        Estimationconfig : general estimation parameters
        testbed : an Optical_System object which describe your testbed

        Load or save

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if isinstance(testbed, OptSy.Optical_System) == False:
            raise Exception("testbed must be an Optical_System objet")


        self.technique = Estimationconfig["estimation"]

        if self.technique == "Perfect":
            self.is_focal_plane = None
            self.is_complex = None

        # this is where you define the pw matrix, the modified Lyot stop
        # or the COFFEE gradiant.

    def estimation(self, testbed, entrance_EF = 0., DM1phase = 0., DM3phase = 0., wavelength = None,  **kwargs):

        if wavelength is None:
            wavelength = testbed.wavelength_0

        if self.technique == "Perfect":
            return testbed.todetector(entrance_EF=entrance_EF, DM1phase = DM1phase, DM3phase = DM3phase)

        return 0