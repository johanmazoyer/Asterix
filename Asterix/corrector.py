# pylint: disable=invalid-name

import os
import numpy as np
from astropy.io import fits

import Asterix.processing_functions as proc
import Asterix.Optical_System_functions as OptSy

import Asterix.WSC_functions as wsc


class Corrector:
    """ --------------------------------------------------
    Corrector Class allows you to define a corrector for one of 2 DM with
    different algorithms.

    Corrector is a class which takes as parameter:
        - the testbed structure
        - the correction parameters

        It must contains 2 functions at least:
        - an initialization (e.g. Jacobian matrix) Corrector.__init__
        The initialization will probaby require previous initialization of
        the testbed and of the estimator

        - an correction function itself with parameters
                - the estimation as a 2D array, potentially 3D for polychromatic correction
        DMVoltage = Corrector.toDM_voltage(estimation)
        It returns the DM Voltage. In all generality, it can one or 2 DMs. Depending on the testbed



    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self,
                 Correctionconfig,
                 testbed,
                 matrix_dir='',
                 save_for_bench=False,
                 realtestbed_dir=''):
        """ --------------------------------------------------
        Initialize the corrector.
        This is where you define the EFC matrix
        For all large files you should use a method of "save to fits" if
        it does not exist "load from fits" if it does, in matrix_dir

        Store in the structure only what you need for estimation. Everything not
        used in self.estimate shoud not be stored

        Parameters
        ----------
        Estimationconfig : general estimation parameters

        testbed : an Optical_System object which describe your testbed


        matrix_dir: path. save all the difficult to measure files here

        save_for_bench. bool default: false
                should we save for the real testbed in realtestbed_dir

        realtestbed_dir: path save all the files the real testbed need to
                            run your code


        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if isinstance(testbed, OptSy.Optical_System) == False:
            raise Exception("testbed must be an Optical_System objet")

        self.correction_algorithm = Correctionconfig["correction_algorithm"].lower()

        if self.correction_algorithm == "efc":
            pass
        else:
            raise Exception("This correction algorithm is not yet implemented")


    def toDM_voltage(self,
                 testbed,
                 estimate,
                 **kwargs):
        """ --------------------------------------------------
        Run an correction from a testbed, and return the DM voltage for one or 2 DMS

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if self.correction_algorithm == "efc":
            pass
        else:
            raise Exception("This correction algorithm is not yet implemented")
