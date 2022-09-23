# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os
import numpy as np
from astropy.io import fits
import skimage.transform

from Asterix import model_dir
from Asterix.optics import OpticalSystem
import Asterix.optics.phase_amplitude_functions as phase_ampl
from Asterix.utils import save_plane_in_fits, crop_or_pad_image, rebin


class Pupil(OpticalSystem):
    """ --------------------------------------------------
    initialize and describe the behavior of single pupil
    pupil is a sub class of OpticalSystem.

    Obviously you can define your pupil
    without that with 2d arrray multiplication (this is a fairly simple object).

    The main advantage of defining them using OpticalSystem is that you can
    use default OpticalSystem functions to obtain PSF, transmission, etc...
    and concatenate them with other elements

    AUTHOR : Johan Mazoyer

    
    -------------------------------------------------- """

    def __init__(self, modelconfig, prad=0., PupType=None, angle_rotation=0, Model_local_dir=None):
        """ --------------------------------------------------
        Initialize a pupil object.           
        TODO: include an SCC Lyot pupil function here !
        TODO: for now pupil .fits are monochromatic but the pupil propagation EF_through
            use wavelenght as a parameter

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
                    general configuration parameters (sizes and dimensions)
                        to initialize OpticalSystem class

        prad : float
            Default is the pupil radius in the parameter file (self.prad)
            radius in pixels of the round pupil.

        PupType : string (default None) 
            Currently known possiibilities are 
            "RoundPup", "ClearPlane", "RomanPup", "RomanLyot", "RomanPupTHD2", "RomanLyotTHD2"

            if not one of those , it will try a full path to a fits file given by the user:

            The pupil .fits files should be 2D and square with an even number of pix.
            with even number of pix and centered between 4 pixels.
            The array size will be assumed to correspond to the size of the entrance pupil 
            and will be rescaled self.prad and then padded to self.dim_overpad_pupil

            This is a bit dangerous because your .fits file might must be defined
            the right way so be careful
        
        angle_rotation: float (default 0)
            angle of rotation of the pupil in degrees in counter-clockwise direction.
            this is only used if the pupil is not clear or empty


        Model_local_dir: string, default None
                    directory to save things you can measure yourself
                    and can save to save time

        -------------------------------------------------- """
        # Initialize the OpticalSystem class and inherit properties
        super().__init__(modelconfig)

        if (Model_local_dir is not None) and not os.path.exists(Model_local_dir):
            print("Creating directory " + Model_local_dir + " ...")
            os.makedirs(Model_local_dir)

        if prad == 0:
            prad = self.prad

        # known cases, with known responses
        # default case: round pupil
        if (PupType is None) or (PupType == "RoundPup"):
            self.pup = phase_ampl.roundpupil(self.dim_overpad_pupil, prad)
            angle_rotation = 0
            if isinstance(prad, int) or prad.is_integer():
                self.string_os += '_RoundPup' + str(int(prad))
            else:
                self.string_os += '_RoundPup' + str(round(prad, 1))

        # Clear (in case we want to define an empty pupil plane)
        elif PupType == "Clear":
            self.pup = np.ones((self.dim_overpad_pupil, self.dim_overpad_pupil))
            angle_rotation = 0
            self.string_os += '_ClearPlane'

        else:
            # In those cases, we are using a fits to create the pupil
            # in these first cases, we use a known .fits with hardcoded file name
            if PupType == "RomanPup":
                pup_fits = fits.getdata(os.path.join(model_dir, "roman_pup_500pix_center4pixels.fits"))
                self.string_os += '_RomanPup' + str(int(prad))

            elif PupType == "RomanPupTHD2":
                pup_fits = fits.getdata(os.path.join(model_dir, "roman_pup_thd2_500pix_center4pixels.fits"))
                self.string_os += '_RomanPupTHD2' + str(int(prad))

            elif PupType == "RomanLyot":
                pup_fits = fits.getdata(os.path.join(model_dir, "roman_lyot_500pix_center4pixels.fits"))
                self.string_os += '_RomanLyot'

            elif PupType == "RomanLyotTHD2":
                pup_fits = fits.getdata(os.path.join(model_dir, "roman_lyot_thd2_500pix_center4pixels.fits"))
                self.string_os += '_RomanLyotTHD2'

            # finally in this last case, we use an unknown .fits defined by user
            else:
                if not os.path.exists(PupType):

                    print("""
                            filename_instr_XXX parameters must either be a known keyword 
                            'RoundPup', 'Clear', 'RomanPup', 'RomanLyot' , 'RomanPupTHD2', 'THD2',
                            or an exisiting full path .fits name. This is not the case here,
                            the name  '{0}' is not a known keyword and is not an existing filename
                            """.format(PupType))
                    print("")
                    print("")
                    raise

                # this is an existing fits file
                # we start by a bunch of tests to check
                # that pupil has a certain acceptable form.
                # print("we load the pupil: " + filename)
                # print("we assume it is centered in its array")
                pup_fits = fits.getdata(PupType)

                if len(pup_fits.shape) != 2:
                    raise Exception("file " + PupType + " should be a 2D array")

                if pup_fits.shape[0] != pup_fits.shape[1]:
                    raise Exception("file " + PupType + " appears to be not square")

                self.string_os += '_Fits'

            if angle_rotation != 0:
                pup_fits = skimage.transform.rotate(pup_fits, angle_rotation, preserve_range=True)
                self.string_os += 'Rot' + str(int(angle_rotation))

                fits.writeto(os.path.join(Model_local_dir,
                                          PupType + 'Rot' + str(int(angle_rotation)) + '.fits'),
                             pup_fits,
                             overwrite=True)

            # we have the fits, we now rescale to good size
            if pup_fits.shape[0] == 2 * self.prad:
                pup_fits_right_size = pup_fits
            else:
                #Rescale to the pupil size
                find_divisors = list()
                for i in range(60, pup_fits.shape[0] + 1):
                    if pup_fits.shape[0] % i == 0:
                        find_divisors.append(i)

                if not int(2 * self.prad) in find_divisors:
                    raise Exception(
                        "Choose a divisor of the .fits file size ({0}) for diam_pup_in_pix parameter: {1}".
                        format(pup_fits.shape[0], find_divisors))
                
                pup_fits_right_size = rebin(pup_fits,
                                                     int(pup_fits.shape[0] / (2 * self.prad)),
                                                     center_on_pixel=False)

            self.pup = crop_or_pad_image(pup_fits_right_size, self.dim_overpad_pupil)

        #initialize the max and sum of PSFs for the normalization to contrast
        self.measure_normalization()

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field through the pupil
        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF:    2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
                        Electric field in the pupil plane a the entrance of the system.

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m.

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.

        dir_save_all_planes : default None. 
                                directory to save all plane
                                in fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Electric field in the pupil plane a the exit of the system


        -------------------------------------------------- """

        if save_all_planes_to_fits == True and dir_save_all_planes == None:
            raise Exception("save_all_planes_to_fits = True can generate a lot of .fits files" +
                            "please define a clear directory using dir_save_all_planes kw argument")

        # call the OpticalSystem super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)
        if wavelength is None:
            wavelength = self.wavelength_0

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_pupil' + '_wl{}'.format(int(wavelength * 1e9))
            save_plane_in_fits(dir_save_all_planes, name_plane, entrance_EF)

        if len(self.pup.shape) == 2:
            exit_EF = entrance_EF * self.pup

        elif len(self.pup.shape) == 3:
            if self.pup.shape != self.nb_wav:
                raise Exception("I'm confused, your pupil seem to be polychromatic" +
                                "(pup.shape=3) but the # of WL (pup.shape[0]={}) ".format(self.pup.shape[0]) +
                                "is different from the system # of WL (nb_wav={})".format(self.nb_wav))
            exit_EF = entrance_EF * self.pup[self.wav_vec.tolist().index(wavelength)]
        else:
            raise Exception("pupil dimension are not acceptable")

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_pupil' + '_wl{}'.format(int(wavelength * 1e9))
            save_plane_in_fits(dir_save_all_planes, name_plane, exit_EF)

        return exit_EF