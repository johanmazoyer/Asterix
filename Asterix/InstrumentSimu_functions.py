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


##############################################
##############################################
### Optical_System
class Optical_System:
    """ --------------------------------------------------
    Super class Optical_System allows to pass parameters to all sub class. We can then creat blocks inside this super class
    An Optical_System start and end in the pupil plane. 
    The entrance and exit pupil plane must always of the same size (dim_overpad_pupil)    
    With these convention, they can be easily assemble to create complex optical systems. 
    
    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self, modelconfig):
        """ --------------------------------------------------
        Initialize Optical_System objects 
        
        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        #pupil in pixel
        self.prad = int(modelconfig["diam_pup_in_pix"] / 2)

        # 1.25 is hard coded for now. TODO Fix that.
        # All pupils in the code must have this dimensions, so that the blocks can be easily switch
        self.dim_overpad_pupil = int(self.prad * 1.25) * 2

        #Lambda over D in pixels in the focal plane
        # at the reference wavelength
        #image size and resolution in detector
        self.dim_im = modelconfig["dim_im"]
        self.science_sampling = modelconfig["science_sampling"]

        #pupil in meters
        self.diam_pup_in_m = modelconfig["diam_pup_in_m"]

        # Exit pupil radius
        self.exitpup_rad = self.prad
        # this is the exit pupil radius, that is used to define the L/D
        # in self.todetector function.
        # by default this is the entrance pupil rad. of course, this can be changed

        # wavelength
        self.Delta_wav = modelconfig["Delta_wav"]
        self.nb_wav = modelconfig["nb_wav"]
        self.wavelength_0 = modelconfig["wavelength_0"]

        if self.Delta_wav != 0:
            if (self.nb_wav % 2 == 0) or self.nb_wav < 2:
                raise Exception(
                    "please set nb_wav parameter to an odd number > 1")

            self.wav_vec = np.linspace(self.wavelength_0 - self.Delta_wav / 2,
                                       self.wavelength_0 + self.Delta_wav / 2,
                                       num=self.nb_wav,
                                       endpoint=True)
        else:
            self.wav_vec = np.array([self.wavelength_0])

    #We define functions that all Optical_System object can use.
    # These can be overwritten for a subclass if need be

    def EF_through(self,
                   entrance_EF=1.,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance pupil to exit pupil

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in fits if save_all_planes_to_fits = True
            

        **kwargs: other kw parameters can be passed for other Optical_System objects EF_trough function

        NEED TO BE DEFINED FOR ALL Optical_System

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if isinstance(entrance_EF, float) or isinstance(entrance_EF, np.float):
            entrance_EF = np.full(
                (self.dim_overpad_pupil, self.dim_overpad_pupil),
                np.float(entrance_EF))

        if isinstance(entrance_EF, np.ndarray) == False:
            raise Exception(
                "entrance_EF should be a float of a numpy array of floats")

        if save_all_planes_to_fits == True and dir_save_all_planes == None:
            raise Exception(
                "save_all_planes_to_fits = True can generate a lot of .fits files"
                +
                "please define a clear directory using dir_save_all_planes kw argument"
            )

        exit_EF = entrance_EF
        return exit_EF

    def todetector(self,
                   entrance_EF=1.,
                   wavelength=None,
                   center_on_pixel=False,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance plane through the system and then
        to Science focal plane.
        TODO can be made polychromatic but maybe not because people 
        will do bad things with it like the intensity of summed EF :-)
        
        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 

        center_on_pixel : bool Default False
            If True, the PSF will be centered on a pixel
            If False, the PSF will be centered between 4 pixels
        This of course assume that no tip-tilt have been introduced in the entrance_EF 
        or during self.EF_through

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in fits if save_all_planes_to_fits = True

        **kwargs: other kw parameters can be passed direclty to self.EF_through function

        Returns
        ------
        ef_focal_plane : 2D array of size [self.dim_im, self.dim_im]
        Electric field in the focal plane.
            the lambda / D is defined such as self.wavelength_0 /  (2*self.exitpup_rad) = self.science_sampling pixels
        
        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """
        if center_on_pixel == True:
            Psf_offset = (0, 0)
        else:
            Psf_offset = (-0.5, -0.5)

        if wavelength == None:
            wavelength = self.wavelength_0

        lambda_ratio = wavelength / self.wavelength_0

        exit_EF = self.EF_through(
            entrance_EF=entrance_EF,
            wavelength=wavelength,
            save_all_planes_to_fits=save_all_planes_to_fits,
            dir_save_all_planes=dir_save_all_planes,
            **kwargs)

        focal_plane_EF = prop.mft(exit_EF,
                                  self.exitpup_rad * 2,
                                  self.dim_im,
                                  self.dim_im / self.science_sampling *
                                  lambda_ratio,
                                  X_offset_output=Psf_offset[0],
                                  Y_offset_output=Psf_offset[1],
                                  inverse=False)

        if save_all_planes_to_fits == True:
            who_called_me = self.__class__.__name__
            name_plane = 'EF_FP_after_' + who_called_me + '_obj' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      focal_plane_EF)

        return focal_plane_EF

    def todetector_Intensity(self,
                             entrance_EF=1.,
                             wavelengths=None,
                             center_on_pixel=False,
                             save_all_planes_to_fits=False,
                             dir_save_all_planes=None,
                             **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance plane through the system, then
        to Science focal plane and measure intensity

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelengths : float or float array of wavelength in m. Default is all wavelenthg in self.wav_vec

        center_on_pixel : bool Default False
            If True, the PSF will be centered on a pixel
            If False, the PSF will be centered between 4 pixels
        This of course assume that no tip-tilt have been introduced in the entrance_EF 
        or during self.EF_through

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in fits if save_all_planes_to_fits = True

        **kwargs: other kw parameters can be passed direclty to self.EF_through function

        Returns
        ------
        focal_plane_Intensity : 2D array of size [self.dim_im, self.dim_im]
        Intensity in the focal plane.
            the lambda / D is defined such as self.wavelength_0 /  (2*self.exitpup_rad) = self.science_sampling pixels
        
        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if wavelengths == None:
            wavelength_vec = self.wav_vec

        elif isinstance(wavelengths, float) or isinstance(
                wavelengths, np.float):
            wavelength_vec = [wavelengths]
        else:
            wavelength_vec = wavelengths
        #TODO check if wavelengths have good format float array and raise
        # exception if not

        focal_plane_Intensity = np.zeros((self.dim_im, self.dim_im))

        for wav in wavelength_vec:
            focal_plane_Intensity += np.abs(
                self.todetector(
                    entrance_EF=entrance_EF,
                    wavelength=wav,
                    center_on_pixel=center_on_pixel,
                    save_all_planes_to_fits=save_all_planes_to_fits,
                    dir_save_all_planes=dir_save_all_planes,
                    **kwargs))**2

        if save_all_planes_to_fits == True:
            who_called_me = self.__class__.__name__
            name_plane = 'Int_FP_after_' + who_called_me + '_obj'
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      focal_plane_Intensity)

        return focal_plane_Intensity

    def transmission(self, **kwargs):
        """
        measure ratio of photons lost when
        crossing the system compared to a clear aperture of radius self.prad

        **kwargs: other kw parameters can be passed direclty to self.EF_through function
        
        Returns
        ------ 
        float is the ratio exit flux  / clear entrance pupil flux
    
        AUTHOR : Johan Mazoyer

        """
        clear_entrance_pupil = phase_ampl.roundpupil(self.dim_overpad_pupil,
                                                     self.prad)

        exit_EF = self.EF_through(**kwargs)

        throughput_loss = np.sum(
            np.abs(exit_EF)) / np.sum(clear_entrance_pupil)

        return throughput_loss

    def EF_from_phase_and_ampl(self,
                               phase_abb=0.,
                               ampl_abb=0.,
                               wavelength=None):
        """ --------------------------------------------------
        Create an electrical field from an phase and amplitude aberrations
        
        Parameters
        ----------
        phase_abb : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]. real
            if 0, no phase aberration (default)  
        
        phase_abb : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]. real
            if 0, no amplitude aberration (default)  
            

        wavelength : float. Default is the reference self.wavelength_0
             current wavelength in m. 


        Returns
        ------
        EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system
        
        AUTHOR : Johan Mazoyer
        """
        if np.iscomplexobj(phase_abb) or np.iscomplexobj(ampl_abb):
            raise Exception(
                "phase_abb and ampl_abb must be real arrays or float, not complex"
            )

        if (phase_abb == 0.).all() and (ampl_abb == 0).all():
            return 1.

        if wavelength == None:
            wavelength = self.wavelength_0
        lambda_ratio = wavelength / self.wavelength_0

        return (1 + ampl_abb) * np.exp(1j * phase_abb / lambda_ratio)


##############################################
##############################################
### PUPIL
class pupil(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of single pupil
    pupil is a sub class of Optical_System.  
    
    Obviously you can define your pupil
    withot that with 2d arrray multiplication (this is a fairly simple object).
    
    The main advantage of defining them using Optical_System is that you can 
    use default Optical_System functions to obtain PSF, transmission, etc... 
    and concatenate them with other elements
    
    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self,
                 modelconfig,
                 prad=0.,
                 model_dir="",
                 filename="",
                 noPup=False):
        """ --------------------------------------------------
        Initialize a pupil object.


        
        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions) 
                        to initialize Optical_System class

        prad : int Default is the pupil prad in the parameter
            radius in pixels of the round pupil.
        
        directory : string (default "")
            name of the directory where filename is

        filename : string (default "")
            name of the .fits file
            The pupil .fits files should be be 2D and square([dim_fits,dim_fits]) 
            and assumed to be centered between 4 pixels.
            if dim_fits < dim_overpad_pupil then the pupil is zero-padded
            if dim_fits > dim_overpad_pupil we raise an Exception

            TODO: include here function random_phase_map, scale_amplitude_abb, shift_phase_ramp 
            TODO: include an SCC Lyot pupil function here !
            TODO: for now pupil .fits are monochromatic but the pupil propagation EF_through 
            use wavelenght as a parameter

        
        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """
        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)
        if prad == 0:
            prad = self.prad

        self.exitpup_rad = prad

        self.pup = np.full((self.dim_overpad_pupil, self.dim_overpad_pupil),
                           1.)

        if filename != "":

            # we start by a bunch of tests to check
            # that pupil has a certain acceptable form.
            print("we load the pupil: " + filename)
            print("we assume it is centered in its array")
            pup_fits = fits.getdata(os.path.join(model_dir, filename))

            if len(pup_fits.shape) != 2:
                raise Exception("file " + filename + " should be a 2D array")

            if pup_fits.shape[0] != pup_fits.shape[1]:
                raise Exception("file " + filename +
                                " appears to be not square")

            # this assume that the pupil file is squared
            # and is centered in the file

            if pup_fits.shape[0] == self.prad:
                pup_fits_right_size = pup_fits
            else:
                #Rescale to the pupil size
                pup_fits_right_size = skimage.transform.rescale(
                    pup_fits,
                    2 * prad / pup_fits.shape[0],
                    preserve_range=True,
                    anti_aliasing=True,
                    multichannel=False)

            self.pup = proc.crop_or_pad_image(pup_fits_right_size,
                                              self.dim_overpad_pupil)

        else:  # no filename
            if noPup == False:
                self.pup = phase_ampl.roundpupil(self.dim_overpad_pupil, prad)

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None):
        """ --------------------------------------------------
        Propagate the electric field through the pupil

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 
        
        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # call the Optical_System super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_pupil' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)

        if wavelength == None:
            wavelength = self.wavelength_0

        if len(self.pup.shape) == 2:
            exit_EF = entrance_EF * self.pup

        elif len(self.pup.shape) == 3:
            if self.pup.shape != self.nb_wav:
                raise Exception(
                    "I'm confused, your pupil seem to be polychromatic" +
                    "(pup.shape=3) but the # of WL (pup.shape[0]={}) ".format(
                        self.pup.shape[0]) +
                    "is different from the system # of WL (nb_wav={})".format(
                        self.nb_wav))
            else:
                exit_EF = entrance_EF * self.pup[self.wav_vec.tolist().index(
                    wavelength)]
        else:
            raise Exception("pupil dimension are not acceptable")

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_pupil' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)

        return exit_EF

    def random_phase_map(self, phaserms, rhoc, slope):
        """ --------------------------------------------------
        Create a random phase map, whose PSD decrease in f^(-slope)
        average is null and stadard deviation is phaserms
        
        Parameters
        ----------
        phaserms : float
            standard deviation of aberration
        rhoc : float
            See Borde et Traub 2006
        slope : float
            Slope of the PSD
        pupil : 2D array
            

        Returns
        ------
        phase : 2D array
            Static random phase map (or OPD) generated 
        -------------------------------------------------- """

        # create a circular pupil of the same radius of the given pupil
        # this will be the pupil over which phase rms = phaserms
        pup = phase_ampl.roundpupil(self.prad, self.dim_overpad_pupil)

        xx, yy = np.meshgrid(
            np.arange(self.dim_overpad_pupil) - self.dim_overpad_pupil / 2,
            np.arange(self.dim_overpad_pupil) - self.dim_overpad_pupil / 2)
        rho = np.hypot(yy, xx)
        PSD0 = 1
        PSD = PSD0 / (1 + (rho / rhoc)**slope)
        sqrtPSD = np.sqrt(2 * PSD)

        randomphase = np.random.randn(
            self.dim_overpad_pupil,
            self.dim_overpad_pupil) + 1j * np.random.randn(
                self.dim_overpad_pupil, self.dim_overpad_pupil)
        phase = np.real(np.fft.ifft2(np.fft.fftshift(sqrtPSD * randomphase)))
        phase = phase - np.mean(phase[np.where(pup == 1.)])
        phase = phase / np.std(phase[np.where(pup == 1.)]) * phaserms
        return phase


##############################################
##############################################
### CORONAGRAPHS
class coronagraph(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of a coronagraph system (from apod plane to the Lyot plane)
    coronagraph is a sub class of Optical_System.  
    

    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self, modelconfig, coroconfig, model_dir):
        """ --------------------------------------------------
        Initialize a coronograph object
        
        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)
        coroconfig : coronagraph parameters
        model_dir : if needed, we load the mask in this directory
        
        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        #pupil and Lyot stop in m
        self.diam_lyot_in_m = coroconfig["diam_lyot_in_m"]

        #Lyot stop in pixel
        self.lyotrad = int(self.prad * self.diam_lyot_in_m /
                           self.diam_pup_in_m)

        self.exitpup_rad = self.lyotrad

        #coronagraph
        self.corona_type = coroconfig["corona_type"].lower()

        if self.corona_type == "fqpm" or self.corona_type == "knife":
            # dim_fp_fft definition only use if prop_apod2lyot == 'fft'
            self.dim_fp_fft = np.zeros(len(self.wav_vec), dtype=np.int)
            for i, wav in enumerate(self.wav_vec):
                self.dim_fp_fft[i] = int(
                    np.ceil(self.prad * self.science_sampling *
                            self.diam_lyot_in_m / self.diam_pup_in_m *
                            self.wavelength_0 / wav)) * 2
                # we take the ceil to be sure that we measure at least the good resolution
                # We do not need to be exact, the mft in science_focal_plane will be

        if self.corona_type == "fqpm":
            self.prop_apod2lyot = 'mft'
            self.err_fqpm = coroconfig["err_fqpm"]
            self.achrom_fqpm = coroconfig["achrom_fqpm"]
            self.FPmsk = self.FQPM()
            self.perfect_coro = True

        elif self.corona_type == "classiclyot":
            self.prop_apod2lyot = 'mft-babinet'
            self.Lyot_fpm_sampling = 30  # hard coded for now, this is very internal cooking
            self.rad_lyot_fpm = coroconfig["rad_lyot_fpm"]
            self.FPmsk = self.ClassicalLyot()
            self.perfect_coro = False

        elif self.corona_type == "knife":
            self.prop_apod2lyot = 'fft'
            self.coro_position = coroconfig["coro_position"].lower()
            self.knife_coro_offset = coroconfig["knife_coro_offset"]
            self.FPmsk = self.KnifeEdgeCoro()
            self.perfect_coro = False

        elif self.corona_type == "vortex":
            self.prop_apod2lyot = 'fft'
            phasevortex = 0  # to be defined
            self.FPmsk = np.exp(1j * phasevortex)
            self.perfect_coro = True

        # We need a pupil only to measure the response
        # of the coronograph to a clear pupil to remove it
        # if perfect corono. THIS IS NOT THE ENTRANCE PUPIL,
        # this is a clear pupil of the same size
        self.clearpup = pupil(modelconfig, prad=self.prad)

        # Plane at the entrance of the coronagraph. In THD2, this is an empty plane.
        # In Roman this is where is the apodiser
        self.apod_pup = pupil(modelconfig,
                              prad=self.prad,
                              model_dir=model_dir,
                              filename=coroconfig["filename_instr_apod"],
                              noPup=True)

        self.lyot_pup = pupil(modelconfig,
                              prad=self.lyotrad,
                              model_dir=model_dir,
                              filename=coroconfig["filename_instr_lyot"])

        if self.perfect_coro == True:
            # do a propagation once with self.perfect_Lyot_pupil = 0 to
            # measure the Lyot pupil that will be removed after
            self.perfect_Lyot_pupil = 0
            self.perfect_Lyot_pupil = self.EF_through(
                entrance_EF=self.clearpup.EF_through())

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   noFPM=False,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None):
        """ --------------------------------------------------
        Propagate the electric field from apod plane before the apod
        pupil to Lyot plane after Lyot pupil

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 
        
        noFPM : bool (default: False)
            if True, remove the FPM if one want to measure a un-obstructed PSF

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer 
        -------------------------------------------------- """

        # call the Optical_System super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_apod' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)

        if wavelength == None:
            wavelength = self.wavelength_0

        if noFPM:
            FPmsk = 1.
        else:
            FPmsk = self.FPmsk[self.wav_vec.tolist().index(wavelength)]

        lambda_ratio = wavelength / self.wavelength_0

        input_wavefront_after_apod = self.apod_pup.EF_through(
            entrance_EF=entrance_EF, wavelength=wavelength)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_apod' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      input_wavefront_after_apod)

        if self.prop_apod2lyot == "fft":
            dim_fp_fft_here = self.dim_fp_fft[self.wav_vec.tolist().index(
                wavelength)]
            input_wavefront_pad = proc.crop_or_pad_image(
                entrance_EF, dim_fp_fft_here)
            # Phase ramp to center focal plane between 4 pixels

            maskshifthalfpix = phase_ampl.shift_phase_ramp(
                dim_fp_fft_here, 0.5, 0.5)
            maskshifthalfpix_invert = phase_ampl.shift_phase_ramp(
                dim_fp_fft_here, -0.5, -0.5)

            #Apod plane to focal plane
            corono_focal_plane = np.fft.fft2(
                np.fft.fftshift(input_wavefront_pad * maskshifthalfpix))

            if save_all_planes_to_fits == True:
                name_plane = 'EF_FP_before_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane)

                ame_plane = 'FPM' + '_wl{}'.format(int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          FPmsk)

                name_plane = 'EF_FP_after_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane * FPmsk)

            # Focal plane to Lyot plane
            lyotplane_before_lyot = np.fft.fftshift(
                np.fft.ifft2(
                    corono_focal_plane * FPmsk)) * maskshifthalfpix_invert
            # we take the convention that when we are in pupil plane the field must be
            # "non-shifted". Because the shift in pupil plane is resolution dependent
            # which depend on the method (fft is not exactly science resolution because
            # of rounding issues, mft-babinet does not use the science resolution, etc).
            # these shift in both direction should be included in apod and pup multiplication
            # to save time

        elif self.prop_apod2lyot == "mft-babinet":
            #Apod plane to focal plane

            corono_focal_plane = prop.mft(
                input_wavefront_after_apod,
                self.dim_overpad_pupil,
                self.dim_fpm,
                self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                X_offset_output=-0.5,
                Y_offset_output=-0.5,
                inverse=False)

            if save_all_planes_to_fits == True:
                name_plane = 'EF_FP_before_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane)

                name_plane = 'FPM' + '_wl{}'.format(int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          FPmsk)

                name_plane = 'EF_FP_after_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane * FPmsk)

            # Focal plane to Lyot plane
            lyotplane_before_lyot_central_part = prop.mft(
                corono_focal_plane * (1 - FPmsk),
                self.dim_fpm,
                self.dim_overpad_pupil,
                self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                X_offset_input=-0.5,
                Y_offset_input=-0.5,
                inverse=True)

            # Babinet's trick
            lyotplane_before_lyot = input_wavefront_after_apod - lyotplane_before_lyot_central_part

        elif self.prop_apod2lyot == "mft":
            # Apod plane to focal plane

            corono_focal_plane = prop.mft(input_wavefront_after_apod,
                                          2 * self.prad,
                                          self.dim_im,
                                          self.dim_im / self.science_sampling *
                                          lambda_ratio,
                                          X_offset_output=-0.5,
                                          Y_offset_output=-0.5,
                                          inverse=False)

            if save_all_planes_to_fits == True:
                name_plane = 'EF_FP_before_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane)

                name_plane = 'FPM' + '_wl{}'.format(int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          FPmsk)

                name_plane = 'EF_FP_after_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane * FPmsk)

            # Focal plane to Lyot plane
            lyotplane_before_lyot = proc.crop_or_pad_image(
                prop.mft(corono_focal_plane * FPmsk,
                         self.dim_im,
                         2 * self.prad,
                         self.dim_im / self.science_sampling * lambda_ratio,
                         X_offset_input=-0.5,
                         Y_offset_input=-0.5,
                         inverse=True), self.dim_overpad_pupil)

        else:
            raise Exception(
                self.prop_apod2lyot +
                " is not a known prop_apod2lyot propagation mehtod")

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_LS' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      lyotplane_before_lyot)

        # crop to the dim_overpad_pupil expeted size
        lyotplane_before_lyot_crop = proc.crop_or_pad_image(
            lyotplane_before_lyot, self.dim_overpad_pupil)

        # Field after filtering by Lyot stop
        lyotplane_after_lyot = self.lyot_pup.EF_through(
            entrance_EF=lyotplane_before_lyot_crop, wavelength=wavelength)

        if (self.perfect_coro == True) & (noFPM == False):
            lyotplane_after_lyot = lyotplane_after_lyot - self.perfect_Lyot_pupil

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_LS' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      lyotplane_after_lyot)

        return lyotplane_after_lyot

    def FQPM(self):
        """ --------------------------------------------------
        Create a Four Quadrant Phase Mask coronagraph 
        

        Returns
        ------
        FQPM : list of len(self.wav_vec) 2D arrays giving the complex transmission of the
            FQPM mask
        
        AUTHOR : Axel Potier
        Modified by Johan Mazoyer
        -------------------------------------------------- """

        if self.prop_apod2lyot == "fft":
            maxdimension_array_fpm = np.max(self.dim_fp_fft)
        else:
            maxdimension_array_fpm = self.dim_im
            print("you should really not simulate FQPM with MFT")

            self.dim_fp_fft = np.zeros(len(self.wav_vec), dtype=np.int)
            for i, wav in enumerate(self.wav_vec):
                self.dim_fp_fft[i] = int(
                    np.ceil(self.prad * self.science_sampling *
                            self.diam_lyot_in_m / self.diam_pup_in_m *
                            self.wavelength_0 / wav)) * 2

        xx, yy = np.meshgrid(
            np.arange(maxdimension_array_fpm) - (maxdimension_array_fpm) / 2,
            np.arange(maxdimension_array_fpm) - (maxdimension_array_fpm) / 2)

        fqpm_thick_vert = np.zeros(
            (maxdimension_array_fpm, maxdimension_array_fpm))
        fqpm_thick_vert[np.where(xx < 0)] = 1
        fqpm_thick_hor = np.zeros(
            (maxdimension_array_fpm, maxdimension_array_fpm))
        fqpm_thick_hor[np.where(yy >= 0)] = 1
        fqpm_thick = fqpm_thick_vert - fqpm_thick_hor

        fqpm = list()
        for i, wav in enumerate(self.wav_vec):
            if self.prop_apod2lyot == "fft":
                dim_fp = self.dim_fp_fft[i]
            else:
                dim_fp = self.dim_im
                print("really you should not do that")

            phase = np.zeros((dim_fp, dim_fp))
            fqpm_thick_cut = proc.crop_or_pad_image(fqpm_thick, dim_fp)
            phase[np.where(fqpm_thick_cut != 0)] = (np.pi + self.err_fqpm)
            if self.achrom_fqpm == False:
                phase = phase * self.wavelength_0 / wav

            fqpm.append(np.exp(1j * phase))

        return fqpm

    def KnifeEdgeCoro(self):
        """ --------------------------------------------------
        Create a Knife edge coronagraph of size (dim_im,dim_im)
    
        Returns
        ------
        shift(Knife) :list of len(self.wav_vec) 2D arrays giving the complex transmission of the
            Knife edge coronagraph mask. TODO A CHECKER YA PTET UN SOUCIS DE SHIFT
        
        AUTHOR : Axel Potier
        Modified by Johan Mazoyer
        -------------------------------------------------- """

        if len(self.wav_vec) == 1:
            raise Exception(
                "KnifeEdgeCoro only working in monochromatic as of now")
        maxdimension_array_fpm = np.max(self.dim_fp_fft)

        # self.coro_position can be 'left', 'right', 'top' or 'bottom'
        # to define the orientation of the coronagraph

        #  Number of pixels per resolution element at central wavelength
        ld_p_0 = self.science_sampling * self.diam_lyot_in_m / self.diam_pup_in_m

        xx, yy = np.meshgrid(np.arange(maxdimension_array_fpm),
                             np.arange(maxdimension_array_fpm))

        Knife = np.zeros((maxdimension_array_fpm, maxdimension_array_fpm))
        if self.coro_position == "right":
            Knife[np.where(xx > (maxdimension_array_fpm / 2 +
                                 self.knife_coro_offset * ld_p_0))] = 1
        if self.coro_position == "left":
            Knife[np.where(xx < (maxdimension_array_fpm / 2 -
                                 self.knife_coro_offset * ld_p_0))] = 1
        if self.coro_position == "bottom":
            Knife[np.where(yy > (maxdimension_array_fpm / 2 +
                                 self.knife_coro_offset * ld_p_0))] = 1
        if self.coro_position == "top":
            Knife[np.where(yy < (maxdimension_array_fpm / 2 -
                                 self.knife_coro_offset * ld_p_0))] = 1

        return [Knife]

    def ClassicalLyot(self):
        """ --------------------------------------------------
        Create a classical Lyot coronagraph of radius rad_LyotFP 0

        rad_LyotFP : int, radius of the Lyot focal plane
    
        Returns
        ------
         classical Lyot : 2D array

        AUTHOR : Johan Mazoyer 
        -------------------------------------------------- """

        ld_p = self.Lyot_fpm_sampling * self.diam_lyot_in_m / self.diam_pup_in_m

        rad_LyotFP_pix = self.rad_lyot_fpm * ld_p

        self.dim_fpm = 2 * int(rad_LyotFP_pix)
        ClassicalLyotstop = 1 - phase_ampl.roundpupil(self.dim_fpm,
                                                      rad_LyotFP_pix)

        return [ClassicalLyotstop]

    ##############################################
    ##############################################
    ### Propagation through coronagraph


##############################################
##############################################
### Deformable mirror
class deformable_mirror(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of a deformable mirror 
    (in pupil plane or out of pupil plane)
    coronagraph is a sub class of Optical_System.  
    

    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self,
                 modelconfig,
                 DMconfig,
                 Name_DM='DM3',
                 load_fits=False,
                 save_fits=False,
                 model_dir='',
                 Model_local_dir=''):
        """ --------------------------------------------------
        Initialize a deformable mirror object
        TODO handle misregistration that is currently not working 
        
        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)
        DMconfig : DM parameters
        Name_DM : the name of the DM, which allows to find it in the parameter file
        load_fits : bool, default = False if true, we do not measure the DM init fits, we load them
        save_fits : bool, default = False if true, we save the DM init fits for future use
        we measure and save the pushact functions

        model_dir: directory to find Measured positions for each actuator in pixel and 
                    influence fun. ie Things you cannot measure yourself and need to be given
        Model_local_dir: directory to save things you can measure yourself 
                    and can save to save time

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        self.exitpup_rad = self.prad

        self.Name_DM = Name_DM
        self.z_position = DMconfig[self.Name_DM + "_z_position"]
        self.active = DMconfig[self.Name_DM + "_active"]
        self.creating_pushact = DMconfig[self.Name_DM + "_creating_pushact"]
        
        MinimumSurfaceRatioInThePupil = DMconfig[
            "MinimumSurfaceRatioInThePupil"]
        DMconfig[self.Name_DM + "_misregistration"] = False
        # no misregistration in the initialization part, only in the correction part

        if self.active == False:
            print(self.Name_DM + ' is not activated')
            return

        # We need a pupil in creatingpushact_inpup() and for
        # which in pup. THIS IS NOT THE ENTRANCE PUPIL,
        # this is a clear pupil of the same size
        self.clearpup = pupil(modelconfig, prad=self.prad)

        #define self.pradDM and check that the pupil is large enough
        # for out of pupil propagation
        if self.z_position != 0:
            dx, dxout = prop.prop_fresnel(self.dim_overpad_pupil,
                                          self.wavelength_0,
                                          self.z_position,
                                          self.diam_pup_in_m / 2,
                                          self.prad,
                                          retscale=1)
            #radius of the pupil in pixel in the DM plane
            self.pradDM = self.prad * dx / dxout

            if dx > 2 * dxout:
                print(dx, dxout)
                raise Exception(
                    "Need to enhance the pupil size in pixel for Fresnel propagation"
                )
        else:
            # radius of the pupil in pixel in the DM plane.
            # by default the size of the pupil
            self.pradDM = self.prad

        # create, save or load the DM_pushact and DM_pushact_inpup functions
        # from the influence function

        # DM_pushact is always in the DM plane
        self.DM_pushact = self.creatingpushact(DMconfig,
                                               load_fits=load_fits,
                                               save_fits=save_fits,
                                               model_dir=model_dir,
                                               Model_local_dir=Model_local_dir)

        if self.z_position != 0:
            # DM_pushact_inpup is always in the pupil plane
            self.DM_pushact_inpup = self.creatingpushact_inpup(
                load_fits=load_fits,
                save_fits=save_fits,
                Model_local_dir=Model_local_dir)
        else:
            # if the DM plane IS the pupil plane
            self.DM_pushact_inpup = self.DM_pushact
            # This is a duplicate of the same file but coherent and
            # allows you to easily concatenate wherever are the DMs

        # create or load 'which actuators are in pupil'
        self.WhichInPupil = self.creatingWhichinPupil(
            MinimumSurfaceRatioInThePupil,
            load_fits=load_fits,
            save_fits=save_fits,
            Model_local_dir=Model_local_dir)

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   DMphase=0.,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None):
        """ --------------------------------------------------
        Propagate the electric field through the DM.
        if z_DM = 0, then it's just a phase multiplication
        if z_DM != 0, this is where we do the fresnel

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 
        
        DMphase : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                    Can also be a float scalar in which case DM_phase is constant
                    default is 0. 
        
        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer 
        -------------------------------------------------- """

        # call the Optical_System super function to check
        # and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength == None:
            wavelength = self.wavelength_0

        if isinstance(DMphase, float) or isinstance(DMphase, np.float):
            DMphase = np.full((self.dim_overpad_pupil, self.dim_overpad_pupil),
                              np.float(DMphase))

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_' + self.Name_DM + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)
            name_plane = 'phase_' + self.Name_DM + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane, DMphase)

        # if the DM is not active or if the surface is 0
        # we save some time : the EF is not modified
        if self.active == False or (DMphase == 0.).all():
            return entrance_EF

        lambda_ratio = wavelength / self.wavelength_0

        if self.z_position == 0:
            EF_after_DM = entrance_EF * np.exp(1j * DMphase / lambda_ratio)

        else:
            EF_after_DM = self.prop_pup_to_DM_and_back(
                entrance_EF,
                DMphase / lambda_ratio,
                wavelength,
                save_all_planes_to_fits=save_all_planes_to_fits,
                dir_save_all_planes=dir_save_all_planes)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_' + self.Name_DM + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)

        return EF_after_DM

    def creatingpushact(self,
                        DMconfig,
                        load_fits=False,
                        save_fits=False,
                        model_dir='',
                        Model_local_dir=''):
        """ --------------------------------------------------
        OPD map induced in the DM plane for each actuator

        Parameters
        ----------
        DMconfig : structure with all information on DMs
        load_fits : bool, default = False if true, we do not measure the DM init fits, we load them
        save_fits : bool, default = False if true, we save the DM init fits for future use
        model_dir: directory to find Measured positions for each actuator in pixel and 
                    influence fun. ie Things you cannot measure yourself and need to be given
        Model_local_dir: directory to save things you can measure yourself 
                    and can save to save time

        Error on the model of the DM

        This function can to cleaned (names and comment). 
            
        Returns
        ------
        pushact : 
        -------------------------------------------------- """

        Name_pushact_fits = self.Name_DM + "_PushActInPup_radpup" + str(
            int(self.pradDM)) + "_dimpuparray" + str(
                int(self.dim_overpad_pupil))

        if (load_fits == True) or (self.creating_pushact == False and os.path.exists(Model_local_dir + Name_pushact_fits + '.fits')):
            return fits.getdata(os.path.join(Model_local_dir ,Name_pushact_fits + '.fits')) 

        diam_pup_in_pix = 2 * self.prad
        diam_pup_in_m = self.diam_pup_in_m
        dim_array = self.dim_overpad_pupil

        pitchDM = DMconfig[self.Name_DM + "_pitch"]
        filename_ActuN = DMconfig[self.Name_DM + "_filename_ActuN"]
        filename_grid_actu = DMconfig[self.Name_DM + "_filename_grid_actu"]
        filename_actu_infl_fct = DMconfig[self.Name_DM +
                                          "_filename_actu_infl_fct"]
        ActuN = DMconfig[self.Name_DM + "_ActuN"]
        y_ActuN = DMconfig[self.Name_DM + "_y_ActuN"]
        x_ActuN = DMconfig[self.Name_DM + "_x_ActuN"]
        xy_ActuN = [x_ActuN, y_ActuN]

        misregistration = DMconfig[self.Name_DM + "_misregistration"]
        if misregistration == True:
            xerror = DMconfig[self.Name_DM + "_xerror"]
            yerror = DMconfig[self.Name_DM + "_yerror"]
            angerror = DMconfig[self.Name_DM + "_angerror"]
            gausserror = DMconfig[self.Name_DM + "_gausserror"]
        else:
            xerror = 0.
            yerror = 0.
            angerror = 0.
            gausserror = 0.

        #Measured positions for each actuator in pixel
        measured_grid = fits.getdata(model_dir + filename_grid_actu)
        #Ratio: pupil radius in the measured position over
        # pupil radius in the numerical simulation
        sampling_simu_over_measured = diam_pup_in_pix / 2 / fits.getheader(
            model_dir + filename_grid_actu)['PRAD']
        if filename_ActuN != "":
            im_ActuN = fits.getdata(model_dir + filename_ActuN)
            im_ActuN_dim = proc.crop_or_pad_image(im_ActuN, dim_array)

            ytmp, xtmp = np.unravel_index(
                np.abs(im_ActuN_dim).argmax(), im_ActuN_dim.shape)
            # shift by (0.5,0.5) pixel because the pupil is
            # centered between pixels
            xy_ActuN = [xtmp - 0.5, ytmp - 0.5]

        # Convert the measred positions of actuators to positions for numerical simulation
        simu_grid = measured_grid * 0
        for i in np.arange(measured_grid.shape[1]):
            simu_grid[:,
                      i] = measured_grid[:,
                                         i] - measured_grid[:, int(
                                             ActuN)] + xy_ActuN
        simu_grid = simu_grid * sampling_simu_over_measured

        # Influence function and the pitch in pixels
        actshape = fits.getdata(model_dir + filename_actu_infl_fct)
        pitch_actshape = fits.getheader(model_dir +
                                        filename_actu_infl_fct)['PITCH']

        # Scaling the influence function to the desired dimension
        # for numerical simulation
        resizeactshape = skimage.transform.rescale(
            actshape,
            diam_pup_in_pix / diam_pup_in_m * pitchDM / pitch_actshape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
            multichannel=False)

        # Gauss2Dfit for centering the rescaled influence function
        Gaussian_fit_param = proc.gauss2Dfit(resizeactshape)
        dx = Gaussian_fit_param[3]
        dy = Gaussian_fit_param[4]
        xycent = len(resizeactshape) / 2
        resizeactshape = nd.interpolation.shift(resizeactshape,
                                                (xycent - dx, xycent - dy))

        # Put the centered influence function inside an array (2*prad x 2*prad)
        actshapeinpupil = np.zeros((dim_array, dim_array))
        if len(resizeactshape) < dim_array:
            actshapeinpupil[0:len(resizeactshape),
                            0:len(resizeactshape
                                  )] = resizeactshape / np.amax(resizeactshape)
            xycenttmp = len(resizeactshape) / 2
        else:
            actshapeinpupil = resizeactshape[
                0:dim_array, 0:dim_array] / np.amax(resizeactshape)
            xycenttmp = dim_array / 2

        # Fill an array with the influence functions of all actuators
        pushact = np.zeros((simu_grid.shape[1], dim_array, dim_array))
        for i in np.arange(pushact.shape[0]):
            if gausserror == 0:
                Psivector = nd.interpolation.shift(
                    actshapeinpupil,
                    (simu_grid[1, i] + dim_array / 2 - xycenttmp + yerror,
                     simu_grid[0, i] + dim_array / 2 - xycenttmp + xerror))

                # Add an error on the orientation of the grid
                if angerror != 0:
                    Psivector = nd.rotate(Psivector,
                                          angerror,
                                          order=5,
                                          cval=0,
                                          reshape=False)[0:dim_array,
                                                         0:dim_array]
            else:
                # Add an error on the sizes of the influence functions
                Psivector = nd.interpolation.shift(
                    actshapeinpupil,
                    (simu_grid[1, i] + dim_array / 2 - xycenttmp,
                     simu_grid[0, i] + dim_array / 2 - xycenttmp))

                xo, yo = np.unravel_index(Psivector.argmax(), Psivector.shape)
                x, y = np.mgrid[0:dim_array, 0:dim_array]
                xy = (x, y)
                Psivector = proc.twoD_Gaussian(xy,
                                               1,
                                               1 + gausserror,
                                               1 + gausserror,
                                               xo,
                                               yo,
                                               0,
                                               0,
                                               flatten=False)
            Psivector[np.where(Psivector < 1e-4)] = 0

            pushact[i] = Psivector

        if save_fits == True and misregistration == False:
            fits.writeto(Model_local_dir + Name_pushact_fits + '.fits',
                         pushact,
                         overwrite=True)

        return pushact

    def creatingpushact_inpup(self,
                              load_fits=False,
                              save_fits=False,
                              Model_local_dir=''):
        """ --------------------------------------------------
        OPD map induced by out-of-pupil DM in the pupil plane for each actuator
        ## Create the influence functions of an out-of-pupil DM
        ## in the pupil plane

        Parameters
        ----------
        load_fits : bool, default = False if true, we do not measure creatingpushact_inpup fits, we load it
        save_fits : bool, default = False if true, we save the creatingpushact_inpup fits for future use
        Model_local_dir: directory to save things you can measure yourself 
                    and can save to save time
        Returns
        ------
        pushact_inpup : Map of the complex phase induced in pupil plane
        -------------------------------------------------- """

        Name_pushact_fits = self.Name_DM + "_PushActInPup_radpup" + str(
            int(self.pradDM)) + "_dimpuparray" + str(
                int(self.dim_overpad_pupil))
        
        if (load_fits == True) or (self.creating_pushact == False and os.path.exists(Model_local_dir + Name_pushact_fits + '.fits')):
            DM_pushact_inpup_real = fits.getdata(os.path.join(Model_local_dir ,Name_pushact_fits+ '_inPup_real.fits'))
            DM_pushact_inpup_imag = fits.getdata(os.path.join(Model_local_dir ,Name_pushact_fits +'_inPup_imag.fits'))

            return DM_pushact_inpup_real + 1j * DM_pushact_inpup_imag

        dim_entrancepupil = self.dim_overpad_pupil
        # do we really need a pupil here ?!? seems to me it would be more general
        # with all the actuators ?
        EF_inDMplane, dxout = prop.prop_fresnel(self.clearpup.pup,
                                                self.wavelength_0,
                                                self.z_position,
                                                self.diam_pup_in_m / 2,
                                                self.prad)
        pushact_inpup = np.zeros(
            (self.DM_pushact.shape[0], dim_entrancepupil, dim_entrancepupil),
            dtype=complex)

        for i in np.arange(self.DM_pushact.shape[0]):
            EF_back_in_pup_plane, dxpup = prop.prop_fresnel(
                EF_inDMplane *
                proc.crop_or_pad_image(self.DM_pushact[i], dim_entrancepupil),
                self.wavelength_0, -self.z_position, self.diam_pup_in_m / 2,
                self.prad)
            pushact_inpup[i] = EF_back_in_pup_plane

        if save_fits == True:
            fits.writeto(Model_local_dir + Name_pushact_fits +
                         '_inPup_real.fits',
                         np.real(pushact_inpup),
                         overwrite=True)
            fits.writeto(Model_local_dir + Name_pushact_fits +
                         '_inPup_imag.fits',
                         np.imag(pushact_inpup),
                         overwrite=True)

        return pushact_inpup

    def creatingWhichinPupil(self,
                             cutinpupil,
                             load_fits=False,
                             save_fits=False,
                             Model_local_dir=''):
        """ --------------------------------------------------
        Create a vector with the index of all the actuators located in the entrance pupil
        
        Parameters:
        ----------
        cutinpupil: float, minimum surface of an actuator inside the pupil to be taken into account (between 0 and 1, in ratio of an actuator perfectly centered in the entrance pupil)
        load_fits : bool, default = False if true, we do not measure WhichInPup fits, we load it
        save_fits : bool, default = False if true, we save the WhichInPup fits for future use
        Model_local_dir: directory to save things you can measure yourself 
                    and can save to save time
        Return:
        ------
        WhichInPupil: 1D array, index of all the actuators located inside the pupil
        -------------------------------------------------- """

        Name_WhichInPup_fits = self.Name_DM + "_Whichact_for" + str(
            cutinpupil) + '_radpup' + str(self.prad)

        if load_fits == True:
            return useful.check_and_load_fits(Model_local_dir,
                                              Name_WhichInPup_fits)

        WhichInPupil = []

        # TODO to check if we need this line. thisis important because if the
        # pupil is no the size we expect it, we might exclude act that should not
        tmp_entrancepupil = proc.crop_or_pad_image(
            self.clearpup.pup, self.DM_pushact_inpup.shape[2])

        for i in np.arange(self.DM_pushact_inpup.shape[0]):
            Psivector = self.DM_pushact_inpup[i]
            cut = cutinpupil * np.sum(np.abs(Psivector))

            if np.sum(Psivector * tmp_entrancepupil) > cut:
                WhichInPupil.append(i)

        WhichInPupil = np.array(WhichInPupil)

        if save_fits == True:
            fits.writeto(Model_local_dir + Name_WhichInPup_fits + '.fits',
                         WhichInPupil,
                         overwrite=True)

        return WhichInPupil

    def prop_pup_to_DM_and_back(self,
                                entrance_EF,
                                phase_DM,
                                wavelength,
                                save_all_planes_to_fits=False,
                                dir_save_all_planes=None):
        """ --------------------------------------------------
        Propagate the field towards an out-of-pupil plane (DM1 plane),
        add the DM1 phase, and propagate to the next pupil plane (DM3 plane)

        Parameters
        ----------
        pupil_wavefront : 2D array (float, double or complex)
                    Wavefront in the pupil plane

        phase_DM1 : 2D array
                    Phase introduced by DM1
        
        wavelength : float
                    wavelength in m
        
        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in fits if save_all_planes_to_fits = True
        
        Returns
        ------
        UDM3 : 2D array (complex)
                Wavefront in the pupil plane after DM1
                (corresponds to DM3 plane on THD2 bench)

        AUTHOR : Raphaël Galicher

        REVISION HISTORY :
        Revision 1.1  2021-02-10 Raphaël Galicher
            Initial revision
        Revision 2.0 2021-02-28 Johan Mazoyer 
            Make it more general for all DMs, put in the struc
        -------------------------------------------------- """

        # Propagation in DM plane out of pupil
        EF_inDMplane, _ = prop.prop_fresnel(entrance_EF, wavelength,
                                            self.z_position,
                                            self.diam_pup_in_m / 2., self.prad)

        # Add DM phase
        if save_all_planes_to_fits == True:
            name_plane = 'EF_before_DM_in_' + self.Name_DM + 'plane_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      EF_inDMplane)

        EF_inDMplane_after_DM = EF_inDMplane * np.exp(1j * phase_DM)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_after_DM_in_' + self.Name_DM + 'plane_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      EF_inDMplane)
        # and propagate to next pupil plane
        EF_back_in_pup_plane, _ = prop.prop_fresnel(EF_inDMplane_after_DM,
                                                    wavelength,
                                                    -self.z_position,
                                                    self.diam_pup_in_m / 2.,
                                                    self.prad)

        return EF_back_in_pup_plane


    def voltage_to_phase(self, actu_vect,  wavelength = None):
        """ --------------------------------------------------
        Generate the phase applied on one DM for a give vector of actuator amplitude
        
        Parameters:
        ----------
        actu_vect : 1D array
                    values of the amplitudes for each actuator

        Return:
        ------
            2D array
            phase map in the same unit as actu_vect times DM_pushact)
        -------------------------------------------------- """

        if wavelength == None:
            wavelength = self.wavelength_0
        
        DM_pushact_reshaped = self.DM_pushact.reshape(
            self.DM_pushact.shape[0],
            self.DM_pushact.shape[1] * self.DM_pushact.shape[2])

        surface_reshaped = np.dot(actu_vect, DM_pushact_reshaped)

        surface_DM = surface_reshaped.reshape(self.DM_pushact.shape[1],
                                               self.DM_pushact.shape[2])

        phase_on_DM = surface_DM * 2 * np.pi * 1e-9 / wavelength

        return phase_on_DM


##############################################
##############################################
### Test bed
class THD2_testbed(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of the THD testbed 
    

    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self,
                 modelconfig,
                 DMconfig,
                 coroconfig,
                 load_fits=False,
                 save_fits=False,
                 model_dir='',
                 Model_local_dir=''):
        """ --------------------------------------------------
        Initialize a the DM system and the coronagraph
        
        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)
        DMconfig : DM parameters
        coroconfig : coronagraph parameters
        load_fits : bool, default = False if true, we do not measure initialization fits, we load it
        save_fits : bool, default = False if true, we save the initialization fits for future use
        
        model_dir: directory to find Measured positions for each actuators and 
                    influence fun and complex corona mask. 
                    ie all the things you cannot measure yourself and need to be given
        Model_local_dir: directory to save things you can measure yourself 
                    and can save to save time

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        self.entrancepupil = pupil(modelconfig,
                                   prad=self.prad,
                                   model_dir=model_dir,
                                   filename=modelconfig["filename_instr_pup"])
        self.DM1 = deformable_mirror(modelconfig,
                                     DMconfig,
                                     Name_DM='DM1',
                                     load_fits=load_fits,
                                     save_fits=save_fits,
                                     model_dir=model_dir,
                                     Model_local_dir=Model_local_dir)
        self.DM3 = deformable_mirror(modelconfig,
                                     DMconfig,
                                     load_fits=load_fits,
                                     save_fits=save_fits,
                                     Name_DM='DM3',
                                     model_dir=model_dir,
                                     Model_local_dir=Model_local_dir)
        self.corono = coronagraph(modelconfig, coroconfig, model_dir=model_dir)

        self.exitpup_rad = self.corono.lyotrad

        # Measure the PSF and store max and Sum
        self.maxPSF, self.sumPSF = self.max_sum_PSF()

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   DM1phase=0.,
                   DM3phase=0.,
                   noFPM=False,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None):
        """ --------------------------------------------------
        Propagate the electric field through the 2 DMs and the coronograph

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 
        
        DM1phase : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                    Can also be a float scalar in which case DM1phase is constant
                    default is 0. 
        
        DM3phase : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
            Can also be a float scalar in which case DM3phase is constant
            default is 0. 
        
        noFPM : bool (default: False)
            if True, remove the FPM if one want to measure a un-obstructed PSF
        
        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in fits if save_all_planes_to_fits = True
                    

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system


        AUTHOR : Johan Mazoyer 
        -------------------------------------------------- """

        # call the Optical_System super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength == None:
            wavelength = self.wavelength_0

        EF_afterentrancepup = self.entrancepupil.EF_through(
            entrance_EF=entrance_EF,
            wavelength=wavelength,
            save_all_planes_to_fits=save_all_planes_to_fits,
            dir_save_all_planes=dir_save_all_planes)

        EF_afterDM1 = self.DM1.EF_through(
            entrance_EF=EF_afterentrancepup,
            wavelength=wavelength,
            DMphase=DM1phase,
            save_all_planes_to_fits=save_all_planes_to_fits,
            dir_save_all_planes=dir_save_all_planes)

        EF_afterDM3 = self.DM3.EF_through(
            entrance_EF=EF_afterDM1,
            wavelength=wavelength,
            DMphase=DM3phase,
            save_all_planes_to_fits=save_all_planes_to_fits,
            dir_save_all_planes=dir_save_all_planes)

        EF_afterCorono = self.corono.EF_through(
            entrance_EF=EF_afterDM3,
            wavelength=wavelength,
            noFPM=noFPM,
            save_all_planes_to_fits=save_all_planes_to_fits,
            dir_save_all_planes=dir_save_all_planes)
        return EF_afterCorono

    def max_sum_PSF(self):
        """ --------------------------------------------------
        Measure the non-coronagraphic PSF with no focal plane mask
        and with flat DMs and return max and sum
        
        Returns
        ------
        np.amax(PSF): max of the non-coronagraphic PSF
        np.sum(PSF): sum of the non-coronagraphic PSF

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """
        PSF = self.todetector_Intensity(center_on_pixel=True, noFPM=True)

        return np.amax(PSF), np.sum(PSF)
