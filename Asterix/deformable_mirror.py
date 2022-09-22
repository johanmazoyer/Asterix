# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os
import time
import numpy as np
from astropy.io import fits

from Asterix.optical_systems import OpticalSystem, model_dir
from Asterix.pupil import Pupil
import Asterix.processing_functions as proc
import Asterix.propagation_functions as prop
import Asterix.phase_amplitude_functions as phase_ampl
import Asterix.save_and_read as saveread

class DeformableMirror(OpticalSystem):
    """ --------------------------------------------------
    initialize and describe the behavior of a deformable mirror
    (in pupil plane or out of pupil plane)
    coronagraph is a sub class of OpticalSystem.


    AUTHOR : Johan Mazoyer


    -------------------------------------------------- """

    def __init__(self, modelconfig, DMconfig, Name_DM='DM3', Model_local_dir=None):
        """ --------------------------------------------------
        Initialize a deformable mirror object

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
                general configuration parameters (sizes and dimensions)

        DMconfig : dict
                DM configuration parameters dictionary 

        Name_DM : string
                the name of the DM, which allows to find it in the parameter file
                we measure and save the pushact functions

        Model_local_dir: path
                directory to save things you can measure yourself
                    and can save to save time
        
        
        -------------------------------------------------- """

        # Initialize the OpticalSystem class and inherit properties
        super().__init__(modelconfig)

        if not os.path.exists(Model_local_dir):
            print("Creating directory " + Model_local_dir + " ...")
            os.makedirs(Model_local_dir)

        self.Model_local_dir = Model_local_dir

        self.Name_DM = Name_DM
        self.z_position = DMconfig[self.Name_DM + "_z_position"]
        self.active = DMconfig[self.Name_DM + "_active"]

        self.WhichInPup_threshold = DMconfig["MinimumSurfaceRatioInThePupil"]

        # For intialization, we assume no misregistration, we introduce it after
        # estimation and correction matrices are created.
        self.misregistration = False

        if DMconfig[self.Name_DM + "_Generic"] == True:
            self.total_act = DMconfig[self.Name_DM + "_Nact1D"]**2
            self.number_act = self.total_act
            self.active_actuators = np.arange(self.number_act)
        else:
            # first thing we do is to open filename_grid_actu to check the number of
            # actuator of this DM. We need the number of act to read and load pushact .fits
            self.total_act = fits.getdata(model_dir + DMconfig[self.Name_DM + "_filename_grid_actu"]).shape[1]

            if DMconfig[self.Name_DM + "_filename_active_actu"] != "":
                self.active_actuators = fits.getdata(model_dir +
                                                     DMconfig[self.Name_DM +
                                                              "_filename_active_actu"]).astype(int)
                self.number_act = len(self.active_actuators)

            else:
                self.number_act = self.total_act
                self.active_actuators = np.arange(self.number_act)

        self.string_os += '_' + self.Name_DM + "_z" + str(int(self.z_position * 1000)) + "_Nact" + str(
            int(self.number_act))

        if DMconfig[self.Name_DM + "_Generic"] == True:
            self.string_os += "Gen"

        if self.active == False:
            print(self.Name_DM + ' is not activated')
            return

        self.DMconfig = DMconfig

        # We need a pupil in creatingpushact_inpup() and for
        # which in pup. THIS IS NOT THE ENTRANCE PUPIL,
        # this is a round pupil of the same size
        self.clearpup = Pupil(modelconfig, PupType="RoundPup", prad=self.prad)

        # create the DM_pushact, surface of the DM for each individual act
        # DM_pushact is always in the DM plane
        self.DM_pushact = self.creatingpushact(DMconfig)

        # create or load 'which actuators are in pupil'
        self.WhichInPupil = self.id_in_pupil_actuators()

        self.misregistration = DMconfig[self.Name_DM + "_misregistration"]
        # now if we relaunch self.DM_pushact, and if misregistration = True
        # it will be different due to misregistration

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   DMphase=0.,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field through the DM.
        if z_DM = 0, then it's just a phase multiplication
        if z_DM != 0, this is where we do the fresnel

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF:    2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil] or float scalar in which case entrance_EF is constant
                        default is 1.
                        Electric field in the pupil plane a the entrance of the system.

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                    current wavelength in m.

        DMphase : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex or float scalar in which case DM_phase is constant
                    default is 0.

                    CAREFUL !! If the DM is part of a testbed. this variable name is changed
                    to DMXXphase (DMXX: name of the DM) to avoid confusion with

        save_all_planes_to_fits: Bool, default False
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.

        dir_save_all_planes : path, default None 
                            directory to save all plane in
                                    fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system


        -------------------------------------------------- """

        if save_all_planes_to_fits == True and dir_save_all_planes == None:
            raise Exception("save_all_planes_to_fits = True can generate a lot of .fits files" +
                            "please define a clear directory using dir_save_all_planes kw argument")

        # call the OpticalSystem super function to check
        # and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength is None:
            wavelength = self.wavelength_0

        if isinstance(DMphase, (int, float, np.float)):
            DMphase = np.full((self.dim_overpad_pupil, self.dim_overpad_pupil), np.float(DMphase))

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_' + self.Name_DM + '_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, entrance_EF)
            name_plane = 'phase_' + self.Name_DM + '_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, DMphase)

        # if the DM is not active or if the surface is 0
        # we save some time : the EF is not modified
        if self.active == False or (DMphase == 0.).all():
            return entrance_EF

        if self.z_position == 0:
            EF_after_DM = entrance_EF * self.EF_from_phase_and_ampl(phase_abb=DMphase, wavelengths=wavelength)

        else:
            EF_after_DM = self.prop_pup_to_DM_and_back(entrance_EF,
                                                       DMphase,
                                                       wavelength,
                                                       save_all_planes_to_fits=save_all_planes_to_fits,
                                                       dir_save_all_planes=dir_save_all_planes)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_' + self.Name_DM + '_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, EF_after_DM)

        return EF_after_DM

    def creatingpushact(self, DMconfig):
        """ --------------------------------------------------
        OPD map induced in the DM plane for each actuator.

        This large array is initialized at the beginning and will be use
        to transorm a voltage into a phase for each DM. This is saved
        in .fits to save times if the parameter have not changed

        In case of "misregistration = True" we measure it once for
        creating the interaction matrix and then once again, between
        the matrix measrueemnt and the correction with a small mismatch
        to simulate its effect.

        AUTHOR : Axel Potier

        Parameters
        ----------
        DMconfig : dict
            DM configuration parameters dictionary 

        Returns
        ------
        pushact : 3D numpy array 
                    of size [self.number_act, self.dim_overpad_pupil, self.dim_overpad_pupil]  
                    contains all the DM OPD map induced in the DM plane for each actuator.

        
        -------------------------------------------------- """
        start_time = time.time()
        Name_pushact_fits = "PushAct_" + self.Name_DM

        if DMconfig[self.Name_DM + "_Generic"] == True:
            Name_pushact_fits += "Gen"

        Name_pushact_fits += "_Nact" + str(int(self.number_act)) + '_dimPP' + str(int(
            self.dim_overpad_pupil)) + '_prad' + str(int(self.prad))

        if (self.misregistration is False) and (os.path.exists(self.Model_local_dir + Name_pushact_fits +
                                                               '.fits')):
            pushact3d = fits.getdata(os.path.join(self.Model_local_dir, Name_pushact_fits + '.fits'))
            print("Load " + Name_pushact_fits)
            return pushact3d

        if self.misregistration:
            xerror = float(DMconfig[self.Name_DM + "_xerror"])
            yerror = float(DMconfig[self.Name_DM + "_yerror"])
            angerror = float(DMconfig[self.Name_DM + "_angerror"])
            gausserror = float(DMconfig[self.Name_DM + "_gausserror"])
        else:
            xerror = 0.
            yerror = 0.
            angerror = 0.
            gausserror = 0.

        diam_pup_in_pix = int(2 * self.prad)
        diam_pup_in_m = self.diam_pup_in_m
        dim_array = self.dim_overpad_pupil

        filename_actu_infl_fct = DMconfig[self.Name_DM + "_filename_actu_infl_fct"]

        if DMconfig[self.Name_DM + "_Generic"] == False:
            #Measured positions for each actuator in pixel with (0,0) = center of pupil
            simu_grid = fits.getdata(
                model_dir + DMconfig[self.Name_DM + "_filename_grid_actu"]) * diam_pup_in_pix + dim_array / 2
            # the DM pitchs are read in the header
            pitchDMX = fits.getheader(model_dir +
                                      DMconfig[self.Name_DM + "_filename_grid_actu"])["PitchV"] * 1e-6
            pitchDMY = fits.getheader(model_dir +
                                      DMconfig[self.Name_DM + "_filename_grid_actu"])["PitchH"] * 1e-6
        else:
            # in this case we have a generic Nact1DxNact1D DM in which the pupil is centered
            # the pitch is read in the parameter file
            Nact1D = DMconfig[self.Name_DM + "_Nact1D"]
            pitchDM = DMconfig[self.Name_DM + "_pitch"]
            simu_grid = generic_actuator_position(Nact1D, pitchDM, diam_pup_in_m,
                                                       diam_pup_in_pix) + dim_array / 2
            pitchDMX = pitchDMY = pitchDM

        # Influence function and the pitch in pixels
        actshape = fits.getdata(model_dir + filename_actu_infl_fct)
        pitch_actshape = fits.getheader(model_dir + filename_actu_infl_fct)['PITCH']

        # Scaling the influence function to the desired dimension
        # for numerical simulation
        resizeactshape = proc.ft_zoom_out(actshape,
                                          (diam_pup_in_pix / diam_pup_in_m * pitchDMX / pitch_actshape,
                                           diam_pup_in_pix / diam_pup_in_m * pitchDMY / pitch_actshape))

        # make sure the actuator shape is in a squarre array of enven dimension (useful for the fft shift).
        # We do not care exactly about the centering since we recenter the actuator just after
        dim_even = int(np.ceil(np.max(resizeactshape.shape) / 2 + 1)) * 2
        resizeactshape = proc.crop_or_pad_image(resizeactshape, dim_even)

        # Gauss2Dfit for centering the rescaled influence function
        Gaussian_fit_param = proc.gauss2Dfit(resizeactshape)
        dx = Gaussian_fit_param[3]
        dy = Gaussian_fit_param[4]
        xycent = len(resizeactshape) / 2

        # Center the actuator shape on a pixel and normalize
        resizeactshape = proc.ft_subpixel_shift(resizeactshape, xshift=xycent - dx,
                                                yshift=xycent - dy) / np.amax(resizeactshape)

        # Put the centered influence function inside an array (self.dim_overpad_pupil x self.dim_overpad_pupil)
        actshapeinpupil = proc.crop_or_pad_image(resizeactshape, dim_array)
        xycenttmp = len(actshapeinpupil) / 2

        # Fill an array with the influence functions of all actuators
        pushact3d = np.zeros((simu_grid.shape[1], dim_array, dim_array))

        # do the first FT only once
        ft_actu = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(actshapeinpupil), norm="ortho"))

        for i in np.arange(pushact3d.shape[0]):

            # Add an error on the orientation of the grid
            if angerror != 0:
                simu_grid[1, i] = simu_grid[1, i] * np.cos(np.radians(angerror)) - simu_grid[0, i] * np.sin(
                    np.radians(angerror))
                simu_grid[0, i] = simu_grid[1, i] * np.sin(np.radians(angerror)) + simu_grid[0, i] * np.cos(
                    np.radians(angerror))

            Psivector = proc.ft_subpixel_shift(ft_actu,
                                               xshift=simu_grid[1, i] - xycenttmp + xerror * pitch_actshape,
                                               yshift=simu_grid[0, i] - xycenttmp + yerror * pitch_actshape,
                                               fourier=True,
                                               norm="ortho")

            if gausserror != 0:
                # Add an error on the sizes of the influence functions

                xy0 = np.unravel_index(Psivector.argmax(), Psivector.shape)
                x, y = np.mgrid[0:dim_array, 0:dim_array]
                xy = (x, y)
                Psivector = proc.twoD_Gaussian(xy,
                                               1,
                                               1 + gausserror,
                                               1 + gausserror,
                                               xy0[0],
                                               xy0[1],
                                               0,
                                               0,
                                               flatten=False)
            Psivector[np.where(Psivector < 1e-4)] = 0

            pushact3d[i] = Psivector

        # we exclude the actuators non active
        pushact3d = pushact3d[self.active_actuators]

        if self.misregistration is False and (
                not os.path.exists(self.Model_local_dir + Name_pushact_fits + '.fits')):
            fits.writeto(self.Model_local_dir + Name_pushact_fits + '.fits', pushact3d)
            print("time for " + Name_pushact_fits + " (s):", round(time.time() - start_time))

        return pushact3d

    def id_in_pupil_actuators(self):
        """ --------------------------------------------------
        Create a vector with the index of all the actuators located in the entrance pupil
        
        AUTHOR: Johan Mazoyer

        Parameters
        ----------
        cutinpupil: float
                    minimum surface of an actuator inside the pupil to be taken into account
                    (between 0 and 1, ratio of an actuator perfectly centered in the entrance pupil)

        Returns
        ------
        WhichInPupil: 1D array
                index of all the actuators located inside the pupil

        
        -------------------------------------------------- """
        start_time = time.time()
        Name_WhichInPup_fits = "WhichInPup_" + self.Name_DM

        if self.DMconfig[self.Name_DM + "_Generic"] == True:
            Name_WhichInPup_fits += "Gen"

        Name_WhichInPup_fits += "_Nact" + str(int(self.number_act)) + '_dimPP' + str(
            int(self.dim_overpad_pupil)) + '_prad' + str(int(self.prad)) + "_thres" + str(
                self.WhichInPup_threshold)

        if os.path.exists(self.Model_local_dir + Name_WhichInPup_fits + '.fits'):
            print("Load " + Name_WhichInPup_fits)
            return fits.getdata(self.Model_local_dir + Name_WhichInPup_fits + '.fits')

        if self.z_position != 0:

            Pup_inDMplane = proc.crop_or_pad_image(
                prop.prop_angular_spectrum(self.clearpup.pup, self.wavelength_0, self.z_position,
                                           self.diam_pup_in_m / 2, self.prad), self.dim_overpad_pupil)
        else:
            Pup_inDMplane = self.clearpup.pup

        WhichInPupil = []
        Sum_actu_with_pup = np.zeros(self.number_act)

        for num_actu in np.arange(self.number_act):
            Sum_actu_with_pup[num_actu] = np.sum(np.abs(self.DM_pushact[num_actu] * Pup_inDMplane))

        Max_val = np.max(Sum_actu_with_pup)
        for num_actu in np.arange(self.number_act):
            if Sum_actu_with_pup[num_actu] > Max_val * self.WhichInPup_threshold:
                WhichInPupil.append(num_actu)

        WhichInPupil = np.array(WhichInPupil)

        fits.writeto(self.Model_local_dir + Name_WhichInPup_fits + '.fits', WhichInPupil, overwrite=True)
        print("time for " + Name_WhichInPup_fits + " (s):", round(time.time() - start_time))

        return WhichInPupil

    def prop_pup_to_DM_and_back(self,
                                entrance_EF,
                                phase_DM,
                                wavelength,
                                save_all_planes_to_fits=False,
                                dir_save_all_planes=None):
        """ --------------------------------------------------
        Propagate the field towards an out-of-pupil plane ,
        add the DM phase, and propagate to the next pupil plane
        
        AUTHOR : Raphaël Galicher, Johan Mazoyer

        REVISION HISTORY :
            Revision 1.1  2021-02-10 Raphaël Galicher (Initial revision)
            Revision 2.0 2021-02-28 Johan Mazoyer (Make it more general for all DMs, put in the struc)

        Parameters
        ----------
        pupil_wavefront : 2D array (float, double or complex)
                    Wavefront in the pupil plane

        phase_DM : 2D array
                    Phase introduced by out of PP DM

        wavelength : float
                    wavelength in m

        save_all_planes_to_fits: Bool, default False
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        
        dir_save_all_planes : string default None. 
                            directory to save all plane in fits if save_all_planes_to_fits = True

        Returns
        ------
        EF_back_in_pup_plane : 2D array (complex)
                            Wavefront in the pupil plane following the DM

            

        -------------------------------------------------- """

        EF_inDMplane = proc.crop_or_pad_image(
            prop.prop_angular_spectrum(entrance_EF, wavelength, self.z_position, self.diam_pup_in_m / 2.,
                                       self.prad), self.dim_overpad_pupil)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_before_DM_in_' + self.Name_DM + 'plane_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, EF_inDMplane)

        # Add DM phase at the right WL
        EF_inDMplane_after_DM = EF_inDMplane * self.EF_from_phase_and_ampl(phase_abb=phase_DM,
                                                                           wavelengths=wavelength)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_after_DM_in_' + self.Name_DM + 'plane_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, EF_inDMplane)

        # and propagate to next pupil plane

        EF_back_in_pup_plane = proc.crop_or_pad_image(
            prop.prop_angular_spectrum(EF_inDMplane_after_DM, wavelength, -self.z_position,
                                       self.diam_pup_in_m / 2., self.prad), self.dim_overpad_pupil)

        return EF_back_in_pup_plane

    def voltage_to_phase(self, actu_vect, einstein_sum=False):
        """ --------------------------------------------------
        Generate the phase applied on one DM for a give vector of actuator amplitude
        We decided to do it without matrix multiplication to save time because a
        lot of the time we have lot of zeros in it

        The phase is define at the reference wl and multiply by wl_ratio in DM.EF_through

        AUTHOR: Johan Mazoyer

        Parameters
        ----------
        actu_vect : 1D array
                    values of the amplitudes for each actuator
        einstein_sum : boolean, default false
                        Use numpy Einstein sum to sum the pushact[i]*actu_vect[i]
                        gives the same results as normal sum. Seems ot be faster for unique actuator
                        but slower for more complex phases

        Returns
        ------
            DM_phase: 2D array
                        phase map in the same unit as actu_vect * DM_pushact)
        

        -------------------------------------------------- """

        where_non_zero_voltage = np.where(actu_vect != 0)

        #opd is in nanometer
        # DM_pushact is in opd nanometer
        opd_to_phase = 2 * np.pi * 1e-9 / self.wavelength_0

        if einstein_sum == True or len(where_non_zero_voltage[0]) < 3:
            phase_on_DM = np.einsum('i,ijk->jk', actu_vect[where_non_zero_voltage],
                                    self.DM_pushact[where_non_zero_voltage]) * opd_to_phase
        else:
            phase_on_DM = np.zeros((self.dim_overpad_pupil, self.dim_overpad_pupil))
            for i in where_non_zero_voltage[0]:
                phase_on_DM += self.DM_pushact[i, :, :] * actu_vect[i] * opd_to_phase

        return phase_on_DM

    def create_DM_basis(self, basis_type='actuator'):
        """ --------------------------------------------------
        Create a DM basis.
        TODO do a zernike basis ?

        AUTHOR: Johan Mazoyer

        Parameters
        ----------
        basis_type: string, default 'actuator' 
            the type of basis. 'fourier' or 'actuator'

        Returns
        ------
        basis: 2d numpy array 
            basis [Size basis, Number of active act in the DM]


        -------------------------------------------------- """
        if basis_type == 'actuator':
            # no need to remove the inactive actuators,
            # they are already removed in pushact
            basis_size = len(self.WhichInPupil)
            basis = np.zeros((basis_size, self.number_act))
            for i in range(basis_size):
                basis[i][self.WhichInPupil[i]] = 1

        elif basis_type == 'fourier':
            start_time = time.time()
            # activeact = [value for value in self.active_actuators]

            sqrtnbract = int(np.sqrt(self.total_act))
            Name_FourrierBasis_fits = "Fourier_basis_" + self.Name_DM + '_prad' + str(
                self.prad) + '_nact' + str(sqrtnbract) + 'x' + str(sqrtnbract)

            cossinbasis = phase_ampl.sine_cosine_basis(sqrtnbract)

            basis_size = cossinbasis.shape[0]
            basis = np.zeros((basis_size, self.number_act))

            for i in range(basis_size):
                vec = cossinbasis[i].flatten()[self.active_actuators]
                basis[i] = vec

            start_time = time.time()
            # This is a very time consuming part of the code.
            # from N voltage vectors with the sine and cosine value, we go N times through the
            # voltage_to_phase functions. For this reason we save the Fourrier base 2D phases on each DMs
            # in a specific .fits file
            if not os.path.exists(self.Model_local_dir + Name_FourrierBasis_fits + '.fits'):
                phasesFourrier = np.zeros((basis_size, self.dim_overpad_pupil, self.dim_overpad_pupil))
                print("Start " + Name_FourrierBasis_fits)
                for i in range(basis_size):
                    phasesFourrier[i] = self.voltage_to_phase(basis[i])
                    if i % 10:
                        saveread._progress(i, basis_size, status='')
                fits.writeto(self.Model_local_dir + Name_FourrierBasis_fits + '.fits', phasesFourrier)
            print("time for " + Name_FourrierBasis_fits, time.time() - start_time)

        else:
            raise Exception(basis_type + " is is not a valid basis_type")

        return basis

def generic_actuator_position(Nact1D, pitchDM, diam_pup_in_m, diam_pup_in_pix):
    """ --------------------------------------------------
    Create a grid of position of actuators for generic  DM.
    The DM will then be automatically defined as squared with Nact1D x Nact1D actuators
    and the pupil centered on this DM.
    
    We need N_act1D > diam_pup_in_m / DM_pitch, so that the DM is larger than the pupil.

    at the end compare to the result of actuator_position in case of DM3, it 
    should be relatively close. If we can, we should try that actu 0 is 
    relatively at the same pos. Test with huge DM pitch (2 actus in the pup)


    AUTHOR: Johan Mazoyer
    
    Parameters
    ----------
    Nact1D : int 
            Numnber of actuators of a square DM in one of the principal direction
    pitchDM: float
            Pitch of the DM (distance between actuators),in meter
    diam_pup_in_m : float
            Diameter of the pupil in meter
    diam_pup_in_pix : int 
            Diameter of the pupil in pixel
    
    Returns
    ------
    simu_grid : 2D array 
                Array of shape is 2 x Nb_actuator
                x and y positions of each actuator for simulation
    
    
    -------------------------------------------------- """
    if Nact1D * pitchDM < diam_pup_in_m:
        raise Exception("""Nact1D*pitchDM < diam_pup_in_m: The DM is smaller than the pupil""")

    pitchDM_in_pix = pitchDM * diam_pup_in_pix / diam_pup_in_m

    pos_actu_in_pitch = np.zeros((2, Nact1D**2))
    for i in range(Nact1D**2):
        pos_actu_in_pitch[:, i] = np.array([i // Nact1D, i % Nact1D])

    # relative positions in pixel of the actuators
    pos_actu_in_pix = pos_actu_in_pitch * pitchDM_in_pix

    if Nact1D % 2 == 1:
        # if Nact1D if odd, then the center of the DM is the
        # actuator number (Nact1D**2 -1) /2
        #
        # 20 21 22 23 24
        # 15 16 17 18 19
        # 10 11 12 13 14
        # 5  6  7  8  9
        # 0  1  2  3  4
        #
        # 6 7 8
        # 3 4 5
        # 0 1 2
        pos_actu_center_pos = np.copy(pos_actu_in_pix[:, (Nact1D**2 - 1) // 2])
        center_pup = np.array([0.5, 0.5])

        for i in range(Nact1D**2):
            pos_actu_in_pix[:, i] = pos_actu_in_pix[:, i] - pos_actu_center_pos + center_pup

    else:

        #if Nact1D is even, the center of the DM is in between 4 actuators
        # (Nact1D -2) //2 * (Nact1D) +  Nact1D//2 -1    is in (-1/2 act, -1/2 act)
        # (Nact1D -2) //2 * (Nact1D) +  Nact1D//2       is in (-1/2 act, +1/2 act)

        # Nact1D //2 * Nact1D +  Nact1D//2 - 1          is in (+1/2 act, -1/2 act)
        # Nact1D //2 * Nact1D +  Nact1D//2              is in (+1/2 act, +1/2 act)

        #  30 31 32 33 34 35
        #  24 25 26 27 28 29
        #  18 19 20 21 22 23
        #  12 13 14 15 16 17
        #  6  7  8  9  10 11
        #  0  1  2  3  4  5

        # 12 13 14 15
        # 8  9  10 11
        # 4  5  6  7
        # 0  1  2  3

        # 2 3
        # 0 1

        pos_actuhalfactfromcenter = np.copy(pos_actu_in_pix[:, Nact1D // 2 * Nact1D + Nact1D // 2])
        halfactfromcenter = np.array([0.5 * pitchDM_in_pix, 0.5 * pitchDM_in_pix])

        center_pup = np.array([0.5, 0.5])

        for i in range(Nact1D**2):
            pos_actu_in_pix[:,
                            i] = pos_actu_in_pix[:,
                                                 i] - pos_actuhalfactfromcenter + halfactfromcenter + center_pup
            pos_actu_in_pix[0, i] *= -1
    return pos_actu_in_pix


## Currently unused
# def actuator_position(measured_grid, measured_ActuN, ActuN, sampling_simu_over_measured):
#     """ --------------------------------------------------
#     Convert the measured positions of actuators to positions for numerical simulation
    
#     AUTHOR: Axel Potier
    
#     Parameters
#     ----------
#     measured_grid : 2D array (float) 
#                     array of shape 2 x Nb_actuator
#                     x and y measured positions for each actuator (unit = pixel)
#     measured_ActuN: 1D array (float) 
#                     arrayof shape 2. x and y positions of actuator ActuN same unit as measured_grid
#     ActuN:          int
#                     Index of the actuator ActuN (corresponding to measured_ActuN)
#     sampling_simu_over_measured : float
#                     Ratio of sampling in simulation grid over sampling in measured grid
    
    
#     Returns
#     ------
#     simu_grid : 2D array 
#                 Array of shape is 2 x Nb_actuator
#                 x and y positions of each actuator for simulation
#                 same unit as measured_ActuN


#     -------------------------------------------------- """
#     simu_grid = measured_grid * 0
#     for i in np.arange(measured_grid.shape[1]):
#         simu_grid[:, i] = measured_grid[:, i] - measured_grid[:, int(ActuN)] + measured_ActuN
#     simu_grid = simu_grid * sampling_simu_over_measured

#     saveread._quickfits(simu_grid)
#     return simu_grid
