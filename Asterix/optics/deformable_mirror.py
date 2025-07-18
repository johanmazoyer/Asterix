import os
import time
from datetime import datetime
import numpy as np
from astropy.io import fits

from Asterix import model_dir

import Asterix.utils.gaussians as gauss
from Asterix.utils import save_plane_in_fits, progress, ft_subpixel_shift, ft_zoom_out, crop_or_pad_image, from_param_to_header

import Asterix.optics.optical_systems as optsy
import Asterix.optics.pupil as pupil
import Asterix.optics.propagation_functions as prop
import Asterix.optics.phase_amplitude_functions as phase_ampl


class DeformableMirror(optsy.OpticalSystem):
    """Initialize and describe the behavior of a deformable mirror (in pupil
    plane or out of pupil plane) coronagraph is a sub class of OpticalSystem.

    AUTHOR : Johan Mazoyer
    """

    def __init__(self, modelconfig, DMconfig, Name_DM='DM2', Model_local_dir=None, silence=False):
        """Initialize a deformable mirror object.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
            general configuration parameters (sizes and dimensions)
        DMconfig : dict
            DM configuration parameters dictionary
        Name_DM : string
            The name of the DM, which allows to find it in the parameter file
            we measure and save the pushact functions
        Model_local_dir : string
            Directory output path for model-related files created on the file for later reuse.
        silence : boolean, default False.
            Whether to silence print outputs.
        """

        # Initialize the OpticalSystem class and inherit properties
        super().__init__(modelconfig)

        if not os.path.exists(Model_local_dir):
            if not silence:
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

        if DMconfig[self.Name_DM + "_Generic"]:
            self.total_act = DMconfig[self.Name_DM + "_Nact1D"]**2

        else:
            # first thing we do is to open filename_grid_actu to check the number of
            # actuator of this DM. We need the number of act to read and load pushact .fits
            self.total_act = fits.getdata(os.path.join(model_dir,
                                                       DMconfig[self.Name_DM + "_filename_grid_actu"])).shape[1]

        if DMconfig[self.Name_DM + "_filename_active_actu"] != "":
            self.active_actuators = fits.getdata(
                os.path.join(model_dir, DMconfig[self.Name_DM + "_filename_active_actu"])).astype(int)
            self.number_act = len(self.active_actuators)

        else:
            self.number_act = self.total_act
            self.active_actuators = np.arange(self.number_act)

        self.string_os += '_' + self.Name_DM + "_z" + str(int(self.z_position * 1000)) + "_Nact" + str(
            int(self.number_act))

        if DMconfig[self.Name_DM + "_Generic"]:
            self.string_os += "Gen"

        if not self.active:
            if not silence:
                print(self.Name_DM + ' is not activated')
            return

        self.DMconfig = DMconfig

        # We need a pupil in creatingpushact_inpup() and for
        # which in pup. THIS IS NOT THE ENTRANCE PUPIL,
        # this is a round pupil of the same size
        self.clearpup = pupil.Pupil(modelconfig, PupType="RoundPup", prad=self.prad)

        # create the DM_pushact, surface of the DM for each individual act
        # DM_pushact is always in the DM plane
        self.DM_pushact = self.creatingpushact(DMconfig, silence=silence)

        # create or load 'which actuators are in pupil'
        self.WhichInPupil = self.id_in_pupil_actuators(silence=silence)

        self.misregistration = DMconfig[self.Name_DM + "_misregistration"]
        # now if we relaunch self.DM_pushact, and if misregistration = True
        # it will be different due to misregistration

        # initialize DM basis that will be defined in the corrector initialization once the
        # full testbed is created
        self.basis = None
        self.basis_size = None
        self.basis_type = None

    def EF_through(self, entrance_EF=1., wavelength=None, DMphase=0., dir_save_all_planes=None, **kwargs):
        """
        Propagate the electric field through the DM.
        if z_DM = 0, then it's just a phase multiplication
        if z_DM != 0, this is where we do the fresnel

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF : 2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil] or complex/float scalar (entrance_EF is constant)
            Default is 1. Electric field in the pupil plane a the entrance of the system.
        wavelength : float. Default is self.wavelength_0 the reference wavelength
            Current wavelength in m.
        DMphase : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil], or complex/float scalar (DM_phase is constant), default is 0
            Phase on DM
            CAREFUL !! If the DM is part of a testbed. this variable name is changed
            to DMXXphase (DMXX: name of the DM) to avoid confusion
        dir_save_all_planes : string or None, default None
            if not None, directory to save all planes in fits for debugging purposes.
            This can generate a lot of fits especially if in a loop, use with caution.

        Returns
        --------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system

        """

        # call the OpticalSystem super function to check
        # and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength is None:
            wavelength = self.wavelength_0

        if isinstance(DMphase, (int, float)):
            DMphase = np.full((self.dim_overpad_pupil, self.dim_overpad_pupil), float(DMphase))

        if dir_save_all_planes is not None:
            name_plane = 'EF_PP_before_' + self.Name_DM + f'_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, entrance_EF)
            name_plane = 'phase_' + self.Name_DM + f'_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, DMphase)

        # if the DM is not active or if the surface is 0
        # we save some time : the EF is not modified
        if (not self.active) or (DMphase == 0.).all():
            return entrance_EF

        if self.z_position == 0:
            EF_after_DM = entrance_EF * self.EF_from_phase_and_ampl(phase_abb=DMphase, wavelengths=wavelength)

        else:
            EF_after_DM = self.prop_pup_to_DM_and_back(entrance_EF,
                                                       DMphase,
                                                       wavelength,
                                                       dir_save_all_planes=dir_save_all_planes)

        if dir_save_all_planes is not None:
            name_plane = 'EF_PP_after_' + self.Name_DM + f'_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, EF_after_DM)

        return EF_after_DM

    def creatingpushact(self, DMconfig, silence=False):
        """OPD map induced in the DM plane for each actuator.

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
        silence : boolean, default False.
            Whether to silence print outputs.

        Returns
        --------
        pushact : 3D numpy arrayof size [self.number_act, self.dim_overpad_pupil, self.dim_overpad_pupil]
            DM OPD maps induced in the DM plane for each actuator.
        """
        start_time = time.time()
        Name_pushact_fits = "PushAct_" + self.Name_DM

        if DMconfig[self.Name_DM + "_Generic"]:
            Name_pushact_fits += "Gen"

        Name_pushact_fits += "_Nact" + str(int(self.number_act)) + '_dimPP' + str(int(
            self.dim_overpad_pupil)) + '_prad' + str(int(self.prad))

        header = fits.Header()
        header.insert(0, ('date_mat', datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "pushact creation date"))

        necessary_dm_param = dict()
        for key in self.DMconfig:
            if (self.Name_DM in key) and ('error' not in key) and ('misregistration' not in key) and ('z_position'
                                                                                                      not in key):
                necessary_dm_param[key] = self.DMconfig[key]
        necessary_dm_param["prad"] = self.prad
        necessary_dm_param["dim_overpad_pupil"] = self.dim_overpad_pupil
        necessary_dm_param["diam_pup_in_m"] = self.diam_pup_in_m
        header = from_param_to_header(necessary_dm_param, header)

        # Loading any existing matrix and comparing their headers to make sure they are created
        # using the same set of parameters
        bool_already_existing_matrix = False
        if os.path.exists(os.path.join(self.Model_local_dir, Name_pushact_fits + ".fits")):
            header_existing = fits.getheader(os.path.join(self.Model_local_dir, Name_pushact_fits + ".fits"))
            # remove the basic kw created automatically  when saving the fits file
            for keyw in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3']:
                header_existing.remove(keyw)
            # we compare the header (ignoring the date)
            bool_already_existing_matrix = fits.HeaderDiff(header_existing, header,
                                                           ignore_keywords=['DATE_MAT']).identical

        if (not self.misregistration) and (bool_already_existing_matrix):
            pushact3d = fits.getdata(os.path.join(self.Model_local_dir, Name_pushact_fits + '.fits'))
            if not silence:
                print("Load " + Name_pushact_fits + ".fits file")
            return pushact3d

        if not silence:
            print("Start " + Name_pushact_fits + " (wait a few seconds)")

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

        if not DMconfig[self.Name_DM + "_Generic"]:
            # Measured positions for each actuator in pixel with (0,0) = center of pupil
            simu_grid = fits.getdata(os.path.join(
                model_dir, DMconfig[self.Name_DM + "_filename_grid_actu"])) * diam_pup_in_pix + dim_array / 2
            # the DM pitchs are read in the header
            pitchDMX = fits.getheader(os.path.join(model_dir,
                                                   DMconfig[self.Name_DM + "_filename_grid_actu"]))["PitchV"] * 1e-6
            pitchDMY = fits.getheader(os.path.join(model_dir,
                                                   DMconfig[self.Name_DM + "_filename_grid_actu"]))["PitchH"] * 1e-6
        else:
            # in this case we have a generic Nact1DxNact1D DM in which the pupil is centered
            # the pitch is read in the parameter file
            Nact1D = DMconfig[self.Name_DM + "_Nact1D"]
            pitchDM = DMconfig[self.Name_DM + "_pitch"]
            simu_grid = generic_actuator_position(Nact1D, pitchDM, diam_pup_in_m, diam_pup_in_pix) + dim_array / 2
            pitchDMX = pitchDMY = pitchDM

        # Influence function and the pitch in pixels
        actshape = fits.getdata(os.path.join(model_dir, filename_actu_infl_fct))
        pitch_actshape = fits.getheader(os.path.join(model_dir, filename_actu_infl_fct))['PITCH']

        # Scaling the influence function to the desired dimension
        # for numerical simulation
        resizeactshape = ft_zoom_out(actshape, (diam_pup_in_pix / diam_pup_in_m * pitchDMX / pitch_actshape,
                                                diam_pup_in_pix / diam_pup_in_m * pitchDMY / pitch_actshape))

        # make sure the actuator shape is in a squarre array of enven dimension (useful for the fft shift).
        # We do not care exactly about the centering since we recenter the actuator just after
        dim_even = int(np.ceil(np.max(resizeactshape.shape) / 2 + 1)) * 2
        resizeactshape = crop_or_pad_image(resizeactshape, dim_even)

        # Gauss2Dfit for centering the rescaled influence function
        Gaussian_fit_param = gauss.gauss2Dfit(resizeactshape)
        dx = Gaussian_fit_param[3]
        dy = Gaussian_fit_param[4]
        xycent = len(resizeactshape) / 2

        # Center the actuator shape on a pixel and normalize
        resizeactshape = ft_subpixel_shift(resizeactshape, xshift=xycent - dx,
                                           yshift=xycent - dy) / np.amax(resizeactshape)

        # Put the centered influence function inside an array (self.dim_overpad_pupil x self.dim_overpad_pupil)
        actshapeinpupil = crop_or_pad_image(resizeactshape, dim_array)
        xycenttmp = len(actshapeinpupil) / 2

        # Fill an array with the influence functions of all actuators
        pushact3d = np.zeros((simu_grid.shape[1], dim_array, dim_array))


        for i in np.arange(pushact3d.shape[0]):
            # Add an error on the orientation of the grid
            if angerror != 0:
                simu_grid[
                    1,
                    i] = simu_grid[1, i] * np.cos(np.radians(angerror)) - simu_grid[0, i] * np.sin(np.radians(angerror))
                simu_grid[
                    0,
                    i] = simu_grid[1, i] * np.sin(np.radians(angerror)) + simu_grid[0, i] * np.cos(np.radians(angerror))

            Psivector = ft_subpixel_shift(actshapeinpupil,
                                          xshift=simu_grid[1, i] - xycenttmp + xerror * pitch_actshape,
                                          yshift=simu_grid[0, i] - xycenttmp + yerror * pitch_actshape,
                                          norm="ortho")

            if gausserror != 0:
                # Add an error on the sizes of the influence functions

                xy0 = np.unravel_index(Psivector.argmax(), Psivector.shape)
                x, y = np.mgrid[0:dim_array, 0:dim_array]
                xy = (x, y)
                Psivector = gauss.twoD_Gaussian(xy,
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

        if (not self.misregistration) and (not bool_already_existing_matrix):

            fits.writeto(os.path.join(self.Model_local_dir, Name_pushact_fits + '.fits'),
                         pushact3d,
                         header,
                         overwrite=True)
            if not silence:
                print("Time for " + Name_pushact_fits + " (s):", round(time.time() - start_time))

        return pushact3d

    def id_in_pupil_actuators(self, silence=False):
        """Create a vector with the index of all the actuators located in the entrance pupil.

        AUTHOR: Johan Mazoyer

        Returns
        --------
        WhichInPupil : 1D array
            Index of all the actuators located inside the pupil.
        silence : boolean, default False.
            Whether to silence print outputs.
        """

        if self.WhichInPup_threshold <= 0:
            return np.arange(self.number_act)

        if self.z_position != 0:

            Pup_inDMplane = crop_or_pad_image(
                prop.prop_angular_spectrum(self.clearpup.pup,
                                           self.wavelength_0,
                                           self.z_position,
                                           self.diam_pup_in_m / 2,
                                           self.prad,
                                           dtype_complex=self.dtype_complex), self.dim_overpad_pupil)
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

        return np.array(WhichInPupil)

    def prop_pup_to_DM_and_back(self, entrance_EF, phase_DM, wavelength, dir_save_all_planes=None):
        """Propagate the fied towards an out-of-pupil plane , add the DM
        phase, and propagate to the next pupil plane.

        AUTHOR : Raphaël Galicher, Johan Mazoyer

        REVISION HISTORY :
            Revision 1.1  2021-02-10 Raphaël Galicher (Initial revision)
            Revision 2.0 2021-02-28 Johan Mazoyer (Make it more general for all DMs, put in the struc)

        Parameters
        ----------
        pupil_wavefront : 2D array (float, double or complex)
            Wavefront in the pupil plane.
        phase_DM : 2D array
            Phase introduced by out of PP DM.
        wavelength : float
            Wavelength in m.
        dir_save_all_planes : string or None, default None
            If not None, absolute directory to save all planes in fits for debugging purposes.
            This can generate a lot of fits especially if in a loop, use with caution.

        Returns
        --------
        EF_back_in_pup_plane : 2D array (complex)
            Wavefront in the pupil plane following the DM
        """

        EF_inDMplane = crop_or_pad_image(
            prop.prop_angular_spectrum(entrance_EF,
                                       wavelength,
                                       self.z_position,
                                       self.diam_pup_in_m / 2.,
                                       self.prad,
                                       dtype_complex=self.dtype_complex), self.dim_overpad_pupil)

        if dir_save_all_planes is not None:
            name_plane = 'EF_before_DM_in_' + self.Name_DM + f'plane_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, EF_inDMplane)

        # Add DM phase at the right WL
        EF_inDMplane_after_DM = EF_inDMplane * self.EF_from_phase_and_ampl(phase_abb=phase_DM, wavelengths=wavelength)

        if dir_save_all_planes is not None:
            name_plane = 'EF_after_DM_in_' + self.Name_DM + f'plane_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, EF_inDMplane)

        # and propagate to next pupil plane

        EF_back_in_pup_plane = crop_or_pad_image(
            prop.prop_angular_spectrum(EF_inDMplane_after_DM,
                                       wavelength,
                                       -self.z_position,
                                       self.diam_pup_in_m / 2.,
                                       self.prad,
                                       dtype_complex=self.dtype_complex), self.dim_overpad_pupil)

        return EF_back_in_pup_plane

    def voltage_to_phase(self, actu_vect, einstein_sum=False):
        """Generate the phase applied on one DM for a give vector of actuator
        amplitude We decided to do it without matrix multiplication to save
        time because a lot of the time we have lot of zeros in it.

        The phase is define at the reference wl and multiply by wl_ratio in DM.EF_through

        AUTHOR: Johan Mazoyer

        Parameters
        ----------
        actu_vect : 1D array
            Values of the amplitudes for each actuator.
        einstein_sum : boolean, default false
            Use numpy Einstein sum to sum the pushact[i]*actu_vect[i]
            gives the same results as normal sum. Seems ot be faster for unique actuator
            but slower for more complex phases.

        Returns
        --------
        DM_phase: 2D array
            phase map in the same unit as actu_vect * DM_pushact.
        """

        where_non_zero_voltage = np.where(actu_vect != 0)
        if len(where_non_zero_voltage[0]) == 0:
            return np.zeros((self.dim_overpad_pupil, self.dim_overpad_pupil))

        # opd is in nanometer
        # DM_pushact is in opd nanometer
        opd_to_phase = 2 * np.pi * 1e-9 / self.wavelength_0

        if einstein_sum or len(where_non_zero_voltage[0]) < 3:
            phase_on_DM = np.einsum('i,ijk->jk', actu_vect[where_non_zero_voltage],
                                    self.DM_pushact[where_non_zero_voltage]) * opd_to_phase
        else:
            phase_on_DM = np.zeros((self.dim_overpad_pupil, self.dim_overpad_pupil))
            for i in where_non_zero_voltage[0]:
                phase_on_DM += self.DM_pushact[i, :, :] * actu_vect[i] * opd_to_phase

        return phase_on_DM

    def create_DM_basis(self, basis_type='actuator', silence=False):
        """Create a DM basis. TODO do a zernike basis ?

        AUTHOR: Johan Mazoyer

        Parameters
        ----------
        basis_type : string, default 'actuator'
            the type of basis. 'fourier' or 'actuator'.
        silence : boolean, default False.
            Whether to silence print outputs.

        Returns
        --------
        basis: 2d numpy array
            Basis [Size basis, Number of active act in the DM].
        """
        if basis_type == 'actuator':
            # no need to remove the inactive actuators,
            # they are already removed in pushact
            basis_size = len(self.WhichInPupil)
            basis = np.zeros((basis_size, self.number_act))
            for i in range(basis_size):
                basis[i][self.WhichInPupil[i]] = 1

        elif basis_type == 'fourier':
            start_time = time.time()

            sqrtnbract = int(np.sqrt(self.total_act))

            cossinbasis = phase_ampl.sine_cosine_basis(sqrtnbract)

            basis_size = cossinbasis.shape[0]
            basis = np.zeros((basis_size, self.number_act))

            for i in range(basis_size):
                vec = cossinbasis[i].flatten()[self.active_actuators]
                basis[i] = vec

            # This is a very time consuming part of the code.
            # from N voltage vectors with the sine and cosine value, we go N times through the
            # voltage_to_phase functions. For this reason we save the Fourrier base 2D phases on each DMs
            # in a specific .fits file that is read during the creation of the matrix in
            # wf_control_functions.create_singlewl_interaction_matrix.py

            Name_FourrierBasis_fits = "Fourier_phases_" + self.Name_DM + '_prad' + str(
                self.prad) + '_nact' + str(sqrtnbract) + 'x' + str(sqrtnbract)

            header = fits.Header()
            header.insert(0, ('date_mat', datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "fourier basis creation date"))

            necessary_dm_param = dict()
            for key in self.DMconfig:
                if (self.Name_DM in key) and ('error' not in key) and ('misregistration' not in key) and ('z_position'
                                                                                                          not in key):
                    header[key] = self.DMconfig[key]
                    necessary_dm_param[key] = self.DMconfig[key]
            necessary_dm_param["prad"] = self.prad
            necessary_dm_param["dim_overpad_pupil"] = self.dim_overpad_pupil
            header = from_param_to_header(necessary_dm_param, header)

            # Loading any existing matrix and comparing their headers to make sure they are created
            # using the same set of parameters
            bool_already_existing_matrix = False
            if os.path.exists(os.path.join(self.Model_local_dir, Name_FourrierBasis_fits + ".fits")):
                header_existing = fits.getheader(os.path.join(self.Model_local_dir, Name_FourrierBasis_fits + ".fits"))
                # remove the basic kw created automatically  when saving the fits file
                for keyw in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3']:
                    header_existing.remove(keyw)
                # we comapre the header (ignoring the date)
                bool_already_existing_matrix = fits.HeaderDiff(header_existing, header,
                                                               ignore_keywords=['DATE_MAT']).identical

            if not bool_already_existing_matrix:
                start_time = time.time()
                phasesFourrier = np.zeros((basis_size, self.dim_overpad_pupil, self.dim_overpad_pupil))
                if not silence:
                    print("Start " + Name_FourrierBasis_fits + " (wait a few 10s of seconds)")
                for i in range(basis_size):
                    phasesFourrier[i] = self.voltage_to_phase(basis[i])
                    if i % 10:
                        progress(i, basis_size, status='')
                fits.writeto(os.path.join(self.Model_local_dir, Name_FourrierBasis_fits + '.fits'),
                             phasesFourrier,
                             header,
                             overwrite=True)
                if not silence:
                    print("")
                    print("Time for " + Name_FourrierBasis_fits + " (s):", round(time.time() - start_time))
            else:
                if not silence:
                    print(Name_FourrierBasis_fits + ".fits file already exists.")

        else:
            raise ValueError(basis_type + " is is not a valid basis_type")

        return basis


def generic_actuator_position(Nact1D, pitchDM, diam_pup_in_m, diam_pup_in_pix):
    """Create a grid of position of actuators for generic  DM. The DM will then
    be automatically defined as squared with Nact1D x Nact1D actuators and the
    pupil centered on this DM.

    We need N_act1D > diam_pup_in_m / DM_pitch, so that the DM is larger than the pupil.

    at the end compare to the result of actuator_position in case of DM2, it
    should be relatively close. If we can, we should try that actu 0 is
    relatively at the same pos. Test with huge DM pitch (2 actus in the pup)

    AUTHOR: Johan Mazoyer

    Parameters
    ----------
    Nact1D : int
        Number of actuators of a square DM in one of the principal direction.
    pitchDM: float
        Pitch of the DM (distance between actuators), in meter.
    diam_pup_in_m : float
        Diameter of the pupil in meter.
    diam_pup_in_pix : int
        Diameter of the pupil in pixel.

    Returns
    --------
    simu_grid : 2D array of shape is 2 x Nb_actuator
        x and y positions of each actuator for simulation.
    """

    if Nact1D * pitchDM < diam_pup_in_m:
        raise ValueError("Nact1D*pitchDM < diam_pup_in_m: The DM is smaller than the pupil")

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

        # if Nact1D is even, the center of the DM is in between 4 actuators
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
            pos_actu_in_pix[:, i] = pos_actu_in_pix[:, i] - pos_actuhalfactfromcenter + halfactfromcenter + center_pup
            pos_actu_in_pix[0, i] *= -1
    return pos_actu_in_pix
