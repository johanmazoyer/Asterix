..  _run-asterix-label:

Basic Asterix Tutorial with THD2 model
-----------------------------------------------

This sections is if you want to use the code as it currently set up, in the THD2 configuration.
To run Asterix you first need a parameter .ini file describing all the testbed configuration and a python file to call the main. 
An example of these files are provided in the packages : Example_param_file.ini and example_run_asterix.py

Run Asterix in THD2 mode
+++++++++++++++++++++++++++++++++

If you wish to run a correction with this example configuration 
with the THD2 testbed, just run:

.. code-block:: python

    import os
    from Asterix import main_THD, Asterix_root
    main_THD.runthd2(os.path.join(Asterix_root, 'Example_param_file.ini'))

Please avoid running Asterix directly in the Asterix install directory to avoid saving .fits files everywhere.
The first parameter of the parameter file is ``Data_dir``. By default it is set to '.' which means no target directory
is set and outputs will be saved to some pre-defined default locations. There are several options for this:

1. Create your own parameter file by copying ``Example_param_file.ini`` into another directory and to create
your own python file which calls ``main_THD.runthd2()``.

2. Leave the parameter file entry ``Data_dir`` in whichever file you are reading set to '.', which will look for a
directory called ``asterix_data`` in your home directory and save all data there. If the code can't find it, it will create one.

3. You can create the environment variable ``ASTERIX_DATA_PATH`` on your local machine which will define the top-level
directory to which all your data outputs from Asterix will be saved.

.. code-block:: python

    from Asterix import main_THD
    main_THD.runthd2(os.path.join(path_to_my_param_file, 'my_param_file.ini'))

To run several simulation with slightly different parameters, you can override one of the parameters directly from the main. 
For example:

.. code-block:: python

    from Asterix import main_THD
    main_THD.runthd2(os.path.join(path_to_my_param_file, 'my_param_file.ini'),
                     NewCoronaconfig={"corona_type" : 'fqpm'})

will override the current corona_type parameter and replace it with "fqpm", leaving all other parameters unchanged.

When you run Asterix, the first time several directories will be created:

* Model_local/ contains all .fits files that are essentials to the correction but that can be measured, even if it can take a long time.

* Interaction_Matrices/ is the directory where Matrices are saved, both for estimation (e.g. pair wise) and correction (e.g. Interaction Matrices).

* Results/ where it will save the results of your correction. The code will automatically create a directory in Results/Name_Experiment/ where 'Name_Experiment' is a parameter in the .ini file and will be preceded by a time stamp.

* Labview/ Finally, if the parameter 'onbench' is set to True, the code will create a directory to put matrices in the format needed to control the THD2 testbed. 


As with a lot of functions in Asterix, ``runthd2()`` has been set up with a mode where each optical plane is saved to .fits file for debugging purposes.
This can generate a lot of fits especially if in a loop so use with caution. To use this option, set up the keyword ``dir_save_all_planes`` to an existing path directory.
What is saved if you activate this option in runthd2() was carefully thougth about, to avoid sending thousands of .fits files:

* Dark Hole Mask
* PW Estimate at each iteration.
* 1 full run through testbed (~17 fits for 1 DM testbed -one before and after each planes more or less-) at each iteration.

What is not saved by default, but can be easily done by setting up the keyword ``dir_save_all_planes`` to an existing path directory, individually for these functions:
* PW Matrix (nb_probes*17 fits for 1 DM testbed)
* EFC Matrix (nb_actuator*17 fits for 1 DM testbed)
* PW difference (nb_probes x17 fits for 1 DM testbed), at each iteration


Description of the parameter file
+++++++++++++++++++++++++++++++++++++++++

General configuration
~~~~~~~~~~~~~~~~~~~~~~
Data_dir: string path to directory to save matrices and results.

onbench: bool, On bench or numerical simulation ? If true this will save a matrix 
to be applied on the THD2 testbed

Then there are several part of the confiration file, corresponding to several part of the testbed or correction. 


[modelconfig]
~~~~~~~~~~~~~~~~~~~~~~
First the [modelconfig] section defines the most important parameters that are used by all other subsystems. 
This also defines the also the entrance pupil plane of the testbed.


    - wavelength_0 : float, central wavelength (in meters)

    - Delta_wav : float, Spectral band (in meters)

    - nb_wav : int, Number of monochromatic images in the spectral band (must be odd number). Ignored if Delta_wav = 0

    - dimScience : int, detector science image size (in pixels)

    - Science_sampling : float, Sampling in the detector science image lambda/Entrance_pupil_diameter in pixel
    
    - diam_pup_in_m : float,pupil diameter (in meters)

    - diam_pup_in_pix : int, pupil diameter (in pixels)

    - overpadding_pupilplane_factor : overpadding pupil plane factor if 2: the pupil of diameter 2*diam_pup_in_pix is in a 2*overpadding_pupilplane_factor*diam_pup_in_pix array

    - filename_instr_pup : Instrument entrance pupil definition. Several keywords are already defined :
                                - "Clear" for no pupil at all (clear plane)
                                - "RoundPup" for a round pupil of radius diam_pup_in_m
                                - "RomanPup" for HLC Roman Pupil on THD
                                - "RomanPupTHD2" for HLC Roman Pupil on THD (rotated by -0.9 degrees))

    or you can use this parameter to put an existing full path .fits name that will be used to define the pupil (e.g. filename_instr_pup = "/myfullpath/roman_pup_500pix_center4pixels.fits"). The pupil in the .fits file are assumed to be at the dimesion of the pupil (no overpadding) and will automatically be rescaled at prad.
    
    - entrance_pup_rotation : if the pupil is not clear or round, you can rotate the pupil using this parameter. Angle in degrees in counter-clockwise direction. The rotated pupil will be used for matrix and correction. This will save the rotated pupil in the Model_local/ directory. 


[DMconfig]
~~~~~~~~~~~~~~~~~~~~~~
The [DMconfig] section define the parameter for DMs

    - MinimumSurfaceRatioInThePupil : minimum ratio of energy of the influence function inside the pupil wrt to energy of the influence function so that the actuator are included into the basis. The lowest the more actuators are considered

Parameters that have to be define for each DMs:
    - DM#_active : bool, Switch on/off DM#

    - DM#_z_position : float, in meter, distance from the pupil in a collimated beam

    - DM#_filename_actu_infl_fct: string, filename of the actuator influence function (inside Model directory)
    
    - DM#_Generic: bool, in the case of a generic DM (DM#_Generic =True), we need only one more parameter to define the DM: the number of actuator N_act1D in one of its principal direction. We need N_act1D > diam_pup_in_m / DM_pitch, so that the DM is larger than the pupil. The DM will then be automatically defined as squared with N_act1DxN_act1D actuators and the puil centered on this DM. careful this not change the  DM#_pitch and the aperture diameter. If you want to have more actuators in the pupil, you migh want to change those as well.

If DM#_Generic = True :

    - DM#_pitch: float, in meter, pitch of the DM (distance between actuators)

    - DM#_Nact1D : int, number of actuator in one of its principal direction.

If DM#_Generic = False  :

    - DM#_filename_grid_actu : string, filename of the grid of actuator positions in unit of pupil diameter with (0,0)=center of the pupil.     # This fits must have PITCHV and PITCHH param in the header

    - DM#_filename_active_actu : string, filename to put if not all actuators are active as in the case of circular DM on THD2

Misregistration parameters:   
    - DM#_misregistration : bool, if true, use difference between testbed model use to create matrix and the one use for correction

If DM#_misregistration = True :
    - DM#_xerror: float, x-direction misalignement in actuator pitch between matrix measurement and correction 

    - DM#_yerror: float, y-direction misalignement in actuator pitch between matrix measurement and correction 

    - DM#_angerror: float, rotation misalignement in degree between matrix measurement and correction 

    - DM#_gausserror : float, influence function size error between matrix measurement and correction (1=100% error)


[Coronaconfig]
~~~~~~~~~~~~~~~~~~~~~~
The [Coronaconfig] section contains the coronagraph parameter.

    - filename_instr_apod : string, Apodisation pupil definition (pupil after the DMS at the entrance of the coronagraph in Roman). Several keywords are already defined :
                                - "Clear" for no apodizer at all (clear plane): this is the case in THD2
                                - "RoundPup" for a round pupil of radius diam_pup_in_m
                                - "RomanPup" for HLC Roman Pupil on THD
                                - "RomanPupTHD2" for HLC Roman Pupil on THD (rotated by -0.9 degrees)

    or you can use this parameter to put an existing full path .fits name that will be used to define the pupil (e.g. filename_instr_pup = "/myfullpath/roman_pup_500pix_center4pixels.fits"). The pupil in the .fits file are assumed to be at the dimesion of the pupil (no overpadding) and will automatically be rescaled at prad. If you want this pupil to be smaller than the entrance pupil you have to overpad your .fits file.
    
    - apod_pup_rotation : float, if the pupil is not clear or round, you can rotate the pupil using this parameter. Angle in degrees in counter-clockwise direction. The rotated pupil will be used for matrix and correction. This will save the rotated pupil in the Model_local/ directory. 


    - filename_instr_lyot : string, Lyot pupil definition (pupil after the DMS at the entrance of the coronagraph in Roman). Several keywords are already defined :
                                - "Clear" for no Lyot pupil at all (clear plane)
                                - "RoundPup" for a round pupil of radius diam_pup_in_m
                                - "RomanLyot" for HLC Roman Pupil
                                - "RomanLyotTHD2" for HLC Roman Lyot on THD (rescaled because of the lyot plane dezoom and rotated by -0.9 degrees)
                                
    or you can use this parameter to put an existing full path .fits name that will be used to define the pupil (e.g. filename_instr_pup = "/myfullpath/roman_pup_500pix_center4pixels.fits"). The pupil in the .fits file are assumed to be at the dimesion of the pupil (no overpadding) and will automatically be rescaled at the pupil radius. If you want this pupil to be smaller than the entrance pupil you have to overpad your .fits file.
    
    - lyot_pup_rotation : float, if the pupil is not clear or round, you can rotate the pupil using this parameter. Angle in degrees in counter-clockwise direction. The rotated pupil will be used for matrix and correction. This will save the rotated pupil in the Model_local/ directory. 

    - diam_lyot_in_m : flaot, lyot diameter (in meters). Only use in the case of a RoundPup Lyot stop (filename_instr_lyot = "RoundPup"). Value for THD2 clear Lyot is 8.035mm = 8.1*0.097 (rayon Lyot * de-zoom entrance pupil plane / Lyopt plane)

    - corona_type: Can be fqpm or knife, vortex, lassiclyot or HLC

If knife coronagraph:
    - knife_coro_position: string, where light passes ('left', 'right', 'top', 'bottom')
    - knife_coro_offset : float, offset of the knife in lambda/pupil diameter

If classiclyot or HLC:
    - rad_lyot_fpm: float radius of the classical Lyot FPM in lambda/pupil diameter

If HLC :
    - transmission_fpm: float, we define the transmission in intensity at vawelength0
    - phase_fpm: float, phase shift at vawelength0

If FQPM:
    - err_fqpm = 0 : float, phase error on the pi phase-shift (in rad)
    - achrom_fqpm : bool, if True, Achromatic FQPM, else pi*lamda0/lamda

If Vortex :
    - vortex_charge : even int, charge of the vortex



[Estimationconfig]
~~~~~~~~~~~~~~~~~~~~~~
The [Estimationconfig] section contains the estimator parameters. An estimator is the thing that measure something you want to correct. 

    - estimation: string, FP WF sensing : 'Perfect' or 'pw'

    - Estim_bin_factor : int, We bin the estimation images used for PW / perfect estim by this factor. this way dimEstim = dimScience / Estim_bin_factor and  Estim_sampling = Science_sampling / Estim_bin_factor. Be careful, this raise an error if Estim_sampling < 3

If estimation = 'PW':
    - amplitudePW : float, Amplitude of PW probes (in nm)

    - posprobes : list of int, Actuators used for PW (DM in pupil plane)

    - cut : float, Threshold to remove pixels with bad estimation of the electric field


[Correctionconfig]
~~~~~~~~~~~~~~~~~~~~~~
The [Correctionconfig] section contains the corrector parameters. An estimator receive an estimation and send DM command to correct for it.

    
    - DH_shape :  string, "circle", "square" or "noDH" (all FP is corrected, depending on the DM(s) size).  Not case sensitive

If DH_shape == 'square':
    - corner_pos = list of float 2.7,11.7,-11.7,11.7 [xmin, xmax, ymin, ymax] Position of the corners of the DH in lambda/Entrance_pupil_diameter

If DH_shape == 'circle':
    - DH_side : string, "Full", "Left", "Right", "Top", "Bottom" to correct one side of the fp. Not case sensitive

    - Sep_Min_Max = 3.5,10 : circle inner and outer radii of the circle DH size in lambda/D

    - circ_offset: float, if circ_side != "Full", remove separation closer than circ_offset (in lambda/Entrance_pupil_diameter)
    - circ_angle : float, if circ_side != "Full", we remove the angles closer than circ_angle (in degrees) from the DH 

Matrix parameters:
    - DM_basis : string, Actuator basis. Currently 'fourier' or 'actuator'. Same parameter for all DMs. Not case sensitive

    - MatrixType : string, Type of matrix : Either 'Perfect' Matrix (exp(i.(phi_DM+phi))) or a 'SmallPhase' aberration matrix (phi_DM.exp(i.phi)). Not totally sure what change. Not case sensitive

    - correction_algorithm: 'efc' for Electric Field Conjugation, 'em' for Energy Minimization, 'sm' for Stroke Minimization, or 'steepest'. Not case sensitive

If EFC :
    - amplitudeEFC float, 
    - regularization: string, regularization when truncated modes in the inversion 'truncation' or 'tikhonov'

if  onbench=True   
    - Nbmodes_OnTestbed : int, number of mode for the inversion


[Loopconfig]
~~~~~~~~~~~~~~~~~~~~~~
Configuration of the loop. The loop is an estimation and a correction which send a command to the DM
    
    - Number_matrix : int>1, Number of time we recompute the Interraction Matrix
    
    - Nbiter_corr: integer or a list of integers, number of iterations in each loop. if you want several iterations with different mode ex: 2,3,2

    - Nbmode_corr :  integer or a list of integers, EFC modes !! Must be of the same size than Nbiter_corr !! ex 330, 340, 350 

    - gain: float, between 0 and 1, EFC correction gain
    
    - Linesearch : bool, if true, the code will find the best EFC modes for each iteration in Nbiter_corr (Nbmode_corr is not used in this case). The best modes is chosen in a list automatically selected depending on hte Number of modes of the system



[SIMUconfig]
~~~~~~~~~~~~~~~~~~~~~~
Finally the last parameter section is dependent on the experiement you are launching. Aberrations, noise, etc

    - Name_Experiment : string use to save the results
    
Amplitude aberrations:

    - set_amplitude_abb: bool if true, add Amplitude aberrations
    - set_random_ampl : Bool. If true we generate a new amplitude map each time. Else, we load the one in ampl_abb_filename
    - ampl_abb_filename : if 'Amplitudebanc_200pix_center4pixels' take the amplitude of the testbed. If set_random_ampl = False and ampl_abb_filename = '', we take the last generated map of amplitude aberration
    
if set_random_ampl = True
    - ampl_rms : float, amount in % in amplitude (not intensity) (between 0 and 100)
    - ampl_rhoc : float, parameter to multiply the power. See Bordé et al. 2006.
    - ampl_slope : float, power slope of the amplitude aberration

    
Upstream phase aberrations:

    - set_UPphase_abb : bool if true, add phase aberrations in the entrance pupil plane
    - set_UPrandom_phase : Bool. If true we generate a new phase map each time. Else, we load the one in UPphase_abb_filename
    - UPphase_abb_filename : string, Load a phase map with this fits name. If ampl_abb_filename = 'Amplitude_THD2' we load the THD2 amplitude map. If set_random_ampl = False and ampl_abb_filename = '', we take the last generated map of amplitude aberrations.
    

if set_UPrandom_phase = True:
    - UPopd_rms: float phase rms  in meter
    - UPphase_rhoc: parameter to multiply the power. See Bordé et al. 2006.
    - UPphase_slope power slope of the up phase aberration
    
Downstream phase aberrations:

    - set_DOphase_abb : bool if true, add phase aberrations in the Lyot pupil plane
    
    - set_DOrandom_phase : Bool. If true we generate a new phase map each time. Else, we load the one in DOphase_abb_filename
    
    - DOphase_abb_filename : string, Load a phase map with this fits name. If set_random_ampl = False and ampl_abb_filename = '', we take the last generated map of amplitude aberration
    

if set_DOrandom_phase = True:
    - DOopd_rms: float phase rms  in meter
    - DOphase_rhoc: parameter to multiply the power. See Bordé et al. 2006.
    - DOphase_slope power slope of the up phase aberration

Photon Noise:

    - nb_photons : float, number of photon entering the telescope. If 0, no photon noise


