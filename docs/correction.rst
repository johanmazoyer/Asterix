..  _correction-label:

Correction
---------------

This section describes how to correct the electrical field in the focal plane in Asterix. Several correction modes
are possible in Asterix:

- an initialization (e.g. Jacobian matrix) ``Corrector.__init__`` : The initialization requires previous initialization of the testbed and of the estimator.
- a matrix update function ``Corrector.update_matrices`` This function is called once during initialization and then each time we need to recompute the Jacobian in the middle of the correction using different DM voltages as the starting point. It can also be used to update the dark-hole mask.
- a correction function ``Corrector.toDM_voltage`` which takes the results of an estimation and returns the DM voltages.

Dark Hole Mask Definition
+++++++++++++++++++++++++++++++

The estimation estimates the focal plane electrical field in a large zone larger than the
correction zone achievable by the DMs. We can reduce the focal plane correction using a binary mask.

``MaskDH`` is a very small class to do all the mask related stuff: retrieve parameters, measure the mask
and measure the string to save matrices.

.. code-block:: python
    
    from Asterix.wfsc import MaskDH
    # testbed is previously defined

    Correctionconfig = config["Correctionconfig"]
    mask_dh = MaskDH(Correctionconfig) # read the configuration
    science_mask_dh = mask_dh.creatingMaskDH(testbed.dimScience, testbed.Science_sampling, **kwargs) # create a mask with a given size and resolution


                                            
Several shapes are possible for the DH using the input parameter ``DH_shape``:

- "square" DH. Size can be defined using the position of the corners in :math:`{\lambda}`/ D with the parameter ``corner_pos``: [xmin, xmax, ymin, ymax], with 0 being the star position. ``DH_side`` parameter is not used and the symmetry of the DH is only set using the corners positions.
- "circle" DH Size can be defined using the parameters:

    - ``Sep_Min_Max`` : 2 element array [iwa, owa], inner and outer working angle of the dark hole
    - ``circ_offset`` : if half DH, we remove separations closer than circ_offset (in :math:`{\lambda}`/ D) from the DH 
    - ``circ_angle`` : if half DH, we remove the angles closer than circ_angle (in degrees) from the DH 
    - ``DH_side`` : can be set to "top", "bottom", "left", "right" and "full" to create full and half dark hole.
- "noDH" DH. In this mode, the mask is just 1 everywhere. 

``creatingMaskDH()`` function has been set up with a mode where each optical plane is saved to a .fits file for debugging purposes.
To use this option, set up the keyword ``dir_save_all_planes`` to an existing path directory.

Additional details can be found directly in :ref:`the code documentation <dh-label>`.

Interaction Matrix
+++++++++++++++++++++++++++++++

Most correction algorithms requires the measurement of an Interaction Matrix.
The specific function doing this is located in WSC_functions.py : create_interaction_matrix()

We  save the matrix in .fits independently for each DMs so that you do not have to recalculate if you go 
from 1 to 2 DMs (provided that this is the only change in the testbed of course).

The Matrix is not limited to the DH size but to the whole FP [dimEstim, dimEstim]. 
First half is real part, second half is imag part. The matrix size is therefore [total(DM.basis_size), 2*dimEstim^2]

Finally we crop the matrix to the DHmask size in a second part. This save time because we do not need to recalculate
the DH for another mask size or direction. 

This code works for all testbeds without prior assumption (if we have at least 1 DM of course). We have optimized 
the code to only propagate once through optical elements before the activated DM and repeat only what is after 
the activated DM.

There are two main parameters for this part: 

- ``DM_basis`` define the actuator basis that you use to describe the DM movement when building the matrix. It can currenlty takes 2 modes: "actuator" (all actuators are pushed one after another) and "fourier",  where we use sine and cosine. Finally, there is a ``amplitudeEFC`` parameter in "actuator" mode which set the level to which you can push the actuators.
- ``MatrixType`` is defining the type of estimation we do to measure the FP for each DM movement. It can currenlty takes 2 modes: "Perfect" (we assume a perfect estimator) and "SmallPhase" (we make a small phase assumption in the matrix. This is the main EFC mode). A mode to use the actual estimator (a type of empirical matrix, as is currently done for SCC for example) will be implemented later).


The Matrix calculation is done during initialization:

.. code-block:: python

    from Asterix.wfsc import MaskDH, Corrector
    # testbed is previously defined
    # estimator is previously defined

    Correctionconfig = config["Correctionconfig"]

    mask_dh = MaskDH(Correctionconfig)
    maskEstim = mask_dh.creatingMaskDH(estimator.dimEstim, estimator.Estim_sampling)
    
    # Initialize the corrector
    corrector = Corrector(Correctionconfig,
                          thd2,
                          estimator.dimEstim,
                          maskEstim=maskEstim)


Once you have initialized, you can update the matrix during the correction wihtout re-initializing using : 

.. code-block:: python
    
    corrector.update_matrices(testbed,
                              initial_DM_voltage=some_DM_voltage,
                              input_wavefront=some_input_wavefront)

This can be useful to recalculate the jacobian around a non zero DM voltage or if you want to crop the matrix with another dark-hole:

.. code-block:: python

    mask_dh2 = MaskDH(Correctionconfig)
    maskEstim2 = mask_dh2.creatingMaskDH(estimator.dimEstim, estimator.Estim_sampling)
    
    corrector.update_matrices(testbed,
                              maskEstim=maskEstim2)


Correction mode
+++++++++++++++++++++++++++++++

The several correction modes have been developped in Asterix, most of which are described in th review paper  
`Groff et al. (2016) <https://ui.adsabs.harvard.edu//#abs/2016JATIS...2a1009G/abstract>`_ and 
`Potier et al. (2020) (PhD, in French)  <https://tel.archives-ouvertes.fr/tel-03065844>`_. You can choose the method
using the ``correction_algorithm`` parameter. Currently : 'efc', 'sm', 'steepest' and 'em' are supported. 


**Electrical Field Conugation (EFC)**:

Most used method on Asterix. It is a optimizes Singular Value Decomposition, for which you can choose several parameters.

- ``regularization`` parameter ('tikhonov', 'truncation') on the way you can smooth or not the truncation of the modes.
- ``Nbmodes_OnTestbed`` is the number of mode that will be used for the inverse matrix for the THD2 testbed, in the Labviw directory
- ``gain`` is the gain of the loop in EFC
- ``Nbiter_corr`` number of iterations in each loop. Can be a single integer or a list of integer
- ``Nbmode_corr`` number of EFC modes. Can be a single integer or a list of integer. If this is a list, it must be of the same size than ``Nbiter_corr``
- ``Linesearch`` : boolean. If TRue the algorithm test a few inversion modes at each iteration and take the ones that minimize the contrast the most. Very time consuming

**Stroke Minimization (SM)**: 
This is specifically the optimized Stroke Minimization described in `Mazoyer et al. (2018) <http://adsabs.harvard.edu/abs/2018AJ....155....7M>`_.
No parameters except ``Nbiter_corr`` : number of iterations in each loop.

**Energy Minimization (EM)**: 
Same parameters as efc. Does not currently work in polychromatic correction.

**Steepest** : 
Same parameters as efc. Does not currently work in polychromatic correction.

Additional details can be found directly in :ref:`the code documentation <correctionfiles-label>`.

Polychromatic Correction
+++++++++++++++++++++++++++++++

Polychromatic estimation and correction are linked so they are both driven by a single parameter 
in the ``[Estimationconfig]`` :ref:`section <polychromaticestim-label>`.

Correction loop
+++++++++++++++++++++++++++++++

``correction_loop.py`` contains 3 functions. The first one is ``correction_loop_1matrix()`` which is a for loop repeated
``Number_matrix`` of times , which updates the interaction matrix and runs ``correction_loop_1matrix()`` in each iteration.


The ``correction_loop_1matrix()`` function is a loop running ``Nbiter_corr`` times. For each iteration, the following steps are done:

* estimation
* correction
* application on DM and measurement of focal plane.

The results are stored in a dictionary and then sent to ``save_loop_results()`` for plotting and saving in the folder
named '/Results/timestamp-Name_experiement' where ``Name_Experiment`` is a parameter from the configuration file. All saved .fits files have all parameters in their headers.
The config (with updated parameters) is also saved in a .ini file, so you can run the same experiment again at a later time.

Additional details can be found directly in :ref:`the code documentation <correction-loop-label>`.
