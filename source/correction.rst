.. _correction-label:

Correction
---------------

This section describes how to correct the electrical field in focal plane in Asterix. Several correction modes 
are possible in Asterix. Files can be found in :ref:`correctionfiles-label`;

- an initialization (e.g. Jacobian matrix) ``Corrector.__init__`` : The initialization requires previous initialization of the testbed and of the estimator.
- a matrix update function ``Corrector.update_matrices`` This function is called once during initalization abd then each time we need to recompute the Jacobian in the middle of the correction using usign differents DM voltages as starting point.
- a correction function ``Corrector.toDM_voltage`` which takes the results of an estimation and returns the DM Voltage. 

Dark Hole Mask Definition
+++++++++++++++++++++++++++++++

The estimation estimates the focal plane electrical field in a large zone larger than the
correction zone achievable by the DMs. We can reduce the focal plane correction using binary mask

``MaskDH`` is a very small class to do all the mask related stuff: retrieve parameters and measure the mask 
and measure the string to save matrices.

.. code-block:: python
    
    from Asterix.wfsc import MaskDH
    # testbed is previously defined

    Correctionconfig = config["Correctionconfig"]
    mask_dh = MaskDH(Correctionconfig) # read the configuration
    science_mask_dh = mask_dh.creatingMaskDH(testbed.dimScience, testbed.Science_sampling, **kwargs) # create a mask at a given size and resolution


                                            
Several shape are possible for the DH using the parameter ``DH_shape``:

- "square" DH. Size can be defined using the position of the corners in :math:`{\lambda}`/ D with the parameter parameter ``corner_pos``: [xmin, xmax, ymin, ymax], with 0 beeing the star position. ``DH_side`` parameter is not used and the symetry of the DH is only set using the corners positions.
- "circle" DH Size can be defined using parameters 
    - ``Sep_Min_Max`` : 2 element array [iwa, owa] inner and outer working angle of the dark hole
    - ``circ_offset`` : if half DH, we remove separations closer than circ_offset (in :math:`{\lambda}`/ D) from the DH 
    - ``circ_angle`` : if half DH, we remove the angles closer than circ_angle (in degrees) from the DH 
    - ``DH_side`` : can be set to "top", "bottom", "left", "right" and "full" to create full and half dark hole.
- "noDH" DH. In this mode, the mask is just 1 everywhere. 

``creatingMaskDH()`` function has been set up with a mode where each optical plane is save to .fits for debugging purposes.
To use this options use keywords ``save_all_planes_to_fits = True`` and set up the directory ``dir_save_all_planes``.

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

    #initalize the corrector
    correc = Corrector(Correctionconfig,
                       testbed,
                       mask_dh,
                       estimator)


Once you have initialized, you can update the matrix during the correction wihtout re-initializing using : 

.. code-block:: python
    
    corrector.update_matrices(testbed,
                              estimator,
                              initial_DM_voltage=initial_DM_voltage,
                              input_wavefront=1.)


This can be useful if the strokes are too high and makes the algorithm not as efficient. 


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
Same parameters as efc

**Steepest** : 
Same parameters as efc

Correction loop
+++++++++++++++++++++++++++++++

``correction_loop.py`` contains 3 functions. The first one is ``correction_loop_1matrix()`` which is a for loop repeated
``Number_matrix`` , which update the Interference Matrix and run the ``correction_loop_1matrix()`` at each iteration.


The ``correction_loop_1matrix()`` function is a loop running ``Nbiter_corr`` times that is basically doing:

* estimation

* correction

* application on DM and measure of DM

The results are stored in a dictionnary then sent to ``save_loop_results()`` for ploting and saving in the folder
named '/Results/Name_experiement' where ``Name_Experiment`` is a parameter. All .fits saved have all parameters in the header. 
The config (with updated parameters) is also saved in a .ini file, so you can run the same experiment. 
