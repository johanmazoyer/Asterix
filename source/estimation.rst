.. _estimation-label:

Estimation
---------------

This section describes how to estimate the electrical field in focal plane in Asterix. Several estimation mode 
are possible in Asterix. Files can be found in :ref:`estimationfiles-label`. 

It contains 2 functions at least:

- an initialization ``Estimator.__init__()`` The initialization will require previous initialization of the testbed (see previous section) and the [Estimationconfig] part of the parameter file.  It set up everything you need for the estimation (e.g. the PW matrix). 

- an estimatation function itself with parameters:
        - the entrance EF
        - DM voltages

It returns the estimation as a 2D complex array. The size in pixel of the output is 
set by the ``Estim_bin_factor`` parameter and is ``dimScience`` / ``Estim_bin_factor``.

.. code-block:: python

    # testbed is previously defined
    Estimationconfig = config["Estimationconfig"]

    myestim = Estimator(Estimationconfig,
                    testbed)
    resultatestimation = myestim.estimate(
                        testbed,
                        voltage_vector=init_voltage,
                        entrance_EF=input_wavefront)



Perfect Estimation
+++++++++++++++++++++++

This is a perfect estimation in focal plane of the electrical field in focal plane. You can use 
this estimation by setting the parameter ``estimation='Perfect'`` before initialization. However, 
this estimation can be also done wihtout initialization or if another estimation have been initialized: 

.. code-block:: python

    # testbed is previously defined
    Estimationconfig = config["Estimationconfig"]    
    
    # we initialize in perfect mode
    Estimationconfig.update({'estimation': "Perfect"})
    myestim = Estimator(Estimationconfig,
                    testbed)
    resultatestimation = myestim.estimate(
                        testbed,
                        voltage_vector=init_voltage,
                        entrance_EF=input_wavefront)
    # this is a perfect FP estimation

    # we re- initialize in pair-wise mode
    Estimationconfig.update({'estimation': "pw"})
    myestim = Estimator(Estimationconfig,
                    testbed)

    resultatestimation = myestim.estimate(
                        testbed,
                        voltage_vector=init_voltage,
                        entrance_EF=input_wavefront)
    # this is a pair-wise FP estimation

    resultatestimation = myestim.estimate(
                    testbed,
                    voltage_vector=init_voltage,
                    entrance_EF=input_wavefront,
                    perfect_estimation=True)
    # this is also a perfect FP estimation, without 
    # re-initializing the estimator

The perfect estimation is exactly equivalent to propagate the light throught the testbed and then
resized to the ``Estim_sampling``: 

.. code-block:: python

    import Asterix.processing_functions as proc
    # testbed is previously defined

    resultatestimation = proc.resizing(testbed.todetector(voltage_vector=init_voltage,
                                    entrance_EF=input_wavefront),myestim.dimEstim) 


All estimators are done this way (first obtains images in the focal plane at the ``Science_sampling`` and 
then resizing) to ensure that the behavior is equivalent to waht would be done on a real testbed

Pair Wise Estimation
+++++++++++++++++++++++

The Pair wise estimation version we used is defined in 
`Potier et al. (2020) <http://adsabs.harvard.edu/abs/2020A%26A...635A.192P>`_ 
The probe used are actuators, which can be chosen using ``posprobes`` parameter. If you choose 
2 random actuators, it can be useful to check the .fits file starting in *EigenValPW* in 
Interaction_Matrices directory. This is the map of the inverse singular values for each 
pixels and it shows if all of the part of the DH are covered by the estimation (see Fig. 4 in Potier et al. 2020).


COFFEE Estimation
+++++++++++++++++++++++
Currenlty not available

SCC Estimation
+++++++++++++++++++++++
Currenlty not available

Polychromatic Estimation
+++++++++++++++++++++++
Currenlty not available