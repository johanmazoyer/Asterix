.. _create-my-testbed-label:

Create and modify testbeds
---------------

Optical System
+++++++++++++++++++++++

Asterix have been thought from the beginning to be able to easily adapt to new configurations of the testbed 
wihtout major changes. This modularity is based on the ``Asterix.Optical_System_functions.Optical_System`` class.
An Optical_System is a part of the testbed which stars and ends in a pupil plane, which allows them to be easily 
concatenated. To be able to be well concatenated, they must all be using the same general parameters (physical 
and numerical parameters). These parameters are stored in the first part of the parameter file, common to 
all ``Optical_System``: [modelconfig]. 
Among those parameters, you have the size in pixels of all pupils. This is the size of the entrance pupil, which
will set up all other dimensions. The pupil planes is slightly overpadded compared to this radius because 
some ``Optical_System`` require it. By convention, all pupil planes are centered bewteen the 4 central pixels. 


Parameter file can be read using ``Asterix.fits_functions.read_parameter_file`` function. We can create a generic
``Optical_System`` like this. 

.. code-block:: python
    
    import Asterix.fits_functions as useful
    import Asterix.Optical_System_functions as OptSy

    config = useful.read_parameter_file(parameter_file)
    modelconfig = config["modelconfig"]

    generic_os = OptSy.Optical_System(modelconfig)


For the moment, this ``generic_os`` does not do anything, it's like an empty pupil plane. 

Each ``Optical_System`` has an attribute function ``EF_through`` which describes the effect this Optical_System has 
on the electrical field when going through it. Once this function is defined, all Optical_System have access to 
a library of functions, common to all ``Optical_System`` : ``todetector`` (electrical field in the next focal plane), 
``todetector_Intensity`` (Intensity in the next focal plane), ``transmission`` (measure ratio of photons lost 
when crossing the system), etc.

.. code-block:: python
    
    exit_EF = generic_os.EF_through() # electrical field after the system 
                                    # (by default, entrance field is 1.)
    EF_FP = generic_os.todetector() # electrical field in the next focal plane
    PSF = generic_os.todetector_Intensity() #  Intensity in the next focal plane


Finally, for all optical system, you can use generic functions like to creaet and manipulate phase and amplitude screens

.. code-block:: python
    
    SIMUconfig = config["SIMUconfig"] #parameters for phase aberrations

    phase_abb_up = generic_os.generate_phase_aberr(SIMUconfig)
    input_wavefront = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_up)

    PSF = generic_os.todetector_Intensity(entrance_EF = input_wavefront) #  Intensity for this aberrations



In the next section, we will present the existing ``Optical_System``s (``Optical_System.pupil``, 
``Optical_System.coronagraph``, ``Optical_System.deformable_mirror``) and the 
way to concatenate them easily to create an ``Optical_System.Testbed``. 

Finally, ``Optical_System`` have been set up with a mode where each optical plane is save to .fits for debugging purposes.
This can generate a lot of fits especially if in a loop so be careful. 
To use this options use keywords ``save_all_planes_to_fits = True`` and set up the directory ``dir_save_all_planes``

Function documentation can be found in Section :ref:`os-label`. 

Pupil
+++++++++++++++++++++++

``Optical_System.pupil`` is the most simple type of ``Optical_System``. It initializes and describes the behavior 
of single pupil pupil is a sub class of ``Optical_System``. Obviously you can define your pupil without that 
with 2d arrray multiplication (this is a fairly simple object). The main advantage of defining them using 
``Optical_System`` is that you can use default ``Optical_System`` functions to obtain PSF, transmission, etc...
and concatenate them with other elements. 

.. code-block:: python
    
    import Asterix.fits_functions as useful
    import Asterix.Optical_System_functions as OptSy

    config = useful.read_parameter_file(parameter_file)
    modelconfig = config["modelconfig"]

    pup_round = OptSy.pupil(modelconfig)

    # Because this is an Optical_System, you can access attribute functions: 
    
    exit_EF = pup_round.EF_through() # electrical field after the system 
                                    #(by default, entrance field is 1.)
    EF_FP = pup_round.todetector() # electrical field in the next focal plane
    PSF = pup_round.todetector_Intensity() #  Intensity in the next focal plane


You can define a different radius than the pupil one in the parameter file

.. code-block:: python

    pup_round = OptSy.pupil(modelconfig, prad = 43)

Some specific aperture types are defined that you can access using the keyword ``PupType``

.. code-block:: python

    pup_roman = OptSy.pupil(modelconfig, PupType = "RomanPup")

Currently supported ``PupType`` are : "RoundPup", "CleanPlane" (empty pupil plane), "RomanPup", "RomanLyot".

You can finally defined your own pupils from a .fits using the keyword ``filename``. In this case, you have to 
manually set up the pupil prad, by definition, it will assume the same size as entrance pupil in the parameter file.

Function documentation can be found in Section :ref:`pupil-label`. 


Coronagraph
+++++++++++++++++++++++

``Optical_System.coronagraph`` is a sub class of ``Optical_System`` which initializes and describes the behavior 
of a coronagraph system (from apodization plane at the entrance of the coronagraph to the Lyot plane). Function
documentation can be found in Section :ref:`coronagraph-label`. 


.. code-block:: python
    
    import Asterix.fits_functions as useful
    import Asterix.Optical_System_functions as OptSy

    config = useful.read_parameter_file(parameter_file)
    modelconfig = config["modelconfig"]
    Coronaconfig = config["Coronaconfig"]

    corono = OptSy.coronagraph(modelconfig, Coronaconfig)
    
    exit_EF = corono.EF_through() # electrical field after the system 
                                    #(by default, entrance field is 1.)
    EF_FP = corono.todetector() # electrical field in the next focal plane
    PSF = corono.todetector_Intensity() #  Intensity in the next focal plane

Type of coronagraph can be changed with ``corona_type`` parameter.  Currently supported ``corona_type`` 
are 'fqpm' or 'knife', 'classiclyot' or 'HLC'. Focal plane functions are automatically normalized in contrast
by default. For details about the way to normalize in polychromatic light, see ``measure_normalization`` 
and ``todetector_Intensity`` documention in :ref:`os-label`


Deformable Mirror
+++++++++++++++++++++++

``Optical_System.coronagraph`` is a subclass of ``Optical_System`` which initializes and describes the behavior 
of a deformable mirror (DM) system. 


.. code-block:: python
    
    import Asterix.fits_functions as useful
    import Asterix.Optical_System_functions as OptSy

    config = useful.read_parameter_file(parameter_file)
    modelconfig = config["modelconfig"]
    DMconfig = config["DMconfig"]

    DM1 = OptSy.deformable_mirror(modelconfig,
                                    DMconfig,
                                    Name_DM='DM1',
                                    Model_local_dir=Model_local_dir)

For the moment, Asterix does not allows you to rapidly create generic DMs, and the DMs are modelled to the 2 DMs
currently used on the THD2 testbed. We will soon allow the definition a "generic" DM with only parameter the number 
of actuators accross the pupil.

Function documentation can be found in Section :ref:`deformable-mirror-label`. 


Concatenate your Optical_Systems
++++++++++++++++++++++++++++++++++++++++++++++

This is a particular subclass of Optical System, because we do not know what is inside
It can only be initialized by giving a list of Optical Systems and it will create a
"testbed" with contains all the Optical Systems and associated EF_through functions.

.. code-block:: python
    
    import Asterix.fits_functions as useful
    import Asterix.Optical_System_functions as OptSy

    config = useful.read_parameter_file(parameter_file)
    modelconfig = config["modelconfig"]
    Coronaconfig = config["Coronaconfig"]
    DMconfig = config["DMconfig"]

    pup_round = OptSy.pupil(modelconfig)

    DM34act = OptSy.deformable_mirror(modelconfig,
                                    DMconfig,
                                    Name_DM='DM1',
                                    Model_local_dir=Model_local_dir)

    DM32act = OptSy.deformable_mirror(modelconfig,
                                    DMconfig,
                                    Name_DM='DM3',
                                    Model_local_dir=Model_local_dir)

    corono = OptSy.coronagraph(modelconfig, Coronaconfig)
    # and then just concatenate
    testbed = OptSy.Testbed([pup_round, DM34act, DM32act, corono],
                            ["entrancepupil", "DM1", "DM3", "corono"])



The whole point of this system is that it can be easily changed. For example, we can add another DM32act DM
just like that:

.. code-block:: python

    testbed = OptSy.Testbed([pup_round, DM34act, DM32act, DM32act, corono],
                        ["entrancepupil", "DM1", "DM3", "DM4", "corono"])


or a specific pupil in the entrance plane of the coronagraph (e.g. like the Roman configuration).

.. code-block:: python

    pup_roman = OptSy.pupil(modelconfig, PupType = "RomanPup")
    testbed = OptSy.Testbed([pup_round, DM34act, DM32act,pup_roman, corono],
                                ["entrancepupil", "DM1", "DM3", "romanpupil" , "corono"])
    


Function documentation can be found in Section :ref:`testbed-label`. 