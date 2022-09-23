
Asterix
===================================

Introduction
-----------------------
Asterix is a python based library for simulating high contrast instruments and testbeds with a strong
focus on focal plane wavefront sensing and correction algorithms. 

Asterix is publicaly available on `GitHub <https://github.com/johanmazoyer/Asterix>`_ and all contributions are welcome!

The development of Asterix is led by Johan Mazoyer with major contributions Axel Potier, Iva Laginja, 
and Raphaël Galicher from LESIA (Paris Observatory)


Setup
--------
.. toctree::
    :maxdepth: 3

    install



Basic Usage
----------------

.. toctree::
    :maxdepth: 3

    run_asterix
    create_my_testbed
    estimation
    correction

Ideas for Asterix improvements
---------------------------------
.. toctree::
    :maxdepth: 3
    
    todo


Annex: Asterix Functions
==================


main_THD
-----------------
.. automodule:: Asterix.main_THD
    :members:

.. callgraph:: Asterix.main_THD.runthd2
    :toctree: api
    :zoomable:
    :direction: horizontal


correction_loop
-----------------
.. automodule:: Asterix.loop.correction_loop
    :members:
    :show-inheritance:

.. automodule:: Asterix.loop.save_and_read
    :members:
    :show-inheritance:


.. callgraph:: Asterix.loop.correction_loop.correction_loop
    :toctree: api
    :zoomable:
    :direction: horizontal

optical_systems
-----------------

OpticalSystem: main class
++++++++++++++++++++++++++
.. _os-label:
.. autoclass:: Asterix.optical_systems.OpticalSystem
    :members:
    :show-inheritance:


OpticalSystem: pupil subclass
++++++++++++++++++++++++++
.. _pupil-label:
.. autoclass:: Asterix.optical_systems.Pupil
    :members:
    :show-inheritance:

OpticalSystem: coronagraph subclass
++++++++++++++++++++++++++
.. _coronagraph-label:
.. autoclass:: Asterix.optical_systems.Coronagraph
    :members:
    :show-inheritance:

.. callgraph:: Asterix.optical_systems.coronagraph.Coronagraph.__init__
    :toctree: api
    :zoomable:
    :direction: horizontal

OpticalSystem: DeformableMirror subclass
++++++++++++++++++++++++++
.. _deformable-mirror-label:
.. autoclass:: Asterix.optical_systems.DeformableMirror
    :members:
    :show-inheritance:

.. callgraph:: Asterix.optical_systems.deformable_mirror.DeformableMirror.__init__
    :toctree: api
    :zoomable:
    :direction: horizontal

OpticalSystem: Testbed subclass
++++++++++++++++++++++++++
.. _testbed-label:
.. autoclass:: Asterix.optical_systems.Testbed
    :members:
    :show-inheritance:



estimator
-----------------
.. _estimationfiles-label:
.. automodule:: Asterix.wfsc.estimator
    :members:

.. _estimationfiles-label:
.. automodule:: Asterix.wfsc.wf_sensing_functions
    :members:

.. callgraph:: Asterix.wfsc.estimator.Estimator.__init__
    :toctree: api
    :zoomable:
    :direction: horizontal

.. callgraph:: Asterix.wfsc.estimator.Estimator.estimate
    :toctree: api
    :zoomable:
    :direction: horizontal


corrector.py
-----------------
.. _estimationfiles-label:
.. automodule:: Asterix.wfsc.corrector
    :members:

.. _estimationfiles-label:
.. automodule:: Asterix.wfsc.wf_control_functions
    :members:

.. _estimationfiles-label:
.. automodule:: Asterix.wfsc.thd_quick_invert
    :members:

.. callgraph:: Asterix.wfsc.corrector.Corrector.update_matrices 
    :toctree: api
    :zoomable:
    :direction: horizontal

.. callgraph:: Asterix.wfsc.corrector.Corrector.toDM_voltage
    :toctree: api
    :zoomable:
    :direction: horizontal


MaskDH.py
-----------------
.. automodule:: Asterix.wfsc.MaskDH
    :members:

propagation_functions.py
-----------------
.. automodule:: Asterix.optics.propagation_functions
    :members:

phase_amplitude_functions.py
-----------------
.. automodule:: Asterix.optics.phase_amplitude_functions
    :members:

processing_functions.py
-----------------
.. automodule:: Asterix.utils.processing_functions
    :members:


save_and_read.py
-----------------
.. automodule:: Asterix.utils.save_and_read
    :members:



.. toctree::
    :maxdepth: 2
    :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
