
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


Main_THD.py
-----------------
.. automodule:: Asterix.Main_THD
    :members:

.. callgraph:: Asterix.Main_THD.runthd2
    :toctree: api
    :zoomable:
    :direction: horizontal


correction_loop.py
-----------------
.. automodule:: Asterix.correction_loop
    :members:

.. callgraph:: Asterix.correction_loop.CorrectionLoop
    :toctree: api
    :zoomable:
    :direction: horizontal


Optical_System_functions.py
-----------------

OpticalSystem: main class
++++++++++++++++++++++++++
.. _os-label:
.. autoclass:: Asterix.Optical_System_functions.OpticalSystem
    :members:
    :show-inheritance:


OpticalSystem: pupil subclass
++++++++++++++++++++++++++
.. _pupil-label:
.. autoclass:: Asterix.Optical_System_functions.Pupil
    :members:
    :show-inheritance:

OpticalSystem: coronagraph subclass
++++++++++++++++++++++++++
.. _coronagraph-label:
.. autoclass:: Asterix.Optical_System_functions.Coronagraph
    :members:
    :show-inheritance:

.. callgraph:: Asterix.Optical_System_functions.Coronagraph.__init__
    :toctree: api
    :zoomable:
    :direction: horizontal

OpticalSystem: DeformableMirror subclass
++++++++++++++++++++++++++
.. _deformable-mirror-label:
.. autoclass:: Asterix.Optical_System_functions.DeformableMirror
    :members:
    :show-inheritance:

.. callgraph:: Asterix.Optical_System_functions.DeformableMirror.__init__
    :toctree: api
    :zoomable:
    :direction: horizontal

OpticalSystem: Testbed subclass
++++++++++++++++++++++++++
.. _testbed-label:
.. autoclass:: Asterix.Optical_System_functions.Testbed
    :members:
    :show-inheritance:


WSC_functions.py
-----------------
.. automodule:: Asterix.WSC_functions
    :members:


estimator.py
-----------------
.. _estimationfiles-label:
.. automodule:: Asterix.estimator
    :members:

.. callgraph:: Asterix.estimator.Estimator.__init__
    :toctree: api
    :zoomable:
    :direction: horizontal

.. callgraph:: Asterix.estimator.Estimator.estimate
    :toctree: api
    :zoomable:
    :direction: horizontal


corrector.py
-----------------
.. _correctionfiles-label:
.. automodule:: Asterix.corrector
    :members:

.. callgraph:: Asterix.corrector.Corrector.update_matrices 
    :toctree: api
    :zoomable:
    :direction: horizontal

.. callgraph:: Asterix.corrector.Corrector.toDM_voltage
    :toctree: api
    :zoomable:
    :direction: horizontal


MaskDH.py
-----------------
.. automodule:: Asterix.MaskDH
    :members:

propagation_functions.py
-----------------
.. automodule:: Asterix.propagation_functions
    :members:

phase_amplitude_functions.py
-----------------
.. automodule:: Asterix.phase_amplitude_functions
    :members:

processing_functions.py
-----------------
.. automodule:: Asterix.processing_functions
    :members:


fits_functions.py
-----------------
.. automodule:: Asterix.fits_functions
    :members:



.. toctree::
    :maxdepth: 2
    :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
