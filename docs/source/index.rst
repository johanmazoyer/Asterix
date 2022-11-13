
Asterix: Simulate your High-Contrast Instruments
===================================

Asterix is a python based library for simulating high contrast instruments and testbeds with a strong
focus on focal plane wavefront sensing and correction algorithms. 

Asterix is publicaly available on `GitHub <https://github.com/johanmazoyer/Asterix>`_ and all contributions are welcome!

The development of Asterix is led by Johan Mazoyer with major contributions Axel Potier, Iva Laginja, 
and Raphaël Galicher from LESIA (Paris Observatory)

.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    install


.. toctree::
    :maxdepth: 1
    :caption: Basic Usage

    run_asterix
    create_my_testbed
    estimation
    correction


.. toctree::
    :maxdepth: 1
    :caption: Ideas for Asterix improvements
    
    todo

.. toctree::
   :maxdepth: 1
   :caption: Asterix Package

   modules_docs/main_thd
   modules_docs/correction_loop



Correction loop
-----------------


HCI metric plots
-----------------
.. automodule:: Asterix.utils.hci_metrics
    :members:
    :show-inheritance:


Coronagraphis instrument simulation
------------------------------------

OpticalSystem: main class
++++++++++++++++++++++++++
.. _os-label:
.. autoclass:: Asterix.optics.OpticalSystem
    :members:
    :show-inheritance:


OpticalSystem: pupil subclass
++++++++++++++++++++++++++++++
.. _pupil-label:
.. autoclass:: Asterix.optics.Pupil
    :members:
    :show-inheritance:

OpticalSystem: coronagraph subclass
+++++++++++++++++++++++++++++++++++++
.. _coronagraph-label:
.. autoclass:: Asterix.optics.Coronagraph
    :members:
    :show-inheritance:

.. callgraph:: Asterix.optics.coronagraph.Coronagraph.__init__
    :toctree: api
    :zoomable:
    :direction: horizontal

OpticalSystem: DeformableMirror subclass
++++++++++++++++++++++++++++++++++++++++++
.. _deformable-mirror-label:
.. autoclass:: Asterix.optics.DeformableMirror
    :members:
    :show-inheritance:

.. callgraph:: Asterix.optics.deformable_mirror.DeformableMirror.__init__
    :toctree: api
    :zoomable:
    :direction: horizontal

OpticalSystem: Testbed subclass
+++++++++++++++++++++++++++++++++
.. _testbed-label:
.. autoclass:: Asterix.optics.Testbed
    :members:
    :show-inheritance:


Propagation functions
++++++++++++++++++++++++++
.. automodule:: Asterix.optics.propagation_functions
    :members:

Phase and amplitude functions
+++++++++++++++++++++++++++++++++
.. automodule:: Asterix.optics.phase_amplitude_functions
    :members:


WF Estimation
-----------------
.. _estimationfiles-label:
.. automodule:: Asterix.wfsc.estimator
    :members:

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


WF Correction
-----------------
.. _correctionfiles-label:
.. automodule:: Asterix.wfsc.corrector
    :members:

.. automodule:: Asterix.wfsc.wf_control_functions
    :members:

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


DH Mask
--------
.. automodule:: Asterix.wfsc.MaskDH
    :members:

Utils: processing
------------------
.. automodule:: Asterix.utils.processing_functions
    :members:


Utils: Save and read
----------------------
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
