.. _run-asterix-label:

Basic Asterix Tutorial with THD2 model
--------------------------------------

To run Asterix you first need a parameter .ini file describing all the testbed configuration. An exmaple of this file 
is provided in the packages Example_param_file.ini. If you wish to run a correction with this example configuration 
with the THD2 testbed, just run:

.. code-block:: python

    import os
    from Asterix import Main_THD
    Asterixroot = os.path.dirname(os.path.realpath(__file__))
    Main_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini')

Please avoid running Asterix directly in the Asterix install directory to avoid saving .fits files everywhere.
The first parameter of the parameter file is ``Data_dir``. By default it is set to '.' which means it will save the files
in the directory you currently are when you run this code. A good practice is therefore to create your own parameter file by
copying Example_param_file.ini in another directory and create you own calling python file which call Main_THD.runthd2.

.. code-block:: python

    from Asterix import Main_THD
    Main_THD.runthd2(path_to_my_param_file + 'my_param_file.ini')

To run several simulation with slightly different parameters, you can override one of the parameters directly from the main. 
For example:

.. code-block:: python

    from Asterix import Main_THD
    Main_THD.runthd2(path_to_my_param_file + 'my_param_file.ini'
        NewCoronaconfig={"corona_type" : 'fqpm'})

will overide the current corona_type parameter and replace it with "fqpm", leaving all other parameter unchanged


When you run Asterix, the first time several directories will be created:
* Model_local/ This directory contains all .fits file that are essentials to the correction but that can be measured, even if it can take a long time
* Interaction_Matrices/ is the directoy where Matrices are savec, both for estimation (e.g. pair wise) and correction (e.g. Interaction Matrices)
* Results/ where it will save the results of your correction. The code will automatically create a directory in Results/Name_Experiment/ where 'Name_Experiment' is a parameter in the .ini file
* Labview/ Finally, if the parameter 'onbench' is set to True, the code will create a directory to put matrices in the format needed to control the THD2 testbed. 


