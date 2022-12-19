.. _install-label:


Conda envs
--------------------------

You can of course install Asterix in your own python environment. However, to avoid unpredictable 
conflicts with other packages, we recommend the creation of a specific environment first. 
If you are already familiar with conda environments you can skip this section. 

Conda is an open source package management system and environment management system. It quickly 
installs, runs and updates packages and their dependencies on your local computer and allows 
switches between environments.

By creating clean python environments for each of your projects (especially packages in continuous 
development by non developers like Asterix), you minimize the risk of of creating conflicts which 
will hinder the use of Asterx and/or on your other projects.

First download and install miniconda3:
https://docs.conda.io/en/latest/miniconda.html

You can use the very useful `Conda Cheat Sheet <https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf>`_
which lists the most common conda command lines you can use.
 
Install Asterix
-----------------

First ``cd`` in the directory where you want to download Asterix. To install the most up to date 
developer version, clone the Asterix repository :

.. code-block:: bash

    $ git clone https://github.com/johanmazoyer/Asterix.git

This clones the repository using HTTPS authentication. Once the repository is cloned onto your computer, ``cd Asterix`` into it. 
Then type:

.. code-block:: bash

    $ conda env create --file environment.yml

This will automatically create a python environement named ``asterix`` with only the required python packages for Asterix, at their
latest stable version. Before installing Asterix and everytime you want to use it, you need to activate this environement:

.. code-block:: bash

    $ conda activate asterix

Run the setup file to install Asterix. Due to the continually developing nature of Asterix, you should always use the current version of the code on
`GitHub <https://github.com/johanmazoyer/Asterix>`_, keep it updated frequently and install in developer's mode, allowing easy update when pulled:

.. code-block:: bash
    
    $ pip install -e '.'


Dependencies
-------------
The installation of Asterix requires the commmon python packages, documented in the setup.py file, which are useful for most astronomical 
data analysis. They will be automatically installed in the setup. 

We recommend you use a version of Python > 3.10 to use Asterix. As Asterix can be computationally expensive, we recommend a 
powerful computer to optimize the correction. This will depend on specific test you want to perform (number of deformable mirrors, 
monochromatic or polychromatic correction, etc.).

