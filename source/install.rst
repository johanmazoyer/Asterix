.. _install-label:


Set up a Conda environment
--------------------------

You can of course install Asterix in your own python environment. However, to avoid unpredictable 
conflicts with other packages, we recommend the creation of a specific environment first. 
If you are already familiar with conda environments you can skip this section. 

Conda is an open source package management system and environment management system. It quickly 
installs, runs and updates packages and their dependencies on your local computer and allows 
switches between environments.

By creating clean python environments for each you projects (especially packages in continuous 
development by non developers like Asterix), you minimize the risk of of creating conflicts which 
will hinder the prevent the use of Asterx and/or on your other projects.

First download and install miniconda3:
https://docs.conda.io/en/latest/miniconda.html

You can now create an environement for installing Asterix:

    $ conda create --name asterix-env python=3.8 numpy scipy astropy matplotlib configobj

This will automatically create a python environement with only the required python packages for Asterix, at their
latest stable version. Before installing Asterix and everytime you want to use it you need to activate this environement:

    $ conda activate asterix-env


You can use the very useful `Conda Cheat Sheet <https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf>`_
which lists the most common conda command lines you can use.
 

Install Asterix
-----------------

Due to the continually developing nature of Asterix, you should always use the current version of the code on
`GitHub <https://github.com/johanmazoyer/Asterix>`_ and keep it updated frequently. 

First ``cd`` in the directory where you want to download Asterix. To install the most up to date 
developer version, clone the Asterix repository :

    $ git clone https://github.com/johanmazoyer/Asterix.git

This clones the repository using HTTPS authentication. Once the repository is cloned onto your computer, ``cd Asterix`` into it.
If you are using a specific environement for Asterix (see previous section), now is the time to Activate it:
    
    $ conda activate asterix-env

Run the setup file:

    $ python setup.py develop

If you use multiple versions/environements of python, you will need to run ``setup.py`` with each version of python
(this should not apply to most people).




Dependencies
-------------
The installation of Asterix requires the following packages, which are useful for most astronomical data analysis. They will be automatically 
installed in the setup. 

* numpy
* scipy
* scikit-image
* astropy
* matplotlib
* configobj

We recommend yuo use a version of Python > 3.5 to use Asterix. As Asterix can be computationally expensive, we recommend a 
powerful computer to optimize the correction. This will depend on specific test you want to perform (number of deformable mirrors, 
monochromatic or polychromatic correctin, etc.).

