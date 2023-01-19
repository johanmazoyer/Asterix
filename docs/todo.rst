..  _todo-label:

Contributing
---------------

If you want to develop for Asterix, you might need packages to test and build the documentation. Run:

.. code-block:: bash
    
    $  pip install -e '.[dev,docs]'

To contribute to Asterix, please follow the following steps:
    1. Make sure your local `master` branch is up-to-date by pulling.
    2. Create a new branch off `master` with a name of your choice and commit your work.
    3. When done, open a PR and request a review after ensuring your branch is up-to-date with the base branch you're merging into (usually `master`) and after running the pytests and the linter locally as follows. To run the pytests:

    .. code-block:: bash

        $ pytest

    To run the flake8 linter:

    .. code-block:: bash

        $ flake8 . --max-line-length=121 --count --statistics


    4. Iterate on the review, once it's approved it will be immediately merged.

**Documentation**

Please update any affected documentation when working on a PR. Docstrings must follow the `numpydoc guidelines <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
To check your updates, you can locally generate the documentation before pushing to master:

.. code-block:: bash

        $ cd docs/
        $ make html

General guidelines:
    * Do not touch other people's branches.
    * Do not touch Draft PRs.
    * If you approve a PR, you can immediately merge it.

Releases
----------

Since Asterix is the baseline simulator for the THD2 testbed, we need to ensure that there is always a version of the
package available that is capable of producing interaction matrices that successfully create
dark holes on the testbed. Such versions are marked by releases.

The requirements for a release are that such a matrix can be generated without changing any parameters (just using the
ones provided in a savef parameter file), and the necessary files are generated without errors. Further, this needs to
be possible for matrices for full as well as half dark holes, both for the FQPM and the WV coronagraph. To ensure this,
each release requires a set of both simulation tests as well as tests on hardware.

To create a new release, follow these steps:
    1. Create a branch off of master starting with "release", for example "release-v0.0.0"
    2. Open a PR of your branch against master. This will will load the release PR template that lists all necessary tests
    3. Work through the necessary test and documentation steps - you will need to work on the THD2 testbed for this, and might have to switch from one coronagraph to another
    4. Once all points are checked, request a review and follow the standard review process (see top of this page)

You can check the release requirements from the template in this file:  
https://github.com/johanmazoyer/Asterix/blob/add_pr_template_releases/.github/pull_request_templates/release.md

You will need to check the raw file to see all comments.

The parameter files for THD2 testbed configurations are:
    - `thd2_setups/fqpm_parameters.ini` for the FPQM coronagraph
    - `thd2_setups/wrapped_vortex_parameters.ini` for the wrapped vortex coronagraph

Remember that if you make changes to your code (or the parameter file), your tests become invalid and you have to start from scratch.
This means it is recommended that once you start a release you push through it without changing anything else unless it
is absolutely necessary. Otherwise, just leave other issues for a future PR.

Envisioned improvements
-------------------------

These are future improvements to Asterix currenly envisionned (not particularly by order of priority):

- add a parameter to do an offset en between PSF and detector (in lambda /D): during matrix measurement and a different one during correction 
- add a parameter to do an offset en between corono and detector (in lambda /D): during matrix measurement and a different one during correction 
- COFFEE estimation (curently underway)
- add coronagraphs (apodizations for APLC and Vortex)
- SCC estimation (on hold, not priority and required a lot of changes to the code core)
- add tools for quick DH analysis (off axis PSF for throughput measurement)

If you want to participate please contact us ! 

**To Check**: These are part of the code that need to be particularly checked and tested :

- tools to rotate pupil / apod / Lyot have not been properly checked
 
**To Discuss**: These are part of the code that should be discussed between the authors:

- Should we remove the intial FP field G0 in all cases when we measure the interaction matrices ?
- normalisation of amplitude map. Currenlty std have been set to 0.1
