..  _todo-label:

Contributing
---------------

If you want to develop for Asterix, you might need packages to test and build the documentation. Run:

.. code-block:: bash
    
    $  pip install -e '.[dev,docs]'

To contribute to Asterix, please follow the following steps:
    1. Make sure your local `master` branch is up-to-date by pulling.
    2. Create a new branch off `master` with a name of your choice and commit your work.
    3. When done, open a PR and request a review after ensuring your branch is up-to-date with the base branch you're merging into (usually `master`) and after running the pytests and the linter locally. You can then the run the pytests

    .. code-block:: bash

        $ pytest

    or run the flake8 linter:

    .. code-block:: bash

        $ flake8 . --max-line-length=121 --count --statistics


    4. Iterate on the review, once it's approved it will be immediately merged.


Generale guidelines:
    * Do not touch other people's branches.
    * Do not touch Draft PRs.
    * If you approve a PR, you can immediately merge it.

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
