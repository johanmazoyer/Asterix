.. _todo-label:

**To Do**: These are future improvements to Asterix currenly envisionned (not particularly by order of priority):

- add a parameter to do an offset en between PSF and detector (in lambda /D): during matrix measurement and a different one during correction 
- add a parameter to do an offset en between corono and detector (in lambda /D): during matrix measurement and a different one during correction 
- add tools to rotate pupil. Save the rotated pupil. add a angle parameter. during matrix measurement (not necessary to add a different one during correction)
- add tools to rotate lyot. Save the rotated pupil. add a angle parameter. during matrix measurement (not necessary to add a different one during correction)
- polychromatic correction (concatenation of matrices at multiple wavelengths)
- polychromatic estimation for current estimator
- COFFEE estimation (curently underway)
- add coronagraphs (apodizations for APLC and Vortex)
- SCC estimation (on hold, not priority and required a lot of changes to the code core)
- add tools for quick DH analysis (Contrast curves, off axis PSF for throughput measurement)

If you want to participate please contact us ! 


**To Check**: These are part of the code that need to be particularly checked and tested :

- the vortex coronagraph
- the way I do Fourrier Basis currently vs on the testbed (elegant MFT way + Saving it because it's too long + Using the number of actu in the DM direction and not in the pupil, which is smaller and decentered + is it the same way it is done on the testbed)

**To Discuss**: These are part of the code that should be discussed between the authors:

- Should we remove the intial FP field G0 in all cases when we measure the interaction matrices ?
- discuss thisloop_expected_iteration_number.

