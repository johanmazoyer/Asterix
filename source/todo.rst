.. _todo-label:

**To Do**: These are future improvements to Asterix currenly envisionned (not particularly by order of priority):

- polychromatic correction (concatenation of matrices at multiple wavelengths)
- polychromatic estimation for current estimator
- COFFEE estimation (curently underway)
- add coronagraphs (apodizations for APLC and Vortex, Roman SPC)
- SCC estimation
- add tools for quick DH analysis (Contrast curves, off axis PSF for throughput measurement)

If you want to participate please contact us ! 


**To Check**: These are part of the code that need to be particularly checked and tested :

- the Fresnel transform to see if it really does what we think it does in all cases
- the vortex coronagraph
- "generic DM" tool to create DM with centered sqaure any number of actuators




**To Discuss**: These are part of the code that should be discussed between the authors:

- Should we remove the intial FP field G0 in all cases when we measure the interaction matrices ?
- discuss thisloop_expected_iteration_number.
- the way i do the basis 
- the way I do Fourrier Basis currently vs on the testbed (elegant MFT way + Saving it because it's too long + Using the number of actu in the DM direction and not in the pupil, which is smaller and decentered + is it the same way it is done on the testbed)
