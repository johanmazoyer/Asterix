..  _todo-label:

Open issues
---------------

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
